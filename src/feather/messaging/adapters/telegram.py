"""Telegram Bot API adapter using long polling.

Reference (verified 2026-04-29 against ``https://core.telegram.org/bots/api``):

- ``GET /bot<token>/getUpdates?offset=&timeout=&limit=&allowed_updates=``
  with ``offset = max(update_id) + 1``. Holds the connection up to
  ``timeout`` seconds; we use 25 s with a 35 s client read timeout.
- ``POST /bot<token>/sendMessage`` with ``chat_id`` and ``text``. Max
  text length 4096; HTTP 429 returns ``parameters.retry_after`` seconds.
- ``GET /bot<token>/getMe`` validates a token (used at connect time).
- ``POST /bot<token>/deleteWebhook`` clears any previously-configured
  webhook so ``getUpdates`` can run.

Long polling does not require a public URL — connecting just needs a
bot token from ``@BotFather``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx

from feather.messaging.base import BaseMessagingAdapter
from feather.messaging.models import (
    AdapterState,
    IncomingMessage,
    OutgoingMessage,
    Platform,
)

if TYPE_CHECKING:
    from feather.messaging.service import MessagingService

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.telegram.org"
_TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]+$")
_MAX_TEXT = 4096
_POLL_TIMEOUT_S = 25
_HTTP_READ_TIMEOUT_S = 35.0
_HTTP_CONNECT_TIMEOUT_S = 10.0
_BACKOFF_INITIAL_S = 1.0
_BACKOFF_MAX_S = 60.0
_ALLOWED_CHAT_TYPES = {"private", "group", "supergroup"}


class TelegramAdapter(BaseMessagingAdapter):
    """Telegram bot adapter (long polling)."""

    platform = Platform.TELEGRAM

    def __init__(
        self,
        service: "MessagingService",
        config: dict[str, Any],
    ) -> None:
        super().__init__(service.router)
        self._service = service
        self._http = service.http_client
        token = str(config.get("bot_token", "")).strip()
        if not token:
            raise ValueError("telegram: bot_token is required")
        if not _TOKEN_RE.match(token):
            raise ValueError(
                "telegram: bot_token must look like '<digits>:<token>' "
                "(see @BotFather)"
            )
        self._token = token
        self._bot_username = str(config.get("bot_username", "")).strip()
        self._poll_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._offset: int | None = None

    @property
    def _api_base(self) -> str:
        return f"{_BASE_URL}/bot{self._token}"

    def _redact(self, value: object) -> str:
        """Return ``value`` as a string with the bot token scrubbed.

        Telegram's ``httpx`` errors include the request URL, which
        contains the token. Logging or re-raising that string would
        leak the credential to the on-screen TUI conversation, the
        clipboard transcript, or ``.feather/logs/feather.log`` (review
        fix C1).
        """

        text = str(value)
        return text.replace(self._token, "<redacted-token>")

    async def start(self) -> None:
        """Validate the token and start the polling loop.

        Any failure after the polling task is created cleans the task up
        before re-raising — without this, ``MessagingService.connect``
        would leak an orphan task hammering Telegram with the in-memory
        token (review fix C2).
        """

        self._set_state(AdapterState.STARTING, detail="validating token")
        try:
            me = await self._call("getMe", method="GET")
        except httpx.HTTPError as exc:
            redacted = self._redact(exc)
            self._set_state(
                AdapterState.ERROR,
                detail="getMe network error",
                last_error=redacted,
            )
            raise ValueError(f"telegram: connect failed: {redacted}") from None
        if not me or not me.get("ok"):
            description = (
                me.get("description") if isinstance(me, dict) else "unknown"
            )
            self._set_state(
                AdapterState.ERROR,
                detail="invalid bot token",
                last_error=str(description),
            )
            raise ValueError(
                f"telegram: getMe rejected token: {description}"
            )
        result = me.get("result", {}) or {}
        self._bot_username = (
            str(result.get("username") or "")
            or self._bot_username
        )

        # Drop any previously-configured webhook; long polling cannot
        # coexist with a registered webhook.
        try:
            await self._call(
                "deleteWebhook",
                method="POST",
                json={"drop_pending_updates": False},
            )
        except httpx.HTTPError as exc:  # pragma: no cover - logged then ignored
            logger.warning(
                "telegram.deleteWebhook_failed err=%s", self._redact(exc)
            )

        self._stop_event.clear()
        self._poll_task = asyncio.create_task(self._poll_forever())
        try:
            chats = await self._service.store.count_chats_for_platform(
                Platform.TELEGRAM
            )
            self._set_chat_count(chats)
        except Exception:
            # Cancel the freshly-spawned polling task so a partial start
            # failure does not leave it orphaned (review fix C2).
            self._stop_event.set()
            task = self._poll_task
            self._poll_task = None
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            raise
        self._set_state(
            AdapterState.RUNNING,
            detail=(
                f"polling as @{self._bot_username}"
                if self._bot_username
                else "polling"
            ),
        )

    async def stop(self) -> None:
        """Cancel the polling task and clear state."""

        self._set_state(AdapterState.STOPPING, detail="shutting down")
        self._stop_event.set()
        task = self._poll_task
        self._poll_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._set_state(AdapterState.DISCONNECTED, detail="stopped")

    async def send_outgoing(self, outgoing: OutgoingMessage) -> None:
        """Send the assistant's reply to the originating chat."""

        if not outgoing.text:
            return
        for chunk in _chunk_text(outgoing.text, _MAX_TEXT):
            await self._send_message_with_retry(outgoing.chat_id, chunk)

    async def _send_message_with_retry(
        self, chat_id: str, text: str
    ) -> None:
        """Send one message and honour Telegram's 429 retry_after."""

        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._call(
                    "sendMessage",
                    method="POST",
                    json={"chat_id": chat_id, "text": text},
                    raise_for_status=False,
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    "telegram.sendMessage_network_error chat=%s err=%s attempt=%s",
                    chat_id,
                    self._redact(exc),
                    attempt,
                )
                if attempt >= 3:
                    return
                await asyncio.sleep(min(_BACKOFF_INITIAL_S * 2 ** (attempt - 1), 10.0))
                continue
            if isinstance(response, dict) and response.get("ok"):
                return
            params = (
                response.get("parameters") if isinstance(response, dict) else None
            )
            if isinstance(params, dict) and "retry_after" in params:
                wait = float(params["retry_after"])
                logger.info(
                    "telegram.sendMessage_throttled chat=%s wait=%s",
                    chat_id,
                    wait,
                )
                await asyncio.sleep(min(wait, _BACKOFF_MAX_S))
                continue
            description = (
                response.get("description")
                if isinstance(response, dict)
                else None
            )
            logger.warning(
                "telegram.sendMessage_failed chat=%s error=%s",
                chat_id,
                description,
            )
            return

    async def _poll_forever(self) -> None:
        """Run getUpdates until ``stop`` is called."""

        backoff = _BACKOFF_INITIAL_S
        while not self._stop_event.is_set():
            # Yield to the event loop at every iteration so cancellation
            # is delivered promptly even when all HTTP calls are mocked
            # (e.g. via respx) and would otherwise complete synchronously.
            await asyncio.sleep(0)
            try:
                payload = await self._fetch_updates()
            except asyncio.CancelledError:
                raise
            except httpx.HTTPError as exc:
                logger.warning("telegram.poll_network_err=%s", self._redact(exc))
                await self._sleep_with_stop(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX_S)
                continue
            except Exception as exc:  # noqa: BLE001
                # Review fix m1: any non-network exception (malformed
                # payload, JSON decode error, etc.) used to silently
                # kill the task. Now we log it AND continue polling so
                # one bad payload does not take the bot offline.
                logger.exception(
                    "telegram.poll_unhandled_err=%s",
                    self._redact(exc),
                )
                await self._sleep_with_stop(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX_S)
                continue
            backoff = _BACKOFF_INITIAL_S
            if not isinstance(payload, dict) or not payload.get("ok"):
                description = (
                    payload.get("description")
                    if isinstance(payload, dict)
                    else "unknown"
                )
                logger.warning(
                    "telegram.poll_api_error description=%s", description
                )
                await self._sleep_with_stop(2.0)
                continue
            updates = payload.get("result") or []
            if not isinstance(updates, list):
                continue
            for update in updates:
                if not isinstance(update, dict):
                    continue
                # Dispatch FIRST, advance offset only on success — under
                # cancellation between advance and dispatch the message
                # would otherwise be silently dropped (review fix C5).
                # ``asyncio.shield`` ensures an in-flight dispatch is
                # not torn down mid-handler when stop() is called.
                update_id = update.get("update_id")
                try:
                    await asyncio.shield(self._dispatch_update(update))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "telegram.dispatch_unhandled_err=%s",
                        self._redact(exc),
                    )
                    # Advance offset anyway — re-fetching the same bad
                    # payload would just hit the same exception in a
                    # tight loop. The router's ``messaging_inbound_dedup``
                    # would reject a redelivery either way.
                if isinstance(update_id, int):
                    self._offset = max(
                        self._offset or 0, update_id + 1
                    )

    async def _fetch_updates(self) -> dict[str, Any] | None:
        params: dict[str, Any] = {
            "timeout": _POLL_TIMEOUT_S,
            "limit": 100,
            "allowed_updates": ["message"],
        }
        if self._offset is not None:
            params["offset"] = self._offset
        timeout = httpx.Timeout(
            connect=_HTTP_CONNECT_TIMEOUT_S,
            read=_HTTP_READ_TIMEOUT_S,
            write=10.0,
            pool=10.0,
        )
        response = await self._http.get(
            f"{self._api_base}/getUpdates",
            params=params,
            timeout=timeout,
        )
        if response.status_code == 429:
            retry_after = float(response.headers.get("retry-after", "1") or 1)
            logger.info("telegram.poll_throttled wait=%s", retry_after)
            await asyncio.sleep(min(retry_after, _BACKOFF_MAX_S))
            return None
        response.raise_for_status()
        return response.json()

    async def _dispatch_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            return
        chat = message.get("chat") or {}
        chat_type = chat.get("type")
        if chat_type not in _ALLOWED_CHAT_TYPES:
            return
        chat_id_raw = chat.get("id")
        if chat_id_raw is None:
            return
        chat_id = str(chat_id_raw)
        sender = message.get("from") or {}
        display_name = (
            sender.get("first_name")
            or sender.get("username")
            or chat.get("title")
            or chat_id
        )
        native_id = str(message.get("message_id") or update.get("update_id") or "")
        timestamp_ms: int | None = None
        date = message.get("date")
        if isinstance(date, int):
            timestamp_ms = date * 1000
        incoming = IncomingMessage(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            sender_display_name=str(display_name),
            text=text,
            native_message_id=f"telegram:{native_id}",
            timestamp_ms=timestamp_ms,
        )
        try:
            await self._router.handle_incoming(incoming)
        except Exception:  # noqa: BLE001
            logger.exception(
                "telegram.dispatch_failed chat=%s native=%s",
                chat_id,
                native_id,
            )
        # Refresh chat count opportunistically.
        chats = await self._service.store.count_chats_for_platform(
            Platform.TELEGRAM
        )
        self._set_chat_count(chats)

    async def _sleep_with_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _call(
        self,
        method_name: str,
        *,
        method: str,
        json: dict[str, Any] | None = None,
        raise_for_status: bool = True,
    ) -> dict[str, Any]:
        url = f"{self._api_base}/{method_name}"
        timeout = httpx.Timeout(
            connect=_HTTP_CONNECT_TIMEOUT_S,
            read=15.0,
            write=10.0,
            pool=10.0,
        )
        response = await self._http.request(
            method.upper(),
            url,
            json=json,
            timeout=timeout,
        )
        if raise_for_status:
            response.raise_for_status()
        return response.json()


def _chunk_text(text: str, limit: int) -> list[str]:
    """Split ``text`` so each chunk is ≤ ``limit`` chars, preferring newlines."""

    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


__all__ = ("TelegramAdapter",)
