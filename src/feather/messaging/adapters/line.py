"""LINE Messaging API adapter (webhook + reply / push).

Reference (verified 2026-04-29 against
``https://developers.line.biz/en/reference/messaging-api/``):

- Webhook signature: ``X-Line-Signature: base64( HMAC_SHA256(raw_body,
  channel_secret) )``. Validate against the **raw** body — parsing
  before validation breaks the signature.
- ``POST https://api.line.me/v2/bot/message/reply`` with
  ``replyToken`` + ``messages[]``. Reply token is one-shot, valid for
  ~60 s.
- ``POST https://api.line.me/v2/bot/message/push`` for unsolicited or
  follow-up messages. Used for assistant replies that don't fit in one
  reply call.
- Max text length per message: 2000 characters. Max 5 messages per
  reply / push call.
- Verification button: LINE sends a POST with ``events: []``. Just
  return 200.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from typing import TYPE_CHECKING, Any

import httpx
from aiohttp import web

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

_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
_PUSH_URL = "https://api.line.me/v2/bot/message/push"
_DEFAULT_PATH = "/line/webhook"
_MAX_TEXT = 2000
_MAX_MESSAGES_PER_CALL = 5
_HTTP_TIMEOUT_S = 15.0


class LineAdapter(BaseMessagingAdapter):
    """LINE webhook adapter."""

    platform = Platform.LINE

    def __init__(
        self,
        service: "MessagingService",
        config: dict[str, Any],
    ) -> None:
        super().__init__(service.router)
        self._service = service
        self._http = service.http_client
        secret = str(config.get("channel_secret", "")).strip()
        token = str(config.get("channel_token", "")).strip()
        if not secret:
            raise ValueError("line: channel_secret is required")
        if not token:
            raise ValueError("line: channel_token is required")
        self._channel_secret = secret
        self._channel_token = token
        self._webhook_path = (
            str(config.get("webhook_path", _DEFAULT_PATH)).strip()
            or _DEFAULT_PATH
        )

    async def start(self) -> None:
        """Register the webhook handler with the shared server."""

        self._set_state(AdapterState.STARTING, detail="registering webhook")
        await self._service.webhook_server.register(
            "POST", self._webhook_path, self._handle_request
        )
        url = (
            f"{self._service.webhook_server.base_url}{self._webhook_path}"
        )
        self._set_state(
            AdapterState.RUNNING,
            detail=f"webhook listening on {url}",
        )
        chats = await self._service.store.count_chats_for_platform(
            Platform.LINE
        )
        self._set_chat_count(chats)

    async def stop(self) -> None:
        """Deregister the webhook handler."""

        self._set_state(AdapterState.STOPPING, detail="removing webhook route")
        await self._service.webhook_server.unregister(
            "POST", self._webhook_path
        )
        self._set_state(AdapterState.DISCONNECTED, detail="stopped")

    async def send_outgoing(self, outgoing: OutgoingMessage) -> None:
        """Reply via the reply token if available, push otherwise."""

        if not outgoing.text:
            return
        chunks = _chunk_text(outgoing.text, _MAX_TEXT)
        reply_token = None
        if isinstance(outgoing.reply_context, dict):
            reply_token = outgoing.reply_context.get("reply_token")

        # First batch (up to 5 messages) goes through the reply API
        # while the token is still fresh; remaining batches push.
        first_batch = chunks[:_MAX_MESSAGES_PER_CALL]
        rest = chunks[_MAX_MESSAGES_PER_CALL:]
        if reply_token:
            await self._reply(reply_token, first_batch)
        else:
            await self._push(outgoing.chat_id, first_batch)
        while rest:
            batch = rest[:_MAX_MESSAGES_PER_CALL]
            rest = rest[_MAX_MESSAGES_PER_CALL:]
            await self._push(outgoing.chat_id, batch)

    async def _reply(self, reply_token: str, texts: list[str]) -> None:
        if not texts:
            return
        body = {
            "replyToken": reply_token,
            "messages": [{"type": "text", "text": t} for t in texts],
        }
        await self._post_with_auth(_REPLY_URL, body)

    async def _push(self, recipient: str, texts: list[str]) -> None:
        if not texts:
            return
        body = {
            "to": recipient,
            "messages": [{"type": "text", "text": t} for t in texts],
        }
        await self._post_with_auth(_PUSH_URL, body)

    async def _post_with_auth(
        self, url: str, body: dict[str, Any]
    ) -> None:
        headers = {
            "Authorization": f"Bearer {self._channel_token}",
            "Content-Type": "application/json",
        }
        try:
            response = await self._http.post(
                url,
                json=body,
                headers=headers,
                timeout=_HTTP_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            logger.warning("line.send_network_error url=%s err=%s", url, exc)
            return
        if response.status_code >= 400:
            logger.warning(
                "line.send_failed url=%s status=%s body=%s",
                url,
                response.status_code,
                response.text[:200],
            )

    async def _handle_request(self, request: web.Request) -> web.Response:
        """aiohttp handler — validate signature, dispatch events."""

        raw = await request.read()
        signature = request.headers.get("X-Line-Signature", "")
        if not signature or not _verify_signature(
            self._channel_secret, raw, signature
        ):
            return web.Response(
                status=400,
                text="invalid signature",
            )
        try:
            payload = _decode_json(raw)
        except ValueError:
            return web.Response(status=400, text="invalid json")
        events = payload.get("events") or []
        if not isinstance(events, list):
            return web.Response(status=200, text="ok")
        for event in events:
            if not isinstance(event, dict):
                continue
            try:
                await self._dispatch_event(event)
            except Exception:  # noqa: BLE001
                # Redact the replyToken — it's a 60-second one-shot
                # capability that lets anyone holding it post one
                # message as the bot to the originating user (review
                # fix m11).
                event_id = event.get("webhookEventId") or event.get("type")
                logger.exception(
                    "line.dispatch_failed event_id=%s",
                    event_id,
                )
        chats = await self._service.store.count_chats_for_platform(
            Platform.LINE
        )
        self._set_chat_count(chats)
        return web.Response(status=200, text="ok")

    async def _dispatch_event(self, event: dict[str, Any]) -> None:
        if event.get("type") != "message":
            return
        message = event.get("message") or {}
        if message.get("type") != "text":
            return
        text = message.get("text") or ""
        if not text.strip():
            return
        source = event.get("source") or {}
        if source.get("type") != "user":
            # Group/room support is non-goal this round.
            return
        user_id = source.get("userId")
        if not isinstance(user_id, str) or not user_id:
            return
        native_id = str(
            event.get("webhookEventId") or message.get("id") or ""
        )
        timestamp_ms = (
            int(event["timestamp"])
            if isinstance(event.get("timestamp"), int)
            else None
        )
        reply_context: dict[str, Any] | None = None
        reply_token = event.get("replyToken")
        if isinstance(reply_token, str) and reply_token:
            reply_context = {"reply_token": reply_token}
        incoming = IncomingMessage(
            platform=Platform.LINE,
            chat_id=user_id,
            sender_display_name=user_id,  # LINE webhook does not include name.
            text=text,
            native_message_id=f"line:{native_id}",
            timestamp_ms=timestamp_ms,
            reply_context=reply_context,
        )
        await self._router.handle_incoming(incoming)


def _verify_signature(secret: str, body: bytes, header_value: str) -> bool:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, header_value)


def _decode_json(raw: bytes) -> dict[str, Any]:
    """Parse a JSON object body or raise ValueError.

    Reject non-object payloads (review fix M5).
    """

    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object at the top level")
    return data


def _chunk_text(text: str, limit: int) -> list[str]:
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


__all__ = ("LineAdapter",)
