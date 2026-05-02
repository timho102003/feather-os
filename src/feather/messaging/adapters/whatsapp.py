"""WhatsApp Business Cloud API adapter (webhook).

Reference (verified 2026-04-29 against Meta for Developers docs):

- GET handshake: ``hub.mode=subscribe``, ``hub.verify_token``,
  ``hub.challenge``. Echo the challenge with status 200 if the verify
  token matches; 403 otherwise.
- POST signature: ``X-Hub-Signature-256: sha256=<hex>``. Compute
  ``HMAC-SHA256(raw_body, app_secret).hexdigest()`` and compare.
- Inbound payload: ``entry[].changes[].value.messages[]``. Each message
  has ``id`` (WAMID), ``from`` (E.164), ``timestamp`` (string seconds),
  ``type`` (we only handle ``text``), ``text.body``. Status events live
  under ``value.statuses[]`` and are ignored by the router.
- Send: ``POST https://graph.facebook.com/<version>/<phone_number_id>/messages``
  with ``Authorization: Bearer <access_token>`` and the
  ``messaging_product=whatsapp`` JSON shape.
- 24-hour customer service window: replies inside the window can be
  free-form text; outside requires templates. Since our adapter only
  responds to a just-arrived message, we are always inside the window.
- Idempotency: Meta retries webhook deliveries, so the router dedups on
  ``messaging_inbound_dedup`` keyed by ``id``.
"""

from __future__ import annotations

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

_DEFAULT_PATH = "/whatsapp/webhook"
_DEFAULT_GRAPH_VERSION = "v22.0"
_MAX_TEXT = 4096
_HTTP_TIMEOUT_S = 15.0


class WhatsAppAdapter(BaseMessagingAdapter):
    """WhatsApp Cloud API webhook adapter."""

    platform = Platform.WHATSAPP

    def __init__(
        self,
        service: "MessagingService",
        config: dict[str, Any],
    ) -> None:
        super().__init__(service.router)
        self._service = service
        self._http = service.http_client
        phone_id = str(config.get("phone_number_id", "")).strip()
        token = str(config.get("access_token", "")).strip()
        verify = str(config.get("verify_token", "")).strip()
        app_secret = str(config.get("app_secret", "")).strip()
        if not phone_id:
            raise ValueError("whatsapp: phone_number_id is required")
        if not token:
            raise ValueError("whatsapp: access_token is required")
        if not verify:
            raise ValueError("whatsapp: verify_token is required")
        if not app_secret:
            raise ValueError("whatsapp: app_secret is required")
        self._phone_number_id = phone_id
        self._access_token = token
        self._verify_token = verify
        self._app_secret = app_secret
        self._graph_version = (
            str(config.get("graph_version", _DEFAULT_GRAPH_VERSION)).strip()
            or _DEFAULT_GRAPH_VERSION
        )
        self._webhook_path = (
            str(config.get("webhook_path", _DEFAULT_PATH)).strip()
            or _DEFAULT_PATH
        )

    async def start(self) -> None:
        """Register GET (handshake) + POST (events) handlers."""

        self._set_state(AdapterState.STARTING, detail="registering webhook")
        await self._service.webhook_server.register(
            "GET", self._webhook_path, self._handle_get
        )
        await self._service.webhook_server.register(
            "POST", self._webhook_path, self._handle_post
        )
        url = (
            f"{self._service.webhook_server.base_url}{self._webhook_path}"
        )
        self._set_state(
            AdapterState.RUNNING,
            detail=f"webhook listening on {url}",
        )
        chats = await self._service.store.count_chats_for_platform(
            Platform.WHATSAPP
        )
        self._set_chat_count(chats)

    async def stop(self) -> None:
        self._set_state(AdapterState.STOPPING, detail="removing webhook route")
        await self._service.webhook_server.unregister(
            "POST", self._webhook_path
        )
        await self._service.webhook_server.unregister(
            "GET", self._webhook_path
        )
        self._set_state(AdapterState.DISCONNECTED, detail="stopped")

    async def send_outgoing(self, outgoing: OutgoingMessage) -> None:
        if not outgoing.text:
            return
        url = (
            f"https://graph.facebook.com/{self._graph_version}/"
            f"{self._phone_number_id}/messages"
        )
        for chunk in _chunk_text(outgoing.text, _MAX_TEXT):
            body = {
                "messaging_product": "whatsapp",
                "to": outgoing.chat_id,
                "type": "text",
                "text": {"body": chunk},
            }
            try:
                response = await self._http.post(
                    url,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self._access_token}",
                        "Content-Type": "application/json",
                    },
                    timeout=_HTTP_TIMEOUT_S,
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    "whatsapp.send_network_error err=%s chat=%s",
                    exc,
                    outgoing.chat_id,
                )
                return
            if response.status_code >= 400:
                logger.warning(
                    "whatsapp.send_failed status=%s body=%s",
                    response.status_code,
                    response.text[:200],
                )
                return

    async def _handle_get(self, request: web.Request) -> web.Response:
        """Webhook verification handshake (GET)."""

        params = request.query
        if (
            params.get("hub.mode") == "subscribe"
            and params.get("hub.verify_token") == self._verify_token
        ):
            challenge = params.get("hub.challenge", "")
            return web.Response(status=200, text=challenge)
        return web.Response(status=403, text="forbidden")

    async def _handle_post(self, request: web.Request) -> web.Response:
        """Inbound event payload (POST)."""

        raw = await request.read()
        signature = request.headers.get("X-Hub-Signature-256", "")
        if not _verify_signature(self._app_secret, raw, signature):
            return web.Response(status=403, text="invalid signature")
        try:
            payload = _decode_json(raw)
        except ValueError:
            return web.Response(status=400, text="invalid json")
        if payload.get("object") != "whatsapp_business_account":
            return web.Response(status=200, text="ok")
        for entry in payload.get("entry") or []:
            if not isinstance(entry, dict):
                continue
            for change in entry.get("changes") or []:
                if not isinstance(change, dict):
                    continue
                value = change.get("value")
                if not isinstance(value, dict):
                    continue
                metadata = value.get("metadata") or {}
                # Fail closed: only accept events whose metadata
                # explicitly names our configured phone_number_id
                # (review fix M6). A forged payload omitting the field
                # would otherwise pass through.
                if (
                    not isinstance(metadata, dict)
                    or metadata.get("phone_number_id") != self._phone_number_id
                ):
                    continue
                for message in value.get("messages") or []:
                    if not isinstance(message, dict):
                        continue
                    try:
                        await self._dispatch_message(message)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "whatsapp.dispatch_failed message=%s",
                            message,
                        )
        chats = await self._service.store.count_chats_for_platform(
            Platform.WHATSAPP
        )
        self._set_chat_count(chats)
        return web.Response(status=200, text="ok")

    async def _dispatch_message(self, message: dict[str, Any]) -> None:
        if message.get("type") != "text":
            return
        text_block = message.get("text") or {}
        body = text_block.get("body") if isinstance(text_block, dict) else None
        if not isinstance(body, str) or not body.strip():
            return
        sender = message.get("from")
        if not isinstance(sender, str) or not sender:
            return
        native_id = str(message.get("id") or "")
        ts = message.get("timestamp")
        timestamp_ms: int | None = None
        try:
            if ts is not None:
                timestamp_ms = int(ts) * 1000
        except (TypeError, ValueError):
            timestamp_ms = None
        incoming = IncomingMessage(
            platform=Platform.WHATSAPP,
            chat_id=sender,
            sender_display_name=sender,
            text=body,
            native_message_id=f"whatsapp:{native_id}",
            timestamp_ms=timestamp_ms,
        )
        await self._router.handle_incoming(incoming)


def _verify_signature(secret: str, body: bytes, header_value: str) -> bool:
    if not header_value or not header_value.startswith("sha256="):
        return False
    expected = (
        "sha256="
        + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    )
    return hmac.compare_digest(expected, header_value)


def _decode_json(raw: bytes) -> dict[str, Any]:
    """Parse a JSON object body or raise ValueError.

    Reject non-object payloads (lists, strings, nulls) explicitly so
    downstream handlers can rely on ``payload.get(...)`` (review fix M5).
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


__all__ = ("WhatsAppAdapter",)
