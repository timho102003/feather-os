"""Integration tests for the WhatsApp Cloud API adapter."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from multidict import MultiDict

from feather.messaging.adapters.whatsapp import WhatsAppAdapter
from feather.messaging.models import IncomingMessage, OutgoingMessage, Platform
from feather.messaging.router import MessagingRouter
from feather.messaging.service import MessagingService
from feather.messaging.store import MessagingStore
from feather.messaging.webhook_server import WebhookServer


_PHONE_ID = "1234567890"
_TOKEN = "EAAToken"
_VERIFY = "feather-verify-token"
_APP_SECRET = "app-secret"


def _sign(body: bytes) -> str:
    digest = hmac.new(_APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


async def _make_adapter(tmp_path: Path):
    store = MessagingStore(tmp_path / "feather.db")
    await store.initialize()
    inbound: list[IncomingMessage] = []

    class _Router(MessagingRouter):
        def __init__(self) -> None:  # noqa: D401 - test double
            self.adapters: dict[Platform, Any] = {}

        def register_adapter(self, p, s):  # type: ignore[override]
            self.adapters[p] = s

        def unregister_adapter(self, p):  # type: ignore[override]
            self.adapters.pop(p, None)

        async def handle_incoming(self, msg):  # type: ignore[override]
            inbound.append(msg)

    router = _Router()  # type: ignore[abstract]
    server = WebhookServer(host="127.0.0.1", port=0)
    http = httpx.AsyncClient()
    service = MessagingService(
        store=store,
        router=router,  # type: ignore[arg-type]
        webhook_server=server,
        http_client=http,
        adapter_factories={
            Platform.WHATSAPP: lambda svc, cfg: WhatsAppAdapter(svc, cfg)
        },
        own_http_client=True,
    )
    adapter = WhatsAppAdapter(
        service,
        {
            "phone_number_id": _PHONE_ID,
            "access_token": _TOKEN,
            "verify_token": _VERIFY,
            "app_secret": _APP_SECRET,
        },
    )
    return adapter, service, inbound


class _FakePayload:
    """Minimal aiohttp payload that supports ``request.read()``."""

    def __init__(self, body: bytes) -> None:
        self._body = body
        self._consumed = False

    async def readany(self) -> bytes:
        if self._consumed:
            return b""
        self._consumed = True
        return self._body

    async def read(self, n: int = -1) -> bytes:
        del n
        if self._consumed:
            return b""
        self._consumed = True
        return self._body

    def at_eof(self) -> bool:
        return self._consumed


def _post_request(body: bytes, signature: str | None) -> web.Request:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if signature is not None:
        headers["X-Hub-Signature-256"] = signature
    return make_mocked_request(
        "POST",
        "/whatsapp/webhook",
        headers=headers,
        payload=_FakePayload(body),
    )


def _wa_inbound_body(text: str = "hello") -> bytes:
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "1555",
                                "phone_number_id": _PHONE_ID,
                            },
                            "messages": [
                                {
                                    "id": "wamid.ABC",
                                    "from": "15551234567",
                                    "timestamp": "1700000000",
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }
    return json.dumps(payload).encode()


async def test_whatsapp_get_handshake_echoes_challenge(
    tmp_path: Path,
) -> None:
    adapter, service, _ = await _make_adapter(tmp_path)
    try:
        request = make_mocked_request(
            "GET",
            "/whatsapp/webhook?hub.mode=subscribe&hub.verify_token="
            + _VERIFY
            + "&hub.challenge=42",
        )
        response = await adapter._handle_get(request)
        assert response.status == 200
        assert response.text == "42"
    finally:
        await service.shutdown()


async def test_whatsapp_get_handshake_rejects_wrong_token(
    tmp_path: Path,
) -> None:
    adapter, service, _ = await _make_adapter(tmp_path)
    try:
        request = make_mocked_request(
            "GET",
            "/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=WRONG"
            "&hub.challenge=42",
        )
        response = await adapter._handle_get(request)
        assert response.status == 403
    finally:
        await service.shutdown()


async def test_whatsapp_post_rejects_missing_signature(
    tmp_path: Path,
) -> None:
    adapter, service, inbound = await _make_adapter(tmp_path)
    try:
        body = _wa_inbound_body()
        request = _post_request(body, signature=None)
        response = await adapter._handle_post(request)
        assert response.status == 403
        assert inbound == []
    finally:
        await service.shutdown()


async def test_whatsapp_post_rejects_wrong_signature(
    tmp_path: Path,
) -> None:
    adapter, service, inbound = await _make_adapter(tmp_path)
    try:
        body = _wa_inbound_body()
        bad = "sha256=" + ("0" * 64)
        request = _post_request(body, signature=bad)
        response = await adapter._handle_post(request)
        assert response.status == 403
        assert inbound == []
    finally:
        await service.shutdown()


async def test_whatsapp_dispatches_text_message_with_valid_signature(
    tmp_path: Path,
) -> None:
    adapter, service, inbound = await _make_adapter(tmp_path)
    try:
        body = _wa_inbound_body(text="ping")
        request = _post_request(body, signature=_sign(body))
        response = await adapter._handle_post(request)
        assert response.status == 200
        assert len(inbound) == 1
        assert inbound[0].text == "ping"
        assert inbound[0].chat_id == "15551234567"
    finally:
        await service.shutdown()


async def test_whatsapp_skips_status_only_payloads(tmp_path: Path) -> None:
    adapter, service, inbound = await _make_adapter(tmp_path)
    try:
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "WABA",
                    "changes": [
                        {
                            "value": {
                                "metadata": {"phone_number_id": _PHONE_ID},
                                "statuses": [
                                    {
                                        "id": "wamid.X",
                                        "status": "delivered",
                                        "timestamp": "1700000001",
                                    }
                                ],
                            },
                            "field": "messages",
                        }
                    ],
                }
            ],
        }
        body = json.dumps(payload).encode()
        request = _post_request(body, signature=_sign(body))
        response = await adapter._handle_post(request)
        assert response.status == 200
        assert inbound == []
    finally:
        await service.shutdown()


async def test_whatsapp_rejects_event_for_other_phone_number(
    tmp_path: Path,
) -> None:
    adapter, service, inbound = await _make_adapter(tmp_path)
    try:
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "WABA",
                    "changes": [
                        {
                            "value": {
                                "metadata": {
                                    "phone_number_id": "OTHER_NUMBER"
                                },
                                "messages": [
                                    {
                                        "id": "wamid.Y",
                                        "from": "15550000000",
                                        "timestamp": "1700000002",
                                        "type": "text",
                                        "text": {"body": "hello"},
                                    }
                                ],
                            },
                            "field": "messages",
                        }
                    ],
                }
            ],
        }
        body = json.dumps(payload).encode()
        request = _post_request(body, signature=_sign(body))
        response = await adapter._handle_post(request)
        assert response.status == 200
        assert inbound == []
    finally:
        await service.shutdown()


@respx.mock
async def test_whatsapp_send_outgoing_posts_to_messages_endpoint(
    tmp_path: Path,
) -> None:
    adapter, service, _ = await _make_adapter(tmp_path)
    url = (
        f"https://graph.facebook.com/v22.0/{_PHONE_ID}/messages"
    )
    route = respx.post(url).mock(
        return_value=httpx.Response(200, json={"messages": [{"id": "x"}]})
    )
    try:
        await adapter.send_outgoing(
            OutgoingMessage(
                platform=Platform.WHATSAPP,
                chat_id="15551234567",
                text="hi",
            )
        )
        assert route.call_count == 1
        body = json.loads(route.calls[0].request.content)
        assert body == {
            "messaging_product": "whatsapp",
            "to": "15551234567",
            "type": "text",
            "text": {"body": "hi"},
        }
        auth = route.calls[0].request.headers.get("Authorization")
        assert auth == f"Bearer {_TOKEN}"
    finally:
        await service.shutdown()
