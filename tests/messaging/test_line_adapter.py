"""Integration tests for the LINE webhook adapter."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from feather.messaging.adapters.line import LineAdapter
from feather.messaging.models import IncomingMessage, OutgoingMessage, Platform
from feather.messaging.router import MessagingRouter
from feather.messaging.service import MessagingService
from feather.messaging.store import MessagingStore
from feather.messaging.webhook_server import WebhookServer


_SECRET = "channel-secret"
_TOKEN = "channel-token"


def _sign(body: bytes) -> str:
    digest = hmac.new(_SECRET.encode(), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


async def _make_adapter(tmp_path: Path):
    """Build a LINE adapter wired to a recording router.

    Webhook server is constructed but never started — tests invoke
    ``adapter._handle_request`` directly to keep the integration
    in-process and deterministic.
    """

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
            Platform.LINE: lambda svc, cfg: LineAdapter(svc, cfg)
        },
        own_http_client=True,
    )
    config = {"channel_secret": _SECRET, "channel_token": _TOKEN}
    adapter = LineAdapter(service, config)
    return adapter, service, inbound


def _line_body(text: str = "hello", token: str = "tok-1") -> bytes:
    payload = {
        "destination": "U0",
        "events": [
            {
                "type": "message",
                "replyToken": token,
                "timestamp": 1700000000000,
                "source": {"type": "user", "userId": "U-alice"},
                "webhookEventId": "evt-1",
                "message": {"id": "m-1", "type": "text", "text": text},
            }
        ],
    }
    return json.dumps(payload).encode("utf-8")


def _make_request(body: bytes, signature: str | None) -> web.Request:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if signature is not None:
        headers["X-Line-Signature"] = signature
    return make_mocked_request(
        "POST", "/line/webhook", headers=headers, payload=_FakePayload(body)
    )


class _FakePayload:
    """Minimal aiohttp payload that supports ``request.read()``.

    aiohttp's ``Request.read`` actually calls ``payload.readany()`` and
    keeps reading until empty, so a single-shot ``readany`` that returns
    the full body the first time and an empty bytes object on the
    second call is sufficient.
    """

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


async def test_line_rejects_missing_signature(tmp_path: Path) -> None:
    adapter, service, inbound = await _make_adapter(tmp_path)
    try:
        body = _line_body()
        request = _make_request(body, signature=None)
        response = await adapter._handle_request(request)
        assert response.status == 400
        assert inbound == []
    finally:
        await service.shutdown()


async def test_line_rejects_bad_signature(tmp_path: Path) -> None:
    adapter, service, inbound = await _make_adapter(tmp_path)
    try:
        body = _line_body()
        request = _make_request(body, signature="not-a-valid-sig")
        response = await adapter._handle_request(request)
        assert response.status == 400
        assert inbound == []
    finally:
        await service.shutdown()


async def test_line_dispatches_text_message_with_valid_signature(
    tmp_path: Path,
) -> None:
    adapter, service, inbound = await _make_adapter(tmp_path)
    try:
        body = _line_body(text="hi feather")
        request = _make_request(body, signature=_sign(body))
        response = await adapter._handle_request(request)
        assert response.status == 200
        assert len(inbound) == 1
        assert inbound[0].text == "hi feather"
        assert inbound[0].chat_id == "U-alice"
        assert inbound[0].reply_context == {"reply_token": "tok-1"}
    finally:
        await service.shutdown()


async def test_line_ignores_non_text_messages(tmp_path: Path) -> None:
    adapter, service, inbound = await _make_adapter(tmp_path)
    try:
        payload = {
            "destination": "U0",
            "events": [
                {
                    "type": "message",
                    "replyToken": "tok",
                    "timestamp": 1,
                    "source": {"type": "user", "userId": "U-bob"},
                    "message": {"id": "m", "type": "sticker"},
                    "webhookEventId": "evt-2",
                }
            ],
        }
        body = json.dumps(payload).encode()
        request = _make_request(body, signature=_sign(body))
        response = await adapter._handle_request(request)
        assert response.status == 200
        assert inbound == []
    finally:
        await service.shutdown()


async def test_line_ignores_group_chats(tmp_path: Path) -> None:
    adapter, service, inbound = await _make_adapter(tmp_path)
    try:
        payload = {
            "destination": "U0",
            "events": [
                {
                    "type": "message",
                    "replyToken": "tok",
                    "timestamp": 1,
                    "source": {"type": "group", "groupId": "G-1"},
                    "message": {"id": "m", "type": "text", "text": "hi"},
                }
            ],
        }
        body = json.dumps(payload).encode()
        request = _make_request(body, signature=_sign(body))
        response = await adapter._handle_request(request)
        assert response.status == 200
        assert inbound == []  # Group/room support is not implemented yet.
    finally:
        await service.shutdown()


async def test_line_verification_request_returns_200(tmp_path: Path) -> None:
    adapter, service, _ = await _make_adapter(tmp_path)
    try:
        body = json.dumps({"destination": "U0", "events": []}).encode()
        request = _make_request(body, signature=_sign(body))
        response = await adapter._handle_request(request)
        assert response.status == 200
    finally:
        await service.shutdown()


@respx.mock
async def test_line_send_outgoing_uses_reply_when_token_present(
    tmp_path: Path,
) -> None:
    adapter, service, _ = await _make_adapter(tmp_path)
    reply_route = respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=httpx.Response(200, json={})
    )
    push_route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=httpx.Response(200, json={})
    )
    try:
        await adapter.send_outgoing(
            OutgoingMessage(
                platform=Platform.LINE,
                chat_id="U-alice",
                text="hello back",
                reply_context={"reply_token": "tok-1"},
            )
        )
        assert reply_route.call_count == 1
        assert push_route.call_count == 0
        body = json.loads(reply_route.calls[0].request.content)
        assert body["replyToken"] == "tok-1"
        assert body["messages"][0]["text"] == "hello back"
    finally:
        await service.shutdown()


@respx.mock
async def test_line_send_outgoing_falls_back_to_push_without_token(
    tmp_path: Path,
) -> None:
    adapter, service, _ = await _make_adapter(tmp_path)
    reply_route = respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=httpx.Response(200, json={})
    )
    push_route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=httpx.Response(200, json={})
    )
    try:
        await adapter.send_outgoing(
            OutgoingMessage(
                platform=Platform.LINE,
                chat_id="U-alice",
                text="unsolicited",
            )
        )
        assert reply_route.call_count == 0
        assert push_route.call_count == 1
        body = json.loads(push_route.calls[0].request.content)
        assert body["to"] == "U-alice"
    finally:
        await service.shutdown()


@respx.mock
async def test_line_send_outgoing_chunks_long_text_into_multiple_messages(
    tmp_path: Path,
) -> None:
    adapter, service, _ = await _make_adapter(tmp_path)
    reply_route = respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=httpx.Response(200, json={})
    )
    push_route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=httpx.Response(200, json={})
    )
    try:
        # 5 chunks (>2000 chars each) → first batch of 5 via reply,
        # rest via push. Build text just over 5 * 2000 = 10000 chars
        # plus a bit so we definitely overflow into a 6th chunk.
        long_text = "a" * 10500
        await adapter.send_outgoing(
            OutgoingMessage(
                platform=Platform.LINE,
                chat_id="U-alice",
                text=long_text,
                reply_context={"reply_token": "tok-1"},
            )
        )
        # Reply should have been called once (with up to 5 messages),
        # push at least once for the spill-over.
        assert reply_route.call_count == 1
        assert push_route.call_count >= 1
        first_body = json.loads(reply_route.calls[0].request.content)
        # All of reply's messages individually <= 2000 chars.
        for m in first_body["messages"]:
            assert len(m["text"]) <= 2000
        assert len(first_body["messages"]) <= 5
    finally:
        await service.shutdown()
