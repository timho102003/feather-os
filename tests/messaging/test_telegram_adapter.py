"""Tests for the Telegram long-polling adapter.

These tests do not hit the network — they use ``respx`` to mock the
Telegram HTTP API. The polling task is started, observed for one cycle,
then stopped.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from feather.messaging.adapters.telegram import TelegramAdapter
from feather.messaging.models import (
    AdapterState,
    IncomingMessage,
    OutgoingMessage,
    Platform,
)
from feather.messaging.router import MessagingRouter
from feather.messaging.store import MessagingStore
from feather.messaging.webhook_server import WebhookServer
from feather.messaging.service import MessagingService


_TOKEN = "1234:abcd-efgh"


async def _service(tmp_path: Path):
    store = MessagingStore(tmp_path / "feather.db")
    await store.initialize()

    inbound: list[IncomingMessage] = []

    async def handle_incoming(msg: IncomingMessage) -> None:
        inbound.append(msg)

    # Replace router behaviour with a recorder; we don't need real
    # session creation for these tests.
    class _RecordingRouter(MessagingRouter):
        def __init__(self) -> None:  # noqa: D401 - one-shot test double
            self.adapters: dict[Platform, Any] = {}

        def register_adapter(self, platform, sender):  # type: ignore[override]
            self.adapters[platform] = sender

        def unregister_adapter(self, platform):  # type: ignore[override]
            self.adapters.pop(platform, None)

        async def handle_incoming(self, msg):  # type: ignore[override]
            await handle_incoming(msg)

    router = _RecordingRouter()  # type: ignore[abstract]
    server = WebhookServer(host="127.0.0.1", port=0)
    http = httpx.AsyncClient()
    service = MessagingService(
        store=store,
        router=router,  # type: ignore[arg-type]
        webhook_server=server,
        http_client=http,
        adapter_factories={
            Platform.TELEGRAM: lambda svc, cfg: TelegramAdapter(svc, cfg)
        },
        own_http_client=True,
    )
    return service, router, inbound, http


def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "result": payload}


@respx.mock
async def test_telegram_connect_validates_token_and_starts_polling(
    tmp_path: Path,
) -> None:
    service, router, inbound, http = await _service(tmp_path)

    base = f"https://api.telegram.org/bot{_TOKEN}"
    respx.get(f"{base}/getMe").mock(
        return_value=httpx.Response(
            200, json=_ok({"id": 1, "is_bot": True, "username": "feather_bot"})
        )
    )
    respx.post(f"{base}/deleteWebhook").mock(
        return_value=httpx.Response(200, json=_ok(True))
    )
    # First poll returns one update; second poll never returns (we stop).
    poll_route = respx.get(f"{base}/getUpdates").mock(
        return_value=httpx.Response(
            200,
            json={
                "ok": True,
                "result": [
                    {
                        "update_id": 42,
                        "message": {
                            "message_id": 7,
                            "date": 1700000000,
                            "from": {"id": 99, "first_name": "Alice"},
                            "chat": {"id": 99, "type": "private"},
                            "text": "hi feather",
                        },
                    }
                ],
            },
        )
    )

    try:
        status = await service.connect(Platform.TELEGRAM, {"bot_token": _TOKEN})
        assert status.state == AdapterState.RUNNING
        assert "feather_bot" in status.detail

        # Wait briefly for the polling task to make at least one fetch.
        for _ in range(20):
            if inbound:
                break
            await asyncio.sleep(0.05)
        assert inbound, "expected one inbound message after first poll"
        assert inbound[0].text == "hi feather"
        assert inbound[0].chat_id == "99"
        assert inbound[0].native_message_id.startswith("telegram:")
    finally:
        await service.shutdown()


@respx.mock
async def test_telegram_connect_rejects_bad_token_format(
    tmp_path: Path,
) -> None:
    service, _, _, _ = await _service(tmp_path)
    try:
        with pytest.raises(ValueError, match="bot_token"):
            await service.connect(
                Platform.TELEGRAM, {"bot_token": "not-a-token"}
            )
    finally:
        await service.shutdown()


@respx.mock
async def test_telegram_connect_rejects_when_getme_fails(
    tmp_path: Path,
) -> None:
    service, _, _, _ = await _service(tmp_path)
    base = f"https://api.telegram.org/bot{_TOKEN}"
    respx.get(f"{base}/getMe").mock(
        return_value=httpx.Response(
            401, json={"ok": False, "description": "Unauthorized"}
        )
    )
    try:
        with pytest.raises(Exception):
            await service.connect(Platform.TELEGRAM, {"bot_token": _TOKEN})
    finally:
        await service.shutdown()


@respx.mock
async def test_telegram_send_outgoing_chunks_long_text(
    tmp_path: Path,
) -> None:
    service, _, _, _ = await _service(tmp_path)
    base = f"https://api.telegram.org/bot{_TOKEN}"
    respx.get(f"{base}/getMe").mock(
        return_value=httpx.Response(
            200, json=_ok({"id": 1, "is_bot": True, "username": "feather_bot"})
        )
    )
    respx.post(f"{base}/deleteWebhook").mock(
        return_value=httpx.Response(200, json=_ok(True))
    )
    respx.get(f"{base}/getUpdates").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": []})
    )
    send_route = respx.post(f"{base}/sendMessage").mock(
        return_value=httpx.Response(200, json=_ok({"message_id": 1}))
    )

    try:
        await service.connect(Platform.TELEGRAM, {"bot_token": _TOKEN})
        adapter = service.adapter(Platform.TELEGRAM)
        assert adapter is not None
        long_text = "a" * 5000  # > 4096 chars → must split
        await adapter.send_outgoing(
            OutgoingMessage(
                platform=Platform.TELEGRAM, chat_id="42", text=long_text
            )
        )
        # Two send calls for one 5000-char message split at 4096.
        send_calls = [c for c in send_route.calls]
        assert len(send_calls) == 2
        first_body = json.loads(send_calls[0].request.content)
        second_body = json.loads(send_calls[1].request.content)
        assert len(first_body["text"]) <= 4096
        assert len(second_body["text"]) <= 4096
        assert first_body["text"] + second_body["text"] == long_text
    finally:
        await service.shutdown()


@respx.mock
async def test_telegram_send_honours_429_retry_after(
    tmp_path: Path,
) -> None:
    service, _, _, _ = await _service(tmp_path)
    base = f"https://api.telegram.org/bot{_TOKEN}"
    respx.get(f"{base}/getMe").mock(
        return_value=httpx.Response(
            200, json=_ok({"id": 1, "is_bot": True, "username": "feather_bot"})
        )
    )
    respx.post(f"{base}/deleteWebhook").mock(
        return_value=httpx.Response(200, json=_ok(True))
    )
    respx.get(f"{base}/getUpdates").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": []})
    )
    # First sendMessage is throttled, second succeeds.
    send_route = respx.post(f"{base}/sendMessage").mock(
        side_effect=[
            httpx.Response(
                429,
                json={
                    "ok": False,
                    "description": "Too Many Requests",
                    "parameters": {"retry_after": 0},
                },
            ),
            httpx.Response(200, json=_ok({"message_id": 1})),
        ]
    )

    try:
        await service.connect(Platform.TELEGRAM, {"bot_token": _TOKEN})
        adapter = service.adapter(Platform.TELEGRAM)
        assert adapter is not None
        await adapter.send_outgoing(
            OutgoingMessage(
                platform=Platform.TELEGRAM, chat_id="42", text="ping"
            )
        )
        assert send_route.call_count == 2
    finally:
        await service.shutdown()


@respx.mock
async def test_telegram_polling_advances_offset_between_batches(
    tmp_path: Path,
) -> None:
    service, _, inbound, _ = await _service(tmp_path)
    base = f"https://api.telegram.org/bot{_TOKEN}"
    respx.get(f"{base}/getMe").mock(
        return_value=httpx.Response(
            200, json=_ok({"id": 1, "is_bot": True, "username": "f"})
        )
    )
    respx.post(f"{base}/deleteWebhook").mock(
        return_value=httpx.Response(200, json=_ok(True))
    )
    poll_route = respx.get(f"{base}/getUpdates").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "ok": True,
                    "result": [
                        {
                            "update_id": 100,
                            "message": {
                                "message_id": 1,
                                "date": 1,
                                "from": {"id": 1, "first_name": "A"},
                                "chat": {"id": 1, "type": "private"},
                                "text": "first",
                            },
                        }
                    ],
                },
            ),
            httpx.Response(
                200,
                json={
                    "ok": True,
                    "result": [
                        {
                            "update_id": 101,
                            "message": {
                                "message_id": 2,
                                "date": 2,
                                "from": {"id": 1, "first_name": "A"},
                                "chat": {"id": 1, "type": "private"},
                                "text": "second",
                            },
                        }
                    ],
                },
            ),
            httpx.Response(200, json={"ok": True, "result": []}),
        ]
    )

    try:
        await service.connect(Platform.TELEGRAM, {"bot_token": _TOKEN})
        for _ in range(40):
            if len(inbound) >= 2:
                break
            await asyncio.sleep(0.05)
        assert len(inbound) >= 2
        # Second poll request must include offset=101 (max id + 1).
        second_request = poll_route.calls[1].request
        assert "offset=101" in str(second_request.url)
    finally:
        await service.shutdown()


@respx.mock
async def test_telegram_skips_channel_chat_type(tmp_path: Path) -> None:
    service, _, inbound, _ = await _service(tmp_path)
    base = f"https://api.telegram.org/bot{_TOKEN}"
    respx.get(f"{base}/getMe").mock(
        return_value=httpx.Response(
            200, json=_ok({"id": 1, "is_bot": True, "username": "f"})
        )
    )
    respx.post(f"{base}/deleteWebhook").mock(
        return_value=httpx.Response(200, json=_ok(True))
    )
    respx.get(f"{base}/getUpdates").mock(
        return_value=httpx.Response(
            200,
            json={
                "ok": True,
                "result": [
                    {
                        "update_id": 200,
                        "message": {
                            "message_id": 7,
                            "date": 1,
                            "from": {"id": 1, "first_name": "A"},
                            "chat": {"id": 1, "type": "channel"},
                            "text": "ignore me",
                        },
                    }
                ],
            },
        )
    )

    try:
        await service.connect(Platform.TELEGRAM, {"bot_token": _TOKEN})
        await asyncio.sleep(0.3)
        assert inbound == []
    finally:
        await service.shutdown()
