"""Regression tests for the messaging red-team review.

Each test corresponds to a shadow test (S1-S7) raised by the reviewer.
Failures here would mean a known-bad behaviour has been re-introduced.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from aiohttp.test_utils import make_mocked_request

from feather.messaging.adapters.line import LineAdapter
from feather.messaging.adapters.telegram import TelegramAdapter
from feather.messaging.adapters.whatsapp import WhatsAppAdapter
from feather.messaging.models import (
    AdapterState,
    IncomingMessage,
    OutgoingMessage,
    Platform,
)
from feather.messaging.router import MessagingRouter
from feather.messaging.service import MessagingService
from feather.messaging.store import MessagingStore
from feather.messaging.webhook_server import WebhookServer
from feather.models import AgentOutcome, AgentRunResult


_TOKEN = "12345:my-secret-token"


class _FakePayload:
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


async def _service(tmp_path: Path) -> tuple[MessagingService, list, MessagingStore]:
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
            Platform.TELEGRAM: lambda svc, cfg: TelegramAdapter(svc, cfg),
            Platform.LINE: lambda svc, cfg: LineAdapter(svc, cfg),
            Platform.WHATSAPP: lambda svc, cfg: WhatsAppAdapter(svc, cfg),
        },
        own_http_client=True,
    )
    return service, inbound, store


# ---------- S1 / S2: Telegram token must not leak ----------


@respx.mock
async def test_S1_telegram_token_redacted_in_connect_error(
    tmp_path: Path,
) -> None:
    """Bad-token errors must NOT include the token in the message."""

    service, _, _ = await _service(tmp_path)
    base = f"https://api.telegram.org/bot{_TOKEN}"
    respx.get(f"{base}/getMe").mock(
        return_value=httpx.Response(
            401, json={"ok": False, "description": "Unauthorized"}
        )
    )
    try:
        with pytest.raises(Exception) as excinfo:
            await service.connect(Platform.TELEGRAM, {"bot_token": _TOKEN})
        assert "my-secret-token" not in str(excinfo.value), (
            f"token leaked: {excinfo.value!s}"
        )
    finally:
        await service.shutdown()


@respx.mock
async def test_S2_telegram_polling_logs_do_not_contain_token(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """sendMessage / poll error logs must scrub the token from URLs."""

    service, _, _ = await _service(tmp_path)
    base = f"https://api.telegram.org/bot{_TOKEN}"
    respx.get(f"{base}/getMe").mock(
        return_value=httpx.Response(
            200,
            json={
                "ok": True,
                "result": {"id": 1, "is_bot": True, "username": "f"},
            },
        )
    )
    respx.post(f"{base}/deleteWebhook").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": True})
    )
    respx.get(f"{base}/getUpdates").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": []})
    )
    # Make sendMessage 5xx so the network-error path runs.
    respx.post(f"{base}/sendMessage").mock(
        side_effect=httpx.ConnectError("boom")
    )
    try:
        await service.connect(Platform.TELEGRAM, {"bot_token": _TOKEN})
        adapter = service.adapter(Platform.TELEGRAM)
        assert adapter is not None
        with caplog.at_level(logging.WARNING):
            await adapter.send_outgoing(
                OutgoingMessage(
                    platform=Platform.TELEGRAM, chat_id="42", text="hi"
                )
            )
        assert "my-secret-token" not in caplog.text
    finally:
        await service.shutdown()


# ---------- S3: router lock dict must not grow unboundedly ----------


async def test_S3_router_does_not_leak_session_locks(tmp_path: Path) -> None:
    """After many distinct chats, the router must not retain dead sessions."""

    store = MessagingStore(tmp_path / "feather.db")
    await store.initialize()
    sessions = iter(f"sess-{i}" for i in range(10000))

    async def create_session() -> str:
        return next(sessions)

    async def is_busy(_session_id: str) -> bool:
        return False

    async def run_agent(session_id: str, text: str) -> AgentRunResult:
        return AgentRunResult(
            status=AgentOutcome.COMPLETED,
            session_id=session_id,
            assistant_text=f"echo: {text}",
        )

    from feather.core.input_queue import UserInputQueue
    queue = UserInputQueue()
    router = MessagingRouter(
        store=store,
        create_session=create_session,
        run_agent=run_agent,
        is_session_busy=is_busy,
        input_queue=queue,
    )

    async def sender(_out: OutgoingMessage) -> None:
        return None

    router.register_adapter(Platform.TELEGRAM, sender)

    for i in range(500):
        await router.handle_incoming(
            IncomingMessage(
                platform=Platform.TELEGRAM,
                chat_id=f"chat-{i}",
                sender_display_name="x",
                text="hello",
                native_message_id=f"native-{i}",
            )
        )

    # The post-fix router uses an ``_active_sessions`` set scoped to
    # currently-running runs only. After 500 sequential calls, none
    # should be active.
    active = getattr(router, "_active_sessions", set())
    assert len(active) == 0, (
        f"router._active_sessions held {len(active)} entries; "
        "the structure leaks across calls"
    )
    # Belt-and-braces: also assert no resurrection of the old leaky
    # ``_session_locks`` dict.
    legacy_locks = getattr(router, "_session_locks", None)
    assert legacy_locks is None or len(legacy_locks) == 0


# ---------- S4: telegram start failure must not orphan polling task ----------


@respx.mock
async def test_S4_telegram_start_failure_does_not_orphan_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If start() raises after task creation, the task must be cancelled."""

    service, _, _ = await _service(tmp_path)
    base = f"https://api.telegram.org/bot{_TOKEN}"
    respx.get(f"{base}/getMe").mock(
        return_value=httpx.Response(
            200,
            json={
                "ok": True,
                "result": {"id": 1, "is_bot": True, "username": "f"},
            },
        )
    )
    respx.post(f"{base}/deleteWebhook").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": True})
    )
    respx.get(f"{base}/getUpdates").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": []})
    )
    # Force the post-task-creation step to raise.
    original = service.store.count_chats_for_platform

    async def boom(*a, **k):
        raise RuntimeError("simulated DB lock")

    monkeypatch.setattr(service.store, "count_chats_for_platform", boom)
    try:
        with pytest.raises(RuntimeError):
            await service.connect(Platform.TELEGRAM, {"bot_token": _TOKEN})
        # The adapter should not be registered.
        assert service.adapter(Platform.TELEGRAM) is None
        # Give any orphan polling task a brief chance to run, then
        # collect outstanding tasks. The fixture's test loop cancels
        # them on teardown, but we assert no telegram task is still
        # running.
        await asyncio.sleep(0.1)
        named = [
            t
            for t in asyncio.all_tasks()
            if "_poll_forever" in str(t.get_coro())
        ]
        assert named == [], f"orphaned polling task: {named!r}"
    finally:
        # Restore for safe shutdown.
        monkeypatch.setattr(
            service.store, "count_chats_for_platform", original
        )
        await service.shutdown()


# ---------- S5: dedup must release on failure so retries work ----------


async def test_S5_dedup_released_when_agent_run_fails(tmp_path: Path) -> None:
    store = MessagingStore(tmp_path / "feather.db")
    await store.initialize()
    from feather.core.input_queue import UserInputQueue
    queue = UserInputQueue()

    calls: list[str] = []
    sessions = iter(f"sess-{i}" for i in range(10))

    async def create_session() -> str:
        return next(sessions)

    async def is_busy(_session_id: str) -> bool:
        return False

    async def run_agent(session_id: str, text: str) -> AgentRunResult:
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return AgentRunResult(
            status=AgentOutcome.COMPLETED,
            session_id=session_id,
            assistant_text="ok",
        )

    router = MessagingRouter(
        store=store,
        create_session=create_session,
        run_agent=run_agent,
        is_session_busy=is_busy,
        input_queue=queue,
    )

    async def sender(_out: OutgoingMessage) -> None:
        return None

    router.register_adapter(Platform.TELEGRAM, sender)

    msg = IncomingMessage(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        sender_display_name="x",
        text="ping",
        native_message_id="dup-1",
    )
    await router.handle_incoming(msg)
    await router.handle_incoming(msg)

    assert calls == ["ping", "ping"], (
        f"expected the failed message to retry on redelivery, got {calls!r}"
    )


# ---------- S6: whatsapp metadata gate must fail closed ----------


_APP_SECRET = "app-secret"
_PHONE_ID = "1234567890"
_VERIFY = "verify-token"


def _wa_sign(body: bytes) -> str:
    return (
        "sha256="
        + hmac.new(
            _APP_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
    )


def _wa_post_request(body: bytes) -> Any:
    return make_mocked_request(
        "POST",
        "/whatsapp/webhook",
        headers={
            "X-Hub-Signature-256": _wa_sign(body),
            "Content-Type": "application/json",
        },
        payload=_FakePayload(body),
    )


async def test_S6_whatsapp_rejects_payload_without_phone_number_id(
    tmp_path: Path,
) -> None:
    service, inbound, _ = await _service(tmp_path)
    adapter = WhatsAppAdapter(
        service,
        {
            "phone_number_id": _PHONE_ID,
            "access_token": "tok",
            "verify_token": _VERIFY,
            "app_secret": _APP_SECRET,
        },
    )
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA",
                "changes": [
                    {
                        "value": {
                            "metadata": {},  # no phone_number_id
                            "messages": [
                                {
                                    "id": "wamid.X",
                                    "from": "15550000000",
                                    "timestamp": "1700000001",
                                    "type": "text",
                                    "text": {"body": "hi"},
                                }
                            ],
                        }
                    }
                ],
            }
        ],
    }
    body = json.dumps(payload).encode()
    request = _wa_post_request(body)
    try:
        response = await adapter._handle_post(request)
        assert response.status == 200
        assert inbound == [], (
            "WhatsApp inbound dispatched despite missing phone_number_id; "
            "metadata gate is failing open"
        )
    finally:
        await service.shutdown()


# ---------- S7: line dispatch error log must redact replyToken ----------


_SECRET = "channel-secret"


def _line_sign(body: bytes) -> str:
    import base64

    digest = hmac.new(
        _SECRET.encode(), body, hashlib.sha256
    ).digest()
    return base64.b64encode(digest).decode("ascii")


async def test_S7_line_dispatch_error_does_not_log_reply_token(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _, _ = await _service(tmp_path)
    adapter = LineAdapter(
        service,
        {"channel_secret": _SECRET, "channel_token": "tok"},
    )

    async def boom(self_, event):  # noqa: ANN001
        raise RuntimeError("intentional dispatch failure")

    monkeypatch.setattr(LineAdapter, "_dispatch_event", boom)

    body = json.dumps(
        {
            "destination": "U0",
            "events": [
                {
                    "type": "message",
                    "replyToken": "secret-reply-tok-XYZ",
                    "timestamp": 1,
                    "source": {"type": "user", "userId": "U-alice"},
                    "message": {
                        "id": "m-1",
                        "type": "text",
                        "text": "hi",
                    },
                    "webhookEventId": "evt-1",
                }
            ],
        }
    ).encode()
    request = make_mocked_request(
        "POST",
        "/line/webhook",
        headers={
            "X-Line-Signature": _line_sign(body),
            "Content-Type": "application/json",
        },
        payload=_FakePayload(body),
    )

    try:
        with caplog.at_level(logging.ERROR):
            await adapter._handle_request(request)
        assert "secret-reply-tok-XYZ" not in caplog.text, (
            "LINE dispatch error log leaked the replyToken"
        )
    finally:
        await service.shutdown()


# ---------- Bonus: WhatsApp signature must reject non-dict JSON cleanly ----------


async def test_whatsapp_post_rejects_non_object_json(tmp_path: Path) -> None:
    service, _, _ = await _service(tmp_path)
    adapter = WhatsAppAdapter(
        service,
        {
            "phone_number_id": _PHONE_ID,
            "access_token": "tok",
            "verify_token": _VERIFY,
            "app_secret": _APP_SECRET,
        },
    )
    body = b"[]"  # JSON array, not object
    request = make_mocked_request(
        "POST",
        "/whatsapp/webhook",
        headers={
            "X-Hub-Signature-256": _wa_sign(body),
            "Content-Type": "application/json",
        },
        payload=_FakePayload(body),
    )
    try:
        response = await adapter._handle_post(request)
        # 400 (rejected) is the right answer; 500 from AttributeError is not.
        assert response.status in (200, 400)
    finally:
        await service.shutdown()
