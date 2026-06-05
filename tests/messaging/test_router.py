"""Tests for :class:`feather.messaging.router.MessagingRouter`."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from feather.core.session.input_queue import UserInputQueue
from feather.messaging.models import (
    IncomingMessage,
    OutgoingMessage,
    Platform,
)
from feather.messaging.router import MessagingRouter
from feather.messaging.store import MessagingStore
from feather.models import AgentOutcome, AgentRunResult


async def _build(tmp_path: Path):
    store = MessagingStore(tmp_path / "feather.db")
    await store.initialize()
    queue = UserInputQueue()
    sessions: list[str] = []
    runs: list[tuple[str, str]] = []
    sent: list[OutgoingMessage] = []

    async def create_session() -> str:
        sid = f"sess-{len(sessions) + 1}"
        sessions.append(sid)
        return sid

    busy_state = {"value": False}

    async def is_busy(_session_id: str) -> bool:
        return busy_state["value"]

    async def run_agent(session_id: str, text: str) -> AgentRunResult:
        runs.append((session_id, text))
        return AgentRunResult(
            status=AgentOutcome.COMPLETED,
            session_id=session_id,
            assistant_text=f"echo: {text}",
        )

    async def sender(out: OutgoingMessage) -> None:
        sent.append(out)

    router = MessagingRouter(
        store=store,
        create_session=create_session,
        run_agent=run_agent,
        is_session_busy=is_busy,
        input_queue=queue,
    )
    router.register_adapter(Platform.TELEGRAM, sender)
    return router, store, queue, runs, sent, sessions, busy_state


def _msg(
    *,
    chat: str = "chat-1",
    text: str = "hello",
    native: str = "msg-1",
    platform: Platform = Platform.TELEGRAM,
    sender_name: str = "Alice",
) -> IncomingMessage:
    return IncomingMessage(
        platform=platform,
        chat_id=chat,
        sender_display_name=sender_name,
        text=text,
        native_message_id=native,
    )


async def test_first_message_creates_session_and_sends_reply(
    tmp_path: Path,
) -> None:
    router, store, _, runs, sent, _, _ = await _build(tmp_path)

    await router.handle_incoming(_msg())

    assert len(runs) == 1
    assert runs[0][1] == "hello"
    mapping = await store.get_chat_mapping(Platform.TELEGRAM, "chat-1")
    assert mapping is not None
    assert mapping.session_id == runs[0][0]
    assert len(sent) == 1
    assert sent[0].text == "echo: hello"
    assert sent[0].chat_id == "chat-1"


async def test_second_message_reuses_session(tmp_path: Path) -> None:
    router, _, _, runs, _, _, _ = await _build(tmp_path)

    await router.handle_incoming(_msg(text="first", native="m-1"))
    await router.handle_incoming(_msg(text="second", native="m-2"))

    assert len(runs) == 2
    assert runs[0][0] == runs[1][0]


async def test_duplicate_native_id_is_ignored(tmp_path: Path) -> None:
    router, _, _, runs, _, _, _ = await _build(tmp_path)

    await router.handle_incoming(_msg(text="first", native="dup"))
    await router.handle_incoming(_msg(text="first-again", native="dup"))

    assert len(runs) == 1


async def test_busy_session_enqueues_via_input_queue(
    tmp_path: Path,
) -> None:
    router, _, queue, runs, _, _, busy = await _build(tmp_path)

    # First message creates session and runs.
    await router.handle_incoming(_msg(text="hello", native="m-1"))
    assert len(runs) == 1
    session_id = runs[0][0]

    # Now flip to busy and send another — should enqueue, not run.
    busy["value"] = True
    await router.handle_incoming(_msg(text="while busy", native="m-2"))

    assert len(runs) == 1
    pending = await queue.peek(session_id)
    assert pending == ("while busy",)


async def test_concurrent_messages_serialize_via_router_lock(
    tmp_path: Path,
) -> None:
    """Two simultaneous incoming messages should not overlap runs."""

    router, store, queue, runs, _, _, _ = await _build(tmp_path)

    # Replace run_agent with a slower one that asserts no concurrent
    # invocations.
    in_flight = {"count": 0}

    async def slow_run(session_id: str, text: str) -> AgentRunResult:
        in_flight["count"] += 1
        try:
            assert in_flight["count"] == 1, "two concurrent runs detected"
            await asyncio.sleep(0.05)
            return AgentRunResult(
                status=AgentOutcome.COMPLETED,
                session_id=session_id,
                assistant_text=f"echo: {text}",
            )
        finally:
            in_flight["count"] -= 1

    router._run_agent = slow_run  # type: ignore[assignment]

    # First call seeds the session mapping.
    await router.handle_incoming(_msg(text="first", native="m-1"))
    assert len(runs) == 0  # The replacement runner doesn't append to ``runs``.
    mapping = await store.get_chat_mapping(Platform.TELEGRAM, "chat-1")
    assert mapping is not None
    session_id = mapping.session_id

    # Now fire two concurrent messages — they share the same session.
    await asyncio.gather(
        router.handle_incoming(_msg(text="A", native="m-2")),
        router.handle_incoming(_msg(text="B", native="m-3")),
    )

    # One ran inline; the other should be queued via the input queue.
    pending = await queue.peek(session_id)
    assert len(pending) >= 1


async def test_unknown_platform_is_skipped_silently(tmp_path: Path) -> None:
    router, _, _, runs, sent, _, _ = await _build(tmp_path)

    await router.handle_incoming(
        _msg(platform=Platform.LINE, native="m-line")
    )
    # No sender registered for LINE — run still happens but reply is dropped.
    assert len(runs) == 1
    assert sent == []


async def test_empty_text_is_ignored(tmp_path: Path) -> None:
    router, _, _, runs, sent, _, _ = await _build(tmp_path)

    await router.handle_incoming(_msg(text="   ", native="m-empty"))

    assert runs == []
    assert sent == []


async def test_awaiting_user_question_is_appended_to_reply(
    tmp_path: Path,
) -> None:
    router, _, _, _, sent, _, _ = await _build(tmp_path)

    async def run_with_question(session_id: str, text: str) -> AgentRunResult:
        return AgentRunResult(
            status=AgentOutcome.AWAITING_USER,
            session_id=session_id,
            assistant_text="Sure, but I need clarification.",
            question="What time zone are you in?",
        )

    router._run_agent = run_with_question  # type: ignore[assignment]

    await router.handle_incoming(_msg())

    assert len(sent) == 1
    assert "What time zone" in sent[0].text
    assert "Sure, but I need clarification." in sent[0].text


async def test_run_agent_failure_does_not_send_reply(tmp_path: Path) -> None:
    router, _, _, _, sent, _, _ = await _build(tmp_path)

    async def crashing_run(session_id: str, text: str) -> AgentRunResult:
        raise RuntimeError("boom")

    router._run_agent = crashing_run  # type: ignore[assignment]

    # Must not raise to the adapter.
    await router.handle_incoming(_msg())

    assert sent == []


async def test_unregister_adapter_disables_replies(tmp_path: Path) -> None:
    router, _, _, _, sent, _, _ = await _build(tmp_path)

    router.unregister_adapter(Platform.TELEGRAM)
    await router.handle_incoming(_msg())

    assert sent == []
