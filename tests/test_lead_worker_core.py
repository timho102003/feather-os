"""Tests for the lead worker subprocess core.

The worker entry script (`feather.lead_worker_entry`) is a thin shell —
all the interesting logic lives in :class:`WorkerCore`, which we drive
here with in-memory command/event streams and a fake agent. That keeps
the tests fast and side-effect-free while still exercising real timing
on the heartbeat ticker and real serialization on the stdout JSONL.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from feather.core.input_queue import UserInputQueue
from feather.core.lead_worker_core import WorkerCore
from feather.core.worker_command_codec import (
    EnqueueUserInputCommand,
    ResumeOnInboxCommand,
    RunCommand,
    ShutdownCommand,
    encode_command,
)
from feather.models import (
    AgentOutcome,
    AgentRunResult,
    RuntimeEvent,
    WorkerStatus,
)
from feather.storage.worker_heartbeat_store import WorkerHeartbeatStore


# --------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------- #


@dataclass
class _RecordedRun:
    method: str
    session_id: str
    incoming_text: str | None
    events_emitted: list[RuntimeEvent] = field(default_factory=list)


class _FakeAgent:
    """Stand-in for :class:`BaseAgent` that records calls and replays scripted events."""

    def __init__(
        self,
        *,
        scripted_events: list[RuntimeEvent] | None = None,
        result: AgentRunResult | None = None,
        raise_on_run: BaseException | None = None,
    ) -> None:
        self._scripted_events = list(scripted_events or [])
        self._result = result or AgentRunResult(
            status=AgentOutcome.COMPLETED,
            session_id="s1",
            assistant_text="ok",
            question=None,
            total_tool_calls=0,
        )
        self._raise = raise_on_run
        self.calls: list[_RecordedRun] = []

    async def run(self, session_id, incoming_text, event_handler=None):  # type: ignore[no-untyped-def]
        rec = _RecordedRun(
            method="run", session_id=session_id, incoming_text=incoming_text
        )
        self.calls.append(rec)
        if self._raise is not None:
            raise self._raise
        for event in self._scripted_events:
            if event_handler is not None:
                event_handler(event)
            rec.events_emitted.append(event)
        return self._result

    async def resume_on_inbox(self, session_id, event_handler=None):  # type: ignore[no-untyped-def]
        rec = _RecordedRun(
            method="resume_on_inbox", session_id=session_id, incoming_text=None
        )
        self.calls.append(rec)
        if self._raise is not None:
            raise self._raise
        for event in self._scripted_events:
            if event_handler is not None:
                event_handler(event)
            rec.events_emitted.append(event)
        return self._result


async def _async_lines(lines: list[str]) -> AsyncIterator[str]:
    """Yield the given lines, optionally with awaits in between."""

    for line in lines:
        # Yield to the loop so heartbeats and command dispatch interleave.
        await asyncio.sleep(0)
        yield line


async def _async_lines_with_pause(
    lines: list[str], pause_seconds: float
) -> AsyncIterator[str]:
    """Yield lines, sleeping between each one to allow heartbeat ticks."""

    for line in lines:
        await asyncio.sleep(pause_seconds)
        yield line


def _captured_event_sink() -> tuple[list[dict[str, Any]], Any]:
    """Return (collector_list, sink_callable) — sink appends parsed JSON lines."""

    collected: list[dict[str, Any]] = []

    def sink(line: str) -> None:
        collected.append(json.loads(line))

    return collected, sink


async def _open_heartbeat_store(tmp_path: Path) -> WorkerHeartbeatStore:
    store = WorkerHeartbeatStore(tmp_path / "feather.db")
    await store.initialize()
    return store


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


async def test_run_command_invokes_agent_run_and_emits_events(tmp_path: Path) -> None:
    agent = _FakeAgent(
        scripted_events=[
            RuntimeEvent(kind="assistant_text_delta", text="hi"),
            RuntimeEvent(kind="assistant_text_delta", text="!"),
        ],
    )
    store = await _open_heartbeat_store(tmp_path)
    input_queue = UserInputQueue()
    events, sink = _captured_event_sink()

    try:
        commands = _async_lines(
            [
                encode_command(RunCommand(session_id="s1", incoming_text="hello")),
                encode_command(ShutdownCommand()),
            ]
        )
        core = WorkerCore(
            agent=agent,
            input_queue=input_queue,
            heartbeat_store=store,
            session_id="s1",
            pid=12345,
            heartbeat_interval=0.05,
            command_source=commands,
            event_sink=sink,
        )
        await core.run()
    finally:
        await store.close()

    assert len(agent.calls) == 1
    assert agent.calls[0].method == "run"
    assert agent.calls[0].incoming_text == "hello"

    kinds = [e["kind"] for e in events]
    # Two deltas + one control event for run completion + one for shutdown ack.
    assert "assistant_text_delta" in kinds
    assert "_run_complete" in kinds
    assert "_shutdown_ack" in kinds
    # Ordering: deltas precede the run-complete control event.
    delta_idxs = [i for i, k in enumerate(kinds) if k == "assistant_text_delta"]
    complete_idx = kinds.index("_run_complete")
    assert all(i < complete_idx for i in delta_idxs)


async def test_resume_on_inbox_command_invokes_agent_resume(tmp_path: Path) -> None:
    agent = _FakeAgent()
    store = await _open_heartbeat_store(tmp_path)
    input_queue = UserInputQueue()
    events, sink = _captured_event_sink()

    try:
        commands = _async_lines(
            [
                encode_command(ResumeOnInboxCommand(session_id="s1")),
                encode_command(ShutdownCommand()),
            ]
        )
        core = WorkerCore(
            agent=agent,
            input_queue=input_queue,
            heartbeat_store=store,
            session_id="s1",
            pid=12345,
            heartbeat_interval=0.05,
            command_source=commands,
            event_sink=sink,
        )
        await core.run()
    finally:
        await store.close()

    assert len(agent.calls) == 1
    assert agent.calls[0].method == "resume_on_inbox"
    kinds = [e["kind"] for e in events]
    assert "_run_complete" in kinds


async def test_enqueue_user_input_command_pushes_to_queue_immediately(
    tmp_path: Path,
) -> None:
    """Enqueue runs *off* the command loop so it never waits behind a run."""

    agent = _FakeAgent()
    store = await _open_heartbeat_store(tmp_path)
    input_queue = UserInputQueue()
    events, sink = _captured_event_sink()

    try:
        commands = _async_lines(
            [
                encode_command(EnqueueUserInputCommand(session_id="s1", text="ping")),
                encode_command(EnqueueUserInputCommand(session_id="s1", text="pong")),
                encode_command(ShutdownCommand()),
            ]
        )
        core = WorkerCore(
            agent=agent,
            input_queue=input_queue,
            heartbeat_store=store,
            session_id="s1",
            pid=12345,
            heartbeat_interval=0.05,
            command_source=commands,
            event_sink=sink,
        )
        await core.run()
    finally:
        await store.close()

    drained = await input_queue.drain("s1")
    assert drained == ["ping", "pong"]
    # No agent run should have happened.
    assert agent.calls == []


async def test_heartbeat_writes_periodically_and_marks_stopped_on_shutdown(
    tmp_path: Path,
) -> None:
    agent = _FakeAgent()
    store = await _open_heartbeat_store(tmp_path)
    input_queue = UserInputQueue()
    events, sink = _captured_event_sink()

    try:
        # Pause between commands so the heartbeat ticker fires several times.
        commands = _async_lines_with_pause(
            [encode_command(ShutdownCommand())], pause_seconds=0.25
        )
        core = WorkerCore(
            agent=agent,
            input_queue=input_queue,
            heartbeat_store=store,
            session_id="s-hb",
            pid=42,
            heartbeat_interval=0.05,
            command_source=commands,
            event_sink=sink,
        )
        await core.run()

        final = await store.get("s-hb")
        assert final is not None
        assert final.pid == 42
        assert final.status is WorkerStatus.STOPPED
    finally:
        await store.close()


async def test_eof_on_command_source_triggers_shutdown(tmp_path: Path) -> None:
    """Closing stdin (EOF) is an implicit shutdown signal — no command needed."""

    agent = _FakeAgent()
    store = await _open_heartbeat_store(tmp_path)
    input_queue = UserInputQueue()
    events, sink = _captured_event_sink()

    try:
        commands = _async_lines([])  # no commands; iterator exhausts immediately
        core = WorkerCore(
            agent=agent,
            input_queue=input_queue,
            heartbeat_store=store,
            session_id="s-eof",
            pid=7,
            heartbeat_interval=0.05,
            command_source=commands,
            event_sink=sink,
        )
        # Must complete without hanging.
        await asyncio.wait_for(core.run(), timeout=2.0)

        final = await store.get("s-eof")
        assert final is not None
        assert final.status is WorkerStatus.STOPPED
    finally:
        await store.close()


async def test_invalid_command_line_is_logged_but_does_not_crash(
    tmp_path: Path, caplog
) -> None:
    """A garbage stdin line must not poison the worker."""

    import logging as _logging

    caplog.set_level(_logging.WARNING, logger="feather.core.lead_worker_core")
    agent = _FakeAgent()
    store = await _open_heartbeat_store(tmp_path)
    input_queue = UserInputQueue()
    events, sink = _captured_event_sink()

    try:
        commands = _async_lines(
            [
                "this is not json",
                '{"cmd": "run"}',  # missing required fields
                encode_command(ShutdownCommand()),
            ]
        )
        core = WorkerCore(
            agent=agent,
            input_queue=input_queue,
            heartbeat_store=store,
            session_id="s1",
            pid=1,
            heartbeat_interval=0.05,
            command_source=commands,
            event_sink=sink,
        )
        await core.run()
    finally:
        await store.close()

    # No run should have been attempted — both bad lines were rejected.
    assert agent.calls == []
    # And the worker logged the rejection (at least once).
    assert any("invalid command" in rec.getMessage().lower() for rec in caplog.records)


async def test_run_command_failure_emits_run_failed_control_event(
    tmp_path: Path,
) -> None:
    """If agent.run() raises, the worker must surface a `_run_failed` control event."""

    agent = _FakeAgent(raise_on_run=RuntimeError("boom"))
    store = await _open_heartbeat_store(tmp_path)
    input_queue = UserInputQueue()
    events, sink = _captured_event_sink()

    try:
        commands = _async_lines(
            [
                encode_command(RunCommand(session_id="s1", incoming_text="hi")),
                encode_command(ShutdownCommand()),
            ]
        )
        core = WorkerCore(
            agent=agent,
            input_queue=input_queue,
            heartbeat_store=store,
            session_id="s1",
            pid=1,
            heartbeat_interval=0.05,
            command_source=commands,
            event_sink=sink,
        )
        await core.run()
    finally:
        await store.close()

    kinds = [e["kind"] for e in events]
    assert "_run_failed" in kinds
    failed = next(e for e in events if e["kind"] == "_run_failed")
    assert failed["payload"]["error_class"] == "RuntimeError"
    assert "boom" in failed["payload"]["error_message"]


async def test_shutdown_emits_ack_event(tmp_path: Path) -> None:
    """ShutdownCommand is acknowledged on the event stream so the supervisor can wait for it."""

    agent = _FakeAgent()
    store = await _open_heartbeat_store(tmp_path)
    input_queue = UserInputQueue()
    events, sink = _captured_event_sink()

    try:
        commands = _async_lines([encode_command(ShutdownCommand())])
        core = WorkerCore(
            agent=agent,
            input_queue=input_queue,
            heartbeat_store=store,
            session_id="s1",
            pid=1,
            heartbeat_interval=0.05,
            command_source=commands,
            event_sink=sink,
        )
        await core.run()
    finally:
        await store.close()

    kinds = [e["kind"] for e in events]
    assert "_shutdown_ack" in kinds
