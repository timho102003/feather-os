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
    CONFIG_RELOAD_ACK_KIND,
    ConfigReloadCommand,
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


async def test_request_shutdown_wakes_idle_command_pump(tmp_path: Path) -> None:
    """SIGTERM-style ``request_shutdown`` while stdin is idle must terminate.

    The earlier ``async for`` impl only checked the shutdown flag *after*
    stdin yielded a line, so a SIGTERM arriving while ``readline`` was
    blocked on an idle pipe would never wake the worker until the pipe
    was closed externally. Verifies the explicit ``__anext__`` race in
    ``_command_pump`` against ``_shutdown_event``.
    """

    agent = _FakeAgent()
    store = await _open_heartbeat_store(tmp_path)
    input_queue = UserInputQueue()
    events, sink = _captured_event_sink()

    # An iterator that never yields — the pump must be woken from the outside.
    async def _idle_forever() -> AsyncIterator[str]:
        while True:
            await asyncio.sleep(60)
            yield ""  # pragma: no cover — never reached in this test

    try:
        core = WorkerCore(
            agent=agent,
            input_queue=input_queue,
            heartbeat_store=store,
            session_id="s-idle",
            pid=1,
            heartbeat_interval=0.05,
            command_source=_idle_forever(),
            event_sink=sink,
        )
        # Schedule a shutdown after a brief moment so the pump is parked
        # in its readline race when the flag flips.
        async def _trip_shutdown() -> None:
            await asyncio.sleep(0.1)
            core.request_shutdown()

        await asyncio.wait_for(
            asyncio.gather(core.run(), _trip_shutdown()), timeout=2.0
        )
        final = await store.get("s-idle")
        assert final is not None
        assert final.status is WorkerStatus.STOPPED
    finally:
        await store.close()


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


# ---------------------------------------------------------------------------
# Task 23/25 — _handle_reload_config in WorkerCore
# ---------------------------------------------------------------------------


class _FakeRuntime:
    """Minimal stand-in for FeatherRuntime used in reload handler tests."""

    def __init__(self) -> None:
        self._app_config = _FakeConfig(active_provider="openai")
        self._agents: dict[str, object] = {}
        self._agent_factory = _FakeAgentFactory(should_fail=False)
        self.reload_calls: int = 0
        self.rebuild_calls: list[str] = []

    async def reload_config(self) -> None:
        self.reload_calls += 1
        # Swap the config to simulate a successful disk read.
        self._app_config = _FakeConfig(active_provider="claude")

    @property
    def config(self) -> "_FakeConfig":
        return self._app_config

    def rebuild_agent(self, name: str) -> object:
        self.rebuild_calls.append(name)
        return object()


class _FakeRuntimeFailingReload(_FakeRuntime):
    """Runtime whose reload_config raises to test rollback."""

    async def reload_config(self) -> None:  # type: ignore[override]
        self.reload_calls += 1
        raise ValueError("active_provider=claude but no claude: block in app.yaml")


class _FakeRuntimeFailingRebuild(_FakeRuntime):
    """Runtime whose rebuild raises to test dry-run rollback."""

    async def reload_config(self) -> None:
        self.reload_calls += 1
        self._app_config = _FakeConfig(active_provider="claude")

    def rebuild_agent(self, name: str) -> object:
        raise ValueError("provider not configured")


@dataclass
class _FakeConfig:
    active_provider: str


class _FakeAgentFactory:
    def __init__(self, *, should_fail: bool) -> None:
        self._should_fail = should_fail
        self._app_config: Any = None
        self._provider: Any = None
        self._providers_by_name: dict[str, Any] = {}

    def build(self, name: str) -> object:
        if self._should_fail:
            raise ValueError("factory dry-run failed")
        return object()


async def _run_worker_with_reload(
    tmp_path: Path,
    commands_encoded: list[str],
    *,
    runtime: _FakeRuntime | None = None,
) -> list[dict[str, Any]]:
    """Helper: run WorkerCore with a fake runtime and capture events."""
    agent = _FakeAgent()
    store = await _open_heartbeat_store(tmp_path)
    input_queue = UserInputQueue()
    events, sink = _captured_event_sink()

    try:
        commands = _async_lines(commands_encoded)
        core = WorkerCore(
            agent=agent,
            input_queue=input_queue,
            heartbeat_store=store,
            session_id="s1",
            pid=1,
            heartbeat_interval=0.05,
            command_source=commands,
            event_sink=sink,
            runtime=runtime,
        )
        await core.run()
    finally:
        await store.close()

    return events


async def test_reload_config_live_emits_success_ack(tmp_path: Path) -> None:
    """A live-class ConfigReloadCommand results in a successful ack event."""

    runtime = _FakeRuntime()
    runtime._agents["lead"] = object()

    events = await _run_worker_with_reload(
        tmp_path,
        [
            encode_command(
                ConfigReloadCommand(
                    correlation_id="corr-1",
                    changed_paths=["app.compaction.trigger_ratio"],
                    reload_class="live",
                )
            ),
            encode_command(ShutdownCommand()),
        ],
        runtime=runtime,
    )

    ack_events = [e for e in events if e["kind"] == CONFIG_RELOAD_ACK_KIND]
    assert len(ack_events) == 1
    ack = ack_events[0]["payload"]
    assert ack["ok"] is True
    assert ack["correlation_id"] == "corr-1"
    assert ack["applied_paths"] == ["app.compaction.trigger_ratio"]
    assert ack["error"] is None
    # Live reload calls reload_config but does NOT call rebuild.
    assert runtime.reload_calls == 1
    assert runtime.rebuild_calls == []


async def test_reload_config_next_turn_rebuilds_agents(tmp_path: Path) -> None:
    """next_turn reload calls reload_config and rebuild for every cached agent."""

    runtime = _FakeRuntime()
    runtime._agents["lead"] = object()

    events = await _run_worker_with_reload(
        tmp_path,
        [
            encode_command(
                ConfigReloadCommand(
                    correlation_id="corr-2",
                    changed_paths=["app.active_provider"],
                    reload_class="next_turn",
                )
            ),
            encode_command(ShutdownCommand()),
        ],
        runtime=runtime,
    )

    ack_events = [e for e in events if e["kind"] == CONFIG_RELOAD_ACK_KIND]
    assert len(ack_events) == 1
    ack = ack_events[0]["payload"]
    assert ack["ok"] is True
    assert runtime.reload_calls == 1
    assert "lead" in runtime.rebuild_calls


async def test_reload_config_without_runtime_emits_error_ack(tmp_path: Path) -> None:
    """When no runtime is attached the handler emits an error ack (not a crash)."""

    events = await _run_worker_with_reload(
        tmp_path,
        [
            encode_command(
                ConfigReloadCommand(
                    correlation_id="corr-3",
                    changed_paths=["app.openai.model"],
                    reload_class="live",
                )
            ),
            encode_command(ShutdownCommand()),
        ],
        runtime=None,
    )

    ack_events = [e for e in events if e["kind"] == CONFIG_RELOAD_ACK_KIND]
    assert len(ack_events) == 1
    ack = ack_events[0]["payload"]
    assert ack["ok"] is False
    assert "no runtime" in ack["error"].lower()


async def test_reload_config_rollback_on_reload_failure(tmp_path: Path) -> None:
    """Task 25: when reload_config raises, prior _app_config is restored (rollback)."""

    runtime = _FakeRuntimeFailingReload()
    runtime._agents["lead"] = object()
    prior_config = runtime._app_config

    events = await _run_worker_with_reload(
        tmp_path,
        [
            encode_command(
                ConfigReloadCommand(
                    correlation_id="corr-4",
                    changed_paths=["app.active_provider"],
                    reload_class="live",
                )
            ),
            encode_command(ShutdownCommand()),
        ],
        runtime=runtime,
    )

    ack_events = [e for e in events if e["kind"] == CONFIG_RELOAD_ACK_KIND]
    assert len(ack_events) == 1
    ack = ack_events[0]["payload"]
    assert ack["ok"] is False
    assert "claude" in ack["error"].lower() or "app.yaml" in ack["error"].lower()
    # Config must be rolled back to the prior value.
    assert runtime._app_config is prior_config


async def test_reload_config_rollback_on_next_turn_rebuild_failure(tmp_path: Path) -> None:
    """Task 25: dry-run rebuild failure rolls back _app_config and emits error ack."""

    runtime = _FakeRuntimeFailingRebuild()
    runtime._agents["lead"] = object()
    prior_config = runtime._app_config

    events = await _run_worker_with_reload(
        tmp_path,
        [
            encode_command(
                ConfigReloadCommand(
                    correlation_id="corr-5",
                    changed_paths=["app.active_provider"],
                    reload_class="next_turn",
                )
            ),
            encode_command(ShutdownCommand()),
        ],
        runtime=runtime,
    )

    ack_events = [e for e in events if e["kind"] == CONFIG_RELOAD_ACK_KIND]
    assert len(ack_events) == 1
    ack = ack_events[0]["payload"]
    assert ack["ok"] is False
    assert "provider" in ack["error"].lower()
    # Config was rolled back.
    assert runtime._app_config is prior_config
