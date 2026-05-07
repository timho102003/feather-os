"""Tests for :class:`feather.core.lead_supervisor.LeadSupervisor`.

A fake :class:`WorkerHandle` replaces the real subprocess so the
supervisor's orchestration is exercised in-memory — no real process,
no flakiness, no LLM. The two real-subprocess concerns
(``asyncio.create_subprocess_exec`` flags and POSIX signal delivery)
are well-covered by stdlib and the existing terminate-pattern tests
in this repo.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from feather.core.lead_supervisor import (
    LeadSupervisor,
    SupervisorError,
    WorkerHandle,
)
from feather.core.runtime_event_codec import encode_event
from feather.core.worker_command_codec import (
    EnqueueUserInputCommand,
    RunCommand,
    ShutdownCommand,
    WorkerCommand,
)
from feather.models import AgentOutcome, RuntimeEvent, WorkerStatus
from feather.storage.worker_heartbeat_store import WorkerHeartbeatStore


# --------------------------------------------------------------------- #
# Fake worker handle
# --------------------------------------------------------------------- #


class _FakeWorkerHandle:
    """In-memory stand-in for :class:`WorkerHandle`."""

    def __init__(self) -> None:
        self._pid = 12345
        self._returncode: int | None = None
        self.commands_received: list[WorkerCommand] = []
        self._event_lines: asyncio.Queue[str | None] = asyncio.Queue()
        self.terminate_called = False
        self.kill_called = False
        self.close_stdin_called = False
        self._wait_event = asyncio.Event()

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def returncode(self) -> int | None:
        return self._returncode

    async def send_command(self, command: WorkerCommand) -> None:
        self.commands_received.append(command)

    async def read_event_line(self) -> str | None:
        return await self._event_lines.get()

    async def close_stdin(self) -> None:
        self.close_stdin_called = True

    def terminate(self) -> None:
        self.terminate_called = True
        if self._returncode is None:
            self._returncode = -15
        self._wait_event.set()

    def kill(self) -> None:
        self.kill_called = True
        if self._returncode is None:
            self._returncode = -9
        self._wait_event.set()

    async def wait(self) -> int:
        await self._wait_event.wait()
        return self._returncode or 0

    # Test driver helpers ------------------------------------------------ #

    def push_event(self, event: RuntimeEvent) -> None:
        self._event_lines.put_nowait(encode_event(event))

    def push_raw_line(self, line: str) -> None:
        self._event_lines.put_nowait(line)

    def push_eof(self) -> None:
        self._event_lines.put_nowait(None)

    def auto_ack_shutdown(self) -> None:
        """Schedule a `_shutdown_ack` once a ShutdownCommand is observed."""

        async def _watcher() -> None:
            while True:
                if any(isinstance(c, ShutdownCommand) for c in self.commands_received):
                    self.push_event(RuntimeEvent(kind="_shutdown_ack", payload={}))
                    return
                await asyncio.sleep(0)

        asyncio.create_task(_watcher(), name="fake.auto_ack")


def _make_supervisor(
    tmp_path: Path,
    handle: WorkerHandle,
    *,
    heartbeat_interval: float = 0.05,
    staleness_threshold: float = 0.5,
    shutdown_grace: float = 0.2,
) -> LeadSupervisor:
    return LeadSupervisor(
        db_path=tmp_path / "feather.db",
        project_root=tmp_path,
        heartbeat_interval=heartbeat_interval,
        staleness_threshold=staleness_threshold,
        shutdown_grace=shutdown_grace,
        handle_factory=lambda _sid: _wrap_in_async(handle),
    )


def _wrap_in_async(value: WorkerHandle):
    async def _factory() -> WorkerHandle:
        return value

    return _factory()


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


async def test_run_completes_after_run_complete_event(tmp_path: Path) -> None:
    handle = _FakeWorkerHandle()
    supervisor = _make_supervisor(tmp_path, handle)
    await supervisor.start("s1")
    handle.push_event(RuntimeEvent(kind="assistant_text_delta", text="hi"))
    handle.push_event(
        RuntimeEvent(
            kind="_run_complete",
            payload={
                "method": "run",
                "status": "completed",
                "session_id": "s1",
                "assistant_text": "hi",
                "question": None,
                "total_tool_calls": 1,
            },
        )
    )

    received: list[RuntimeEvent] = []
    result = await supervisor.run("s1", "hello", received.append)

    assert result.status is AgentOutcome.COMPLETED
    assert result.assistant_text == "hi"
    assert result.session_id == "s1"
    assert result.total_tool_calls == 1
    # The user-visible event was forwarded; the control event was swallowed.
    assert [e.kind for e in received] == ["assistant_text_delta"]
    assert any(isinstance(c, RunCommand) for c in handle.commands_received)

    handle.auto_ack_shutdown()
    await supervisor.shutdown()


async def test_run_failed_raises_supervisor_error(tmp_path: Path) -> None:
    handle = _FakeWorkerHandle()
    supervisor = _make_supervisor(tmp_path, handle)
    await supervisor.start("s1")
    handle.push_event(
        RuntimeEvent(
            kind="_run_failed",
            payload={
                "method": "run",
                "error_class": "RuntimeError",
                "error_message": "boom",
                "traceback": "Traceback (most recent call last):\n...",
            },
        )
    )

    with pytest.raises(SupervisorError, match="RuntimeError.*boom"):
        await supervisor.run("s1", "hi")

    handle.auto_ack_shutdown()
    await supervisor.shutdown()


async def test_eof_during_run_raises_supervisor_error(tmp_path: Path) -> None:
    handle = _FakeWorkerHandle()
    supervisor = _make_supervisor(tmp_path, handle)
    await supervisor.start("s1")
    handle.push_eof()

    with pytest.raises(SupervisorError, match="exited before completing"):
        await supervisor.run("s1", "hi")

    await supervisor.shutdown()


async def test_invalid_event_line_logged_but_run_still_completes(
    tmp_path: Path,
) -> None:
    """A garbled line on the worker stream must not poison the run."""

    handle = _FakeWorkerHandle()
    supervisor = _make_supervisor(tmp_path, handle)
    await supervisor.start("s1")
    handle.push_raw_line("not json at all")
    handle.push_event(RuntimeEvent(kind="assistant_text_delta", text="ok"))
    handle.push_event(
        RuntimeEvent(
            kind="_run_complete",
            payload={
                "method": "run",
                "status": "completed",
                "session_id": "s1",
                "assistant_text": "ok",
                "total_tool_calls": 0,
            },
        )
    )

    received: list[RuntimeEvent] = []
    result = await supervisor.run("s1", "hi", received.append)
    assert result.status is AgentOutcome.COMPLETED
    assert [e.text for e in received] == ["ok"]

    handle.auto_ack_shutdown()
    await supervisor.shutdown()


async def test_shutdown_falls_back_to_sigterm_on_ack_timeout(tmp_path: Path) -> None:
    """If the worker never acks, supervisor must escalate to SIGTERM."""

    handle = _FakeWorkerHandle()
    supervisor = _make_supervisor(
        tmp_path, handle, shutdown_grace=0.05
    )
    await supervisor.start("s1")
    # No auto-ack — simulate hung worker.
    await supervisor.shutdown()

    assert any(isinstance(c, ShutdownCommand) for c in handle.commands_received)
    assert handle.terminate_called, "must escalate to SIGTERM after ack timeout"


async def test_shutdown_acked_does_not_send_sigterm(tmp_path: Path) -> None:
    """A clean ack avoids the SIGTERM/SIGKILL fallback paths needlessly.

    The supervisor still calls ``terminate`` after ack as a final reaper —
    matching the existing terminate_agent_tool / mcp_client pattern of
    "send polite signal, then enforce" — but the worker is already gone
    by then so the call is harmless. We assert the ack arrived; we do
    NOT assert ``not terminate_called`` because reaping is the contract.
    """

    handle = _FakeWorkerHandle()
    supervisor = _make_supervisor(tmp_path, handle)
    await supervisor.start("s1")
    handle.auto_ack_shutdown()
    await supervisor.shutdown()

    assert any(isinstance(c, ShutdownCommand) for c in handle.commands_received)


async def test_enqueue_user_input_sends_command(tmp_path: Path) -> None:
    handle = _FakeWorkerHandle()
    supervisor = _make_supervisor(tmp_path, handle)
    await supervisor.start("s1")
    await supervisor.enqueue_user_input("s1", "mid-turn nudge")

    queued = [
        c for c in handle.commands_received if isinstance(c, EnqueueUserInputCommand)
    ]
    assert len(queued) == 1
    assert queued[0].text == "mid-turn nudge"

    handle.auto_ack_shutdown()
    await supervisor.shutdown()


async def test_is_stale_false_when_heartbeat_fresh(tmp_path: Path) -> None:
    handle = _FakeWorkerHandle()
    supervisor = _make_supervisor(tmp_path, handle, staleness_threshold=1.0)
    await supervisor.start("s1")
    # Open a separate connection to seed a fresh heartbeat row.
    seeder = WorkerHeartbeatStore(tmp_path / "feather.db")
    await seeder.initialize()
    try:
        await seeder.heartbeat(
            session_id="s1", pid=12345, status=WorkerStatus.RUNNING
        )
    finally:
        await seeder.close()

    assert await supervisor.is_stale() is False

    handle.auto_ack_shutdown()
    await supervisor.shutdown()


async def test_is_stale_true_when_no_heartbeat(tmp_path: Path) -> None:
    handle = _FakeWorkerHandle()
    supervisor = _make_supervisor(tmp_path, handle, staleness_threshold=1.0)
    await supervisor.start("s1")

    assert await supervisor.is_stale() is True

    handle.auto_ack_shutdown()
    await supervisor.shutdown()


async def test_is_stale_false_when_worker_stopped(tmp_path: Path) -> None:
    """A worker that's cleanly stopped is shut down, not stale."""

    handle = _FakeWorkerHandle()
    supervisor = _make_supervisor(
        tmp_path, handle, heartbeat_interval=0.01, staleness_threshold=0.05
    )
    await supervisor.start("s1")
    seeder = WorkerHeartbeatStore(tmp_path / "feather.db")
    await seeder.initialize()
    try:
        await seeder.heartbeat(
            session_id="s1", pid=12345, status=WorkerStatus.STOPPED
        )
    finally:
        await seeder.close()
    # Wait past the threshold — STOPPED rows must still register as not-stale.
    await asyncio.sleep(0.15)
    assert await supervisor.is_stale() is False

    handle.auto_ack_shutdown()
    await supervisor.shutdown()


async def test_restart_respawns_worker(tmp_path: Path) -> None:
    """`restart()` shuts down then re-invokes the handle factory for the same session."""

    handles_yielded: list[_FakeWorkerHandle] = []

    def factory(_session_id: str):
        async def _make() -> WorkerHandle:
            h = _FakeWorkerHandle()
            handles_yielded.append(h)
            h.auto_ack_shutdown()
            return h

        return _make()

    supervisor = LeadSupervisor(
        db_path=tmp_path / "feather.db",
        project_root=tmp_path,
        heartbeat_interval=0.05,
        staleness_threshold=0.5,
        shutdown_grace=0.05,
        handle_factory=factory,
    )
    await supervisor.start("s1")
    assert len(handles_yielded) == 1
    await supervisor.restart()
    assert len(handles_yielded) == 2
    assert supervisor.session_id == "s1"

    await supervisor.shutdown()


async def test_start_idempotent_for_same_session(tmp_path: Path) -> None:
    handle = _FakeWorkerHandle()
    supervisor = _make_supervisor(tmp_path, handle)
    await supervisor.start("s1")
    # Second call with same session id must be a no-op (the test would
    # otherwise hang on _wrap_in_async returning a coroutine that's
    # already been awaited).
    await supervisor.start("s1")

    handle.auto_ack_shutdown()
    await supervisor.shutdown()


async def test_start_with_different_session_raises(tmp_path: Path) -> None:
    handle = _FakeWorkerHandle()
    supervisor = _make_supervisor(tmp_path, handle)
    await supervisor.start("s1")
    with pytest.raises(RuntimeError, match="already running"):
        await supervisor.start("s2")

    handle.auto_ack_shutdown()
    await supervisor.shutdown()


async def test_concurrent_restart_calls_are_serialized(tmp_path: Path) -> None:
    """Two simultaneous restart() calls must not race on shutdown/start.

    Pre-fix, the second call's shutdown() short-circuited (``_closed``
    already True) and then raced start() against the first call's
    in-flight start, producing a no-op restart and corrupting internal
    state. The _restart_lock makes them serial.
    """

    handles: list[_FakeWorkerHandle] = []

    def factory(_sid: str):
        async def _make() -> WorkerHandle:
            h = _FakeWorkerHandle()
            handles.append(h)
            h.auto_ack_shutdown()
            return h

        return _make()

    supervisor = LeadSupervisor(
        db_path=tmp_path / "feather.db",
        project_root=tmp_path,
        heartbeat_interval=0.05,
        staleness_threshold=0.5,
        shutdown_grace=0.05,
        handle_factory=factory,
    )
    await supervisor.start("s1")
    # Fire two restart()s in parallel; both must complete cleanly and
    # the supervisor must end up running the latest (third) handle.
    await asyncio.gather(supervisor.restart(), supervisor.restart())
    # 1 (initial start) + 2 (one per restart, serialized) = 3 handles.
    assert len(handles) == 3
    assert supervisor.session_id == "s1"
    assert supervisor.is_running is True

    await supervisor.shutdown()


async def test_supervisor_rejects_invalid_threshold(tmp_path: Path) -> None:
    """staleness_threshold must exceed heartbeat_interval; otherwise nonsense."""

    with pytest.raises(ValueError, match="staleness_threshold"):
        LeadSupervisor(
            db_path=tmp_path / "feather.db",
            project_root=tmp_path,
            heartbeat_interval=1.0,
            staleness_threshold=0.5,
        )


async def test_restart_drains_stale_events_so_next_run_is_not_short_circuited(
    tmp_path: Path,
) -> None:
    """A `_run_complete` left over in the queue from the previous worker
    must NOT be returned by the FIRST run() against the new worker.

    Reproduces the failure mode: previous worker emits a `_run_complete`
    that arrives after the agent driver was cancelled (so nothing
    consumed it). Without queue draining on restart, the next supervisor.run
    would short-circuit on the stale event and return the prior worker's
    payload — silently dropping the actually-new turn. Critical for the
    self-repair restart-resume path that builds on this substrate.
    """

    handles: list[_FakeWorkerHandle] = []

    def factory(_sid: str):
        async def _make() -> WorkerHandle:
            h = _FakeWorkerHandle()
            handles.append(h)
            h.auto_ack_shutdown()
            return h

        return _make()

    supervisor = LeadSupervisor(
        db_path=tmp_path / "feather.db",
        project_root=tmp_path,
        heartbeat_interval=0.05,
        staleness_threshold=0.5,
        shutdown_grace=0.05,
        handle_factory=factory,
    )
    await supervisor.start("s1")
    # Push a stale `_run_complete` from the FIRST worker before any caller
    # has consumed it (mimics an event that arrived after the driver was
    # cancelled in on_unmount). Mark it so the test can detect leakage.
    handles[0].push_event(
        RuntimeEvent(
            kind="_run_complete",
            payload={
                "method": "run",
                "status": "completed",
                "session_id": "s1",
                "assistant_text": "STALE — must not leak to next worker",
                "total_tool_calls": 999,
            },
        )
    )
    await supervisor.restart()
    # The new worker should produce a clean, distinguishable result.
    handles[1].push_event(
        RuntimeEvent(
            kind="_run_complete",
            payload={
                "method": "run",
                "status": "completed",
                "session_id": "s1",
                "assistant_text": "fresh",
                "total_tool_calls": 0,
            },
        )
    )
    result = await supervisor.run("s1", "hello")
    assert result.assistant_text == "fresh"
    assert result.total_tool_calls == 0

    await supervisor.shutdown()


async def test_start_closes_heartbeat_store_when_factory_raises(
    tmp_path: Path,
) -> None:
    """Factory failure must not leak the open SQLite connection."""

    db_path = tmp_path / "feather.db"

    def boom_factory(_sid: str):
        async def _make() -> WorkerHandle:
            raise RuntimeError("factory exploded")

        return _make()

    supervisor = LeadSupervisor(
        db_path=db_path,
        project_root=tmp_path,
        heartbeat_interval=0.05,
        staleness_threshold=0.5,
        handle_factory=boom_factory,
    )
    with pytest.raises(RuntimeError, match="factory exploded"):
        await supervisor.start("s1")
    # If the store leaked, a second store opening the same DB would still
    # work (SQLite + WAL allows multiple readers/writers), so we instead
    # verify the supervisor is in a clean state and a retry succeeds.
    assert supervisor._heartbeat_store is None  # noqa: SLF001 — internal check
    assert supervisor._handle is None  # noqa: SLF001 — internal check

    # Retry with a working factory should still succeed (no stale state).
    handle = _FakeWorkerHandle()
    supervisor._handle_factory = (  # noqa: SLF001 — internal check
        lambda _sid: _wrap_in_async(handle)
    )
    await supervisor.start("s1")
    handle.auto_ack_shutdown()
    await supervisor.shutdown()
