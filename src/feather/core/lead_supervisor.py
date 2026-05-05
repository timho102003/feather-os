"""Supervisor that owns the lead-worker subprocess from the TUI side.

The supervisor lives in the Textual TUI process. It spawns the lead
worker (:mod:`feather.lead_worker_entry`) as a subprocess, writes
commands to its stdin, drains its stdout for ``RuntimeEvent`` lines,
and watches the ``worker_heartbeats`` row for staleness so the user
can be warned when the worker hangs.

The TUI calls into the supervisor through a surface that mirrors
:class:`feather.core.base_agent.BaseAgent`'s shape — :meth:`run`,
:meth:`resume_on_inbox`, :meth:`enqueue_user_input`,
:meth:`has_pending_inbox` — so the existing rendering and input
plumbing keeps working with a one-line swap from the in-process agent
reference to a supervisor reference.

Concurrency model:

* One persistent stdout reader task drains worker events into a queue.
* :meth:`run` and :meth:`resume_on_inbox` await that queue, dispatching
  every UI-visible event through the caller's ``event_handler`` and
  returning when the matching ``_run_complete`` / ``_run_failed``
  control event arrives.
* A separate background task is not used for heartbeat polling here —
  staleness is checked on demand via :meth:`is_stale`. The TUI runs
  its own poll loop already (see :meth:`_inbox_watcher`) and adding
  a parallel one would just duplicate cadence.

Test surface: the worker process is opened via an injectable
:class:`WorkerHandle` factory so tests can drive the supervisor with
in-memory streams instead of a real subprocess.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from feather.core.runtime_event_codec import EventCodecError, decode_event
from feather.core.worker_command_codec import (
    EnqueueUserInputCommand,
    ResumeOnInboxCommand,
    RunCommand,
    ShutdownCommand,
    WorkerCommand,
    encode_command,
)
from feather.models import (
    AgentOutcome,
    AgentRunResult,
    EventHandler,
    RuntimeEvent,
    WorkerStatus,
)
from feather.storage.worker_heartbeat_store import WorkerHeartbeatStore

logger = logging.getLogger(__name__)


_RUN_COMPLETE = "_run_complete"
_RUN_FAILED = "_run_failed"
_SHUTDOWN_ACK = "_shutdown_ack"

_DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 1.0
_DEFAULT_STALENESS_THRESHOLD_SECONDS = 5.0
_DEFAULT_SHUTDOWN_GRACE_SECONDS = 2.0


class SupervisorError(RuntimeError):
    """Raised when the worker dies or violates the protocol mid-run."""


class WorkerHandle(Protocol):
    """Abstract handle for one lead-worker subprocess.

    Production wraps :class:`asyncio.subprocess.Process`; tests pass an
    in-memory fake so the supervisor's orchestration is verifiable
    without a real subprocess.
    """

    @property
    def pid(self) -> int | None: ...

    @property
    def returncode(self) -> int | None: ...

    async def send_command(self, command: WorkerCommand) -> None:
        """Write one encoded command JSONL line to the worker's stdin."""
        ...

    async def read_event_line(self) -> str | None:
        """Return the next stdout line, or ``None`` on EOF."""
        ...

    async def close_stdin(self) -> None:
        """Close stdin so the worker observes EOF (implicit shutdown)."""
        ...

    def terminate(self) -> None:
        """Send SIGTERM."""
        ...

    def kill(self) -> None:
        """Send SIGKILL."""
        ...

    async def wait(self) -> int:
        """Wait for the worker process to exit; return the exit code."""
        ...


HandleFactory = Callable[[str], Awaitable[WorkerHandle]]
"""Callable that, given a session_id, returns a started worker handle."""


class LeadSupervisor:
    """Owns the lead-worker subprocess and exposes a BaseAgent-shaped surface."""

    def __init__(
        self,
        *,
        db_path: Path,
        project_root: Path,
        agent_name: str = "lead",
        heartbeat_interval: float = _DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        staleness_threshold: float = _DEFAULT_STALENESS_THRESHOLD_SECONDS,
        shutdown_grace: float = _DEFAULT_SHUTDOWN_GRACE_SECONDS,
        handle_factory: HandleFactory | None = None,
    ) -> None:
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if staleness_threshold <= heartbeat_interval:
            raise ValueError(
                "staleness_threshold must exceed heartbeat_interval — otherwise "
                "the supervisor can flag a perfectly healthy worker stale."
            )
        self._db_path = db_path
        self._project_root = project_root
        self._agent_name = agent_name
        self._heartbeat_interval = heartbeat_interval
        self._staleness_threshold = staleness_threshold
        self._shutdown_grace = shutdown_grace
        self._handle_factory = handle_factory or self._default_subprocess_factory

        self._handle: WorkerHandle | None = None
        self._session_id: str | None = None
        self._heartbeat_store: WorkerHeartbeatStore | None = None
        self._event_queue: asyncio.Queue[RuntimeEvent | None] = asyncio.Queue()
        self._reader_task: asyncio.Task[None] | None = None
        self._run_lock = asyncio.Lock()
        self._closed = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self, session_id: str) -> None:
        """Spawn the worker subprocess and start draining events.

        Idempotent for the same ``session_id``; calling with a different
        id while a worker is already running raises ``RuntimeError``
        (use :meth:`shutdown` + :meth:`start` to switch sessions).
        """

        if self._handle is not None:
            if self._session_id != session_id:
                raise RuntimeError(
                    f"supervisor already running for session {self._session_id!r}, "
                    f"cannot start({session_id!r})"
                )
            return

        store = WorkerHeartbeatStore(self._db_path)
        await store.initialize()
        self._heartbeat_store = store
        self._handle = await self._handle_factory(session_id)
        self._session_id = session_id
        self._reader_task = asyncio.create_task(
            self._stdout_reader(), name="supervisor.stdout_reader"
        )
        logger.info(
            "lead_supervisor started session_id=%s pid=%s",
            session_id,
            self._handle.pid,
        )

    async def shutdown(self) -> None:
        """Cleanly tear down the worker.

        Sequence: send :class:`ShutdownCommand` and wait for the
        ``_shutdown_ack`` control event; if neither arrives within
        ``shutdown_grace`` seconds, fall back to SIGTERM → 2 s wait →
        SIGKILL — the same dance used everywhere else in the tree
        (``mcp_client``, ``terminate_agent_tool``, ``task_tools``).

        Caller invariant: ``shutdown`` must not run concurrently with an
        in-flight :meth:`run` / :meth:`resume_on_inbox` for the same
        supervisor — both would race on the shared event queue and the
        ``_shutdown_ack`` could be consumed by the wrong waiter. The TUI
        honors this by cancelling its agent-driver task before calling
        ``shutdown`` in ``on_unmount``.
        """

        if self._closed or self._handle is None:
            return
        self._closed = True
        handle = self._handle
        try:
            try:
                await asyncio.wait_for(
                    handle.send_command(ShutdownCommand()),
                    timeout=self._shutdown_grace,
                )
                await asyncio.wait_for(
                    self._drain_until_kind(_SHUTDOWN_ACK),
                    timeout=self._shutdown_grace,
                )
            except (asyncio.TimeoutError, ConnectionError, BrokenPipeError):
                logger.warning(
                    "lead_supervisor shutdown ack timeout — falling back to SIGTERM "
                    "session_id=%s",
                    self._session_id,
                )
            with contextlib.suppress(Exception):
                await handle.close_stdin()
            handle.terminate()
            try:
                await asyncio.wait_for(handle.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "lead_supervisor SIGTERM timeout — escalating to SIGKILL "
                    "session_id=%s pid=%s",
                    self._session_id,
                    handle.pid,
                )
                handle.kill()
                try:
                    await asyncio.wait_for(handle.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.error(
                        "lead_supervisor SIGKILL timeout — giving up wait "
                        "session_id=%s pid=%s",
                        self._session_id,
                        handle.pid,
                    )
        finally:
            if self._reader_task is not None:
                self._reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._reader_task
                self._reader_task = None
            if self._heartbeat_store is not None:
                await self._heartbeat_store.close()
                self._heartbeat_store = None
            self._handle = None

    async def restart(self) -> None:
        """Stop the current worker and respawn it on the same session id."""

        if self._session_id is None:
            raise RuntimeError("supervisor.restart() called before start()")
        sid = self._session_id
        await self.shutdown()
        self._closed = False
        await self.start(sid)

    # ------------------------------------------------------------------ #
    # BaseAgent-shaped surface
    # ------------------------------------------------------------------ #

    async def run(
        self,
        session_id: str,
        incoming_text: str,
        event_handler: EventHandler | None = None,
    ) -> AgentRunResult:
        """Mirror :meth:`BaseAgent.run` from the TUI's perspective."""

        async with self._run_lock:
            handle = self._require_handle()
            await handle.send_command(
                RunCommand(session_id=session_id, incoming_text=incoming_text)
            )
            return await self._await_run_terminal(event_handler, method="run")

    async def resume_on_inbox(
        self,
        session_id: str,
        event_handler: EventHandler | None = None,
    ) -> AgentRunResult | None:
        """Mirror :meth:`BaseAgent.resume_on_inbox` from the TUI's perspective."""

        async with self._run_lock:
            handle = self._require_handle()
            await handle.send_command(ResumeOnInboxCommand(session_id=session_id))
            # The worker signals an empty-inbox no-op via status="no_op" in
            # the control event payload (see _serialize_result in
            # lead_worker_core); _result_from_payload coerces that to a
            # COMPLETED AgentRunResult, which is a valid terminal value to
            # return — no special-casing needed here.
            return await self._await_run_terminal(
                event_handler, method="resume_on_inbox"
            )

    async def enqueue_user_input(self, session_id: str, text: str) -> None:
        """Inject a mid-turn user message into the worker's input queue."""

        handle = self._require_handle()
        await handle.send_command(
            EnqueueUserInputCommand(session_id=session_id, text=text)
        )

    # ------------------------------------------------------------------ #
    # Liveness
    # ------------------------------------------------------------------ #

    async def is_stale(self, *, threshold: float | None = None) -> bool:
        """Return True if the latest worker heartbeat is older than threshold."""

        if self._heartbeat_store is None or self._session_id is None:
            return False
        record = await self._heartbeat_store.get(self._session_id)
        if record is None:
            return True
        # A worker that already wrote ``STOPPED`` is shut down, not stale.
        if record.status is WorkerStatus.STOPPED:
            return False
        cutoff = threshold if threshold is not None else self._staleness_threshold
        age = (datetime.now(UTC) - record.heartbeat_at).total_seconds()
        return age > cutoff

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def is_running(self) -> bool:
        return self._handle is not None and not self._closed

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_handle(self) -> WorkerHandle:
        if self._handle is None or self._closed:
            raise RuntimeError(
                "supervisor.start() must be called before run/resume_on_inbox."
            )
        return self._handle

    async def _stdout_reader(self) -> None:
        """Continuously drain worker stdout, decode events, push to queue."""

        handle = self._handle
        assert handle is not None
        while True:
            try:
                line = await handle.read_event_line()
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except Exception:  # noqa: BLE001
                logger.exception(
                    "lead_supervisor stdout read failed session_id=%s",
                    self._session_id,
                )
                await self._event_queue.put(None)
                return
            if line is None:
                # EOF — the worker is gone.
                await self._event_queue.put(None)
                return
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = decode_event(stripped)
            except EventCodecError as exc:
                logger.warning(
                    "lead_supervisor invalid event line session_id=%s err=%s line=%r",
                    self._session_id,
                    exc,
                    stripped[:200],
                )
                continue
            await self._event_queue.put(event)

    async def _await_run_terminal(
        self,
        event_handler: EventHandler | None,
        *,
        method: str,
    ) -> AgentRunResult:
        """Drain the event stream until a run-terminal control event arrives."""

        while True:
            event = await self._event_queue.get()
            if event is None:
                raise SupervisorError(
                    f"worker exited before completing {method}() session_id={self._session_id!r}"
                )
            if event.kind == _RUN_COMPLETE:
                return _result_from_payload(event.payload or {})
            if event.kind == _RUN_FAILED:
                payload = event.payload or {}
                raise SupervisorError(
                    f"worker {method}() failed: "
                    f"{payload.get('error_class', 'UnknownError')}: "
                    f"{payload.get('error_message', '')}"
                )
            if event.kind == _SHUTDOWN_ACK:
                # The supervisor only awaits shutdown ack via shutdown();
                # if one shows up here, swallow it rather than passing to UI.
                continue
            if event.kind.startswith("_"):
                logger.debug(
                    "lead_supervisor unknown control event kind=%s — swallowed",
                    event.kind,
                )
                continue
            if event_handler is not None:
                try:
                    event_handler(event)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "lead_supervisor event_handler raised — continuing"
                    )

    async def _drain_until_kind(self, kind: str) -> RuntimeEvent | None:
        """Drain the event queue until ``kind`` arrives (or EOF).

        Intermediate events are discarded — only used by :meth:`shutdown`,
        where any pending UI deltas are about to be torn down anyway.
        """

        while True:
            event = await self._event_queue.get()
            if event is None or event.kind == kind:
                return event

    # ------------------------------------------------------------------ #
    # Default production worker spawn
    # ------------------------------------------------------------------ #

    async def _default_subprocess_factory(self, session_id: str) -> WorkerHandle:
        """Spawn ``python -m feather.lead_worker_entry`` as the worker."""

        argv = [
            sys.executable,
            "-m",
            "feather.lead_worker_entry",
            "--session-id",
            session_id,
            "--root",
            str(self._project_root),
            "--heartbeat-interval",
            str(self._heartbeat_interval),
            "--agent-name",
            self._agent_name,
        ]
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._project_root),
        )
        return _SubprocessWorkerHandle(process)


def _result_from_payload(payload: dict[str, Any]) -> AgentRunResult:
    """Reconstruct an :class:`AgentRunResult` from a ``_run_complete`` payload."""

    status_raw = str(payload.get("status", "completed"))
    try:
        status = AgentOutcome(status_raw)
    except ValueError:
        # ``no_op`` and other non-AgentOutcome values default to COMPLETED
        # so the TUI's existing branching keeps working.
        status = AgentOutcome.COMPLETED
    return AgentRunResult(
        status=status,
        session_id=str(payload.get("session_id") or ""),
        assistant_text=str(payload.get("assistant_text") or ""),
        question=payload.get("question"),
        total_tool_calls=int(payload.get("total_tool_calls") or 0),
    )


class _SubprocessWorkerHandle:
    """Production :class:`WorkerHandle` wrapping :class:`asyncio.subprocess.Process`."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._stderr_buffer = bytearray()
        self._stderr_drainer: asyncio.Task[None] | None = None
        if process.stderr is not None:
            self._stderr_drainer = asyncio.create_task(
                self._drain_stderr(),
                name="supervisor.stderr_drain",
            )

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    @property
    def stderr_buffer(self) -> bytes:
        return bytes(self._stderr_buffer)

    async def send_command(self, command: WorkerCommand) -> None:
        if self._process.stdin is None or self._process.stdin.is_closing():
            raise BrokenPipeError("worker stdin is closed")
        line = encode_command(command).encode("utf-8")
        self._process.stdin.write(line)
        self._process.stdin.write(b"\n")
        await self._process.stdin.drain()

    async def read_event_line(self) -> str | None:
        if self._process.stdout is None:
            return None
        raw = await self._process.stdout.readline()
        if not raw:
            return None
        return raw.decode("utf-8", errors="replace")

    async def close_stdin(self) -> None:
        if self._process.stdin is None or self._process.stdin.is_closing():
            return
        self._process.stdin.close()
        with contextlib.suppress(Exception):
            await self._process.stdin.wait_closed()

    def terminate(self) -> None:
        if self._process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            self._process.send_signal(signal.SIGTERM)

    def kill(self) -> None:
        if self._process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            self._process.kill()

    async def wait(self) -> int:
        rc = await self._process.wait()
        if self._stderr_drainer is not None:
            with contextlib.suppress(Exception):
                await self._stderr_drainer
        return rc

    async def _drain_stderr(self) -> None:
        assert self._process.stderr is not None
        with contextlib.suppress(Exception):
            while True:
                chunk = await self._process.stderr.read(8192)
                if not chunk:
                    return
                self._stderr_buffer.extend(chunk)
