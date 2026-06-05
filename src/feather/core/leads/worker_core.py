"""Async coordinator for the lead-worker subprocess.

The worker subprocess (`feather.lead_worker_entry`) is a thin shell that
parses argv, builds the runtime, wires stdin/stdout, installs a SIGTERM
handler, and then hands control to :class:`WorkerCore`. All concurrent
behavior lives here so it can be unit-tested with in-memory streams and
a fake agent.

Three independent coroutines run concurrently:

* **command pump** consumes stdin lines, decodes each into a typed
  :mod:`worker_command_codec` command, and routes it. ``EnqueueUserInput``
  is dispatched directly so a user typing mid-run is not blocked behind
  the in-flight ``Run``/``Resume`` command. Other commands are queued.
* **heartbeat pump** writes a ``RUNNING`` row to ``worker_heartbeats``
  every ``heartbeat_interval`` seconds. Cancelled on shutdown.
* **command loop** drains the queue and serially dispatches
  ``RunCommand`` / ``ResumeOnInboxCommand`` / ``ShutdownCommand`` to the
  agent. Each agent run streams ``RuntimeEvent``s through ``event_sink``
  (one JSON line per event), then a control event ``_run_complete`` or
  ``_run_failed`` marks the call's terminal state.
* ``ConfigReloadCommand`` is handled serially by the command loop between
  turns: it calls :meth:`FeatherRuntime.reload_config` (and optionally
  :meth:`FeatherRuntime.rebuild_agent` for ``next_turn`` class), then
  emits a ``_config_reload_ack`` control event.

Shutdown sources, all unified through ``shutdown_event``:

* explicit :class:`ShutdownCommand` from the supervisor (graceful — the
  in-flight agent run is allowed to finish first);
* EOF on ``command_source`` (supervisor closed the worker's stdin);
* SIGTERM (the script layer flips the event from its signal handler).

After the three pumps unwind, a final ``STOPPED`` heartbeat is written so
the supervisor can distinguish "exited cleanly" from "hung/crashed".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from feather.config.schema import ReloadClass
from feather.core.session.input_queue import UserInputQueue
from feather.core.ipc.event_codec import encode_event
from feather.core.ipc.command_codec import (
    CONFIG_RELOAD_ACK_KIND,
    CommandCodecError,
    ConfigReloadCommand,
    EnqueueUserInputCommand,
    ResumeOnInboxCommand,
    RunCommand,
    ShutdownCommand,
    WorkerCommand,
    decode_command,
)
from feather.models import AgentRunResult, RuntimeEvent, WorkerStatus
from feather.storage.worker_heartbeat_store import WorkerHeartbeatStore

if TYPE_CHECKING:
    from feather.runtime import FeatherRuntime

logger = logging.getLogger(__name__)


EventSink = Callable[[str], None]
"""Callable invoked once per encoded JSONL event line (no trailing newline)."""


class _AgentLike(Protocol):
    """Subset of :class:`feather.core.agent.base.BaseAgent` the worker uses."""

    async def run(  # noqa: D401 — protocol surface
        self,
        session_id: str,
        incoming_text: str,
        event_handler: Callable[[RuntimeEvent], None] | None = None,
    ) -> AgentRunResult: ...

    async def resume_on_inbox(
        self,
        session_id: str,
        event_handler: Callable[[RuntimeEvent], None] | None = None,
    ) -> AgentRunResult | None: ...


class WorkerCore:
    """Coordinator for the lead-worker subprocess's three pumps."""

    def __init__(
        self,
        *,
        agent: _AgentLike,
        input_queue: UserInputQueue,
        heartbeat_store: WorkerHeartbeatStore,
        session_id: str,
        pid: int,
        heartbeat_interval: float,
        command_source: AsyncIterator[str],
        event_sink: EventSink,
        runtime: "FeatherRuntime | None" = None,
    ) -> None:
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        self._agent = agent
        self._input_queue = input_queue
        self._heartbeat_store = heartbeat_store
        self._session_id = session_id
        self._pid = pid
        self._heartbeat_interval = heartbeat_interval
        self._command_source = command_source
        self._event_sink = event_sink
        self._runtime = runtime
        self._command_queue: asyncio.Queue[WorkerCommand] = asyncio.Queue()
        self._shutdown_event = asyncio.Event()

    def request_shutdown(self) -> None:
        """Trip the shutdown flag (safe to call from a signal handler)."""

        self._shutdown_event.set()

    async def run(self) -> None:
        """Run the three pumps until shutdown is requested."""

        # Write an initial heartbeat synchronously so the supervisor can
        # observe "the worker started" without waiting a full interval.
        await self._heartbeat_store.heartbeat(
            session_id=self._session_id,
            pid=self._pid,
            status=WorkerStatus.RUNNING,
        )

        pumps = (
            asyncio.create_task(self._command_pump(), name="worker.cmd_pump"),
            asyncio.create_task(self._heartbeat_pump(), name="worker.hb_pump"),
            asyncio.create_task(self._command_loop(), name="worker.cmd_loop"),
        )
        try:
            await asyncio.gather(*pumps)
        finally:
            # Tear down any pump that's still alive. Cancellations and
            # secondary failures are swallowed so the final STOPPED
            # heartbeat below still fires; the original failure (if any)
            # already surfaced through the gather above.
            for task in pumps:
                if not task.done():
                    task.cancel()
            for task in pumps:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await self._heartbeat_store.heartbeat(
                session_id=self._session_id,
                pid=self._pid,
                status=WorkerStatus.STOPPED,
            )

    # ------------------------------------------------------------------ #
    # Pumps
    # ------------------------------------------------------------------ #

    async def _command_pump(self) -> None:
        # Use explicit ``__anext__`` instead of ``async for`` so each fetch
        # can be raced against ``_shutdown_event.wait()``. With ``async for``
        # the loop body only runs *after* a line arrives, which means a
        # SIGTERM that fires while ``readline`` is blocked on an idle
        # stdin would not wake the pump until the supervisor (or the OS)
        # closed the pipe.
        async def _read_one() -> str:
            return await self._command_source.__anext__()

        try:
            while not self._shutdown_event.is_set():
                next_task = asyncio.create_task(_read_one())
                stop_task = asyncio.create_task(self._shutdown_event.wait())
                done, pending = await asyncio.wait(
                    {next_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if next_task not in done:
                    # Shutdown won the race — drain pending cancellations
                    # so the cancelled readline doesn't surface a warning.
                    with contextlib.suppress(
                        asyncio.CancelledError, Exception
                    ):
                        await next_task
                    return
                try:
                    raw_line = next_task.result()
                except StopAsyncIteration:
                    return
                try:
                    cmd = decode_command(raw_line)
                except CommandCodecError as exc:
                    logger.warning(
                        "worker invalid command, skipping: %s | line=%r",
                        exc,
                        raw_line[:200],
                    )
                    continue
                # EnqueueUserInputCommand bypasses the serial command
                # queue so a user typing mid-run is never blocked behind
                # an in-flight Run/Resume.
                if isinstance(cmd, EnqueueUserInputCommand):
                    await self._input_queue.enqueue(cmd.session_id, cmd.text)
                    continue
                await self._command_queue.put(cmd)
        finally:
            # Source exhausted (EOF) or raised — implicit shutdown.
            self._shutdown_event.set()

    async def _heartbeat_pump(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._heartbeat_interval,
                )
                # shutdown_event was set during the wait — exit.
                return
            except asyncio.TimeoutError:
                # Cadence tick — refresh heartbeat.
                try:
                    await self._heartbeat_store.heartbeat(
                        session_id=self._session_id,
                        pid=self._pid,
                        status=WorkerStatus.RUNNING,
                    )
                except Exception:  # noqa: BLE001
                    # A failed heartbeat write is observability-fatal but
                    # not run-fatal: the supervisor will detect staleness.
                    logger.exception("worker heartbeat write failed")

    async def _command_loop(self) -> None:
        while True:
            cmd = await self._next_command_or_shutdown()
            if cmd is None:
                return
            if isinstance(cmd, ShutdownCommand):
                self._emit_control_event("_shutdown_ack", payload={})
                self._shutdown_event.set()
                return
            if isinstance(cmd, RunCommand):
                await self._dispatch_run(
                    "run",
                    self._agent.run(cmd.session_id, cmd.incoming_text, self._stream_event),
                )
            elif isinstance(cmd, ResumeOnInboxCommand):
                await self._dispatch_run(
                    "resume_on_inbox",
                    self._agent.resume_on_inbox(cmd.session_id, self._stream_event),
                )
            elif isinstance(cmd, ConfigReloadCommand):
                await self._handle_reload_config(cmd)
            # EnqueueUserInputCommand should never reach the loop — the
            # pump dispatches it directly. Defensive log if it does.
            else:  # pragma: no cover — defensive
                logger.error(
                    "worker command_loop received unexpected command type: %r",
                    type(cmd).__name__,
                )

    async def _next_command_or_shutdown(self) -> WorkerCommand | None:
        """Return the next queued command, or ``None`` if shutdown fires first.

        Queued commands always take precedence over shutdown so a graceful
        :class:`ShutdownCommand` (or any work the supervisor enqueued
        before closing stdin) is fully processed before the loop exits.
        """

        if not self._command_queue.empty():
            return await self._command_queue.get()
        get_task = asyncio.create_task(self._command_queue.get())
        stop_task = asyncio.create_task(self._shutdown_event.wait())
        done, pending = await asyncio.wait(
            {get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if get_task in done:
            return get_task.result()
        return None

    # ------------------------------------------------------------------ #
    # Run dispatch helpers
    # ------------------------------------------------------------------ #

    async def _dispatch_run(
        self,
        method_name: str,
        coro: Awaitable[AgentRunResult | None],
    ) -> None:
        """Await an agent-run coroutine and emit its terminal control event."""

        try:
            result = await coro
        except Exception as exc:  # noqa: BLE001
            logger.exception("worker agent.%s crashed: %s", method_name, exc)
            self._emit_control_event(
                "_run_failed",
                payload={
                    "method": method_name,
                    "error_class": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            return
        self._emit_control_event(
            "_run_complete",
            payload=_serialize_result(method_name, result),
        )

    async def _handle_reload_config(self, cmd: ConfigReloadCommand) -> None:
        """Handle a :class:`ConfigReloadCommand` between agent turns.

        Implements a validate-then-swap strategy:

        1. Snapshot the current ``_app_config`` so it can be restored on error.
        2. Call :meth:`FeatherRuntime.reload_config` to swap ``_app_config`` from
           disk.
        3. For ``next_turn`` reload class, perform a **dry-run** agent rebuild
           using the factory before committing.  If the rebuild raises
           (e.g. invalid ``active_provider`` missing its config block), the prior
           config is restored and an error ack is emitted.
        4. Emit a ``_config_reload_ack`` control event with the outcome so the
           supervisor can unblock its waiting :meth:`request_config_reload` call.

        When no runtime is attached (unit-test mode without a real
        :class:`FeatherRuntime`), the ack is emitted with ``ok=False`` and an
        explanatory error so tests that intentionally omit the runtime see a
        clear failure rather than a silent no-op.

        Args:
            cmd: The decoded :class:`ConfigReloadCommand` from the supervisor.
        """

        if self._runtime is None:
            logger.warning(
                "worker received reload_config but no runtime attached "
                "correlation_id=%s — emitting error ack",
                cmd.correlation_id,
            )
            self._emit_control_event(
                CONFIG_RELOAD_ACK_KIND,
                payload={
                    "correlation_id": cmd.correlation_id,
                    "ok": False,
                    "applied_paths": [],
                    "error": "WorkerCore has no runtime attached",
                },
            )
            return

        prior_config = self._runtime.config
        try:
            await self._runtime.reload_config()
            if cmd.reload_class == ReloadClass.NEXT_TURN.value:
                # Two-phase: build each cached agent first to surface a bad
                # config (e.g. active_provider missing its block) BEFORE any
                # cached instance is swapped, so a failing build rolls back
                # cleanly with no partial state.
                cached_agents = list(self._runtime._agents)
                for agent_name in cached_agents:
                    self._runtime._agent_factory.build(agent_name)
                for agent_name in cached_agents:
                    self._runtime.rebuild_agent(agent_name)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "worker reload_config failed — rolling back "
                "correlation_id=%s error=%s",
                cmd.correlation_id,
                exc,
            )
            # Roll back: restore prior config on the runtime so the agent
            # keeps running against the known-good config.
            self._runtime._app_config = prior_config
            self._emit_control_event(
                CONFIG_RELOAD_ACK_KIND,
                payload={
                    "correlation_id": cmd.correlation_id,
                    "ok": False,
                    "applied_paths": [],
                    "error": str(exc),
                },
            )
            return

        logger.info(
            "worker reload_config applied reload_class=%s paths=%r "
            "correlation_id=%s",
            cmd.reload_class,
            cmd.changed_paths,
            cmd.correlation_id,
        )
        self._emit_control_event(
            CONFIG_RELOAD_ACK_KIND,
            payload={
                "correlation_id": cmd.correlation_id,
                "ok": True,
                "applied_paths": list(cmd.changed_paths),
                "error": None,
            },
        )

    def _stream_event(self, event: RuntimeEvent) -> None:
        """Forward one ``RuntimeEvent`` to the event sink as a JSONL line."""

        try:
            line = encode_event(event)
        except Exception:  # noqa: BLE001
            logger.exception("worker event encode failed kind=%s", event.kind)
            return
        try:
            self._event_sink(line)
        except Exception:  # noqa: BLE001
            # An event-sink failure (broken pipe — supervisor died) means
            # the worker is talking into the void. Trip shutdown so the
            # heartbeat ticker also winds down rather than spinning.
            logger.exception("worker event sink failed; shutting down")
            self._shutdown_event.set()

    def _emit_control_event(self, kind: str, *, payload: dict[str, Any]) -> None:
        """Emit a supervisor-internal control event (``_``-prefixed kind)."""

        self._stream_event(RuntimeEvent(kind=kind, payload=payload))


def _serialize_result(
    method_name: str, result: AgentRunResult | None
) -> dict[str, Any]:
    """Project an :class:`AgentRunResult` into a JSON-friendly payload."""

    if result is None:
        return {"method": method_name, "status": "no_op"}
    return {
        "method": method_name,
        "status": result.status.value,
        "session_id": result.session_id,
        "assistant_text": result.assistant_text,
        "question": result.question,
        "total_tool_calls": result.total_tool_calls,
    }
