"""Supervisor-side watcher: trigger worker restart when the lead asks.

The ``request_restart`` tool sets a flag on the session row; this
watcher polls that flag from the supervisor (TUI) process. On a
not-pending → pending transition it:

1. Cancels any in-flight ``LeadSupervisor.run`` (see invariant in
   ``LeadSupervisor.shutdown``).
2. Calls ``LeadSupervisor.restart()`` — graceful SIGTERM/SIGKILL +
   respawn on the same ``session_id`` (so conversation history is
   preserved).
3. Drops a synthetic message into the lead's mailbox so the next
   ``resume_on_inbox`` cycle has something to drain — the lead sees
   "restart succeeded — resume your task" as its next user-equivalent
   input and continues naturally.
4. Clears the flag so the next pending-transition is detected freshly.

Errors during restart are surfaced to the same mailbox so the lead's
next turn shows the failure rather than silently losing state.

Pure orchestration logic (transition detection + dispatch) is split
from the polling loop so unit tests can drive ``run_once`` directly
without timing dependencies.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from feather.core.constants import LEAD_AGENT_NAME
from feather.storage.agent_message_store import AgentMessageStore
from feather.storage.session_store import SessionStore

logger = logging.getLogger(__name__)


_DEFAULT_POLL_INTERVAL_SECONDS = 1.5
# Cap on how long we wait for the in-flight run task to wind down before
# triggering restart anyway. The supervisor's restart() will SIGKILL the
# worker regardless, so a stuck cleanup must not block the watchdog.
_DEFAULT_CANCEL_TIMEOUT_SECONDS = 10.0


@dataclass(slots=True, frozen=True)
class _RestartOutcome:
    """Result of one restart attempt — surfaced into the lead's inbox."""

    succeeded: bool
    message: str


# Cancel callback returns True if it actually cancelled an in-flight task.
CancelInFlightRunCallback = Callable[[], Awaitable[bool]]


class RestartWatcher:
    """Periodic poll of ``sessions.restart_requested_at`` → ``supervisor.restart()``."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        message_store: AgentMessageStore,
        lead_session_id: str,
        restart_fn: Callable[[], Awaitable[None]],
        cancel_in_flight_run: CancelInFlightRunCallback | None = None,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        cancel_timeout_seconds: float = _DEFAULT_CANCEL_TIMEOUT_SECONDS,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if cancel_timeout_seconds <= 0:
            raise ValueError("cancel_timeout_seconds must be positive")
        self._session_store = session_store
        self._message_store = message_store
        self._lead_session_id = lead_session_id
        self._restart = restart_fn
        self._cancel_in_flight = cancel_in_flight_run
        self._poll_interval = poll_interval_seconds
        self._cancel_timeout = cancel_timeout_seconds
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def run_once(self) -> bool:
        """Check the flag once; if set, perform the restart cycle.

        Returns True iff a restart was actually triggered (used by
        tests; production callers ignore the return value).
        """

        flag = await self._session_store.get_restart_request(
            self._lead_session_id
        )
        if flag is None:
            return False
        ts, reason = flag
        logger.info(
            "restart_watcher triggering session=%s ts=%s reason=%s",
            self._lead_session_id,
            ts,
            reason,
        )
        # Cancel any in-flight run so the supervisor.shutdown invariant
        # (no concurrent run/shutdown) holds when restart() runs. Bound
        # the wait — if the run task's cleanup hangs (slow MCP shutdown,
        # gh subprocess timeout, etc) we proceed to restart() anyway,
        # which SIGKILLs the worker as a last resort. Without this cap
        # the watcher's poll loop would block for the full cleanup
        # duration and the hang banner could never warn the user.
        if self._cancel_in_flight is not None:
            try:
                await asyncio.wait_for(
                    self._cancel_in_flight(),
                    timeout=self._cancel_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "restart_watcher cancel_in_flight timed out after %.1fs "
                    "— proceeding to restart anyway session=%s",
                    self._cancel_timeout,
                    self._lead_session_id,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "restart_watcher cancel_in_flight_run failed session=%s",
                    self._lead_session_id,
                )
        outcome = await self._do_restart(reason)
        # Clear the flag whether the restart succeeded or failed — the
        # follow-up inbox message tells the lead what happened, and
        # keeping the flag set would cause an infinite restart loop.
        await self._session_store.clear_restart_request(self._lead_session_id)
        await self._post_outcome(outcome)
        return True

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(), name="restart_watcher")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                logger.exception("restart_watcher.stop_task_failed")
            self._task = None

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._poll_interval
                )
                return
            except asyncio.TimeoutError:
                pass
            try:
                await self.run_once()
            except Exception:  # noqa: BLE001
                logger.exception("restart_watcher.tick_failed")

    async def _do_restart(self, reason: str) -> _RestartOutcome:
        try:
            await self._restart()
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "restart_watcher restart_failed session=%s",
                self._lead_session_id,
            )
            return _RestartOutcome(
                succeeded=False,
                message=(
                    f"Worker restart FAILED ({type(exc).__name__}: {exc}). "
                    f"Reason given: {reason}. The session is still on the "
                    "previous worker — investigate, then either retry "
                    "request_restart or ask the user to relaunch feather."
                ),
            )
        return _RestartOutcome(
            succeeded=True,
            message=(
                f"Worker restart succeeded (reason: {reason}). "
                "Resume the task you were working on; any patched "
                "feather/* modules are now reloaded."
            ),
        )

    async def _post_outcome(self, outcome: _RestartOutcome) -> None:
        await self._message_store.send(
            from_session_id=self._lead_session_id,
            # Namespaced "from" name so a future user-defined agent
            # named "system" can't shadow these supervisor-side
            # restart-cycle notifications.
            from_agent_name="__system_restart_watcher",
            to_session_id=self._lead_session_id,
            # The lead's BaseAgent filters by exact name match
            # (case-sensitive SQL). Canonical constant avoids the
            # silent-strand bug.
            to_agent_name=LEAD_AGENT_NAME,
            body=outcome.message,
            expects_response=False,
        )
