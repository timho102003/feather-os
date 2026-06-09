"""Async background trigger for the memory write path.

After every agent ``run_loop`` invocation, ``BaseAgent`` calls
``maybe_schedule`` with the current session/agent. The trigger checks its
config + closed state and fires a tracked, detached :class:`asyncio.Task`
(regardless of the legacy ``background`` flag, so :meth:`drain` always sees
in-flight work).

Three guarantees:

1. The user-facing turn never blocks on memory work — ``maybe_schedule`` is
   synchronous and O(1).
2. Background-task exceptions never propagate to the agent loop —
   :meth:`LiveMemoryTrigger._run` swallows everything except
   ``asyncio.CancelledError`` and logs it.
3. ``runtime.shutdown()`` calls :meth:`drain` to await in-flight tasks
   within ``shutdown_timeout_s``; stragglers are cancelled with a small
   grace window.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from feather.memory.config import MemoryTriggerConfig
from feather.memory.enums import MemoryOwner

if TYPE_CHECKING:
    from feather.memory.service import MemoryService

logger = logging.getLogger(__name__)


class MemoryTrigger(ABC):
    """Interface ``BaseAgent`` consumes."""

    @abstractmethod
    def maybe_schedule(
        self, session_id: str, *, agent_model: str, owner: MemoryOwner
    ) -> None: ...

    @abstractmethod
    async def drain(self, timeout_s: float) -> None: ...

    @abstractmethod
    def cancel_all(self) -> None: ...


class NoOpMemoryTrigger(MemoryTrigger):
    """Returned when memory is gated off. All operations are no-ops."""

    def maybe_schedule(
        self, session_id: str, *, agent_model: str, owner: MemoryOwner
    ) -> None:
        return None

    async def drain(self, timeout_s: float) -> None:
        return None

    def cancel_all(self) -> None:
        return None


class LiveMemoryTrigger(MemoryTrigger):
    """Schedule :meth:`MemoryService.extract_and_store` as detached tasks."""

    def __init__(
        self, *, service: "MemoryService", cfg: MemoryTriggerConfig
    ) -> None:
        self._service = service
        self._cfg = cfg
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = False

    # -- scheduling -----------------------------------------------------------

    def maybe_schedule(
        self, session_id: str, *, agent_model: str, owner: MemoryOwner
    ) -> None:
        if self._closed or not self._cfg.enabled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — fall back to creating a fresh task via
            # ensure_future on the default policy. This branch is mostly for
            # safety; in practice maybe_schedule is invoked from inside the
            # agent loop.
            logger.debug("memory.trigger.no_loop")
            return

        # Both modes schedule a detached task; tracking is unconditional so
        # drain()/cancel_all() always see in-flight work. ``background``
        # remains accepted in config but no longer changes behavior.
        task = loop.create_task(
            self._run(session_id, agent_model, owner),
            name=f"memory-extract:{session_id[:8]}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._on_done)

    async def _run(
        self, session_id: str, agent_model: str, owner: MemoryOwner
    ) -> None:
        try:
            report = await self._service.extract_and_store(
                session_id, agent_model=agent_model, owner=owner
            )
            logger.info("memory.trigger.ok", extra=report.to_log_fields())
        except asyncio.CancelledError:
            logger.warning(
                "memory.trigger.cancelled", extra={"session_id": session_id}
            )
            raise
        except Exception:
            logger.exception(
                "memory.trigger.failed", extra={"session_id": session_id}
            )

    def _on_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        # Read exception so Python doesn't complain about it being un-retrieved.
        exc = task.exception()
        if exc is not None:
            logger.debug(
                "memory.trigger.discarded_task_exc", extra={"exc": repr(exc)}
            )

    # -- shutdown -------------------------------------------------------------

    async def drain(self, timeout_s: float) -> None:
        """Await pending tasks up to ``timeout_s``; cancel stragglers."""
        self._closed = True
        if not self._tasks:
            return
        pending = list(self._tasks)
        logger.info(
            "memory.trigger.draining", extra={"in_flight": len(pending)}
        )
        try:
            done, still = await asyncio.wait(pending, timeout=timeout_s)
        except Exception:
            logger.exception("memory.trigger.drain_error")
            still = set(pending)
        if still:
            logger.warning(
                "memory.trigger.drain_timeout",
                extra={
                    "abandoned": len(still),
                    "timeout_s": timeout_s,
                },
            )
            for task in still:
                task.cancel()
            await asyncio.wait(still, timeout=2.0)
        # Belt-and-suspenders: anything left in the set should be removed.
        self._tasks.clear()

    def cancel_all(self) -> None:
        """Synchronous cancel — used by the SIGINT handler."""
        self._closed = True
        for task in list(self._tasks):
            task.cancel()


__all__ = [
    "MemoryTrigger",
    "LiveMemoryTrigger",
    "NoOpMemoryTrigger",
]
