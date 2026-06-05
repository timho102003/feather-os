"""One streaming channel per lead: a headless driver + event fan-out.

The TUI renders a lead by registering an event handler and running a per-lead
driver loop. The API does the same thing without a terminal: each lead gets a
:class:`LeadChannel` that runs queued user messages through the lead's
``LeadHandle`` and broadcasts every :class:`RuntimeEvent` (as a dict) to all
subscribed WebSocket clients — so the browser sees thinking, tool calls, and
results live, exactly like the TUI.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

from feather.api.events import event_to_dict
from feather.models import AgentOutcome, RuntimeEvent

if TYPE_CHECKING:
    from feather.core.leads.manager import LeadHandle
    from feather.runtime import FeatherRuntime

logger = logging.getLogger(__name__)

__all__ = ("LeadChannel",)

_SUBSCRIBER_QUEUE_MAX = 2000


class LeadChannel:
    """Drive one lead and fan its events out to subscribed clients."""

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        handle: "LeadHandle",
        session_id: str,
        runtime: "FeatherRuntime",
    ) -> None:
        self.name = name
        self.display_name = display_name
        self.session_id = session_id
        self.status = "idle"
        self._handle = handle
        self._runtime = runtime
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._run_queue: asyncio.Queue[str] = asyncio.Queue()
        self._driver_task: asyncio.Task[None] | None = None
        # Receive cron / messaging / sub-agent-reaper events too.
        runtime.set_session_event_handler(session_id, self._on_event)

    # --- subscriptions ---------------------------------------------------

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    # --- event fan-out ---------------------------------------------------

    def _on_event(self, event: RuntimeEvent) -> None:
        """Sync, non-blocking: enqueue to every subscriber (drop if slow)."""

        if event.kind in ("tool_started",):
            self.status = "running"
        self._broadcast(event_to_dict(event))

    def _broadcast(self, data: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:  # slow client — drop rather than block
                logger.warning("api.channel.subscriber_dropped_event", extra={"lead": self.name})

    # --- driving ---------------------------------------------------------

    async def send(self, text: str) -> None:
        """Enqueue a user message for this lead's driver (runs as its own turn)."""
        await self._run_queue.put(text)

    async def enqueue_input(self, text: str) -> bool:
        """Inject mid-turn input to steer the agent's *current* turn.

        Routes to the lead's input queue (drained by the run loop) rather than
        the run queue, so it lands in the turn already in flight — TUI parity.
        Returns False if the handle has no input queue wired.
        """
        return await self._handle.enqueue_user_input(text)

    def start(self) -> None:
        if self._driver_task is None:
            self._driver_task = asyncio.create_task(self._driver())

    async def stop(self) -> None:
        if self._driver_task is not None and not self._driver_task.done():
            self._driver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._driver_task
        self._driver_task = None
        try:
            self._runtime.set_session_event_handler(self.session_id, None)
        except Exception:  # noqa: BLE001
            logger.exception("api.channel.unregister_failed", extra={"lead": self.name})

    async def _driver(self) -> None:
        while True:
            text = await self._run_queue.get()
            self.status = "running"
            self._broadcast({"kind": "status", "payload": {"status": "running"}})
            try:
                result = await self._handle.run(text, self._on_event)
                if result is not None and result.status == AgentOutcome.AWAITING_USER:
                    self.status = "awaiting_user"
                    self._broadcast(
                        {"kind": "status", "payload": {"status": "awaiting_user"}}
                    )
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("api.channel.run_failed", extra={"lead": self.name})
                self._broadcast({"kind": "error", "text": f"{type(exc).__name__}: {exc}"})
            self.status = "idle"
            self._broadcast({"kind": "status", "payload": {"status": "idle"}})
