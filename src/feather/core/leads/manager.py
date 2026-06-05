"""Multi-lead orchestration.

A :class:`LeadManager` owns the set of active leads — each a top-level agent
with its own identity ("soul"), durable session, and run handle. A
:class:`LeadHandle` is a uniform surface over the two existing process models
(in-process ``BaseAgent`` vs supervised worker subprocess), each bound to one
lead's ``(name, session_id)``.

Phase A builds and tests this substrate and drives the single default lead;
the same surface scales to N concurrently-running leads for the multi-lead TUI
(Phase B) — ``add_lead`` simply spins up another handle + session + worker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from feather.models import AgentRunResult, EventHandler

if TYPE_CHECKING:
    from feather.core.agent.base import BaseAgent
    from feather.core.leads.supervisor import LeadSupervisor
    from feather.core.session.input_queue import UserInputQueue

logger = logging.getLogger(__name__)

__all__ = (
    "LeadHandle",
    "InProcessLeadHandle",
    "SupervisedLeadHandle",
    "LeadInfo",
    "LeadManager",
)


@runtime_checkable
class LeadHandle(Protocol):
    """Uniform run surface for one lead, bound to its ``(name, session_id)``."""

    name: str
    session_id: str

    async def run(self, text: str, on_event: EventHandler | None) -> AgentRunResult: ...

    async def resume_on_inbox(self, on_event: EventHandler | None) -> AgentRunResult | None: ...

    async def enqueue_user_input(self, text: str) -> bool: ...

    async def shutdown(self) -> None: ...


class InProcessLeadHandle:
    """Drive an in-process ``BaseAgent`` bound to one ``(name, session_id)``.

    Mid-turn user input is routed through the shared runtime input queue (the
    agent has no enqueue method of its own).
    """

    def __init__(
        self,
        *,
        name: str,
        session_id: str,
        agent: "BaseAgent",
        input_queue: "UserInputQueue | None" = None,
    ) -> None:
        self.name = name
        self.session_id = session_id
        self._agent = agent
        self._input_queue = input_queue

    async def run(self, text: str, on_event: EventHandler | None) -> AgentRunResult:
        return await self._agent.run(self.session_id, text, on_event)

    async def resume_on_inbox(self, on_event: EventHandler | None) -> AgentRunResult | None:
        return await self._agent.resume_on_inbox(self.session_id, on_event)

    async def enqueue_user_input(self, text: str) -> bool:
        if self._input_queue is None:
            return False
        return await self._input_queue.enqueue(self.session_id, text)

    async def shutdown(self) -> None:
        return None


class SupervisedLeadHandle:
    """Drive a :class:`LeadSupervisor` worker bound to one ``(name, session_id)``."""

    def __init__(self, *, name: str, session_id: str, supervisor: "LeadSupervisor") -> None:
        self.name = name
        self.session_id = session_id
        self._supervisor = supervisor

    @property
    def supervisor(self) -> "LeadSupervisor":
        """The underlying worker supervisor (for staleness/restart/config-reload)."""
        return self._supervisor

    async def run(self, text: str, on_event: EventHandler | None) -> AgentRunResult:
        return await self._supervisor.run(self.session_id, text, on_event)

    async def resume_on_inbox(self, on_event: EventHandler | None) -> AgentRunResult | None:
        return await self._supervisor.resume_on_inbox(self.session_id, on_event)

    async def enqueue_user_input(self, text: str) -> bool:
        await self._supervisor.enqueue_user_input(self.session_id, text)
        return True

    async def shutdown(self) -> None:
        await self._supervisor.shutdown()


@dataclass(slots=True, frozen=True)
class LeadInfo:
    """Display + identity metadata for one active lead."""

    name: str
    display_name: str
    personality: str
    soul: str
    color: str | None
    emoji: str | None
    session_id: str


class LeadManager:
    """Own the set of active leads: their handles, durable sessions, and info.

    Depends on a small, duck-typed slice of :class:`FeatherRuntime`:
    ``build_agent``, ``input_queue``, ``lead_session_store``, ``agent_catalog``,
    ``default_lead_name``, and ``build_lead_supervisor`` (only when
    ``worker_mode`` is on). This keeps the manager unit-testable with a light
    fake runtime.
    """

    def __init__(self, runtime: object, *, worker_mode: bool) -> None:
        self._runtime = runtime
        self._worker_mode = worker_mode
        self._handles: dict[str, LeadHandle] = {}
        self._infos: dict[str, LeadInfo] = {}

    async def start(self, lead_names: Sequence[str] | None = None) -> None:
        """Bring up the given leads (default: all discovered, else the default)."""

        names = list(lead_names) if lead_names is not None else self._discover_lead_names()
        for name in names:
            await self.add_lead(name)

    def _discover_lead_names(self) -> list[str]:
        leads = [entry.name for entry in self._runtime.agent_catalog.list_leads()]
        return leads or [self._runtime.default_lead_name]

    async def add_lead(self, name: str) -> LeadHandle:
        """Resume-or-create ``name``'s session and build its handle (idempotent)."""

        if name in self._handles:
            return self._handles[name]
        agent, session_id = await self._resolve_session(name)
        handle = await self._build_handle(name, agent, session_id)
        self._handles[name] = handle
        self._infos[name] = self._build_info(name, agent, session_id)
        logger.info(
            "lead_manager.add_lead name=%s session_id=%s worker_mode=%s",
            name, session_id, self._worker_mode,
        )
        return handle

    async def _resolve_session(self, name: str) -> tuple["BaseAgent", str]:
        # Build the in-process agent for this lead (cached by the runtime). Even
        # in worker mode this is how the session row is created before the
        # worker is started — matching today's TUI bootstrap.
        agent = self._runtime.build_agent(name)
        store = self._runtime.lead_session_store
        existing = await store.get(name)
        if existing is not None:
            session_id = await agent.ensure_session_with_id(existing)
        else:
            session_id = await agent.create_session()
            await store.upsert(name, session_id)
        return agent, session_id

    async def _build_handle(self, name: str, agent: "BaseAgent", session_id: str) -> LeadHandle:
        if self._worker_mode:
            supervisor = self._runtime.build_lead_supervisor(name)
            await supervisor.start(session_id)
            return SupervisedLeadHandle(name=name, session_id=session_id, supervisor=supervisor)
        return InProcessLeadHandle(
            name=name,
            session_id=session_id,
            agent=agent,
            input_queue=getattr(self._runtime, "input_queue", None),
        )

    @staticmethod
    def _build_info(name: str, agent: "BaseAgent", session_id: str) -> LeadInfo:
        cfg = agent.config
        return LeadInfo(
            name=name,
            display_name=cfg.name,
            personality=cfg.personality,
            soul=cfg.soul,
            color=cfg.color,
            emoji=cfg.emoji,
            session_id=session_id,
        )

    def list_leads(self) -> list[LeadInfo]:
        """Return info for every active lead, sorted by name."""
        return [self._infos[name] for name in sorted(self._infos)]

    def handle(self, name: str) -> LeadHandle:
        """Return the handle for an active lead (KeyError if not started)."""
        return self._handles[name]

    def info(self, name: str) -> LeadInfo:
        """Return the info for an active lead."""
        return self._infos[name]

    def active_names(self) -> list[str]:
        """Return the names of all active leads, sorted."""
        return sorted(self._handles)

    async def shutdown(self) -> None:
        """Tear down every lead handle (idempotent); swallow per-handle errors."""
        for handle in list(self._handles.values()):
            try:
                await handle.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("lead_manager.shutdown_failed", extra={"lead": handle.name})
        self._handles.clear()
        self._infos.clear()
