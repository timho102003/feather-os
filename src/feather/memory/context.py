"""Per-task context for memory-aware tools.

``recall_memory`` (and any future memory tool that needs to know the active
session) reads ``current_session_id`` to honor ``session_scoped=true``.
``BaseAgent.run_loop`` sets the value at the start of every loop iteration
so the contextvar is always populated for the duration of a tool dispatch.
"""

from __future__ import annotations

from contextvars import ContextVar

current_session_id: ContextVar[str | None] = ContextVar(
    "feather_memory_current_session_id", default=None
)
