"""``manage_memory`` — proactive CRUD over long-term user memory.

Counterpart to :class:`feather.tools.recall_memory_tool.RecallMemoryTool`
(read-only). Where the auto-extractor sweeps the conversation every N user
turns and the classifier infers CREATE/UPDATE/DELETE, this tool lets the
lead act on a *direct* user instruction immediately:

- "remember <X>" → ``CREATE``
- "update what you know about <Y> — it's actually <Z>" → ``UPDATE``
- "forget <X>" / "I never said that" → ``DELETE``

Not useful for ambient observations — the auto-extractor handles those.
The lead-only registration in :class:`feather.core.agent.factory.AgentFactory`
prevents sub-agents from rewriting user memory behind the user's back.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

from feather.memory.enums import MemoryOp, MemoryOwner
from feather.memory.models import AppliedOp
from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.tools.base import BaseTool

logger = logging.getLogger(__name__)


class _ProactiveMemoryService(Protocol):
    """Structural interface the tool depends on (so tests can fake it)."""

    async def proactive_create(
        self,
        *,
        content: str,
        purpose: str,
        session_id: str,
        owner: MemoryOwner = MemoryOwner.USER,
    ) -> AppliedOp: ...

    async def proactive_update(
        self,
        *,
        target_query: str,
        content: str,
        purpose: str,
        session_id: str,
        owner: MemoryOwner = MemoryOwner.USER,
        score_threshold: float = 0.4,
    ) -> AppliedOp: ...

    async def proactive_delete(
        self,
        *,
        target_query: str,
        session_id: str,
        owner: MemoryOwner = MemoryOwner.USER,
        score_threshold: float = 0.4,
    ) -> AppliedOp: ...


class ManageMemoryTool(BaseTool):
    """Direct CRUD on long-term memory, triggered by explicit user request."""

    name = "manage_memory"
    description = (
        "Proactively create, update, or delete a long-term memory about "
        "the user. Use ONLY when the user explicitly asks you to remember, "
        "forget, or correct something (e.g. \"remember I prefer X\", "
        "\"forget what I said about Y\", \"actually it's Z, not W\"). "
        "Do NOT use for ambient observations — the background memory "
        "extractor handles those automatically."
    )
    # OpenAI strict mode: every property MUST appear in `required`.
    # Optional behavior is expressed via `["T", "null"]` unions.
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["operation", "content", "purpose", "target_query"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["CREATE", "UPDATE", "DELETE"],
                "description": (
                    "CREATE = persist a new memory. UPDATE = replace an "
                    "existing memory matched by target_query with new "
                    "content. DELETE = remove the memory matched by "
                    "target_query."
                ),
            },
            "content": {
                "type": ["string", "null"],
                "description": (
                    "The memory text (one self-contained declarative "
                    "sentence in 5W1H form, e.g. 'In project X the user "
                    "prefers Y because Z.'). REQUIRED for CREATE and "
                    "UPDATE; pass null for DELETE."
                ),
            },
            "purpose": {
                "type": ["string", "null"],
                "description": (
                    "How a future agent could use this memory (e.g. 'route "
                    "language-specific suggestions to Rust'). REQUIRED for "
                    "CREATE and UPDATE; pass null for DELETE."
                ),
            },
            "target_query": {
                "type": ["string", "null"],
                "description": (
                    "Natural-language description of the EXISTING memory "
                    "to update or delete (e.g. 'the user's programming "
                    "language preference'). REQUIRED for UPDATE and "
                    "DELETE; pass null for CREATE."
                ),
            },
        },
    }

    def __init__(
        self,
        *,
        service: _ProactiveMemoryService,
        session_id_resolver: Callable[[], str | None],
    ) -> None:
        self._service = service
        self._resolve_session = session_id_resolver

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        del context  # session id comes from the resolver, not the LLM args
        operation_raw = (arguments.get("operation") or "").strip().upper()
        if operation_raw not in {"CREATE", "UPDATE", "DELETE"}:
            return _err(
                f"invalid operation {operation_raw!r}; expected CREATE, UPDATE, or DELETE."
            )

        session_id = self._resolve_session()
        if session_id is None:
            return _err(
                "no active session — manage_memory cannot persist without "
                "a session anchor."
            )

        content = _clean(arguments.get("content"))
        purpose = _clean(arguments.get("purpose"))
        target_query = _clean(arguments.get("target_query"))

        try:
            if operation_raw == "CREATE":
                return await self._do_create(
                    content=content, purpose=purpose, session_id=session_id
                )
            if operation_raw == "UPDATE":
                return await self._do_update(
                    target_query=target_query,
                    content=content,
                    purpose=purpose,
                    session_id=session_id,
                )
            return await self._do_delete(
                target_query=target_query, session_id=session_id
            )
        except ValueError as exc:
            # Service-level validation (e.g. session has no user message).
            return _err(str(exc))
        except Exception as exc:  # noqa: BLE001 — render any failure to the agent
            logger.exception(
                "manage_memory.unexpected_error",
                extra={"session_id": session_id, "operation": operation_raw},
            )
            return _err(f"unexpected error: {exc}")

    # -- per-op handlers ------------------------------------------------------

    async def _do_create(
        self, *, content: str | None, purpose: str | None, session_id: str
    ) -> ToolExecutionResult:
        if content is None:
            return _err("CREATE requires `content` (it was null or empty).")
        if purpose is None:
            return _err("CREATE requires `purpose` (it was null or empty).")
        applied = await self._service.proactive_create(
            content=content, purpose=purpose, session_id=session_id
        )
        return _render_applied("created", applied, fallback_hint=None)

    async def _do_update(
        self,
        *,
        target_query: str | None,
        content: str | None,
        purpose: str | None,
        session_id: str,
    ) -> ToolExecutionResult:
        if target_query is None:
            return _err(
                "UPDATE requires `target_query` describing which existing "
                "memory to replace."
            )
        if content is None:
            return _err("UPDATE requires `content` (the new memory text).")
        if purpose is None:
            return _err("UPDATE requires `purpose`.")
        applied = await self._service.proactive_update(
            target_query=target_query,
            content=content,
            purpose=purpose,
            session_id=session_id,
        )
        return _render_applied(
            "updated",
            applied,
            fallback_hint=(
                "No matching memory found. Consider CREATE instead, or "
                "rephrase target_query to better match the existing memory."
            ),
        )

    async def _do_delete(
        self, *, target_query: str | None, session_id: str
    ) -> ToolExecutionResult:
        if target_query is None:
            return _err(
                "DELETE requires `target_query` describing which memory to forget."
            )
        applied = await self._service.proactive_delete(
            target_query=target_query, session_id=session_id
        )
        return _render_applied(
            "deleted",
            applied,
            fallback_hint=(
                "No matching memory found. Either it was never stored or "
                "the target_query is too far from any existing memory; "
                "rephrase and try again."
            ),
        )


# -- helpers ------------------------------------------------------------------


def _clean(value: object) -> str | None:
    """Return a stripped non-empty string or ``None``.

    The LLM's strict-mode arguments arrive as either a string or ``None`` —
    we normalize whitespace-only strings to ``None`` so per-op validators
    can give one consistent error message.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _err(message: str) -> ToolExecutionResult:
    """Render a user-facing error to the agent prefixed with the tool name."""
    return ToolExecutionResult(output=f"manage_memory: {message}")


def _render_applied(
    verb: str, applied: AppliedOp, *, fallback_hint: str | None
) -> ToolExecutionResult:
    """Format an :class:`AppliedOp` for the agent.

    Success path includes the group_id so subsequent tool calls (e.g. another
    UPDATE) could re-target it precisely if we ever expose that affordance.
    """
    if applied.error is not None:
        body = f"manage_memory: {applied.op.value} not applied — {applied.error}"
        if fallback_hint:
            body += f"\n{fallback_hint}"
        return ToolExecutionResult(output=body)
    group = str(applied.group_id) if applied.group_id is not None else "(no group)"
    chunk_msg = (
        f" ({applied.chunk_count} chunk{'s' if applied.chunk_count != 1 else ''})"
        if applied.op is not MemoryOp.DELETE
        else ""
    )
    return ToolExecutionResult(
        output=f"Memory {verb}: group_id={group}{chunk_msg}."
    )


__all__ = ["ManageMemoryTool"]
