"""``recall_memory`` tool — explicit memory lookup invokable by the lead agent.

The tool does not contextualize the query (the agent supplies it directly).
Optional parameters mirror :class:`MemoryRetrievalConfig` so a session can
override defaults per call. ``session_scoped=true`` filters results to the
current session via a session-id resolver injected at construction time.
"""

from __future__ import annotations

from typing import Any, Callable

from feather.memory.config import MemoryRetrievalConfig
from feather.memory.enums import MemoryOwner
from feather.memory.models import MemorySearchResult
from feather.memory.reader import MemoryReader
from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.tools.base import BaseTool


class RecallMemoryTool(BaseTool):
    """Search long-term memory about the user."""

    name = "recall_memory"
    description = (
        "Search your long-term memory about the user (across all past "
        "sessions). Use when you need a specific fact, preference, or past "
        "decision that might not be in the current conversation. Results "
        "are ranked by relevance."
    )
    # OpenAI strict mode: every property MUST appear in `required`.
    # Optional behavior is expressed via `["T", "null"]` unions instead of
    # omission. The tool's execute() treats null as "use config default".
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query", "top_k", "score_threshold", "session_scoped"],
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The natural-language question or keywords to search "
                    "memory with."
                ),
            },
            "top_k": {
                "type": ["integer", "null"],
                "description": (
                    "How many memories to return; pass null to use the "
                    "configured top_k_tool default (typically 10)."
                ),
            },
            "score_threshold": {
                "type": ["number", "null"],
                "description": (
                    "Minimum cosine similarity (0-1); pass null to use the "
                    "configured score_threshold default."
                ),
            },
            "session_scoped": {
                "type": ["boolean", "null"],
                "description": (
                    "If true, restrict results to memories produced in this "
                    "same session; null or false means cross-session search."
                ),
            },
        },
    }

    def __init__(
        self,
        *,
        reader: MemoryReader,
        cfg: MemoryRetrievalConfig,
        session_id_resolver: Callable[[], str | None],
    ) -> None:
        self._reader = reader
        self._cfg = cfg
        self._resolve_session = session_id_resolver

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        del context  # session id comes via the injected resolver
        query = (arguments.get("query") or "").strip()
        if not query:
            return ToolExecutionResult(output="recall_memory: empty query rejected.")
        top_k = arguments.get("top_k") or self._cfg.top_k_tool
        threshold = arguments.get("score_threshold")
        if threshold is None:
            threshold = self._cfg.score_threshold
        session_scoped = bool(arguments.get("session_scoped"))
        session_id = self._resolve_session() if session_scoped else None

        results = await self._reader.recall(
            query=query,
            top_k=int(top_k),
            score_threshold=float(threshold),
            session_id=session_id,
            owner=MemoryOwner.USER,
        )
        if not results:
            return ToolExecutionResult(
                output="No memories found above threshold."
            )
        return ToolExecutionResult(output=_render_results(query, results))


def _render_results(query: str, results: list[MemorySearchResult]) -> str:
    """Plain-text rendering of recall results. Not the same as the prompt block."""
    lines: list[str] = [f"Found {len(results)} memories (query={query!r}):", ""]
    for i, r in enumerate(results, 1):
        date = r.payload.created_at.strftime("%Y-%m-%d")
        lines.append(f"{i}. [{r.score:.2f}] {r.payload.content}")
        lines.append(f"    purpose: {r.payload.purpose}")
        lines.append(f"    from session {date}")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = ["RecallMemoryTool"]
