"""Memory query builder — contextualizes the latest conversation into a query.

One provider call per agent turn (on the critical path, bounded by
``retrieval_timeout_s`` at the reader layer). Fails open by returning a
:class:`QueryDecision` that falls back to the raw latest user text — the
reader still performs a retrieval, just with a lower-quality query.
"""

from __future__ import annotations

import logging
from typing import Sequence

from feather.memory.config import MemoryOperationModelConfig
from feather.memory.models import QueryBuildResponse, QueryDecision
from feather.models import (
    MessageRole,
    ProviderRequestConfig,
    SessionMessage,
)
from feather.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def _render_recent(
    recent_messages: Sequence[SessionMessage], latest_user_text: str
) -> str:
    """Render recent session messages + the latest user text for the LLM.

    The latest user text is appended separately so the LLM gets an unambiguous
    signal about what message needs contextualization, even if it overlaps
    with the last entry in ``recent_messages``.
    """
    lines: list[str] = []
    for msg in recent_messages:
        role = msg.role.value if isinstance(msg.role, MessageRole) else str(msg.role)
        lines.append(f"{role}: {msg.content}")
    lines.append("")
    lines.append(f"LATEST user message:\n{latest_user_text}")
    return "\n".join(lines)


class MemoryQueryBuilder:
    """Convert recent conversation into a single retrieval query."""

    def __init__(
        self,
        *,
        provider: BaseLLMProvider,
        prompt: str,
        cfg: MemoryOperationModelConfig,
        default_model: str | None = None,
    ) -> None:
        self._provider = provider
        self._prompt = prompt
        self._cfg = cfg
        self._default_model = default_model

    async def build(
        self,
        recent_messages: Sequence[SessionMessage],
        *,
        latest_user_text: str | None = None,
        agent_model: str,
    ) -> QueryDecision:
        """Return a :class:`QueryDecision`.

        Args:
            recent_messages: Non-compact session messages in ascending order
                (oldest first, most recent last).
            latest_user_text: The latest user input for this turn — used as
                the fallback query when the provider fails.
            agent_model: The agent's current conversation model (used when
                ``cfg.model`` is None).
        """
        fallback_text = (latest_user_text or "").strip()
        try:
            rendered = _render_recent(recent_messages, fallback_text)
            request_config = ProviderRequestConfig(
                model=self._cfg.model or self._default_model or agent_model,
                max_output_tokens=self._cfg.max_output_tokens,
                temperature=self._cfg.temperature,
                response_schema=QueryBuildResponse,
            )
            turn = await self._provider.complete(
                instructions=self._prompt,
                input_items=[{"role": "user", "content": rendered}],
                tools=[],
                previous_response_id=None,
                request_config=request_config,
            )
            parsed = QueryBuildResponse.model_validate_json(turn.output_text)
        except Exception:
            logger.warning(
                "memory.query_builder.failed — falling back to raw user text",
                exc_info=True,
            )
            return QueryDecision(
                query=fallback_text, should_skip=False, reasoning="fallback"
            )
        query = parsed.query.strip()
        # If the LLM says skip, honor it; otherwise a blank query falls back
        # so we still run retrieval on *something*.
        if parsed.should_skip:
            return QueryDecision(
                query="", should_skip=True, reasoning=parsed.reasoning
            )
        if not query:
            query = fallback_text
        return QueryDecision(
            query=query, should_skip=False, reasoning=parsed.reasoning
        )


__all__ = ["MemoryQueryBuilder"]
