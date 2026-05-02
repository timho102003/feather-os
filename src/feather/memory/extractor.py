"""Memory extractor — converts a 10-turn window into atomic memories.

Makes ONE provider call with a strict ``ExtractionResponse`` JSON schema and
returns a list of :class:`AtomicMemory` dataclasses. Like compaction, this
call uses a fresh context (``previous_response_id=None``, ``tools=[]``) so
active-agent state is never consumed for memory bookkeeping.
"""

from __future__ import annotations

import logging
from typing import Sequence

from feather.memory.config import MemoryOperationModelConfig
from feather.memory.models import (
    AtomicMemory,
    ExtractionResponse,
    MemoryWindow,
)
from feather.models import (
    MessageRole,
    ProviderRequestConfig,
    SessionMessage,
)
from feather.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def _render_transcript(messages: Sequence[SessionMessage]) -> str:
    """Render session messages into a role-prefixed transcript for the LLM."""
    lines: list[str] = []
    for msg in messages:
        role = msg.role.value if isinstance(msg.role, MessageRole) else str(msg.role)
        # Normalize any inline whitespace at the edges; keep internal structure.
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


class MemoryExtractor:
    """Turn a :class:`MemoryWindow` into ``list[AtomicMemory]`` via one LLM call."""

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

    async def extract(
        self, window: MemoryWindow, agent_model: str
    ) -> list[AtomicMemory]:
        """Run one provider turn with a strict extraction schema.

        Args:
            window: The 10-turn extraction window.
            agent_model: The agent's current conversation model; used when
                ``cfg.model`` is ``None``.

        Returns:
            One ``AtomicMemory`` per extracted fact (possibly empty).

        Raises:
            pydantic.ValidationError: If the provider output doesn't match
                :class:`ExtractionResponse` (should not happen with strict
                mode; still surfaced for tests and non-strict providers).
        """
        transcript = _render_transcript(window.messages)
        request_config = ProviderRequestConfig(
            model=self._cfg.model or self._default_model or agent_model,
            max_output_tokens=self._cfg.max_output_tokens,
            temperature=self._cfg.temperature,
            response_schema=ExtractionResponse,
        )
        turn = await self._provider.complete(
            instructions=self._prompt,
            input_items=[{"role": "user", "content": transcript}],
            tools=[],
            previous_response_id=None,
            request_config=request_config,
        )
        parsed = ExtractionResponse.model_validate_json(turn.output_text)
        out: list[AtomicMemory] = []
        for item in parsed.memories:
            data = item.model_dump()
            # Drop any memory where content degenerated to whitespace.
            if not data.get("content", "").strip():
                continue
            out.append(AtomicMemory(**data))
        logger.info(
            "memory.extract.llm",
            extra={
                "session_id": window.session_id,
                "model": request_config.model,
                "count": len(out),
            },
        )
        return out
