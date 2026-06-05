"""Automatic active-context compaction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from feather.core.prompts.compaction_prompt import COMPACTION_PROMPT
from feather.models import (
    CompactionConfig,
    EventHandler,
    MessageRole,
    ProviderRequestConfig,
    RuntimeEvent,
    SessionMessage,
)
from feather.providers.base import BaseLLMProvider
from feather.storage.session_store import SessionStore

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class CompactionDecision:
    """Result of evaluating whether active history should be compacted."""

    should_compact: bool
    input_tokens: int
    usage_ratio: float


class ContextCompactor:
    """Summarize active conversation history when context usage gets high."""

    def __init__(
        self,
        *,
        config: CompactionConfig,
        provider: BaseLLMProvider,
        session_store: SessionStore,
    ) -> None:
        self._config = config
        self._provider = provider
        self._session_store = session_store

    @property
    def context_window_tokens(self) -> int:
        """Configured context window used to compute usage ratios."""

        return self._config.context_window_tokens

    async def maybe_compact(
        self,
        session_id: str,
        *,
        usage: dict | None,
        event_handler: EventHandler | None = None,
    ) -> bool:
        """Compact the active history if the configured threshold is exceeded."""

        decision = await self.evaluate(session_id, usage=usage)
        if not decision.should_compact:
            return False

        if event_handler is not None:
            event_handler(
                RuntimeEvent(
                    kind="compaction_started",
                    text=f"Compacting context at {decision.usage_ratio:.1%} of the configured window.",
                )
            )

        history = await self._session_store.render_history_for_cache(session_id)
        turn = await self._provider.complete(
            instructions=COMPACTION_PROMPT,
            input_items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Active conversation history to compact:\n\n{history}",
                        }
                    ],
                }
            ],
            tools=[],
            previous_response_id=None,
            request_config=ProviderRequestConfig(
                model=self._config.model,
                max_output_tokens=self._config.max_output_tokens,
                temperature=self._config.temperature,
            ),
        )
        summary = turn.output_text.strip()
        if not summary:
            raise ValueError("Compaction model returned an empty summary.")

        await self._session_store.add_message(
            session_id,
            MessageRole.ASSISTANT,
            summary,
            is_compact=True,
        )
        await self._session_store.update_response_state(session_id, last_response_id=None)
        logger.info(
            "session compacted session_id=%s input_tokens=%s usage_ratio=%.4f summary_chars=%s",
            session_id,
            decision.input_tokens,
            decision.usage_ratio,
            len(summary),
        )

        if event_handler is not None:
            event_handler(
                RuntimeEvent(
                    kind="compaction_finished",
                    text="Active history compacted. Future turns will replay from the latest compact summary.",
                )
            )
        return True

    async def evaluate(self, session_id: str, *, usage: dict | None) -> CompactionDecision:
        """Compute whether a session should be compacted."""

        if not self._config.enabled:
            return CompactionDecision(should_compact=False, input_tokens=0, usage_ratio=0.0)

        active_messages = await self._session_store.list_active_messages(session_id)
        if not self._has_compactable_content(active_messages):
            return CompactionDecision(should_compact=False, input_tokens=0, usage_ratio=0.0)

        input_tokens = self._extract_input_tokens(usage)
        if input_tokens is None:
            input_tokens = self._estimate_input_tokens(active_messages)
        usage_ratio = input_tokens / self._config.context_window_tokens
        return CompactionDecision(
            should_compact=usage_ratio >= self._config.trigger_ratio,
            input_tokens=input_tokens,
            usage_ratio=usage_ratio,
        )

    def _extract_input_tokens(self, usage: dict | None) -> int | None:
        """Extract prompt-side token usage from a provider usage payload."""

        if usage is None:
            return None
        input_tokens = usage.get("input_tokens")
        if isinstance(input_tokens, int):
            return input_tokens
        total_tokens = usage.get("total_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(total_tokens, int) and isinstance(output_tokens, int):
            return max(total_tokens - output_tokens, 0)
        return None

    def _estimate_input_tokens(self, messages: list[SessionMessage]) -> int:
        """Fallback token estimate used when the provider usage payload is unavailable."""

        rendered = "\n".join(
            f"{message.role.value}{'[compact]' if message.is_compact else ''}: {message.content}"
            for message in messages
        )
        # Heuristic fallback only. Real API usage should be preferred when available.
        return max(1, len(rendered) // 4)

    def _has_compactable_content(self, messages: list[SessionMessage]) -> bool:
        """Return true when there is new content beyond an existing compact summary."""

        if not messages:
            return False
        return any(not message.is_compact for message in messages)
