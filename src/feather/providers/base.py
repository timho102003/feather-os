"""Provider abstraction used by the lead agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from feather.models import EventHandler, ModelTurn, ProviderRequestConfig


class BaseLLMProvider(ABC):
    """Abstract interface for one LLM provider.

    ``stateful`` signals whether the provider maintains server-side
    conversation state keyed on ``previous_response_id``. Responses-API
    providers (OpenAI) keep a cursor and only need *new* input items per
    turn; stateless providers (OpenRouter / Chat Completions) require the
    full conversation history to be replayed on every turn. BaseAgent
    reads this attribute to decide whether to rebuild the full history
    from :class:`~feather.storage.session_store.SessionStore` on each
    iteration.
    """

    stateful: bool = True

    @abstractmethod
    async def complete(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        event_handler: EventHandler | None = None,
        request_config: ProviderRequestConfig | None = None,
    ) -> ModelTurn:
        """Run one provider turn.

        Args:
            instructions: Full system instructions.
            input_items: New input items for the provider.
            tools: Registered tool schemas.
            previous_response_id: Optional stateful response cursor.
            event_handler: Optional runtime event sink.
            request_config: Optional per-request generation overrides.

        Returns:
            Normalized model turn.
        """
