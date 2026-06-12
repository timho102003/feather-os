"""Conversation-state strategies: how prior context reaches the provider.

Stateful providers (OpenAI Responses) keep server-side history keyed by
``previous_response_id`` — each turn sends only new items plus the cursor.
Stateless providers (OpenRouter / Claude) get no cursor — each turn must
carry the full structural transcript (assistant tool_calls followed by
matching tool outputs), which these strategies own end to end.

``initial_input_items`` serves ``run()`` (a new user message); ``begin``
serves runs that start with no new input (``resume_on_inbox``).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from feather.models import ModelTurn, SessionRecord

__all__ = (
    "ConversationContext",
    "StatefulConversation",
    "StatelessConversation",
    "model_turn_input_items",
)

HistoryReplayFn = Callable[[], Awaitable[list[dict[str, Any]]]]


def model_turn_input_items(turn: ModelTurn) -> list[dict[str, Any]]:
    """Convert a model turn into replayable provider input items.

    Stateless providers need the assistant side of the transcript to
    continue a tool loop. Responses-stateful providers do not use these
    items because ``previous_response_id`` already points at the assistant
    turn on the provider side.
    """

    if turn.tool_calls:
        items: list[dict[str, Any]] = []
        for index, tool_call in enumerate(turn.tool_calls):
            item: dict[str, Any] = {
                "type": "function_call",
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": json.dumps(
                    tool_call.arguments,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
            if index == 0 and turn.output_text:
                item["content"] = turn.output_text
            items.append(item)
        return items
    if turn.output_text:
        return [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": turn.output_text}],
            }
        ]
    return []


def _has_structural_context(pending_inputs: list[dict[str, Any]]) -> bool:
    """True when pending inputs already carry replayed transcript context."""

    return any(
        item.get("type") in {"message", "function_call"} for item in pending_inputs
    )


class ConversationContext(ABC):
    """Per-run strategy owning replay, cursor, and pause semantics."""

    @abstractmethod
    async def initial_input_items(
        self, session: SessionRecord, new_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Items for run_loop at run() entry: pending → (history?) → new."""

    @abstractmethod
    async def begin(self, input_items: list[dict[str, Any]]) -> None:
        """Once-per-run lifecycle hook at run_loop entry, before the first provider turn."""

    @abstractmethod
    def provider_request(
        self, session: SessionRecord, input_items: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Consume input_items; return (items to send, cursor or None)."""

    @abstractmethod
    def record_turn(
        self, sent_items: list[dict[str, Any]], turn: ModelTurn
    ) -> None:
        """Fold a completed provider turn back into the strategy state."""

    @abstractmethod
    def pause_payload(
        self, tool_outputs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """pending_inputs to persist when pausing for AWAITING_USER."""


class StatefulConversation(ConversationContext):
    """Server-side cursor: replay only when the cursor was reset."""

    def __init__(self, *, replay: HistoryReplayFn) -> None:
        self._replay = replay

    async def initial_input_items(
        self, session: SessionRecord, new_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        items = list(session.pending_inputs)
        if session.last_response_id is None:
            items.extend(await self._replay())
        items.extend(new_items)
        return items

    async def begin(self, input_items: list[dict[str, Any]]) -> None:
        return None

    def provider_request(
        self, session: SessionRecord, input_items: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        return list(input_items), session.last_response_id

    def record_turn(
        self, sent_items: list[dict[str, Any]], turn: ModelTurn
    ) -> None:
        return None

    def pause_payload(
        self, tool_outputs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return list(tool_outputs)


class StatelessConversation(ConversationContext):
    """Full-transcript replay: the in-run structural transcript lives here."""

    def __init__(self, *, replay: HistoryReplayFn) -> None:
        self._replay = replay
        self._transcript: list[dict[str, Any]] | None = None

    async def initial_input_items(
        self, session: SessionRecord, new_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        items = list(session.pending_inputs)
        if not _has_structural_context(items):
            items.extend(await self._replay())
        items.extend(new_items)
        return items

    async def begin(self, input_items: list[dict[str, Any]]) -> None:
        if not input_items:
            self._transcript = await self._replay()

    def provider_request(
        self, session: SessionRecord, input_items: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        if self._transcript is None:
            self._transcript = list(input_items)
        elif input_items:
            self._transcript.extend(input_items)
        return list(self._transcript), None

    def record_turn(
        self, sent_items: list[dict[str, Any]], turn: ModelTurn
    ) -> None:
        transcript = list(sent_items)
        transcript.extend(model_turn_input_items(turn))
        self._transcript = transcript

    def pause_payload(
        self, tool_outputs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        items = list(self._transcript or [])
        items.extend(tool_outputs)
        return items
