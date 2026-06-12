"""Null-object event emission for the agent loop."""

from __future__ import annotations

from typing import Any

from feather.models import EventHandler, RuntimeEvent

__all__ = ("EventEmitter",)


class EventEmitter:
    """Wraps an optional :data:`EventHandler` so emit sites need no None checks.

    Deliberately adds no error handling: a raising handler propagates exactly
    as it did when call sites invoked the handler directly.
    """

    __slots__ = ("_handler",)

    def __init__(self, handler: EventHandler | None) -> None:
        self._handler = handler

    @property
    def handler(self) -> EventHandler | None:
        """The raw handler, for seams that accept ``EventHandler | None``."""

        return self._handler

    @property
    def enabled(self) -> bool:
        """Whether emitted events reach a real handler."""

        return self._handler is not None

    def emit(
        self,
        kind: str,
        *,
        text: str | None = None,
        tool_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Build and deliver one event; no-op without a handler."""

        if self._handler is None:
            return
        self._handler(
            RuntimeEvent(kind=kind, text=text, tool_name=tool_name, payload=payload)
        )

    def forward(self, event: RuntimeEvent) -> None:
        """Pass through a pre-built event (e.g. the inbox preview)."""

        if self._handler is None:
            return
        self._handler(event)
