"""Contextvars + logging filter that stamp every log line with session/agent.

Feather's log lines previously embedded ``session_id=...`` and
``agent=...`` inside the free-form message body — consistent across
:mod:`feather.core.base_agent` but absent from third-party dependencies
(httpx, google-genai, openai) and inconsistent across Feather's own
modules. This module lifts both identifiers into the log record so they
can be rendered in a fixed position by the formatter:

    2026-04-22 07:52:18 | INFO | <agent> | <session_id_8> | feather.core.base_agent | ...

``current_session_id`` is re-exported from
:mod:`feather.memory.context` so existing consumers keep working; the
filter reads both contextvars on every record.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

from feather.memory.context import current_session_id

# Mirror ``current_session_id``'s shape — a role-name contextvar that
# BaseAgent sets on loop entry. None renders as a short placeholder so
# third-party loggers (httpx, openai, google-genai) stay aligned.
current_agent_name: ContextVar[str | None] = ContextVar(
    "feather_current_agent_name", default=None
)


class _ContextFilter(logging.Filter):
    """Attach ``agent_ctx`` and ``session_ctx`` to every ``LogRecord``.

    Using a filter (rather than a custom Formatter) keeps the existing
    third-party handlers working and avoids overriding ``LogRecord``
    attributes the stdlib already owns.
    """

    _PLACEHOLDER = "-"
    _SESSION_PREFIX_LEN = 8  # short prefix keeps lines readable

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        agent = current_agent_name.get() or self._PLACEHOLDER
        session = current_session_id.get() or self._PLACEHOLDER
        if session != self._PLACEHOLDER and len(session) > self._SESSION_PREFIX_LEN:
            session = session[: self._SESSION_PREFIX_LEN]
        record.agent_ctx = agent
        record.session_ctx = session
        return True


def build_context_filter() -> logging.Filter:
    """Return a fresh context filter instance.

    Filters belong on :class:`logging.Handler` instances, not on
    loggers: logger-level filters only run on records emitted at that
    exact logger, so a filter attached to the root logger would miss
    anything emitted by ``feather.foo.bar`` or ``httpx``. Handlers see
    every record that's routed to them, regardless of the originating
    logger, which is what we want for session-tagged log lines.
    """

    return _ContextFilter()
