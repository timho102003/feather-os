"""Per-session queue for user messages injected mid-run.

The CLI always accepts user input. When the lead agent is busy executing a
turn (provider call, tool execution, compaction), newly typed messages are
enqueued here instead of being dropped. Between turns, ``BaseAgent.run_loop``
drains the queue and prepends each message to the next provider input so the
agent can reflect the user's new idea without waiting for the run to finish.

Design notes:

- One FIFO per session, guarded by an ``asyncio.Lock`` — enqueue and drain
  are concurrency-safe against each other.
- Bounded with a sane default (64) so a runaway producer cannot grow memory
  without bound. When full, the oldest message is dropped and a warning is
  logged so the event is observable.
- Messages are plain text; persistence as ``MessageRole.USER`` happens at the
  drain site inside the agent loop, not here.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Iterable

logger = logging.getLogger(__name__)

_DEFAULT_MAX_PER_SESSION = 64


class UserInputQueue:
    """Per-session FIFO of user messages awaiting injection between turns."""

    def __init__(self, max_per_session: int = _DEFAULT_MAX_PER_SESSION) -> None:
        if max_per_session <= 0:
            raise ValueError("max_per_session must be positive")
        self._max_per_session = max_per_session
        self._queues: dict[str, deque[str]] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, session_id: str, text: str) -> bool:
        """Append ``text`` to ``session_id``'s queue.

        Args:
            session_id: Session identifier.
            text: Message content; stripped before enqueue, empty strings
                are ignored.

        Returns:
            ``True`` if the message was enqueued, ``False`` if it was empty.
        """

        cleaned = text.strip()
        if not cleaned:
            return False
        async with self._lock:
            queue = self._queues.setdefault(session_id, deque())
            if len(queue) >= self._max_per_session:
                dropped = queue.popleft()
                logger.warning(
                    "user_input_queue overflow session_id=%s dropped=%s",
                    session_id,
                    dropped[:80],
                )
            queue.append(cleaned)
            depth = len(queue)
        logger.info(
            "user_input_queue enqueue session_id=%s depth=%s",
            session_id,
            depth,
        )
        return True

    async def drain(self, session_id: str) -> list[str]:
        """Return and clear all pending messages for ``session_id``."""

        async with self._lock:
            queue = self._queues.get(session_id)
            if not queue:
                return []
            items = list(queue)
            queue.clear()
        logger.info(
            "user_input_queue drain session_id=%s count=%s",
            session_id,
            len(items),
        )
        return items

    async def peek(self, session_id: str) -> tuple[str, ...]:
        """Return a snapshot of currently queued messages without removing them."""

        async with self._lock:
            queue = self._queues.get(session_id)
            if not queue:
                return ()
            return tuple(queue)

    async def depth(self, session_id: str) -> int:
        """Return the current queue depth for ``session_id``."""

        async with self._lock:
            queue = self._queues.get(session_id)
            return len(queue) if queue else 0

    async def clear(self, session_id: str) -> int:
        """Discard all pending messages for ``session_id``."""

        async with self._lock:
            queue = self._queues.pop(session_id, None)
            return len(queue) if queue else 0

    async def extend(self, session_id: str, texts: Iterable[str]) -> int:
        """Enqueue multiple messages atomically.

        Returns:
            Number of messages actually enqueued (skips empty strings).
        """

        added = 0
        async with self._lock:
            queue = self._queues.setdefault(session_id, deque())
            for text in texts:
                cleaned = text.strip()
                if not cleaned:
                    continue
                if len(queue) >= self._max_per_session:
                    dropped = queue.popleft()
                    logger.warning(
                        "user_input_queue overflow session_id=%s dropped=%s",
                        session_id,
                        dropped[:80],
                    )
                queue.append(cleaned)
                added += 1
        return added
