"""Per-session coordination for serialized agent execution."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator


class SessionRunCoordinator:
    """Provide one shared async lock per session ID."""

    def __init__(self) -> None:
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @asynccontextmanager
    async def acquire(self, session_id: str) -> AsyncIterator[None]:
        """Serialize work for one session."""

        lock = self._locks[session_id]
        async with lock:
            yield

    def is_busy(self, session_id: str) -> bool:
        """Return True when a run is currently in flight for the session.

        Read-only inspection used by the messaging router to decide
        whether to spawn a new run or enqueue via :class:`UserInputQueue`.
        Returns False when no lock has ever been acquired for this id —
        ``defaultdict`` would normally create one, so this method peeks
        without inserting.
        """

        lock = self._locks.get(session_id)
        if lock is None:
            return False
        return lock.locked()
