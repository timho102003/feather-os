"""Per-session coordination for serialized agent execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class SessionRunCoordinator:
    """Provide one shared async lock per session ID.

    Entries are reference-counted and evicted when the last holder or
    waiter releases, so the map is bounded by *concurrently active*
    sessions instead of growing with every session ever seen.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._refcounts: dict[str, int] = {}

    @asynccontextmanager
    async def acquire(self, session_id: str) -> AsyncIterator[None]:
        """Serialize work for one session.

        The refcount is incremented before awaiting the lock, so an
        entry is never evicted while any task holds *or waits on* it; a
        fresh lock object after eviction is safe because nothing
        references the old one. Single event loop — no race between the
        lookup and the increment.
        """

        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        self._refcounts[session_id] = self._refcounts.get(session_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            remaining = self._refcounts[session_id] - 1
            if remaining:
                self._refcounts[session_id] = remaining
            else:
                del self._refcounts[session_id]
                del self._locks[session_id]

    def is_busy(self, session_id: str) -> bool:
        """Return True when a run is currently in flight for the session.

        Read-only inspection used by the messaging router to decide
        whether to spawn a new run or enqueue via
        :class:`UserInputQueue`. A missing entry means no holder and no
        waiter, hence not busy.
        """

        lock = self._locks.get(session_id)
        if lock is None:
            return False
        return lock.locked()
