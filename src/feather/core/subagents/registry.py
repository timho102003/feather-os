"""Runtime registry of live sub-agent subprocesses.

The lead's ``spawn_agent`` tool launches a subprocess and returns
immediately. The registry owns the ``asyncio.subprocess.Process`` handle
and is watched by the runtime's subprocess reaper: when a child exits,
the reaper parses the final envelope from stdout and delivers it to the
parent session's inbox as one last agent message.

The registry is intentionally minimal — it keeps only what the reaper
needs to do its job and to perform clean shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# How long we remember a session_id as "recently exited" after the reaper
# delivers its final report and removes the live entry. Long enough that
# the lead is very unlikely to still be composing a send_message to it
# when the memory of the child evaporates; short enough that recycled
# session_ids eventually stop matching.
_GRAVEYARD_TTL_SECONDS: float = 3600.0
# Cap entries so a long-running session with many dispatches can't grow
# the graveyard unboundedly.
_GRAVEYARD_MAX_ENTRIES: int = 1024


@dataclass(slots=True)
class LiveSubagent:
    """One running sub-agent subprocess.

    ``stdout_buffer`` / ``stderr_buffer`` are filled by background drainer
    tasks kicked off at spawn time — this prevents a chatty child from
    blocking on a full pipe before its envelope can be read. The reaper
    reads from the buffers, not from the live pipes.
    """

    session_id: str
    agent_name: str
    parent_session_id: str
    parent_agent_name: str
    process: asyncio.subprocess.Process
    task_text: str
    correlation_id: str | None = None
    task_id: str | None = None
    task_run_id: str | None = None
    envelope: dict[str, Any] | None = field(default=None)
    started_at: str = ""
    stdout_buffer: bytearray = field(default_factory=bytearray)
    stderr_buffer: bytearray = field(default_factory=bytearray)
    drainers: tuple[asyncio.Task[None], ...] = field(default_factory=tuple)


class SubagentRegistry:
    """Concurrency-safe registry of live sub-agent subprocesses.

    A small "graveyard" remembers sub-agent session_ids that have
    recently exited. Sub-agent processes are short-lived — they finish
    their single ``run_loop`` and exit — after which the reaper delivers
    their final report and :meth:`remove`\\s their entry from the live
    dict. Consumers of :meth:`is_recently_exited` (e.g. ``send_message``)
    can then refuse late writes to a dead inbox instead of silently
    enqueuing messages the sub-agent will never read.
    """

    def __init__(self) -> None:
        self._live: dict[str, LiveSubagent] = {}
        # session_id -> exited_at (monotonic time). OrderedDict so we can
        # evict oldest entries FIFO once the cap is reached.
        self._graveyard: OrderedDict[str, float] = OrderedDict()
        self._lock = asyncio.Lock()

    async def register(self, live: LiveSubagent) -> None:
        async with self._lock:
            if live.session_id in self._live:
                raise ValueError(
                    f"sub-agent session already registered: {live.session_id}"
                )
            self._live[live.session_id] = live
            # If a prior generation of this id was marked exited (only
            # possible after an accidental reuse), drop the old entry so
            # liveness checks on the new one aren't poisoned.
            self._graveyard.pop(live.session_id, None)
        logger.info(
            "subagent_registry register session_id=%s agent=%s parent_session=%s",
            live.session_id,
            live.agent_name,
            live.parent_session_id,
        )

    async def remove(self, session_id: str) -> LiveSubagent | None:
        async with self._lock:
            claimed = self._live.pop(session_id, None)
            if claimed is not None:
                # Remember this id as exited so late send_message calls
                # can be refused with a clear error.
                self._graveyard[session_id] = time.monotonic()
                self._graveyard.move_to_end(session_id)
                self._evict_expired_locked()
            return claimed

    async def snapshot(self) -> list[LiveSubagent]:
        async with self._lock:
            return list(self._live.values())

    async def get(self, session_id: str) -> LiveSubagent | None:
        async with self._lock:
            return self._live.get(session_id)

    async def is_recently_exited(self, session_id: str) -> bool:
        """Return True if ``session_id`` was reaped within the TTL window."""

        async with self._lock:
            self._evict_expired_locked()
            return session_id in self._graveyard

    def _evict_expired_locked(self) -> None:
        """Drop graveyard entries older than the TTL. Caller holds the lock."""

        now = time.monotonic()
        while self._graveyard:
            oldest_key = next(iter(self._graveyard))
            if now - self._graveyard[oldest_key] <= _GRAVEYARD_TTL_SECONDS:
                break
            self._graveyard.popitem(last=False)
        # Cap-based eviction after TTL eviction so TTL always wins.
        while len(self._graveyard) > _GRAVEYARD_MAX_ENTRIES:
            self._graveyard.popitem(last=False)
