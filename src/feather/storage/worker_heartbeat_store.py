"""SQLite-backed liveness store for lead worker subprocesses.

Each lead worker writes one row to ``worker_heartbeats`` keyed by its
``session_id`` and refreshes it on a fixed cadence. The supervisor (the
Textual TUI process that spawned the worker) reads the row to detect
hangs: a heartbeat older than the configured staleness threshold means
the worker is unresponsive and should be surfaced to the user.

The store owns row I/O only. Cadence, threshold, and what the supervisor
does with a stale heartbeat live in the supervisor module — staleness is
policy, not storage.

Design notes:

* ``session_id`` is the primary key, so heartbeats are naturally one row
  per worker. Refresh is an ``INSERT ... ON CONFLICT DO UPDATE`` so the
  call is a single atomic statement under WAL.
* No foreign key to ``sessions`` — mirrors :mod:`agent_message_store`,
  whose high-frequency writer pattern is the closest precedent. The
  store is the only writer; orphaned-row risk is bounded.
* Timestamps are persisted as ISO-8601 UTC strings (the same convention
  the rest of the schema uses) and re-attached to ``UTC`` on read so
  consumers always work with tz-aware ``datetime``s.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from feather.models import WorkerHeartbeat, WorkerStatus
from feather.storage.schema import initialize_database_schema

logger = logging.getLogger(__name__)


class WorkerHeartbeatStore:
    """Persist and query lead-worker liveness heartbeats."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the connection and ensure the schema exists."""

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self._db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA foreign_keys=ON;")
        await self._connection.execute("PRAGMA journal_mode=WAL;")
        # Match the busy_timeout used by the other stores so concurrent
        # writers (worker writing heartbeats while supervisor reads) back
        # off rather than raising under WAL contention bursts.
        await self._connection.execute("PRAGMA busy_timeout=5000;")
        await initialize_database_schema(self._connection)
        await self._connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def heartbeat(
        self,
        *,
        session_id: str,
        pid: int,
        status: WorkerStatus,
    ) -> None:
        """Insert or refresh the heartbeat row for ``session_id``."""

        connection = self._require_connection()
        now_iso = datetime.now(UTC).isoformat()
        await connection.execute(
            """
            INSERT INTO worker_heartbeats (session_id, pid, status, heartbeat_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                pid = excluded.pid,
                status = excluded.status,
                heartbeat_at = excluded.heartbeat_at
            """,
            (session_id, pid, status.value, now_iso),
        )
        await connection.commit()

    async def get(self, session_id: str) -> WorkerHeartbeat | None:
        """Return the heartbeat row for ``session_id`` if present."""

        connection = self._require_connection()
        cursor = await connection.execute(
            "SELECT session_id, pid, status, heartbeat_at "
            "FROM worker_heartbeats WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_heartbeat(row)

    async def clear(self, session_id: str) -> None:
        """Delete the heartbeat row for ``session_id`` (no-op if absent)."""

        connection = self._require_connection()
        await connection.execute(
            "DELETE FROM worker_heartbeats WHERE session_id = ?",
            (session_id,),
        )
        await connection.commit()

    async def count(self) -> int:
        """Return the total number of heartbeat rows (test/diagnostic only)."""

        connection = self._require_connection()
        cursor = await connection.execute(
            "SELECT COUNT(*) FROM worker_heartbeats"
        )
        row = await cursor.fetchone()
        # COUNT(*) always returns exactly one row, so the row is never None.
        assert row is not None
        return int(row[0])

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError(
                "WorkerHeartbeatStore.initialize() must be called before use."
            )
        return self._connection


def _row_to_heartbeat(row: aiosqlite.Row) -> WorkerHeartbeat:
    """Project an ``aiosqlite.Row`` into a :class:`WorkerHeartbeat`."""

    raw_ts = row["heartbeat_at"]
    parsed = datetime.fromisoformat(raw_ts)
    if parsed.tzinfo is None:
        # Older rows or hand-crafted writes might omit the tz suffix;
        # heartbeats are always written in UTC so re-attach it.
        parsed = parsed.replace(tzinfo=UTC)
    return WorkerHeartbeat(
        session_id=row["session_id"],
        pid=int(row["pid"]),
        status=WorkerStatus(row["status"]),
        heartbeat_at=parsed,
    )
