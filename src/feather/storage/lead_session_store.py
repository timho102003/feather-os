"""Durable ``lead_name`` → ``session_id`` pointer.

Each lead (Tim, Sophia, …) keeps its own long-lived conversation. This store
records which session belongs to which lead so the :class:`LeadManager` can
resume the right conversation on every launch instead of starting fresh.

One process — the manager — owns the only writer, so blind last-write-wins is
safe here. Unlike ``session_store``/``cron_store`` this store sets
``busy_timeout`` so a contended write backs off instead of hard-failing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from feather.storage.schema import LEAD_SESSIONS_TABLE

__all__ = ("LeadSessionStore",)


class LeadSessionStore:
    """One aiosqlite connection owning the ``lead_sessions`` pointer table."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the connection and ensure the table exists."""

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self._db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA busy_timeout=5000;")
        await self._connection.execute("PRAGMA journal_mode=WAL;")
        await self._connection.execute(LEAD_SESSIONS_TABLE.create_sql)
        await self._connection.commit()

    async def close(self) -> None:
        """Close the SQLite connection."""

        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def get(self, lead_name: str) -> str | None:
        """Return the session id recorded for ``lead_name``, or ``None``."""

        cursor = await self._require_connection().execute(
            "SELECT session_id FROM lead_sessions WHERE lead_name = ?",
            (lead_name,),
        )
        row = await cursor.fetchone()
        return None if row is None else str(row["session_id"])

    async def upsert(self, lead_name: str, session_id: str) -> None:
        """Record (or replace) the session id for ``lead_name``."""

        now = _now_iso()
        connection = self._require_connection()
        await connection.execute(
            """
            INSERT INTO lead_sessions (lead_name, session_id, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(lead_name) DO UPDATE SET
                session_id = excluded.session_id,
                updated_at = excluded.updated_at
            """,
            (lead_name, session_id, now, now),
        )
        await connection.commit()

    async def list(self) -> list[tuple[str, str]]:
        """Return ``(lead_name, session_id)`` pairs sorted by lead name."""

        cursor = await self._require_connection().execute(
            "SELECT lead_name, session_id FROM lead_sessions ORDER BY lead_name"
        )
        return [(str(r["lead_name"]), str(r["session_id"])) for r in await cursor.fetchall()]

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("LeadSessionStore.initialize() must be called first.")
        return self._connection


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
