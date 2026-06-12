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

import aiosqlite

from feather.storage.base import BaseSQLiteStore
from feather.storage.schema import LEAD_SESSIONS_TABLE

__all__ = ("LeadSessionStore",)


class LeadSessionStore(BaseSQLiteStore):
    """One aiosqlite connection owning the ``lead_sessions`` pointer table."""

    # No foreign_keys pragma: the pointer table references nothing.
    _FOREIGN_KEYS = False

    async def _apply_schema(self, connection: aiosqlite.Connection) -> None:
        """Create only the ``lead_sessions`` table (no full schema)."""

        await connection.execute(LEAD_SESSIONS_TABLE.create_sql)

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


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
