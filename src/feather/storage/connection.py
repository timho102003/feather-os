"""Shared aiosqlite connection bootstrap for feather stores."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

_DEFAULT_BUSY_TIMEOUT_MS = 5000

__all__ = ("open_store_connection",)


async def open_store_connection(
    db_path: Path,
    *,
    foreign_keys: bool = True,
    busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
) -> aiosqlite.Connection:
    """Open one long-lived store connection with the house pragma set.

    Every store opens its own connection to the same ``feather.db``,
    across multiple OS processes. WAL keeps readers from blocking the
    writer; the explicit ``busy_timeout`` makes a contended write back
    off instead of hard-failing with ``database is locked`` (and keeps
    that contract independent of ``sqlite3.connect``'s default timeout).
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(db_path)
    connection.row_factory = aiosqlite.Row
    if foreign_keys:
        await connection.execute("PRAGMA foreign_keys=ON;")
    await connection.execute("PRAGMA journal_mode=WAL;")
    await connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms};")
    return connection
