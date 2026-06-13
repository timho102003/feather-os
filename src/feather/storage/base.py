"""Shared lifecycle for long-lived SQLite-backed stores."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import aiosqlite

from feather.storage.connection import open_store_connection
from feather.storage.schema import initialize_database_schema

__all__ = ("BaseSQLiteStore",)


class BaseSQLiteStore:
    """One long-lived aiosqlite connection with the house open/close protocol.

    Subclasses override ``_FOREIGN_KEYS`` when their tables reference nothing
    (LeadSessionStore) and ``_apply_schema`` when they own a single table
    instead of the full schema.
    """

    _FOREIGN_KEYS: ClassVar[bool] = True

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the connection and ensure required tables exist."""

        self._connection = await open_store_connection(
            self._db_path, foreign_keys=self._FOREIGN_KEYS
        )
        await self._apply_schema(self._connection)
        await self._connection.commit()

    async def _apply_schema(self, connection: aiosqlite.Connection) -> None:
        """Create the tables this store relies on (full schema by default)."""

        await initialize_database_schema(connection)

    async def close(self) -> None:
        """Close the SQLite connection (idempotent)."""

        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError(
                f"{type(self).__name__}.initialize() must be called before use."
            )
        return self._connection
