"""Tests for the shared store-connection bootstrap."""

from __future__ import annotations

from pathlib import Path

from feather.storage.connection import open_store_connection


async def test_open_store_connection_applies_house_pragmas(tmp_path: Path) -> None:
    connection = await open_store_connection(tmp_path / "sub" / "feather.db")
    try:
        for pragma, expected in (
            ("busy_timeout", 5000),
            ("foreign_keys", 1),
        ):
            cursor = await connection.execute(f"PRAGMA {pragma};")
            row = await cursor.fetchone()
            assert int(row[0]) == expected, pragma
        cursor = await connection.execute("PRAGMA journal_mode;")
        row = await cursor.fetchone()
        assert str(row[0]).lower() == "wal"
    finally:
        await connection.close()


async def test_open_store_connection_can_skip_foreign_keys(tmp_path: Path) -> None:
    connection = await open_store_connection(
        tmp_path / "feather.db", foreign_keys=False
    )
    try:
        cursor = await connection.execute("PRAGMA foreign_keys;")
        row = await cursor.fetchone()
        assert int(row[0]) == 0
    finally:
        await connection.close()
