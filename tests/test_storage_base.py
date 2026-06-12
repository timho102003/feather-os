"""Tests for BaseSQLiteStore shared lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.storage.lead_session_store import LeadSessionStore
from feather.storage.session_store import SessionStore


async def test_require_connection_before_initialize_raises_with_class_name_session(
    tmp_path: Path,
) -> None:
    """SessionStore raises with its class name before initialize()."""

    store = SessionStore(tmp_path / "feather.db")
    with pytest.raises(RuntimeError, match=r"SessionStore\.initialize\(\)"):
        await store.get_session("s-1")


async def test_require_connection_before_initialize_raises_with_class_name_lead_session(
    tmp_path: Path,
) -> None:
    """LeadSessionStore raises with its class name before initialize()."""

    store = LeadSessionStore(tmp_path / "feather.db")
    with pytest.raises(RuntimeError, match=r"LeadSessionStore\.initialize\(\)"):
        await store.get("tim")


async def test_initialize_close_close_is_idempotent(tmp_path: Path) -> None:
    """Calling close() twice must not raise."""

    store = SessionStore(tmp_path / "feather.db")
    await store.initialize()
    await store.close()
    await store.close()  # second close must be a no-op


async def test_lead_session_store_opens_without_foreign_keys(tmp_path: Path) -> None:
    """LeadSessionStore disables foreign keys; SessionStore keeps them on."""

    lead_store = LeadSessionStore(tmp_path / "feather_lead.db")
    session_store = SessionStore(tmp_path / "feather_session.db")
    await lead_store.initialize()
    await session_store.initialize()
    try:
        cursor = await lead_store._connection.execute("PRAGMA foreign_keys")
        row = await cursor.fetchone()
        assert int(row[0]) == 0, "LeadSessionStore must have foreign_keys=OFF"

        cursor = await session_store._connection.execute("PRAGMA foreign_keys")
        row = await cursor.fetchone()
        assert int(row[0]) == 1, "SessionStore must have foreign_keys=ON"
    finally:
        await lead_store.close()
        await session_store.close()


async def test_base_applies_full_schema_by_default(tmp_path: Path) -> None:
    """After SessionStore.initialize the core tables are present."""

    store = SessionStore(tmp_path / "feather.db")
    await store.initialize()
    try:
        cursor = await store._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        rows = await cursor.fetchall()
        table_names = {row["name"] for row in rows}
        for expected in ("sessions", "messages", "agent_messages", "tasks"):
            assert expected in table_names, f"Missing table: {expected}"
    finally:
        await store.close()
