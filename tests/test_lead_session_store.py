"""LeadSessionStore maps each lead name to its durable session id."""

from __future__ import annotations

import pytest

from feather.storage.lead_session_store import LeadSessionStore


@pytest.fixture
async def store(tmp_path):
    s = LeadSessionStore(tmp_path / "feather.db")
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()


async def test_get_missing_returns_none(store):
    assert await store.get("tim") is None


async def test_upsert_then_get(store):
    await store.upsert("tim", "sess-1")
    assert await store.get("tim") == "sess-1"


async def test_upsert_is_idempotent_and_updates(store):
    await store.upsert("tim", "sess-1")
    await store.upsert("tim", "sess-2")
    assert await store.get("tim") == "sess-2"
    assert [name for name, _ in await store.list()] == ["tim"]


async def test_list_is_sorted_by_lead_name(store):
    await store.upsert("sophia", "s-2")
    await store.upsert("tim", "s-1")
    assert await store.list() == [("sophia", "s-2"), ("tim", "s-1")]


async def test_busy_timeout_pragma_is_set(store):
    cursor = await store._connection.execute("PRAGMA busy_timeout")
    row = await cursor.fetchone()
    assert int(row[0]) > 0


async def test_use_before_initialize_raises(tmp_path):
    s = LeadSessionStore(tmp_path / "feather.db")
    with pytest.raises(RuntimeError):
        await s.get("tim")
