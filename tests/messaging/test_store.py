"""Tests for :class:`feather.messaging.store.MessagingStore`."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from feather.messaging.models import Platform
from feather.messaging.store import MessagingStore


async def _store(tmp_path: Path) -> MessagingStore:
    db = tmp_path / "feather.db"
    store = MessagingStore(db)
    await store.initialize()
    # The sessions table is created by initialize_database_schema, but
    # the messaging_chats foreign key would fail without a session row,
    # so we drop the constraint at runtime by allowing arbitrary
    # session_ids in tests. SQLite ignores FK by default unless PRAGMA
    # foreign_keys=ON is set.
    return store


async def test_store_round_trips_credentials(tmp_path: Path) -> None:
    store = await _store(tmp_path)

    await store.save_credentials(
        Platform.TELEGRAM, {"bot_token": "1234:abc"}
    )
    record = await store.load_credentials(Platform.TELEGRAM)

    assert record is not None
    assert record.platform == Platform.TELEGRAM
    assert record.config == {"bot_token": "1234:abc"}
    assert record.enabled is True


async def test_store_lists_only_known_platforms(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    await store.save_credentials(Platform.TELEGRAM, {"bot_token": "x"})
    await store.save_credentials(Platform.LINE, {"channel_secret": "y"})

    # Insert an unknown-platform row directly to simulate a future
    # release adding a platform name we don't recognise yet. The store
    # should skip it without crashing.
    async with aiosqlite.connect(store._db_path) as conn:  # type: ignore[attr-defined]
        await conn.execute(
            """
            INSERT INTO messaging_credentials
                (platform, config_json, enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("future_platform", "{}", 1, "now", "now"),
        )
        await conn.commit()

    records = await store.list_credentials()
    platforms = {r.platform for r in records}
    assert platforms == {Platform.TELEGRAM, Platform.LINE}


async def test_store_replaces_credentials_on_save(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    await store.save_credentials(Platform.TELEGRAM, {"bot_token": "first"})
    await store.save_credentials(Platform.TELEGRAM, {"bot_token": "second"})

    record = await store.load_credentials(Platform.TELEGRAM)
    assert record is not None
    assert record.config == {"bot_token": "second"}


async def test_store_deletes_credentials(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    await store.save_credentials(Platform.TELEGRAM, {"bot_token": "x"})
    await store.delete_credentials(Platform.TELEGRAM)

    assert await store.load_credentials(Platform.TELEGRAM) is None


async def test_chat_mapping_create_and_get(tmp_path: Path) -> None:
    store = await _store(tmp_path)

    mapping = await store.upsert_chat_mapping(
        platform=Platform.LINE,
        chat_id="user-1",
        session_id="sess-1",
        display_name="Alice",
    )
    assert mapping.session_id == "sess-1"
    assert mapping.display_name == "Alice"

    found = await store.get_chat_mapping(Platform.LINE, "user-1")
    assert found is not None
    assert found.session_id == "sess-1"


async def test_chat_mapping_upsert_keeps_session_updates_name(
    tmp_path: Path,
) -> None:
    store = await _store(tmp_path)

    await store.upsert_chat_mapping(
        platform=Platform.LINE,
        chat_id="user-1",
        session_id="sess-1",
        display_name="Alice",
    )
    refreshed = await store.upsert_chat_mapping(
        platform=Platform.LINE,
        chat_id="user-1",
        session_id="sess-1",  # Reuse the same session id.
        display_name="Alice Renamed",
    )

    assert refreshed.session_id == "sess-1"
    assert refreshed.display_name == "Alice Renamed"


async def test_count_chats_for_platform(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    await store.upsert_chat_mapping(
        Platform.LINE, "u1", "s1", display_name="a"
    )
    await store.upsert_chat_mapping(
        Platform.LINE, "u2", "s2", display_name="b"
    )
    await store.upsert_chat_mapping(
        Platform.TELEGRAM, "12345", "s3", display_name="c"
    )

    assert await store.count_chats_for_platform(Platform.LINE) == 2
    assert await store.count_chats_for_platform(Platform.TELEGRAM) == 1
    assert await store.count_chats_for_platform(Platform.WHATSAPP) == 0


async def test_claim_inbound_is_atomic(tmp_path: Path) -> None:
    store = await _store(tmp_path)

    first = await store.claim_inbound(Platform.WHATSAPP, "wamid-1")
    second = await store.claim_inbound(Platform.WHATSAPP, "wamid-1")

    assert first is True
    assert second is False


async def test_claim_inbound_distinguishes_platforms(tmp_path: Path) -> None:
    store = await _store(tmp_path)

    assert await store.claim_inbound(Platform.LINE, "id-1") is True
    # Same id, different platform — not a duplicate.
    assert await store.claim_inbound(Platform.WHATSAPP, "id-1") is True


async def test_prune_inbound_older_than_removes_old_rows(
    tmp_path: Path,
) -> None:
    store = await _store(tmp_path)
    # Insert a row with an obviously-stale timestamp.
    async with aiosqlite.connect(store._db_path) as conn:  # type: ignore[attr-defined]
        await conn.execute(
            """
            INSERT INTO messaging_inbound_dedup
                (platform, native_message_id, seen_at)
            VALUES (?, ?, ?)
            """,
            (Platform.LINE.value, "old-id", "2000-01-01T00:00:00+00:00"),
        )
        await conn.commit()
    await store.claim_inbound(Platform.LINE, "fresh-id")

    pruned = await store.prune_inbound_older_than(hours=1)

    assert pruned == 1
    # Fresh id is still present (would-be-duplicate returns False).
    assert await store.claim_inbound(Platform.LINE, "fresh-id") is False
    # Old id is gone (claim succeeds).
    assert await store.claim_inbound(Platform.LINE, "old-id") is True
