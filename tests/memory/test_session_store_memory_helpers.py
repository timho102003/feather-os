"""Tests for SessionStore helpers added for the memory subsystem."""

from __future__ import annotations

from pathlib import Path

from feather.models import MessageRole
from feather.storage.session_store import SessionStore


async def test_get_non_compact_after_returns_everything_when_anchor_is_none(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite")
    await store.initialize()
    try:
        session = await store.create_session("Lead")
        await store.add_message(session.id, MessageRole.USER, "one")
        await store.add_message(session.id, MessageRole.ASSISTANT, "two")
        await store.add_message(
            session.id, MessageRole.ASSISTANT, "summary", is_compact=True
        )
        await store.add_message(session.id, MessageRole.USER, "three")

        msgs = await store.get_non_compact_after(session.id, after_message_id=None)
        contents = [m.content for m in msgs]
        # compact row must be skipped
        assert contents == ["one", "two", "three"]
    finally:
        await store.close()


async def test_get_non_compact_after_returns_only_messages_after_the_anchor(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "db.sqlite")
    await store.initialize()
    try:
        session = await store.create_session("Lead")
        await store.add_message(session.id, MessageRole.USER, "one")
        m2 = await store.add_message(session.id, MessageRole.ASSISTANT, "two")
        await store.add_message(session.id, MessageRole.USER, "three")
        await store.add_message(session.id, MessageRole.ASSISTANT, "four")

        msgs = await store.get_non_compact_after(session.id, after_message_id=m2.id)
        assert [m.content for m in msgs] == ["three", "four"]
    finally:
        await store.close()


async def test_get_non_compact_after_skips_compact_rows(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite")
    await store.initialize()
    try:
        session = await store.create_session("Lead")
        m1 = await store.add_message(session.id, MessageRole.USER, "one")
        await store.add_message(
            session.id, MessageRole.ASSISTANT, "summary", is_compact=True
        )
        await store.add_message(session.id, MessageRole.USER, "two")

        msgs = await store.get_non_compact_after(session.id, after_message_id=m1.id)
        assert [m.content for m in msgs] == ["two"]
    finally:
        await store.close()


async def test_get_non_compact_after_unknown_anchor_returns_everything(tmp_path: Path) -> None:
    """Qdrant might reference a message that no longer exists locally — treat as 'from start'."""
    store = SessionStore(tmp_path / "db.sqlite")
    await store.initialize()
    try:
        session = await store.create_session("Lead")
        await store.add_message(session.id, MessageRole.USER, "one")
        await store.add_message(session.id, MessageRole.ASSISTANT, "two")

        msgs = await store.get_non_compact_after(
            session.id, after_message_id="ghost-id-that-doesnt-exist"
        )
        assert [m.content for m in msgs] == ["one", "two"]
    finally:
        await store.close()


async def test_get_non_compact_after_filters_by_session_id(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite")
    await store.initialize()
    try:
        s1 = await store.create_session("Lead")
        s2 = await store.create_session("Lead")
        await store.add_message(s1.id, MessageRole.USER, "s1-msg")
        await store.add_message(s2.id, MessageRole.USER, "s2-msg")

        msgs = await store.get_non_compact_after(s1.id, after_message_id=None)
        assert [m.content for m in msgs] == ["s1-msg"]
    finally:
        await store.close()


async def test_get_recent_non_compact_returns_last_n_in_order(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite")
    await store.initialize()
    try:
        session = await store.create_session("Lead")
        for i in range(5):
            await store.add_message(session.id, MessageRole.USER, f"m{i}")

        recent = await store.get_recent_non_compact(session.id, limit=3)
        assert [m.content for m in recent] == ["m2", "m3", "m4"]
    finally:
        await store.close()


async def test_get_recent_non_compact_skips_compact_rows(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite")
    await store.initialize()
    try:
        session = await store.create_session("Lead")
        await store.add_message(session.id, MessageRole.USER, "a")
        await store.add_message(session.id, MessageRole.ASSISTANT, "summary", is_compact=True)
        await store.add_message(session.id, MessageRole.USER, "b")
        await store.add_message(session.id, MessageRole.ASSISTANT, "c")

        recent = await store.get_recent_non_compact(session.id, limit=10)
        assert [m.content for m in recent] == ["a", "b", "c"]
    finally:
        await store.close()


async def test_get_recent_non_compact_with_zero_limit_returns_empty(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite")
    await store.initialize()
    try:
        session = await store.create_session("Lead")
        await store.add_message(session.id, MessageRole.USER, "a")
        assert await store.get_recent_non_compact(session.id, limit=0) == []
    finally:
        await store.close()
