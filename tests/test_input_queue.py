"""Tests for the per-session UserInputQueue."""

from __future__ import annotations

import asyncio
import logging

import pytest

from feather.core.input_queue import UserInputQueue


async def test_enqueue_and_drain_preserves_order() -> None:
    q = UserInputQueue()
    for text in ("one", "two", "three"):
        assert await q.enqueue("s1", text) is True
    drained = await q.drain("s1")
    assert drained == ["one", "two", "three"]
    assert await q.drain("s1") == []


async def test_empty_strings_are_ignored() -> None:
    q = UserInputQueue()
    assert await q.enqueue("s1", "") is False
    assert await q.enqueue("s1", "   ") is False
    assert await q.depth("s1") == 0


async def test_whitespace_is_stripped_on_enqueue() -> None:
    q = UserInputQueue()
    await q.enqueue("s1", "  hello  ")
    assert await q.drain("s1") == ["hello"]


async def test_sessions_are_isolated() -> None:
    q = UserInputQueue()
    await q.enqueue("s1", "a")
    await q.enqueue("s2", "b")
    assert await q.drain("s1") == ["a"]
    assert await q.drain("s2") == ["b"]


async def test_overflow_drops_oldest_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    q = UserInputQueue(max_per_session=3)
    caplog.set_level(logging.WARNING)
    for text in ("a", "b", "c"):
        await q.enqueue("s1", text)
    # 4th should push out "a"
    await q.enqueue("s1", "d")
    drained = await q.drain("s1")
    assert drained == ["b", "c", "d"]
    assert any("overflow" in rec.message for rec in caplog.records)


async def test_invalid_max_per_session_raises() -> None:
    with pytest.raises(ValueError):
        UserInputQueue(max_per_session=0)
    with pytest.raises(ValueError):
        UserInputQueue(max_per_session=-1)


async def test_concurrent_enqueues_preserve_all_messages() -> None:
    q = UserInputQueue(max_per_session=1024)
    count = 200

    async def push(i: int) -> None:
        await q.enqueue("s1", f"msg-{i}")

    await asyncio.gather(*(push(i) for i in range(count)))
    drained = await q.drain("s1")
    assert len(drained) == count
    # All unique and none lost
    assert set(drained) == {f"msg-{i}" for i in range(count)}


async def test_concurrent_drain_during_enqueue_never_loses_messages() -> None:
    """Interleave enqueue and drain; the union must contain every message."""

    q = UserInputQueue(max_per_session=1024)
    total = 200
    produced_event = asyncio.Event()
    seen: list[str] = []

    async def producer() -> None:
        for i in range(total):
            await q.enqueue("s1", f"m{i}")
        produced_event.set()

    async def draining_consumer() -> None:
        while not produced_event.is_set():
            seen.extend(await q.drain("s1"))
            await asyncio.sleep(0)
        # Final drain after producer finished.
        seen.extend(await q.drain("s1"))

    await asyncio.gather(producer(), draining_consumer())
    assert len(seen) == total
    assert set(seen) == {f"m{i}" for i in range(total)}


async def test_peek_returns_snapshot_without_mutation() -> None:
    q = UserInputQueue()
    await q.enqueue("s1", "x")
    await q.enqueue("s1", "y")
    snap = await q.peek("s1")
    assert snap == ("x", "y")
    # Still present
    assert await q.depth("s1") == 2


async def test_clear_empties_one_session_only() -> None:
    q = UserInputQueue()
    await q.enqueue("s1", "a")
    await q.enqueue("s2", "b")
    assert await q.clear("s1") == 1
    assert await q.depth("s1") == 0
    assert await q.depth("s2") == 1


async def test_extend_enqueues_many_and_skips_empty() -> None:
    q = UserInputQueue()
    count = await q.extend("s1", ["a", "", "  ", "b"])
    assert count == 2
    assert await q.drain("s1") == ["a", "b"]
