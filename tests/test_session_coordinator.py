"""Lifecycle tests for the per-session run lock map."""

from __future__ import annotations

import asyncio

from feather.core.session.coordinator import SessionRunCoordinator


async def test_lock_entry_evicted_after_release() -> None:
    coordinator = SessionRunCoordinator()
    async with coordinator.acquire("s1"):
        assert coordinator.is_busy("s1")
    assert not coordinator.is_busy("s1")
    assert coordinator._locks == {}
    assert coordinator._refcounts == {}


async def test_concurrent_acquires_serialize_and_share_one_lock() -> None:
    coordinator = SessionRunCoordinator()
    events: list[str] = []
    first_inside = asyncio.Event()
    release_first = asyncio.Event()

    async def first() -> None:
        async with coordinator.acquire("s1"):
            events.append("first-in")
            first_inside.set()
            await release_first.wait()
        events.append("first-out")

    async def second() -> None:
        await first_inside.wait()
        async with coordinator.acquire("s1"):
            events.append("second-in")

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())
    await first_inside.wait()
    await asyncio.sleep(0)  # let second() block on the lock
    assert "s1" in coordinator._locks  # waiter keeps the entry alive
    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert events == ["first-in", "first-out", "second-in"]
    assert coordinator._locks == {}
    assert coordinator._refcounts == {}


async def test_cancelled_waiter_does_not_leak_entry() -> None:
    coordinator = SessionRunCoordinator()
    holder_inside = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with coordinator.acquire("s1"):
            holder_inside.set()
            await release_holder.wait()

    async def waiter() -> None:
        async with coordinator.acquire("s1"):
            pass

    holder_task = asyncio.create_task(holder())
    await holder_inside.wait()
    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    waiter_task.cancel()
    try:
        await waiter_task
    except asyncio.CancelledError:
        pass
    release_holder.set()
    await holder_task
    assert coordinator._locks == {}
    assert coordinator._refcounts == {}


async def test_reacquire_after_eviction_works() -> None:
    coordinator = SessionRunCoordinator()
    async with coordinator.acquire("s1"):
        pass
    async with coordinator.acquire("s1"):
        assert coordinator.is_busy("s1")
    assert not coordinator.is_busy("s1")
