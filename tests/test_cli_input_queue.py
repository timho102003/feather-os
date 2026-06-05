"""Tests for the CLI stdin-dispatch rules around the UserInputQueue.

These tests do not launch the full CLI. They drive the same dispatch logic
by replicating the CLI's state variables (busy_event, awaiting_event,
pending_answer, new_run_queue) and invoking the same decision branches.
The intent is to pin the behaviour of idle-vs-busy-vs-awaiting-user
dispatch so future refactors can't regress it silently.
"""

from __future__ import annotations

import asyncio

from feather.core.session.input_queue import UserInputQueue


async def _dispatch(
    line: str,
    *,
    input_queue: UserInputQueue,
    session_id: str,
    busy: asyncio.Event,
    awaiting: asyncio.Event,
    pending_answer: asyncio.Queue,
    new_run_queue: asyncio.Queue,
) -> str:
    """Mirror of the dispatch rules inside cli.run_cli._stdin_reader."""

    text = line.strip()
    if not text:
        return "empty"
    if awaiting.is_set():
        try:
            pending_answer.put_nowait(text)
            awaiting.clear()
            return "answered"
        except asyncio.QueueFull:
            await input_queue.enqueue(session_id, text)
            return "queued"
    if busy.is_set():
        ok = await input_queue.enqueue(session_id, text)
        return "queued" if ok else "dropped"
    await new_run_queue.put(text)
    return "new-run"


async def test_idle_line_starts_new_run() -> None:
    q = UserInputQueue()
    busy = asyncio.Event()
    awaiting = asyncio.Event()
    pending: asyncio.Queue = asyncio.Queue(maxsize=1)
    new_run: asyncio.Queue = asyncio.Queue()

    res = await _dispatch(
        "hello",
        input_queue=q,
        session_id="s1",
        busy=busy,
        awaiting=awaiting,
        pending_answer=pending,
        new_run_queue=new_run,
    )
    assert res == "new-run"
    assert new_run.get_nowait() == "hello"
    assert await q.depth("s1") == 0


async def test_busy_line_enqueues() -> None:
    q = UserInputQueue()
    busy = asyncio.Event()
    busy.set()
    awaiting = asyncio.Event()
    pending: asyncio.Queue = asyncio.Queue(maxsize=1)
    new_run: asyncio.Queue = asyncio.Queue()

    res = await _dispatch(
        "while busy",
        input_queue=q,
        session_id="s1",
        busy=busy,
        awaiting=awaiting,
        pending_answer=pending,
        new_run_queue=new_run,
    )
    assert res == "queued"
    assert await q.peek("s1") == ("while busy",)
    assert new_run.empty()


async def test_awaiting_line_delivers_answer() -> None:
    q = UserInputQueue()
    busy = asyncio.Event()
    awaiting = asyncio.Event()
    awaiting.set()
    pending: asyncio.Queue = asyncio.Queue(maxsize=1)
    new_run: asyncio.Queue = asyncio.Queue()

    res = await _dispatch(
        "yes",
        input_queue=q,
        session_id="s1",
        busy=busy,
        awaiting=awaiting,
        pending_answer=pending,
        new_run_queue=new_run,
    )
    assert res == "answered"
    assert pending.get_nowait() == "yes"
    assert not awaiting.is_set()
    assert await q.depth("s1") == 0


async def test_answer_slot_full_falls_back_to_queue() -> None:
    """If an answer is already queued (shouldn't happen, but defense-in-depth),
    the extra line must land in the injection queue, not be silently lost."""

    q = UserInputQueue()
    busy = asyncio.Event()
    awaiting = asyncio.Event()
    awaiting.set()
    pending: asyncio.Queue = asyncio.Queue(maxsize=1)
    pending.put_nowait("already-there")
    new_run: asyncio.Queue = asyncio.Queue()

    res = await _dispatch(
        "second",
        input_queue=q,
        session_id="s1",
        busy=busy,
        awaiting=awaiting,
        pending_answer=pending,
        new_run_queue=new_run,
    )
    assert res == "queued"
    assert await q.peek("s1") == ("second",)


async def test_empty_input_is_ignored() -> None:
    q = UserInputQueue()
    busy = asyncio.Event()
    awaiting = asyncio.Event()
    pending: asyncio.Queue = asyncio.Queue(maxsize=1)
    new_run: asyncio.Queue = asyncio.Queue()
    res = await _dispatch(
        "   ",
        input_queue=q,
        session_id="s1",
        busy=busy,
        awaiting=awaiting,
        pending_answer=pending,
        new_run_queue=new_run,
    )
    assert res == "empty"
