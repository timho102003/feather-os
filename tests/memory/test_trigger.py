"""Tests for LiveMemoryTrigger and NoOpMemoryTrigger."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from feather.memory.config import MemoryTriggerConfig
from feather.memory.enums import MemoryOp, MemoryOwner
from feather.memory.models import (
    MemoryExtractionReport,
)
from feather.memory.trigger import LiveMemoryTrigger, NoOpMemoryTrigger


class _FakeService:
    def __init__(
        self,
        *,
        delay: float = 0.0,
        raise_exc: BaseException | None = None,
        block_event: asyncio.Event | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._delay = delay
        self._exc = raise_exc
        self._block_event = block_event

    async def extract_and_store(
        self,
        session_id: str,
        *,
        agent_model: str,
        owner: MemoryOwner,
    ) -> MemoryExtractionReport:
        self.calls.append({"session_id": session_id, "model": agent_model, "owner": owner})
        if self._block_event is not None:
            await self._block_event.wait()
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._exc is not None:
            raise self._exc
        return MemoryExtractionReport.empty(session_id, reason="fake")


# Inline mode (background=False) ---------------------------------------------


async def test_inline_mode_runs_extraction_synchronously() -> None:
    service = _FakeService()
    trigger = LiveMemoryTrigger(
        service=service,  # type: ignore[arg-type]
        cfg=MemoryTriggerConfig(background=False, enabled=True),
    )
    trigger.maybe_schedule(
        session_id="s1", agent_model="m", owner=MemoryOwner.USER
    )
    # In inline mode the schedule wraps the coroutine in a task too — give the
    # event loop one tick to run it.
    await asyncio.sleep(0)
    assert service.calls == [
        {"session_id": "s1", "model": "m", "owner": MemoryOwner.USER}
    ]


async def test_disabled_trigger_never_calls_service() -> None:
    service = _FakeService()
    trigger = LiveMemoryTrigger(
        service=service,  # type: ignore[arg-type]
        cfg=MemoryTriggerConfig(enabled=False, background=True),
    )
    trigger.maybe_schedule(
        session_id="s1", agent_model="m", owner=MemoryOwner.USER
    )
    await asyncio.sleep(0)
    assert service.calls == []


# Background mode ------------------------------------------------------------


async def test_background_schedule_returns_immediately_and_runs_in_a_task() -> None:
    block = asyncio.Event()
    service = _FakeService(block_event=block)
    trigger = LiveMemoryTrigger(
        service=service,  # type: ignore[arg-type]
        cfg=MemoryTriggerConfig(background=True),
    )
    trigger.maybe_schedule(
        session_id="s1", agent_model="m", owner=MemoryOwner.USER
    )
    # Hand control to the loop so the task can start, then verify it's pending.
    await asyncio.sleep(0)
    assert len(trigger._tasks) == 1
    block.set()
    # Drain to completion
    await trigger.drain(timeout_s=1.0)
    assert len(trigger._tasks) == 0
    assert len(service.calls) == 1


async def test_background_task_exception_is_swallowed_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = _FakeService(raise_exc=RuntimeError("boom"))
    trigger = LiveMemoryTrigger(
        service=service,  # type: ignore[arg-type]
        cfg=MemoryTriggerConfig(background=True),
    )
    with caplog.at_level("ERROR"):
        trigger.maybe_schedule(
            session_id="s1", agent_model="m", owner=MemoryOwner.USER
        )
        await trigger.drain(timeout_s=1.0)
    # The exception must NOT propagate out of drain.
    assert any("memory.trigger.failed" in rec.message for rec in caplog.records)
    assert len(trigger._tasks) == 0


# Drain ----------------------------------------------------------------------


async def test_drain_waits_for_in_flight_tasks() -> None:
    block = asyncio.Event()
    service = _FakeService(block_event=block)
    trigger = LiveMemoryTrigger(
        service=service,  # type: ignore[arg-type]
        cfg=MemoryTriggerConfig(background=True),
    )
    trigger.maybe_schedule(
        session_id="s1", agent_model="m", owner=MemoryOwner.USER
    )
    await asyncio.sleep(0)

    # drain should not return until block is released
    drain_task = asyncio.create_task(trigger.drain(timeout_s=1.0))
    await asyncio.sleep(0.02)
    assert not drain_task.done()
    block.set()
    await drain_task
    assert len(trigger._tasks) == 0


async def test_drain_cancels_tasks_after_timeout(caplog: pytest.LogCaptureFixture) -> None:
    block = asyncio.Event()  # never set
    service = _FakeService(block_event=block)
    trigger = LiveMemoryTrigger(
        service=service,  # type: ignore[arg-type]
        cfg=MemoryTriggerConfig(background=True),
    )
    trigger.maybe_schedule(
        session_id="s1", agent_model="m", owner=MemoryOwner.USER
    )
    await asyncio.sleep(0)
    with caplog.at_level("WARNING"):
        await trigger.drain(timeout_s=0.05)
    assert any("drain_timeout" in rec.message for rec in caplog.records)
    # Task either cancelled or completed via cancel — in either case it's gone.
    assert len(trigger._tasks) == 0
    block.set()


async def test_drain_with_no_tasks_is_a_noop() -> None:
    trigger = LiveMemoryTrigger(
        service=_FakeService(),  # type: ignore[arg-type]
        cfg=MemoryTriggerConfig(background=True),
    )
    await trigger.drain(timeout_s=1.0)


# Closed state ---------------------------------------------------------------


async def test_after_drain_new_schedules_are_ignored() -> None:
    """Once the trigger is drained, additional scheduling silently no-ops."""
    service = _FakeService()
    trigger = LiveMemoryTrigger(
        service=service,  # type: ignore[arg-type]
        cfg=MemoryTriggerConfig(background=True),
    )
    await trigger.drain(timeout_s=1.0)
    trigger.maybe_schedule(
        session_id="s2", agent_model="m", owner=MemoryOwner.USER
    )
    await asyncio.sleep(0)
    assert service.calls == []


# cancel_all -----------------------------------------------------------------


async def test_cancel_all_marks_closed_and_cancels_tasks() -> None:
    block = asyncio.Event()
    service = _FakeService(block_event=block)
    trigger = LiveMemoryTrigger(
        service=service,  # type: ignore[arg-type]
        cfg=MemoryTriggerConfig(background=True),
    )
    trigger.maybe_schedule(
        session_id="s1", agent_model="m", owner=MemoryOwner.USER
    )
    await asyncio.sleep(0)
    trigger.cancel_all()
    block.set()
    # Drain silently swallows any cancellations.
    await trigger.drain(timeout_s=1.0)
    # And further scheduling is suppressed.
    trigger.maybe_schedule("s2", agent_model="m", owner=MemoryOwner.USER)
    await asyncio.sleep(0)
    # service was called once for s1 (it had already started when cancelled)
    # but s2 must NOT have been called.
    assert all(call["session_id"] != "s2" for call in service.calls)


# NoOp trigger ---------------------------------------------------------------


async def test_noop_trigger_satisfies_protocol_without_doing_anything() -> None:
    trigger = NoOpMemoryTrigger()
    trigger.maybe_schedule(
        session_id="s1", agent_model="m", owner=MemoryOwner.USER
    )
    await trigger.drain(timeout_s=1.0)
    trigger.cancel_all()
