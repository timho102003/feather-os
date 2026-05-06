"""Tests for the supervisor-side restart watcher."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from feather.core.restart_watcher import RestartWatcher
from feather.storage.agent_message_store import AgentMessageStore
from feather.storage.session_store import SessionStore


async def _open_stores(tmp_path: Path) -> tuple[SessionStore, AgentMessageStore]:
    sess = SessionStore(tmp_path / "feather.db")
    await sess.initialize()
    msg = AgentMessageStore(tmp_path / "feather.db")
    await msg.initialize()
    return sess, msg


async def test_run_once_no_op_when_flag_unset(tmp_path: Path) -> None:
    sess, msg = await _open_stores(tmp_path)
    try:
        session = await sess.create_session("lead")
        restart_fn = AsyncMock()
        watcher = RestartWatcher(
            session_store=sess,
            message_store=msg,
            lead_session_id=session.id,
            restart_fn=restart_fn,
        )
        triggered = await watcher.run_once()
        assert triggered is False
        restart_fn.assert_not_awaited()
        inbox = await msg.inbox(to_session_id=session.id, to_agent_name="Lead")
        assert inbox == []
    finally:
        await sess.close()
        await msg.close()


async def test_run_once_triggers_restart_when_flag_set(tmp_path: Path) -> None:
    sess, msg = await _open_stores(tmp_path)
    try:
        session = await sess.create_session("lead")
        await sess.mark_restart_requested(session.id, "patched compaction.py")
        restart_fn = AsyncMock()
        watcher = RestartWatcher(
            session_store=sess,
            message_store=msg,
            lead_session_id=session.id,
            restart_fn=restart_fn,
        )
        triggered = await watcher.run_once()

        assert triggered is True
        restart_fn.assert_awaited_once()
        # Flag is cleared after the restart cycle so the watcher doesn't
        # spin on the same row.
        assert await sess.get_restart_request(session.id) is None
        # Lead inbox has a system message describing the outcome.
        inbox = await msg.inbox(to_session_id=session.id, to_agent_name="Lead")
        assert len(inbox) == 1
        assert "succeeded" in inbox[0].body.lower()
        assert "patched compaction.py" in inbox[0].body
        assert inbox[0].from_agent_name == "__system_restart_watcher"
    finally:
        await sess.close()
        await msg.close()


async def test_run_once_clears_flag_and_reports_failure_when_restart_raises(
    tmp_path: Path,
) -> None:
    """A failing restart_fn must not leave the flag set (would loop forever)."""

    sess, msg = await _open_stores(tmp_path)
    try:
        session = await sess.create_session("lead")
        await sess.mark_restart_requested(session.id, "patch X")

        async def boom() -> None:
            raise RuntimeError("supervisor exploded")

        watcher = RestartWatcher(
            session_store=sess,
            message_store=msg,
            lead_session_id=session.id,
            restart_fn=boom,
        )
        triggered = await watcher.run_once()
        assert triggered is True
        # Flag MUST be cleared, otherwise the next tick re-fires
        # restart() in a tight loop.
        assert await sess.get_restart_request(session.id) is None
        inbox = await msg.inbox(to_session_id=session.id, to_agent_name="Lead")
        assert len(inbox) == 1
        assert "failed" in inbox[0].body.lower()
        assert "RuntimeError" in inbox[0].body
        assert "supervisor exploded" in inbox[0].body
    finally:
        await sess.close()
        await msg.close()


async def test_run_once_calls_cancel_callback_before_restart(
    tmp_path: Path,
) -> None:
    """If a cancel callback is supplied, it must run before restart() so the
    LeadSupervisor.shutdown invariant ("no concurrent run") holds."""

    sess, msg = await _open_stores(tmp_path)
    try:
        session = await sess.create_session("lead")
        await sess.mark_restart_requested(session.id, "patch")

        order: list[str] = []

        async def cancel() -> bool:
            order.append("cancel")
            return True

        async def restart() -> None:
            order.append("restart")

        watcher = RestartWatcher(
            session_store=sess,
            message_store=msg,
            lead_session_id=session.id,
            restart_fn=restart,
            cancel_in_flight_run=cancel,
        )
        await watcher.run_once()
        assert order == ["cancel", "restart"]
    finally:
        await sess.close()
        await msg.close()


async def test_run_once_proceeds_when_cancel_callback_times_out(
    tmp_path: Path,
) -> None:
    """A stuck cancel callback must NOT block the watcher indefinitely.

    The supervisor's restart() will SIGKILL the worker as a last
    resort, so even if the run task's cleanup hangs forever the watcher
    must proceed. Without the cancel-timeout the poll loop would block
    for the full hang duration and the hang banner could never warn
    the user.
    """

    sess, msg = await _open_stores(tmp_path)
    try:
        session = await sess.create_session("lead")
        await sess.mark_restart_requested(session.id, "patch")

        cancel_started = asyncio.Event()
        restart_called = asyncio.Event()

        async def stuck_cancel() -> bool:
            cancel_started.set()
            # Wait far longer than the test's cancel_timeout — the
            # watcher must time out and proceed regardless.
            await asyncio.sleep(60)
            return True  # pragma: no cover — not reached

        async def restart_fn() -> None:
            restart_called.set()

        watcher = RestartWatcher(
            session_store=sess,
            message_store=msg,
            lead_session_id=session.id,
            restart_fn=restart_fn,
            cancel_in_flight_run=stuck_cancel,
            cancel_timeout_seconds=0.1,
        )
        await asyncio.wait_for(watcher.run_once(), timeout=2.0)
        assert cancel_started.is_set()
        assert restart_called.is_set(), "restart must fire after cancel timeout"
    finally:
        await sess.close()
        await msg.close()


async def test_run_once_does_not_re_trigger_after_clearing(tmp_path: Path) -> None:
    """Two consecutive ticks where the flag was set then cleared must
    only fire one restart."""

    sess, msg = await _open_stores(tmp_path)
    try:
        session = await sess.create_session("lead")
        await sess.mark_restart_requested(session.id, "patch")
        restart_fn = AsyncMock()
        watcher = RestartWatcher(
            session_store=sess,
            message_store=msg,
            lead_session_id=session.id,
            restart_fn=restart_fn,
        )
        first = await watcher.run_once()
        second = await watcher.run_once()
        assert first is True
        assert second is False
        assert restart_fn.await_count == 1
    finally:
        await sess.close()
        await msg.close()
