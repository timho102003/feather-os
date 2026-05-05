"""Tests for the WorkerHeartbeatStore SQLite-backed liveness store."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from feather.models import WorkerStatus
from feather.storage.worker_heartbeat_store import WorkerHeartbeatStore


async def _open(tmp_path: Path) -> WorkerHeartbeatStore:
    store = WorkerHeartbeatStore(tmp_path / "feather.db")
    await store.initialize()
    return store


async def test_heartbeat_writes_and_reads_back(tmp_path: Path) -> None:
    """A single heartbeat round-trips with all fields preserved."""

    store = await _open(tmp_path)
    try:
        await store.heartbeat(
            session_id="s-lead",
            pid=12345,
            status=WorkerStatus.RUNNING,
        )
        row = await store.get("s-lead")
        assert row is not None
        assert row.session_id == "s-lead"
        assert row.pid == 12345
        assert row.status is WorkerStatus.RUNNING
        # heartbeat_at must be a tz-aware UTC datetime within ~5s of "now"
        assert row.heartbeat_at.tzinfo is not None
        delta = abs((datetime.now(UTC) - row.heartbeat_at).total_seconds())
        assert delta < 5.0
    finally:
        await store.close()


async def test_heartbeat_upserts_in_place(tmp_path: Path) -> None:
    """A second heartbeat for the same session_id updates instead of inserting."""

    store = await _open(tmp_path)
    try:
        await store.heartbeat(
            session_id="s-lead", pid=111, status=WorkerStatus.RUNNING
        )
        first = await store.get("s-lead")
        assert first is not None

        await store.heartbeat(
            session_id="s-lead", pid=222, status=WorkerStatus.STOPPING
        )
        second = await store.get("s-lead")
        assert second is not None
        assert second.pid == 222
        assert second.status is WorkerStatus.STOPPING
        # heartbeat_at must advance (or at minimum not regress)
        assert second.heartbeat_at >= first.heartbeat_at

        # Exactly one row per session_id.
        assert await store.count() == 1
    finally:
        await store.close()


async def test_get_returns_none_for_unknown_session(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        assert await store.get("nope") is None
    finally:
        await store.close()


async def test_heartbeats_isolated_per_session(tmp_path: Path) -> None:
    """Heartbeats for different sessions do not interfere."""

    store = await _open(tmp_path)
    try:
        await store.heartbeat(
            session_id="s-a", pid=1, status=WorkerStatus.RUNNING
        )
        await store.heartbeat(
            session_id="s-b", pid=2, status=WorkerStatus.RUNNING
        )
        a = await store.get("s-a")
        b = await store.get("s-b")
        assert a is not None and b is not None
        assert a.pid == 1 and b.pid == 2
        assert await store.count() == 2
    finally:
        await store.close()


async def test_clear_removes_row(tmp_path: Path) -> None:
    """`clear` deletes the heartbeat (used on graceful shutdown bookkeeping)."""

    store = await _open(tmp_path)
    try:
        await store.heartbeat(
            session_id="s-lead", pid=5, status=WorkerStatus.RUNNING
        )
        await store.clear("s-lead")
        assert await store.get("s-lead") is None
    finally:
        await store.close()


async def test_status_enum_round_trips_via_storage(tmp_path: Path) -> None:
    """All defined WorkerStatus values must survive write -> read."""

    store = await _open(tmp_path)
    try:
        for idx, status in enumerate(WorkerStatus):
            sid = f"s-{idx}"
            await store.heartbeat(session_id=sid, pid=1000 + idx, status=status)
            row = await store.get(sid)
            assert row is not None
            assert row.status is status
    finally:
        await store.close()
