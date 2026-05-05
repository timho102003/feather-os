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


async def test_concurrent_write_and_read_under_wal(tmp_path: Path) -> None:
    """A separate connection's writes must be visible to a separate
    connection's reads under WAL — the worker writes from one process
    while the supervisor reads from another, and a stuck contention
    bug here would mask hangs in production.
    """

    import asyncio

    # Two store instances on the same DB file simulate two processes.
    writer = WorkerHeartbeatStore(tmp_path / "feather.db")
    reader = WorkerHeartbeatStore(tmp_path / "feather.db")
    await writer.initialize()
    await reader.initialize()

    try:
        # Writer hammers heartbeats; reader observes the latest pid.
        async def write_loop() -> None:
            for i in range(50):
                await writer.heartbeat(
                    session_id="s-wal", pid=1000 + i, status=WorkerStatus.RUNNING
                )

        async def read_loop() -> int:
            seen = 0
            for _ in range(50):
                row = await reader.get("s-wal")
                if row is not None:
                    seen += 1
                await asyncio.sleep(0)
            return seen

        _, seen = await asyncio.gather(write_loop(), read_loop())
        # The reader must see at least one row — exact count is timing-
        # dependent so we just assert "made forward progress, no deadlock".
        assert seen > 0
        # Final pid in writer's last heartbeat is what reader observes.
        final = await reader.get("s-wal")
        assert final is not None
        assert final.pid == 1049
    finally:
        await writer.close()
        await reader.close()


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
