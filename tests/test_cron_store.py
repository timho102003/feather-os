"""Tests for cron-job persistence and scheduling math."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from feather.models import CronJobStatus, CronScheduleType
from feather.storage.cron_store import CronJobStore
from feather.storage.session_store import SessionStore


async def test_cron_store_creates_updates_and_deletes_jobs(tmp_path: Path) -> None:
    """Cron jobs should persist schedule metadata and support updates/deletes."""

    db_path = tmp_path / "feather.db"
    session_store = SessionStore(db_path)
    cron_store = CronJobStore(db_path)
    await session_store.initialize()
    await cron_store.initialize()

    try:
        session = await session_store.create_session("Lead")
        job = await cron_store.create_job(
            session_id=session.id,
            agent_key="lead",
            name="Daily standup",
            schedule_type=CronScheduleType.CRON,
            schedule_value="0 9 * * 1-5",
            timezone="UTC",
            prompt="Summarize yesterday's work.",
        )

        assert job.name == "Daily standup"
        assert job.status == CronJobStatus.ACTIVE
        assert job.next_run_at is not None

        updated = await cron_store.update_job(
            job.id,
            name="Weekday standup",
            timezone="America/New_York",
            prompt="Summarize yesterday's work and blockers.",
            status=CronJobStatus.PAUSED,
        )

        assert updated.name == "Weekday standup"
        assert updated.timezone == "America/New_York"
        assert updated.prompt == "Summarize yesterday's work and blockers."
        assert updated.status == CronJobStatus.PAUSED

        listed = await cron_store.list_jobs(session_id=session.id)
        assert [record.id for record in listed] == [job.id]

        deleted = await cron_store.delete_job(job.id)
        assert deleted is True
        assert await cron_store.list_jobs(session_id=session.id) == []
    finally:
        await cron_store.close()
        await session_store.close()


async def test_cron_store_tracks_due_jobs_success_and_failures(tmp_path: Path) -> None:
    """Due-job queries and run-state transitions should behave deterministically."""

    db_path = tmp_path / "feather.db"
    session_store = SessionStore(db_path)
    cron_store = CronJobStore(db_path)
    await session_store.initialize()
    await cron_store.initialize()

    try:
        session = await session_store.create_session("Lead")
        due_at = datetime.now(UTC) + timedelta(seconds=2)
        job = await cron_store.create_job(
            session_id=session.id,
            agent_key="lead",
            name="One-time reminder",
            schedule_type=CronScheduleType.ONCE,
            schedule_value=due_at.isoformat(),
            timezone="UTC",
            prompt="Say hello later.",
        )

        due_jobs = await cron_store.list_due_jobs(now=due_at + timedelta(seconds=1), limit=10)
        assert [record.id for record in due_jobs] == [job.id]

        retry_at = due_at + timedelta(minutes=1)
        failed = await cron_store.mark_job_failed(
            job.id,
            error="provider unavailable",
            retry_at=retry_at,
        )
        assert failed.last_error == "provider unavailable"
        assert failed.next_run_at == retry_at.isoformat()
        assert failed.status == CronJobStatus.ACTIVE

        succeeded = await cron_store.mark_job_succeeded(job.id, ran_at=retry_at)
        assert succeeded.status == CronJobStatus.COMPLETED
        assert succeeded.next_run_at is None
        assert succeeded.last_run_at == retry_at.isoformat()
        assert succeeded.last_error is None
    finally:
        await cron_store.close()
        await session_store.close()
