"""Tests for background cron-job dispatch.

Cron now routes through the agent-message mailbox: when a job fires,
the scheduler drops a single message into the target agent's inbox
and marks the job succeeded. The agent's existing
``resume_on_inbox`` path picks it up on its next iteration. So the
scheduler-side test surface is small — does the message land, with
the right shape, addressed to the right agent — without needing an
LLM provider, an agent factory, or any conversation-loop fixtures.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from feather.core.scheduling.cron_scheduler import CronScheduler
from feather.models import AgentMessageStatus, CronScheduleType, SchedulerConfig
from feather.storage.agent_message_store import AgentMessageStore
from feather.storage.cron_store import CronJobStore
from feather.storage.session_store import SessionStore


def _scheduler_config() -> SchedulerConfig:
    return SchedulerConfig(
        enabled=True,
        poll_interval_seconds=2,
        failure_retry_seconds=30,
        max_due_jobs_per_tick=10,
    )


async def _open_stores(
    tmp_path: Path,
) -> tuple[CronJobStore, AgentMessageStore, str]:
    """Open the three SQLite stores cron needs and return a real lead session id.

    The FK on ``cron_jobs.session_id`` requires a row in ``sessions``,
    so seeding one here keeps the test focused on cron behavior rather
    than seeding boilerplate inside each test body.
    """

    db_path = tmp_path / "feather.db"
    session_store = SessionStore(db_path)
    cron_store = CronJobStore(db_path)
    message_store = AgentMessageStore(db_path)
    await session_store.initialize()
    await cron_store.initialize()
    await message_store.initialize()
    session = await session_store.create_session("lead")
    await session_store.close()
    return cron_store, message_store, session.id


async def test_due_job_lands_in_target_agent_inbox(tmp_path: Path) -> None:
    """Firing a due job must produce ONE pending agent_messages row
    addressed to the job's target agent, and mark the job succeeded."""

    cron_store, message_store, session_id = await _open_stores(tmp_path)
    try:
        due_at = datetime.now(UTC) + timedelta(seconds=2)
        job = await cron_store.create_job(
            session_id=session_id,
            agent_key="Lead",
            name="Reminder",
            schedule_type=CronScheduleType.ONCE,
            schedule_value=due_at.isoformat(),
            timezone="UTC",
            prompt="Tell the user the reminder fired.",
        )

        scheduler = CronScheduler(
            config=_scheduler_config(),
            cron_store=cron_store,
            message_store=message_store,
        )
        dispatched = await scheduler.run_pending(
            now=due_at + timedelta(seconds=1)
        )

        assert dispatched == 1
        updated = await cron_store.get_job(job.id)
        assert updated.status.value == "completed"

        inbox = await message_store.inbox(
            to_session_id=session_id, to_agent_name="Lead"
        )
        assert len(inbox) == 1
        msg = inbox[0]
        assert msg.from_agent_name == "__system_cron"
        assert msg.from_session_id == session_id
        assert msg.expects_response is False
        assert msg.status is AgentMessageStatus.PENDING
        assert "<scheduled_task_trigger>" in msg.body
        assert "<cron_job_name>Reminder</cron_job_name>" in msg.body
        assert "Tell the user the reminder fired." in msg.body
    finally:
        await cron_store.close()
        await message_store.close()


async def test_dispatch_fires_triggered_event(tmp_path: Path) -> None:
    """The TUI's runtime-event handler must see ``scheduled_task_triggered``
    so the user gets live feedback that the cron prompt was queued.

    The old design also fired ``scheduled_task_completed`` after the
    LLM finished; that event no longer makes sense (the scheduler does
    not block on the agent), so it must NOT appear here.
    """

    cron_store, message_store, session_id = await _open_stores(tmp_path)
    try:
        due_at = datetime.now(UTC) + timedelta(seconds=1)
        await cron_store.create_job(
            session_id=session_id,
            agent_key="Lead",
            name="Reminder",
            schedule_type=CronScheduleType.ONCE,
            schedule_value=due_at.isoformat(),
            timezone="UTC",
            prompt="prompt",
        )
        events: list[str] = []
        scheduler = CronScheduler(
            config=_scheduler_config(),
            cron_store=cron_store,
            message_store=message_store,
            event_handler_resolver=lambda sid: (
                (lambda ev: events.append(ev.kind)) if sid == session_id else None
            ),
        )
        await scheduler.run_pending(now=due_at + timedelta(seconds=1))

        assert "scheduled_task_triggered" in events
        # Mailbox routing means the scheduler doesn't observe completion.
        assert "scheduled_task_completed" not in events
        assert "scheduled_task_failed" not in events
    finally:
        await cron_store.close()
        await message_store.close()


async def test_mailbox_send_failure_marks_job_failed_with_retry(
    tmp_path: Path,
) -> None:
    """If the mailbox write itself raises, the job must be marked failed
    with a retry_at, and a ``scheduled_task_failed`` event must fire so
    the user knows the prompt didn't land."""

    cron_store, message_store, session_id = await _open_stores(tmp_path)
    try:
        due_at = datetime.now(UTC) + timedelta(seconds=1)
        job = await cron_store.create_job(
            session_id=session_id,
            agent_key="Lead",
            name="Reminder",
            schedule_type=CronScheduleType.ONCE,
            schedule_value=due_at.isoformat(),
            timezone="UTC",
            prompt="prompt",
        )

        # Force the mailbox send to raise — exercise the failure path
        # without simulating disk corruption.
        async def boom(**_kwargs):
            raise RuntimeError("simulated write failure")

        message_store.send = boom  # type: ignore[method-assign]

        events: list[str] = []
        scheduler = CronScheduler(
            config=_scheduler_config(),
            cron_store=cron_store,
            message_store=message_store,
            event_handler_resolver=lambda sid: (
                (lambda ev: events.append(ev.kind)) if sid == session_id else None
            ),
        )
        await scheduler.run_pending(now=due_at + timedelta(seconds=1))

        updated = await cron_store.get_job(job.id)
        # The failure path marks the job for a retry, not "completed".
        assert updated.status.value != "completed"
        assert updated.last_error is not None
        assert "simulated write failure" in updated.last_error
        assert "scheduled_task_failed" in events
    finally:
        await cron_store.close()
        await message_store.close()


async def test_disabled_scheduler_skips_dispatch(tmp_path: Path) -> None:
    """When config.enabled is False, run_pending must no-op even if
    a job is otherwise due — preserves the existing config-gate."""

    cron_store, message_store, session_id = await _open_stores(tmp_path)
    try:
        due_at = datetime.now(UTC) + timedelta(seconds=1)
        await cron_store.create_job(
            session_id=session_id,
            agent_key="Lead",
            name="ReminderDisabled",
            schedule_type=CronScheduleType.ONCE,
            schedule_value=due_at.isoformat(),
            timezone="UTC",
            prompt="prompt",
        )

        scheduler = CronScheduler(
            config=replace(_scheduler_config(), enabled=False),
            cron_store=cron_store,
            message_store=message_store,
        )
        dispatched = await scheduler.run_pending(
            now=due_at + timedelta(seconds=1)
        )
        assert dispatched == 0
        inbox = await message_store.inbox(
            to_session_id=session_id, to_agent_name="Lead"
        )
        assert inbox == []
    finally:
        await cron_store.close()
        await message_store.close()


async def test_cron_message_is_drainable_by_real_lead_base_agent(
    tmp_path: Path,
) -> None:
    """End-to-end pin: cron's mailbox row MUST land in the canonical
    inbox the lead's BaseAgent polls.

    Caught a real bug in review: senders were addressing ``to_agent_name=
    "lead"`` (lowercase) but the lead's BaseAgent filters by its
    ``agent_config.name`` which is ``"Lead"`` (capital L), and the SQL
    filter is case-sensitive — so every cron message used to silently
    strand. This test wires a real ``AgentMessageStore`` and asserts
    inbox visibility under the canonical ``"Lead"`` filter.
    """

    from feather.core.constants import LEAD_AGENT_NAME

    cron_store, message_store, session_id = await _open_stores(tmp_path)
    try:
        due_at = datetime.now(UTC) + timedelta(seconds=1)
        await cron_store.create_job(
            session_id=session_id,
            agent_key="Lead",
            name="EndToEnd",
            schedule_type=CronScheduleType.ONCE,
            schedule_value=due_at.isoformat(),
            timezone="UTC",
            prompt="run the morning standup prompt",
        )
        scheduler = CronScheduler(
            config=_scheduler_config(),
            cron_store=cron_store,
            message_store=message_store,
        )
        await scheduler.run_pending(now=due_at + timedelta(seconds=1))

        # Drain via the EXACT filter ``BaseAgent.has_pending_inbox`` /
        # ``_drain_agent_inbox`` use — to_agent_name = the canonical
        # ``LEAD_AGENT_NAME``. Any mismatch on either side strands.
        canonical = await message_store.inbox(
            to_session_id=session_id, to_agent_name=LEAD_AGENT_NAME
        )
        assert len(canonical) == 1, (
            "cron message must be visible to the lead's canonical inbox "
            "filter; if this asserts 0, the case-mismatch bug is back"
        )

        # Belt-and-braces: the OPPOSITE casing must NOT see the message
        # (i.e. confirm the canonical name is actually load-bearing).
        wrong_case = await message_store.inbox(
            to_session_id=session_id, to_agent_name="lead"
        )
        assert wrong_case == [], (
            "lowercase `lead` filter should return nothing — proves the "
            "case-sensitive SQL behavior the bug exploited"
        )
    finally:
        await cron_store.close()
        await message_store.close()
