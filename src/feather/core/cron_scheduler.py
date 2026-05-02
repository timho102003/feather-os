"""Background scheduler that dispatches due cron jobs into agent sessions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from feather.core.agent_factory import AgentFactory
from feather.models import CronJobRecord, RuntimeEvent, SchedulerConfig
from feather.storage.cron_store import CronJobStore

logger = logging.getLogger(__name__)

SessionEventHandlerResolver = Callable[[str], Callable[[RuntimeEvent], None] | None]


class CronScheduler:
    """Poll persisted cron jobs and dispatch due work through the normal agent loop."""

    def __init__(
        self,
        *,
        config: SchedulerConfig,
        cron_store: CronJobStore,
        agent_factory: AgentFactory,
        event_handler_resolver: SessionEventHandlerResolver | None = None,
    ) -> None:
        self._config = config
        self._cron_store = cron_store
        self._agent_factory = agent_factory
        self._event_handler_resolver = event_handler_resolver
        self._task: asyncio.Task[None] | None = None
        self._tick_lock = asyncio.Lock()
        self._dispatching_job_ids: set[str] = set()

    def set_event_handler_resolver(self, resolver: SessionEventHandlerResolver | None) -> None:
        """Install a resolver that maps session IDs to runtime event handlers."""

        self._event_handler_resolver = resolver

    async def start(self) -> None:
        """Start the background polling loop if enabled."""

        if not self._config.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run_forever(), name="feather-cron-scheduler")

    async def stop(self) -> None:
        """Stop the background polling loop."""

        if self._task is None:
            return
        task = self._task
        self._task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def run_pending(self, *, now: datetime | None = None) -> int:
        """Dispatch all currently due jobs once."""

        if not self._config.enabled:
            return 0

        jobs_to_dispatch: list[CronJobRecord] = []
        async with self._tick_lock:
            due_jobs = await self._cron_store.list_due_jobs(
                now=now,
                limit=self._config.max_due_jobs_per_tick,
            )
            for job in due_jobs:
                if job.id in self._dispatching_job_ids:
                    continue
                self._dispatching_job_ids.add(job.id)
                jobs_to_dispatch.append(job)

        dispatched = 0
        for job in jobs_to_dispatch:
            try:
                await self._dispatch_job(job)
                dispatched += 1
            finally:
                async with self._tick_lock:
                    self._dispatching_job_ids.discard(job.id)
        return dispatched

    async def _run_forever(self) -> None:
        while True:
            try:
                await self.run_pending()
            except Exception:  # noqa: BLE001
                logger.exception("cron scheduler tick failed")
            await asyncio.sleep(self._config.poll_interval_seconds)

    async def _dispatch_job(self, job: CronJobRecord) -> None:
        event_handler = self._resolve_event_handler(job.session_id)
        fired_at = datetime.now(UTC)
        if event_handler is not None:
            event_handler(
                RuntimeEvent(
                    kind="scheduled_task_triggered",
                    text=f"Cron `{job.name}` fired at {fired_at.isoformat()}",
                )
            )

        agent = self._agent_factory.build(job.agent_key)
        try:
            await agent.run(
                job.session_id,
                _render_scheduled_message(job, fired_at=fired_at),
                event_handler,
            )
        except Exception as exc:  # noqa: BLE001
            retry_at = fired_at + timedelta(seconds=self._config.failure_retry_seconds)
            await self._cron_store.mark_job_failed(
                job.id,
                error=str(exc),
                retry_at=retry_at,
            )
            logger.exception("cron job dispatch failed job_id=%s session_id=%s", job.id, job.session_id)
            if event_handler is not None:
                event_handler(
                    RuntimeEvent(
                        kind="scheduled_task_failed",
                        text=(
                            f"Cron `{job.name}` failed and will retry at {retry_at.isoformat()}: {exc}"
                        ),
                    )
                )
            return

        updated = await self._cron_store.mark_job_succeeded(job.id, ran_at=fired_at)
        logger.info(
            "cron job dispatched job_id=%s session_id=%s next_run_at=%s status=%s",
            updated.id,
            updated.session_id,
            updated.next_run_at,
            updated.status.value,
        )
        if event_handler is not None:
            next_run_text = updated.next_run_at or "none"
            event_handler(
                RuntimeEvent(
                    kind="scheduled_task_completed",
                    text=f"Cron `{updated.name}` completed. Next run: {next_run_text}",
                )
            )

    def _resolve_event_handler(self, session_id: str) -> Callable[[RuntimeEvent], None] | None:
        if self._event_handler_resolver is None:
            return None
        return self._event_handler_resolver(session_id)


def _render_scheduled_message(job: CronJobRecord, *, fired_at: datetime) -> str:
    """Render one cron trigger as a structured inbound message."""

    return (
        "<scheduled_task_trigger>\n"
        f"<cron_job_id>{job.id}</cron_job_id>\n"
        f"<cron_job_name>{job.name}</cron_job_name>\n"
        f"<schedule_type>{job.schedule_type.value}</schedule_type>\n"
        f"<schedule_value>{job.schedule_value}</schedule_value>\n"
        f"<timezone>{job.timezone}</timezone>\n"
        f"<fired_at>{fired_at.isoformat()}</fired_at>\n"
        "<instruction>\n"
        f"{job.prompt}\n"
        "</instruction>\n"
        "</scheduled_task_trigger>"
    )
