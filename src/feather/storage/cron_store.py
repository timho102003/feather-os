"""SQLite-backed persistence for scheduled cron jobs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiosqlite
from croniter import croniter

from feather.models import CronJobRecord, CronJobStatus, CronScheduleType
from feather.storage.connection import open_store_connection
from feather.storage.schema import initialize_database_schema

_UNSET = object()


class CronJobStore:
    """Persist cron jobs and their schedule state in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the SQLite connection and ensure the schema exists."""

        self._connection = await open_store_connection(self._db_path)
        await initialize_database_schema(self._connection)
        await self._connection.commit()

    async def close(self) -> None:
        """Close the SQLite connection."""

        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def create_job(
        self,
        *,
        session_id: str,
        agent_key: str,
        name: str,
        schedule_type: CronScheduleType,
        schedule_value: str,
        timezone: str,
        prompt: str,
    ) -> CronJobRecord:
        """Create one scheduled job and compute its next due time."""

        normalized_name = name.strip()
        normalized_prompt = prompt.strip()
        if not normalized_name:
            raise ValueError("Cron job `name` must not be empty.")
        if not normalized_prompt:
            raise ValueError("Cron job `prompt` must not be empty.")

        now = _utc_now()
        next_run_at = self._compute_next_run_iso(
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            timezone=timezone,
            after=now,
        )
        job_id = str(uuid4())
        await self._execute(
            """
            INSERT INTO cron_jobs (
                id,
                session_id,
                agent_key,
                name,
                schedule_type,
                schedule_value,
                timezone,
                prompt,
                status,
                last_run_at,
                next_run_at,
                last_error,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                session_id,
                agent_key,
                normalized_name,
                schedule_type.value,
                schedule_value.strip(),
                timezone.strip(),
                normalized_prompt,
                CronJobStatus.ACTIVE.value,
                None,
                next_run_at,
                None,
                now.isoformat(),
                now.isoformat(),
            ),
        )
        await self._connection.commit()
        return await self.get_job(job_id)

    async def get_job(self, job_id: str) -> CronJobRecord:
        """Fetch one cron job by ID."""

        row = await self._fetchone("SELECT * FROM cron_jobs WHERE id = ?", (job_id,))
        if row is None:
            raise ValueError(f"Unknown cron job: {job_id}")
        return self._row_to_record(row)

    async def list_jobs(
        self,
        *,
        session_id: str | None = None,
        status: CronJobStatus | None = None,
        limit: int | None = None,
    ) -> list[CronJobRecord]:
        """List cron jobs with optional session and status filters."""

        clauses: list[str] = []
        params: list[object] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT * FROM cron_jobs
            {where}
            ORDER BY
                CASE WHEN next_run_at IS NULL THEN 1 ELSE 0 END,
                next_run_at ASC,
                created_at ASC
        """
        if limit is not None:
            query = f"{query} LIMIT ?"
            params.append(limit)
        cursor = await self._connection.execute(query, tuple(params))
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def find_jobs_by_name(self, session_id: str, name: str) -> list[CronJobRecord]:
        """Find jobs in one session by exact name, case-insensitively."""

        cursor = await self._connection.execute(
            """
            SELECT * FROM cron_jobs
            WHERE session_id = ? AND LOWER(name) = LOWER(?)
            ORDER BY created_at ASC
            """,
            (session_id, name.strip()),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def delete_job(self, job_id: str) -> bool:
        """Delete one cron job. Returns whether a row was removed."""

        cursor = await self._connection.execute("DELETE FROM cron_jobs WHERE id = ?", (job_id,))
        await self._connection.commit()
        return (cursor.rowcount or 0) > 0

    async def update_job(
        self,
        job_id: str,
        *,
        name: str | None | object = _UNSET,
        schedule_type: CronScheduleType | None | object = _UNSET,
        schedule_value: str | None | object = _UNSET,
        timezone: str | None | object = _UNSET,
        prompt: str | None | object = _UNSET,
        status: CronJobStatus | None | object = _UNSET,
    ) -> CronJobRecord:
        """Update mutable cron-job fields and recompute next due time when needed."""

        job = await self.get_job(job_id)
        new_name = job.name if name is _UNSET else (name.strip() if name is not None else "")
        new_prompt = job.prompt if prompt is _UNSET else (prompt.strip() if prompt is not None else "")
        new_schedule_type = job.schedule_type if schedule_type is _UNSET else schedule_type
        new_schedule_value = job.schedule_value if schedule_value is _UNSET else schedule_value.strip()
        new_timezone = job.timezone if timezone is _UNSET else timezone.strip()
        new_status = job.status if status is _UNSET else status

        if not new_name:
            raise ValueError("Cron job `name` must not be empty.")
        if not new_prompt:
            raise ValueError("Cron job `prompt` must not be empty.")
        if new_schedule_type is None:
            raise ValueError("Cron job `schedule_type` must not be null.")
        if not new_schedule_value:
            raise ValueError("Cron job `schedule_value` must not be empty.")
        if not new_timezone:
            raise ValueError("Cron job `timezone` must not be empty.")
        if new_status is None:
            raise ValueError("Cron job `status` must not be null.")

        next_run_at = job.next_run_at
        if new_status == CronJobStatus.COMPLETED:
            next_run_at = None
        elif self._schedule_changed(
            job=job,
            schedule_type=new_schedule_type,
            schedule_value=new_schedule_value,
            timezone=new_timezone,
        ) or (job.status != CronJobStatus.ACTIVE and new_status == CronJobStatus.ACTIVE):
            next_run_at = self._compute_next_run_iso(
                schedule_type=new_schedule_type,
                schedule_value=new_schedule_value,
                timezone=new_timezone,
                after=_utc_now(),
            )

        await self._update_job_fields(
            job_id,
            name=new_name,
            schedule_type=new_schedule_type.value,
            schedule_value=new_schedule_value,
            timezone=new_timezone,
            prompt=new_prompt,
            status=new_status.value,
            next_run_at=next_run_at,
        )
        return await self.get_job(job_id)

    async def list_due_jobs(self, *, now: datetime | None = None, limit: int = 10) -> list[CronJobRecord]:
        """Return active jobs due to run at or before the given time."""

        active_now = (now or _utc_now()).isoformat()
        cursor = await self._connection.execute(
            """
            SELECT * FROM cron_jobs
            WHERE status = ? AND next_run_at IS NOT NULL AND next_run_at <= ?
            ORDER BY next_run_at ASC
            LIMIT ?
            """,
            (CronJobStatus.ACTIVE.value, active_now, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def mark_job_succeeded(self, job_id: str, *, ran_at: datetime | None = None) -> CronJobRecord:
        """Advance a job after a successful dispatch."""

        job = await self.get_job(job_id)
        completed_at = ran_at or _utc_now()
        next_run_at: str | None = None
        status = job.status

        if job.schedule_type == CronScheduleType.CRON:
            next_run_at = self._compute_next_run_iso(
                schedule_type=job.schedule_type,
                schedule_value=job.schedule_value,
                timezone=job.timezone,
                after=completed_at,
            )
        else:
            status = CronJobStatus.COMPLETED

        await self._update_job_fields(
            job_id,
            last_run_at=completed_at.isoformat(),
            next_run_at=next_run_at,
            status=status.value,
            last_error=None,
        )
        return await self.get_job(job_id)

    async def mark_job_failed(
        self,
        job_id: str,
        *,
        error: str,
        retry_at: datetime,
    ) -> CronJobRecord:
        """Record a dispatch failure and move the next run to the retry timestamp."""

        job = await self.get_job(job_id)
        if job.status == CronJobStatus.COMPLETED:
            return job
        await self._update_job_fields(
            job_id,
            last_error=error.strip()[:4000] or "Unknown cron dispatch error.",
            next_run_at=retry_at.isoformat(),
            status=CronJobStatus.ACTIVE.value,
        )
        return await self.get_job(job_id)

    async def _update_job_fields(self, job_id: str, **values: object) -> None:
        assignments = ", ".join(f"{key} = ?" for key in values)
        params = list(values.values()) + [_utc_now().isoformat(), job_id]
        await self._execute(
            f"UPDATE cron_jobs SET {assignments}, updated_at = ? WHERE id = ?",
            tuple(params),
        )
        await self._connection.commit()

    async def _execute(self, query: str, params: tuple[object, ...]) -> None:
        if self._connection is None:
            raise RuntimeError("CronJobStore.initialize() must be called before use.")
        await self._connection.execute(query, params)

    async def _fetchone(self, query: str, params: tuple[object, ...]) -> aiosqlite.Row | None:
        if self._connection is None:
            raise RuntimeError("CronJobStore.initialize() must be called before use.")
        cursor = await self._connection.execute(query, params)
        return await cursor.fetchone()

    def _row_to_record(self, row: aiosqlite.Row) -> CronJobRecord:
        return CronJobRecord(
            id=row["id"],
            session_id=row["session_id"],
            agent_key=row["agent_key"],
            name=row["name"],
            schedule_type=CronScheduleType(row["schedule_type"]),
            schedule_value=row["schedule_value"],
            timezone=row["timezone"],
            prompt=row["prompt"],
            status=CronJobStatus(row["status"]),
            last_run_at=row["last_run_at"],
            next_run_at=row["next_run_at"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _schedule_changed(
        self,
        *,
        job: CronJobRecord,
        schedule_type: CronScheduleType,
        schedule_value: str,
        timezone: str,
    ) -> bool:
        return (
            job.schedule_type != schedule_type
            or job.schedule_value != schedule_value
            or job.timezone != timezone
        )

    def _compute_next_run_iso(
        self,
        *,
        schedule_type: CronScheduleType,
        schedule_value: str,
        timezone: str,
        after: datetime,
    ) -> str:
        next_run = self._compute_next_run_datetime(
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            timezone=timezone,
            after=after,
        )
        return next_run.isoformat()

    def _compute_next_run_datetime(
        self,
        *,
        schedule_type: CronScheduleType,
        schedule_value: str,
        timezone: str,
        after: datetime,
    ) -> datetime:
        zone = self._resolve_zone(timezone)
        raw_value = schedule_value.strip()
        if not raw_value:
            raise ValueError("Cron job `schedule_value` must not be empty.")

        if schedule_type == CronScheduleType.CRON:
            base = after.astimezone(zone)
            try:
                next_local = croniter(raw_value, base).get_next(datetime)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid cron expression: {raw_value}") from exc
            if next_local.tzinfo is None:
                next_local = next_local.replace(tzinfo=zone)
            return next_local.astimezone(UTC)

        scheduled_at = self._parse_once_datetime(raw_value, zone)
        if scheduled_at <= after.astimezone(UTC):
            raise ValueError("One-time schedules must be in the future.")
        return scheduled_at

    def _parse_once_datetime(self, value: str, zone: ZoneInfo) -> datetime:
        try:
            scheduled_at = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                "One-time schedules must be ISO 8601 datetimes, for example `2026-04-16T21:30:00-04:00`."
            ) from exc
        if scheduled_at.tzinfo is None:
            scheduled_at = scheduled_at.replace(tzinfo=zone)
        return scheduled_at.astimezone(UTC)

    def _resolve_zone(self, timezone: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone.strip())
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown timezone: {timezone}") from exc


def _utc_now() -> datetime:
    """Return the current UTC timestamp as an aware datetime."""

    return datetime.now(UTC)
