"""Lead-only cron scheduling tools."""

from __future__ import annotations

from typing import Any

from feather.models import CronJobStatus, CronScheduleType, ToolExecutionContext, ToolExecutionResult
from feather.storage.cron_store import CronJobStore
from feather.tools.base import BaseTool


class CreateCronTool(BaseTool):
    """Create a recurring or one-time scheduled job for the active session."""

    name = "create_cron"
    description = "Create a scheduled job for the current session using either a cron expression or one-time ISO datetime."
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short human-readable job name.",
            },
            "schedule_type": {
                "type": "string",
                "enum": [CronScheduleType.CRON.value, CronScheduleType.ONCE.value],
                "description": "Use `cron` for recurring jobs or `once` for one-time ISO datetimes.",
            },
            "schedule_value": {
                "type": "string",
                "description": "Cron expression for recurring jobs, or ISO 8601 datetime for one-time jobs.",
            },
            "timezone": {
                "type": "string",
                "description": "IANA timezone such as `UTC` or `America/New_York`.",
            },
            "prompt": {
                "type": "string",
                "description": "Instruction the scheduler should inject back into the lead session when the job fires.",
            },
        },
        "required": ["name", "schedule_type", "schedule_value", "timezone", "prompt"],
        "additionalProperties": False,
    }

    def __init__(self, cron_store: CronJobStore) -> None:
        self._cron_store = cron_store

    def get_prompt(self) -> str:
        return (
            "- `create_cron`: schedule a future task for the active session. "
            "Use `schedule_type=cron` with a cron expression for recurring work, or `schedule_type=once` "
            "with an ISO 8601 datetime for one-time work. Always include the timezone and the exact prompt "
            "the lead agent should receive when the job fires. After creating the job, confirm the schedule "
            "instead of executing the scheduled instruction immediately."
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        job = await self._cron_store.create_job(
            session_id=context.session_id,
            agent_key="lead",
            name=arguments["name"],
            schedule_type=CronScheduleType(arguments["schedule_type"]),
            schedule_value=arguments["schedule_value"],
            timezone=arguments["timezone"],
            prompt=arguments["prompt"],
        )
        return ToolExecutionResult(output=_render_job_summary("Created cron job", job))


class UpdateCronTool(BaseTool):
    """Update one existing scheduled job."""

    name = "update_cron"
    description = "Update an existing cron job by ID or exact job name within the current session."
    parameters_schema = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": ["string", "null"],
                "description": "Exact cron job ID to update.",
            },
            "job_name": {
                "type": ["string", "null"],
                "description": "Exact cron job name to update when ID is not known.",
            },
            "new_name": {
                "type": ["string", "null"],
                "description": "Optional replacement job name.",
            },
            "schedule_type": {
                "type": ["string", "null"],
                "enum": [CronScheduleType.CRON.value, CronScheduleType.ONCE.value, None],
                "description": "Optional replacement schedule type.",
            },
            "schedule_value": {
                "type": ["string", "null"],
                "description": "Optional replacement cron expression or one-time ISO 8601 datetime.",
            },
            "timezone": {
                "type": ["string", "null"],
                "description": "Optional replacement IANA timezone.",
            },
            "prompt": {
                "type": ["string", "null"],
                "description": "Optional replacement injected prompt.",
            },
            "status": {
                "type": ["string", "null"],
                "enum": [
                    CronJobStatus.ACTIVE.value,
                    CronJobStatus.PAUSED.value,
                    CronJobStatus.COMPLETED.value,
                    None,
                ],
                "description": "Optional replacement job status.",
            },
        },
        "required": [
            "job_id",
            "job_name",
            "new_name",
            "schedule_type",
            "schedule_value",
            "timezone",
            "prompt",
            "status",
        ],
        "additionalProperties": False,
    }

    def __init__(self, cron_store: CronJobStore) -> None:
        self._cron_store = cron_store

    def get_prompt(self) -> str:
        return (
            "- `update_cron`: update a scheduled job by `job_id` when possible, or by exact `job_name` when unique. "
            "Use this to rename a job, change its schedule, change the injected prompt, or pause/reactivate it."
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        job = await _resolve_job(
            self._cron_store,
            session_id=context.session_id,
            job_id=arguments["job_id"],
            job_name=arguments["job_name"],
        )
        update_kwargs: dict[str, Any] = {}
        if arguments["new_name"] is not None:
            update_kwargs["name"] = arguments["new_name"]
        if arguments["schedule_type"] is not None:
            update_kwargs["schedule_type"] = CronScheduleType(arguments["schedule_type"])
        if arguments["schedule_value"] is not None:
            update_kwargs["schedule_value"] = arguments["schedule_value"]
        if arguments["timezone"] is not None:
            update_kwargs["timezone"] = arguments["timezone"]
        if arguments["prompt"] is not None:
            update_kwargs["prompt"] = arguments["prompt"]
        if arguments["status"] is not None:
            update_kwargs["status"] = CronJobStatus(arguments["status"])
        updated = await self._cron_store.update_job(job.id, **update_kwargs)
        return ToolExecutionResult(output=_render_job_summary("Updated cron job", updated))


class DeleteCronTool(BaseTool):
    """Delete one scheduled job."""

    name = "delete_cron"
    description = "Delete an existing cron job by ID or exact job name within the current session."
    parameters_schema = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": ["string", "null"],
                "description": "Exact cron job ID to delete.",
            },
            "job_name": {
                "type": ["string", "null"],
                "description": "Exact cron job name to delete when ID is not known.",
            },
        },
        "required": ["job_id", "job_name"],
        "additionalProperties": False,
    }

    def __init__(self, cron_store: CronJobStore) -> None:
        self._cron_store = cron_store

    def get_prompt(self) -> str:
        return (
            "- `delete_cron`: remove a scheduled job from the active session. "
            "Prefer `job_id`; use exact `job_name` only when it resolves uniquely."
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        job = await _resolve_job(
            self._cron_store,
            session_id=context.session_id,
            job_id=arguments["job_id"],
            job_name=arguments["job_name"],
        )
        deleted = await self._cron_store.delete_job(job.id)
        if not deleted:
            raise ValueError(f"Cron job no longer exists: {job.id}")
        return ToolExecutionResult(
            output=(
                "Deleted cron job.\n"
                f"id: {job.id}\n"
                f"name: {job.name}\n"
                f"schedule: {job.schedule_type.value} {job.schedule_value}\n"
                f"timezone: {job.timezone}"
            )
        )


class ListCronsTool(BaseTool):
    """List scheduled jobs for the current session."""

    name = "list_crons"
    description = "List cron jobs for the current session, optionally filtered by status."
    parameters_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": ["string", "null"],
                "enum": [
                    CronJobStatus.ACTIVE.value,
                    CronJobStatus.PAUSED.value,
                    CronJobStatus.COMPLETED.value,
                    None,
                ],
                "description": "Optional status filter.",
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "Maximum jobs to return. Defaults to 20.",
                "minimum": 1,
                "maximum": 200,
            },
        },
        "required": ["status", "limit"],
        "additionalProperties": False,
    }

    def __init__(self, cron_store: CronJobStore) -> None:
        self._cron_store = cron_store

    def get_prompt(self) -> str:
        return (
            "- `list_crons`: inspect scheduled jobs for the active session. "
            "Use this before updating or deleting when you are not sure which job ID to target."
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        jobs = await self._cron_store.list_jobs(
            session_id=context.session_id,
            status=CronJobStatus(arguments["status"]) if arguments["status"] is not None else None,
            limit=int(arguments["limit"] or 20),
        )
        if not jobs:
            return ToolExecutionResult(output="No cron jobs found for the current session.")

        lines = ["Cron jobs:"]
        for job in jobs:
            lines.extend(
                [
                    f"- id: {job.id}",
                    f"  name: {job.name}",
                    f"  status: {job.status.value}",
                    f"  schedule: {job.schedule_type.value} {job.schedule_value}",
                    f"  timezone: {job.timezone}",
                    f"  next_run_at: {job.next_run_at or '-'}",
                    f"  last_error: {job.last_error or '-'}",
                ]
            )
        return ToolExecutionResult(output="\n".join(lines))


async def _resolve_job(
    cron_store: CronJobStore,
    *,
    session_id: str,
    job_id: str | None,
    job_name: str | None,
) -> Any:
    if job_id is not None:
        job = await cron_store.get_job(job_id)
        if job.session_id != session_id:
            raise ValueError("Cron job does not belong to the current session.")
        return job

    if job_name is None or not job_name.strip():
        raise ValueError("Provide either `job_id` or an exact `job_name`.")

    matches = await cron_store.find_jobs_by_name(session_id, job_name)
    if not matches:
        raise ValueError(f"No cron job found with name `{job_name}` in the current session.")
    if len(matches) > 1:
        ids = ", ".join(match.id for match in matches)
        raise ValueError(
            f"Multiple cron jobs match `{job_name}` in the current session. Use `job_id` instead. Matches: {ids}"
        )
    return matches[0]


def _render_job_summary(prefix: str, job: Any) -> str:
    return (
        f"{prefix}.\n"
        f"id: {job.id}\n"
        f"name: {job.name}\n"
        f"status: {job.status.value}\n"
        f"schedule: {job.schedule_type.value} {job.schedule_value}\n"
        f"timezone: {job.timezone}\n"
        f"next_run_at: {job.next_run_at or '-'}\n"
        f"prompt: {job.prompt}"
    )
