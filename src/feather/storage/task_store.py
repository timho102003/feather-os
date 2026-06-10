"""SQLite-backed persistence for plans, tasks, task runs, and outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite

from feather.models import (
    PlanRecord,
    PlanStatus,
    TaskEventRecord,
    TaskOutputKind,
    TaskOutputRecord,
    TaskRecord,
    TaskRunRecord,
    TaskRunStatus,
    TaskStatus,
)
from feather.storage.connection import open_store_connection
from feather.storage.schema import initialize_database_schema

_UNSET = object()
_TERMINAL_STATUSES = {
    TaskStatus.COMPLETED_WITH_REPORT,
    TaskStatus.COMPLETED_WITH_ARTIFACTS,
    TaskStatus.COMPLETED_WITHOUT_ARTIFACTS,
    TaskStatus.FAILED,
    TaskStatus.STOPPED,
}


class TaskStore:
    """Persist durable task management state in SQLite."""

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

    async def create_plan(
        self,
        *,
        filepath: str,
        title: str,
        summary: str,
        lead_session_id: str,
        status: PlanStatus = PlanStatus.ACTIVE,
    ) -> PlanRecord:
        """Create one plan row."""

        filepath = filepath.strip()
        title = title.strip()
        summary = summary.strip()
        if not filepath:
            raise ValueError("Plan `filepath` must not be empty.")
        if not title:
            raise ValueError("Plan `title` must not be empty.")
        if not lead_session_id:
            raise ValueError("Plan `lead_session_id` must not be empty.")

        now = _utc_now()
        plan_id = str(uuid4())
        await self._execute(
            """
            INSERT INTO plans (
                id, filepath, title, summary, status, lead_session_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan_id,
                filepath,
                title,
                summary,
                status.value,
                lead_session_id,
                now,
                now,
            ),
        )
        await self._connection.commit()
        return await self.get_plan(plan_id)

    async def get_plan(self, plan_id: str) -> PlanRecord:
        """Fetch one plan by ID."""

        row = await self._fetchone("SELECT * FROM plans WHERE id = ?", (plan_id,))
        if row is None:
            raise ValueError(f"Unknown plan: {plan_id}")
        return _row_to_plan(row)

    async def find_plan_by_filepath(
        self, *, lead_session_id: str, filepath: str
    ) -> PlanRecord | None:
        """Find the newest plan matching a filepath within a lead session."""

        cursor = await self._require_connection().execute(
            """
            SELECT * FROM plans
             WHERE lead_session_id = ? AND filepath = ?
             ORDER BY created_at DESC, id DESC
             LIMIT 1
            """,
            (lead_session_id, filepath.strip()),
        )
        row = await cursor.fetchone()
        return None if row is None else _row_to_plan(row)

    async def create_task(
        self,
        *,
        lead_session_id: str,
        title: str,
        description: str = "",
        success_criteria: str = "",
        required_outputs: list[str] | None = None,
        plan_id: str | None = None,
        parent_task_id: str | None = None,
        responsible_agent_name: str | None = None,
        responsible_session_id: str | None = None,
        status: TaskStatus = TaskStatus.QUEUED,
    ) -> TaskRecord:
        """Create one task row."""

        title = title.strip()
        if not title:
            raise ValueError("Task `title` must not be empty.")
        if not lead_session_id:
            raise ValueError("Task `lead_session_id` must not be empty.")
        if plan_id is not None:
            await self.get_plan(plan_id)
        if parent_task_id is not None:
            await self.get_task(parent_task_id)

        now = _utc_now()
        task_id = str(uuid4())
        await self._execute(
            """
            INSERT INTO tasks (
                id, plan_id, parent_task_id, title, description, success_criteria,
                required_outputs, status, responsible_agent_name,
                responsible_session_id, lead_session_id, blocked_question,
                blocked_correlation_id, error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
            """,
            (
                task_id,
                plan_id,
                parent_task_id,
                title,
                description.strip(),
                success_criteria.strip(),
                json.dumps(required_outputs or []),
                status.value,
                _clean_optional(responsible_agent_name),
                _clean_optional(responsible_session_id),
                lead_session_id,
                now,
                now,
            ),
        )
        await self._connection.commit()
        await self.add_event(
            task_id,
            event_type="created",
            message=f"Task created: {title}",
            agent_name=responsible_agent_name,
            session_id=responsible_session_id,
        )
        return await self.get_task(task_id)

    async def get_task(self, task_id: str) -> TaskRecord:
        """Fetch one task by ID."""

        row = await self._fetchone("SELECT * FROM tasks WHERE id = ?", (task_id,))
        if row is None:
            raise ValueError(f"Unknown task: {task_id}")
        return _row_to_task(row)

    async def find_task_by_session(self, session_id: str) -> TaskRecord | None:
        """Find the newest non-terminal task assigned to a session."""

        cursor = await self._require_connection().execute(
            """
            SELECT * FROM tasks
             WHERE responsible_session_id = ?
             ORDER BY
               CASE
                 WHEN status IN (?, ?, ?, ?, ?) THEN 1
                 ELSE 0
               END,
               updated_at DESC,
               created_at DESC
             LIMIT 1
            """,
            (
                session_id,
                *(status.value for status in _TERMINAL_STATUSES),
            ),
        )
        row = await cursor.fetchone()
        return None if row is None else _row_to_task(row)

    async def list_tasks(
        self,
        *,
        lead_session_id: str | None = None,
        plan_id: str | None = None,
        status: TaskStatus | None = None,
        responsible_session_id: str | None = None,
        limit: int = 20,
    ) -> list[TaskRecord]:
        """List tasks with optional filters."""

        clauses: list[str] = []
        params: list[object] = []
        if lead_session_id is not None:
            clauses.append("lead_session_id = ?")
            params.append(lead_session_id)
        if plan_id is not None:
            clauses.append("plan_id = ?")
            params.append(plan_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if responsible_session_id is not None:
            clauses.append("responsible_session_id = ?")
            params.append(responsible_session_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(200, int(limit))))
        cursor = await self._require_connection().execute(
            f"""
            SELECT * FROM tasks
            {where}
            ORDER BY
              CASE status
                WHEN 'running' THEN 0
                WHEN 'blocked_needs_input' THEN 1
                WHEN 'queued' THEN 2
                WHEN 'failed' THEN 3
                ELSE 4
              END,
              updated_at DESC,
              created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows = await cursor.fetchall()
        return [_row_to_task(row) for row in rows]

    async def update_task(
        self,
        task_id: str,
        *,
        status: TaskStatus | None | object = _UNSET,
        responsible_agent_name: str | None | object = _UNSET,
        responsible_session_id: str | None | object = _UNSET,
        blocked_question: str | None | object = _UNSET,
        blocked_correlation_id: str | None | object = _UNSET,
        error: str | None | object = _UNSET,
    ) -> TaskRecord:
        """Update mutable task state fields."""

        await self.get_task(task_id)
        values: dict[str, object | None] = {}
        if status is not _UNSET:
            if status is None:
                raise ValueError("Task `status` must not be null.")
            values["status"] = status.value
        if responsible_agent_name is not _UNSET:
            values["responsible_agent_name"] = _clean_optional(responsible_agent_name)
        if responsible_session_id is not _UNSET:
            values["responsible_session_id"] = _clean_optional(responsible_session_id)
        if blocked_question is not _UNSET:
            values["blocked_question"] = _clean_optional(blocked_question)
        if blocked_correlation_id is not _UNSET:
            values["blocked_correlation_id"] = _clean_optional(blocked_correlation_id)
        if error is not _UNSET:
            values["error"] = _clean_optional(error)
        if not values:
            return await self.get_task(task_id)

        await self._update_fields("tasks", task_id, values)
        updated = await self.get_task(task_id)
        if "status" in values:
            await self.add_event(
                task_id,
                event_type="status",
                message=f"Task status -> {updated.status.value}",
                agent_name=updated.responsible_agent_name,
                session_id=updated.responsible_session_id,
            )
        return updated

    async def create_run(
        self,
        *,
        task_id: str,
        session_id: str,
        agent_name: str,
        pid: int | None,
    ) -> TaskRunRecord:
        """Record one started subprocess attempt for a task."""

        await self.get_task(task_id)
        run_id = str(uuid4())
        now = _utc_now()
        await self._execute(
            """
            INSERT INTO task_runs (
                id, task_id, session_id, agent_name, pid, status, exit_code,
                envelope_status, error, started_at, ended_at
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, NULL)
            """,
            (
                run_id,
                task_id,
                session_id,
                agent_name,
                pid,
                TaskRunStatus.RUNNING.value,
                now,
            ),
        )
        await self._connection.commit()
        await self.add_event(
            task_id,
            event_type="run_started",
            message=f"Run started: {agent_name} {session_id}",
            agent_name=agent_name,
            session_id=session_id,
        )
        return await self.get_run(run_id)

    async def get_run(self, run_id: str) -> TaskRunRecord:
        """Fetch one task run by ID."""

        row = await self._fetchone("SELECT * FROM task_runs WHERE id = ?", (run_id,))
        if row is None:
            raise ValueError(f"Unknown task run: {run_id}")
        return _row_to_run(row)

    async def finish_run(
        self,
        run_id: str,
        *,
        status: TaskRunStatus,
        exit_code: int | None,
        envelope_status: str | None,
        error: str | None = None,
    ) -> TaskRunRecord:
        """Mark one task run ended."""

        now = _utc_now()
        await self._update_fields(
            "task_runs",
            run_id,
            {
                "status": status.value,
                "exit_code": exit_code,
                "envelope_status": _clean_optional(envelope_status),
                "error": _clean_optional(error),
                "ended_at": now,
            },
            touch=False,
        )
        run = await self.get_run(run_id)
        await self.add_event(
            run.task_id,
            event_type="run_finished",
            message=f"Run ended: {status.value} envelope={envelope_status or '-'}",
            agent_name=run.agent_name,
            session_id=run.session_id,
        )
        return run

    async def update_run_pid(self, run_id: str, pid: int | None) -> TaskRunRecord:
        """Attach a subprocess PID to an already-created run."""

        await self._update_fields("task_runs", run_id, {"pid": pid}, touch=False)
        return await self.get_run(run_id)

    async def latest_run_for_task(self, task_id: str) -> TaskRunRecord | None:
        """Return the newest run for a task, if any."""

        cursor = await self._require_connection().execute(
            """
            SELECT * FROM task_runs
             WHERE task_id = ?
             ORDER BY started_at DESC, id DESC
             LIMIT 1
            """,
            (task_id,),
        )
        row = await cursor.fetchone()
        return None if row is None else _row_to_run(row)

    async def add_output(
        self,
        *,
        task_id: str,
        kind: TaskOutputKind,
        path: str | None,
        content: str | None,
        summary: str,
        created_by_session_id: str,
        validated: bool = False,
        is_final: bool = False,
    ) -> TaskOutputRecord:
        """Add one output row for a task."""

        await self.get_task(task_id)
        if path is None and content is None:
            raise ValueError("Task output needs either `path` or `content`.")
        summary = summary.strip()
        if not summary:
            raise ValueError("Task output `summary` must not be empty.")
        output_id = str(uuid4())
        now = _utc_now()
        await self._execute(
            """
            INSERT INTO task_outputs (
                id, task_id, kind, path, content, summary, created_by_session_id,
                validated, is_final, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                output_id,
                task_id,
                kind.value,
                _clean_optional(path),
                _clean_optional(content),
                summary,
                created_by_session_id,
                int(validated),
                int(is_final),
                now,
            ),
        )
        await self._connection.commit()
        await self.add_event(
            task_id,
            event_type="output",
            message=f"Output added: {kind.value} {path or summary}",
            session_id=created_by_session_id,
        )
        return await self.get_output(output_id)

    async def get_output(self, output_id: str) -> TaskOutputRecord:
        """Fetch one task output by ID."""

        row = await self._fetchone("SELECT * FROM task_outputs WHERE id = ?", (output_id,))
        if row is None:
            raise ValueError(f"Unknown task output: {output_id}")
        return _row_to_output(row)

    async def list_outputs(self, task_id: str) -> list[TaskOutputRecord]:
        """List outputs for one task."""

        cursor = await self._require_connection().execute(
            """
            SELECT * FROM task_outputs
             WHERE task_id = ?
             ORDER BY is_final DESC, created_at ASC, id ASC
            """,
            (task_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_output(row) for row in rows]

    async def has_final_output(self, task_id: str) -> bool:
        """Return whether a task has at least one final output."""

        cursor = await self._require_connection().execute(
            "SELECT 1 FROM task_outputs WHERE task_id = ? AND is_final = 1 LIMIT 1",
            (task_id,),
        )
        return await cursor.fetchone() is not None

    async def add_event(
        self,
        task_id: str,
        *,
        event_type: str,
        message: str,
        agent_name: str | None = None,
        session_id: str | None = None,
    ) -> TaskEventRecord:
        """Append one event to a task timeline."""

        event_id = str(uuid4())
        now = _utc_now()
        await self._execute(
            """
            INSERT INTO task_events (
                id, task_id, event_type, message, agent_name, session_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                task_id,
                event_type.strip() or "event",
                message.strip() or "(empty)",
                _clean_optional(agent_name),
                _clean_optional(session_id),
                now,
            ),
        )
        await self._connection.commit()
        return await self.get_event(event_id)

    async def get_event(self, event_id: str) -> TaskEventRecord:
        """Fetch one task event by ID."""

        row = await self._fetchone("SELECT * FROM task_events WHERE id = ?", (event_id,))
        if row is None:
            raise ValueError(f"Unknown task event: {event_id}")
        return _row_to_event(row)

    async def list_events(self, task_id: str, *, limit: int = 20) -> list[TaskEventRecord]:
        """List newest task events in chronological order."""

        cursor = await self._require_connection().execute(
            """
            SELECT * FROM (
                SELECT * FROM task_events
                 WHERE task_id = ?
                 ORDER BY created_at DESC, id DESC
                 LIMIT ?
            )
            ORDER BY created_at ASC, id ASC
            """,
            (task_id, max(1, min(200, int(limit)))),
        )
        rows = await cursor.fetchall()
        return [_row_to_event(row) for row in rows]

    async def _update_fields(
        self,
        table: str,
        row_id: str,
        values: dict[str, object | None],
        *,
        touch: bool = True,
    ) -> None:
        assignments = ", ".join(f"{key} = ?" for key in values)
        params = list(values.values())
        if touch:
            assignments = f"{assignments}, updated_at = ?"
            params.append(_utc_now())
        params.append(row_id)
        await self._execute(f"UPDATE {table} SET {assignments} WHERE id = ?", tuple(params))
        await self._connection.commit()

    async def _execute(self, query: str, params: tuple[object | None, ...]) -> None:
        connection = self._require_connection()
        await connection.execute(query, params)

    async def _fetchone(
        self, query: str, params: tuple[object | None, ...]
    ) -> aiosqlite.Row | None:
        cursor = await self._require_connection().execute(query, params)
        return await cursor.fetchone()

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("TaskStore.initialize() must be called before use.")
        return self._connection


def _row_to_plan(row: aiosqlite.Row) -> PlanRecord:
    return PlanRecord(
        id=row["id"],
        filepath=row["filepath"],
        title=row["title"],
        summary=row["summary"],
        status=PlanStatus(row["status"]),
        lead_session_id=row["lead_session_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_task(row: aiosqlite.Row) -> TaskRecord:
    return TaskRecord(
        id=row["id"],
        plan_id=row["plan_id"],
        parent_task_id=row["parent_task_id"],
        title=row["title"],
        description=row["description"],
        success_criteria=row["success_criteria"],
        required_outputs=json.loads(row["required_outputs"]),
        status=TaskStatus(row["status"]),
        responsible_agent_name=row["responsible_agent_name"],
        responsible_session_id=row["responsible_session_id"],
        lead_session_id=row["lead_session_id"],
        blocked_question=row["blocked_question"],
        blocked_correlation_id=row["blocked_correlation_id"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_run(row: aiosqlite.Row) -> TaskRunRecord:
    return TaskRunRecord(
        id=row["id"],
        task_id=row["task_id"],
        session_id=row["session_id"],
        agent_name=row["agent_name"],
        pid=row["pid"],
        status=TaskRunStatus(row["status"]),
        exit_code=row["exit_code"],
        envelope_status=row["envelope_status"],
        error=row["error"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
    )


def _row_to_output(row: aiosqlite.Row) -> TaskOutputRecord:
    return TaskOutputRecord(
        id=row["id"],
        task_id=row["task_id"],
        kind=TaskOutputKind(row["kind"]),
        path=row["path"],
        content=row["content"],
        summary=row["summary"],
        created_by_session_id=row["created_by_session_id"],
        validated=bool(row["validated"]),
        is_final=bool(row["is_final"]),
        created_at=row["created_at"],
    )


def _row_to_event(row: aiosqlite.Row) -> TaskEventRecord:
    return TaskEventRecord(
        id=row["id"],
        task_id=row["task_id"],
        event_type=row["event_type"],
        message=row["message"],
        agent_name=row["agent_name"],
        session_id=row["session_id"],
        created_at=row["created_at"],
    )


def _clean_optional(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    stripped = value.strip()
    return stripped or None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
