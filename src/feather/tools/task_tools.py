"""Durable task management tools for agents."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from feather.config import load_agent_config
from feather.core.agent_catalog import AgentCatalog
from feather.core.subagent_registry import LiveSubagent, SubagentRegistry
from feather.models import (
    AgentMessage,
    TaskOutputKind,
    TaskRecord,
    TaskRunStatus,
    TaskStatus,
    ToolExecutionContext,
    ToolExecutionResult,
)
from feather.storage.agent_message_store import AgentMessageStore
from feather.storage.task_store import TaskStore
from feather.tools.base import BaseTool
from feather.tools.spawn_agent_tool import launch_subagent_process

logger = logging.getLogger(__name__)

_COMPLETED_STATUSES = {
    TaskStatus.COMPLETED_WITH_REPORT,
    TaskStatus.COMPLETED_WITH_ARTIFACTS,
    TaskStatus.COMPLETED_WITHOUT_ARTIFACTS,
}
_REQUEST_INPUT_WAIT_TIMEOUT_SECONDS = 30 * 60
_REQUEST_INPUT_POLL_SECONDS = 0.5


class TaskCreateTool(BaseTool):
    """Create a durable task, optionally attached to a plan."""

    name = "task_create"
    description = "Create a durable task that can be monitored, resumed, and completed with explicit outputs."
    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short task title."},
            "description": {"type": "string", "description": "Task details."},
            "success_criteria": {
                "type": "string",
                "description": "Concrete completion criteria.",
            },
            "required_outputs": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Required output names or paths, if any.",
            },
            "plan_id": {"type": ["string", "null"], "description": "Existing plan id."},
            "plan_filepath": {
                "type": ["string", "null"],
                "description": "Plan filepath; creates or reuses a plan when plan_id is null.",
            },
            "plan_title": {
                "type": ["string", "null"],
                "description": "Title to use when creating a plan from plan_filepath.",
            },
            "responsible_agent_name": {
                "type": ["string", "null"],
                "description": "Agent assigned to the task, if known.",
            },
            "responsible_session_id": {
                "type": ["string", "null"],
                "description": "Session assigned to the task, if known.",
            },
        },
        "required": [
            "title",
            "description",
            "success_criteria",
            "required_outputs",
            "plan_id",
            "plan_filepath",
            "plan_title",
            "responsible_agent_name",
            "responsible_session_id",
        ],
        "additionalProperties": False,
    }

    def __init__(self, task_store: TaskStore) -> None:
        self._task_store = task_store

    def get_prompt(self) -> str:
        return (
            "- `task_create`: create a durable task with explicit success criteria "
            "and required outputs. Use this before dispatching or tracking work "
            "that may span sub-agents, pauses, or resumes."
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        plan_id = _optional_str(arguments.get("plan_id"))
        plan_filepath = _optional_str(arguments.get("plan_filepath"))
        if plan_id is None and plan_filepath is not None:
            plan = await self._task_store.find_plan_by_filepath(
                lead_session_id=context.session_id,
                filepath=plan_filepath,
            )
            if plan is None:
                plan = await self._task_store.create_plan(
                    filepath=plan_filepath,
                    title=_optional_str(arguments.get("plan_title"))
                    or f"Plan for {_require_str(arguments, 'title')}",
                    summary="Created by task_create.",
                    lead_session_id=context.session_id,
                )
            plan_id = plan.id

        task = await self._task_store.create_task(
            lead_session_id=context.session_id,
            title=_require_str(arguments, "title"),
            description=str(arguments.get("description") or ""),
            success_criteria=str(arguments.get("success_criteria") or ""),
            required_outputs=[
                str(item).strip()
                for item in (arguments.get("required_outputs") or [])
                if str(item).strip()
            ],
            plan_id=plan_id,
            responsible_agent_name=_optional_str(arguments.get("responsible_agent_name")),
            responsible_session_id=_optional_str(arguments.get("responsible_session_id")),
        )
        return ToolExecutionResult(output=_render_task("Created task", task))


class TaskListTool(BaseTool):
    """List durable tasks visible to the current agent."""

    name = "task_list"
    description = "List durable tasks for the current lead session or current sub-agent session."
    parameters_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": ["string", "null"],
                "enum": [status.value for status in TaskStatus] + [None],
                "description": "Optional status filter.",
            },
            "plan_id": {"type": ["string", "null"], "description": "Optional plan filter."},
            "responsible_session_id": {
                "type": ["string", "null"],
                "description": "Optional responsible session filter.",
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "Maximum rows to return. Defaults to 20.",
                "minimum": 1,
                "maximum": 200,
            },
        },
        "required": ["status", "plan_id", "responsible_session_id", "limit"],
        "additionalProperties": False,
    }

    def __init__(self, task_store: TaskStore) -> None:
        self._task_store = task_store

    def get_prompt(self) -> str:
        return "- `task_list`: inspect queued, running, blocked, failed, and completed tasks."

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        # Strict OpenAI tool calling forces every property into `required`,
        # so the model emits `status=""` to signal "no filter". Treat empty
        # string the same as omitted; otherwise TaskStatus("") raises.
        raw_status = arguments.get("status")
        if isinstance(raw_status, str):
            raw_status = raw_status.strip() or None
        status = TaskStatus(raw_status) if raw_status is not None else None
        if _is_lead(context):
            lead_session_id = context.session_id
            responsible_session_id = _optional_str(arguments.get("responsible_session_id"))
        else:
            lead_session_id = None
            responsible_session_id = context.session_id
        tasks = await self._task_store.list_tasks(
            lead_session_id=lead_session_id,
            plan_id=_optional_str(arguments.get("plan_id")),
            status=status,
            responsible_session_id=responsible_session_id,
            limit=int(arguments.get("limit") or 20),
        )
        if not tasks:
            return ToolExecutionResult(output="No tasks found.")
        lines = ["Tasks:"]
        for task in tasks:
            lines.append(_task_line(task))
        return ToolExecutionResult(output="\n".join(lines))


class TaskGetTool(BaseTool):
    """Fetch one durable task with outputs and recent events."""

    name = "task_get"
    description = "Get one durable task by id, including outputs and recent events."
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task id to inspect."},
        },
        "required": ["task_id"],
        "additionalProperties": False,
    }

    def __init__(self, task_store: TaskStore) -> None:
        self._task_store = task_store

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        task = await _resolve_task(
            self._task_store, context, _require_str(arguments, "task_id")
        )
        outputs = await self._task_store.list_outputs(task.id)
        events = await self._task_store.list_events(task.id, limit=8)
        lines = [_render_task("Task", task)]
        if outputs:
            lines.append("outputs:")
            for output in outputs:
                lines.append(
                    f"- {output.kind.value} final={output.is_final} "
                    f"validated={output.validated} path={output.path or '-'} "
                    f"summary={output.summary}"
                )
        if events:
            lines.append("recent_events:")
            for event in events:
                lines.append(
                    f"- {event.created_at} {event.event_type}: {event.message}"
                )
        return ToolExecutionResult(output="\n".join(lines))


class TaskUpdateTool(BaseTool):
    """Update durable task state."""

    name = "task_update"
    description = "Update a durable task's status, assignment, blocked question, or error."
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": ["string", "null"],
                "description": "Task id. Sub-agents may omit to use their current assigned task.",
            },
            "status": {
                "type": ["string", "null"],
                "enum": [status.value for status in TaskStatus] + [None],
                "description": "Optional replacement task status.",
            },
            "responsible_agent_name": {
                "type": ["string", "null"],
                "description": "Optional responsible agent name.",
            },
            "responsible_session_id": {
                "type": ["string", "null"],
                "description": "Optional responsible session id.",
            },
            "blocked_question": {
                "type": ["string", "null"],
                "description": "Question blocking the task, if any.",
            },
            "error": {
                "type": ["string", "null"],
                "description": "Failure/error note, if any.",
            },
        },
        "required": [
            "task_id",
            "status",
            "responsible_agent_name",
            "responsible_session_id",
            "blocked_question",
            "error",
        ],
        "additionalProperties": False,
    }

    def __init__(self, task_store: TaskStore) -> None:
        self._task_store = task_store

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        task = await _resolve_task(
            self._task_store, context, _optional_str(arguments.get("task_id"))
        )
        updates: dict[str, Any] = {}
        if arguments.get("status") is not None:
            updates["status"] = TaskStatus(arguments["status"])
        if arguments.get("responsible_agent_name") is not None:
            updates["responsible_agent_name"] = _optional_str(
                arguments.get("responsible_agent_name")
            )
        if arguments.get("responsible_session_id") is not None:
            updates["responsible_session_id"] = _optional_str(
                arguments.get("responsible_session_id")
            )
        if arguments.get("blocked_question") is not None:
            updates["blocked_question"] = _optional_str(arguments.get("blocked_question"))
        if arguments.get("error") is not None:
            updates["error"] = _optional_str(arguments.get("error"))
        updated = await self._task_store.update_task(task.id, **updates)
        return ToolExecutionResult(output=_render_task("Updated task", updated))


class TaskOutputTool(BaseTool):
    """Record a task report or artifact."""

    name = "task_output"
    description = "Attach a report, artifact, source, or log to a task; mark final outputs explicitly."
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": ["string", "null"],
                "description": "Task id. Sub-agents may omit to use their current assigned task.",
            },
            "kind": {
                "type": "string",
                "enum": [kind.value for kind in TaskOutputKind],
                "description": "Output kind.",
            },
            "path": {"type": ["string", "null"], "description": "Artifact path."},
            "content": {"type": ["string", "null"], "description": "Inline output content."},
            "summary": {"type": "string", "description": "Short output summary."},
            "validated": {
                "type": "boolean",
                "description": "Whether path/content was verified by the agent.",
            },
            "is_final": {
                "type": "boolean",
                "description": "True only for final task deliverables.",
            },
        },
        "required": [
            "task_id",
            "kind",
            "path",
            "content",
            "summary",
            "validated",
            "is_final",
        ],
        "additionalProperties": False,
    }

    def __init__(self, task_store: TaskStore, *, root: Path) -> None:
        self._task_store = task_store
        self._root = root

    def get_prompt(self) -> str:
        return (
            "- `task_output`: record task deliverables. A task is not truly "
            "finished until it has a final task_output. Use `is_final=true` "
            "for the final report/artifact."
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        task = await _resolve_task(
            self._task_store, context, _optional_str(arguments.get("task_id"))
        )
        if task.status == TaskStatus.STOPPED:
            raise ValueError("Cannot add output to a stopped task.")
        if task.status in _COMPLETED_STATUSES and bool(arguments.get("is_final", False)):
            raise ValueError("Task already has a completed status.")
        raw_path = _optional_str(arguments.get("path"))
        if raw_path is not None and bool(arguments.get("validated", False)):
            _validate_path_exists(self._root, raw_path)
        if (
            raw_path is not None
            and bool(arguments.get("is_final", False))
            and not bool(arguments.get("validated", False))
        ):
            raise ValueError("Final artifact outputs must be validated.")
        output = await self._task_store.add_output(
            task_id=task.id,
            kind=TaskOutputKind(arguments["kind"]),
            path=raw_path,
            content=_optional_str(arguments.get("content")),
            summary=_require_str(arguments, "summary"),
            created_by_session_id=context.session_id,
            validated=bool(arguments.get("validated", False)),
            is_final=bool(arguments.get("is_final", False)),
        )
        if output.is_final:
            if output.path is not None:
                status = TaskStatus.COMPLETED_WITH_ARTIFACTS
            else:
                status = TaskStatus.COMPLETED_WITH_REPORT
            await self._task_store.update_task(
                task.id,
                status=status,
                blocked_question=None,
                blocked_correlation_id=None,
                error=None,
            )
        return ToolExecutionResult(
            output=(
                "Recorded task output.\n"
                f"task_id: {task.id}\n"
                f"output_id: {output.id}\n"
                f"kind: {output.kind.value}\n"
                f"is_final: {output.is_final}\n"
                f"path: {output.path or '-'}\n"
                f"summary: {output.summary}"
            )
        )


class RequestInputTool(BaseTool):
    """Ask Lead for input and wait briefly for a correlated response."""

    name = "request_input"
    description = "Ask Lead for input needed to continue the current task; waits for a correlated answer when possible."
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": ["string", "null"],
                "description": "Task id. Sub-agents may omit to use their current assigned task.",
            },
            "question": {"type": "string", "description": "The blocking question."},
            "context": {"type": "string", "description": "Why this answer is needed."},
            "options": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Optional answer choices.",
            },
            "default": {
                "type": ["string", "null"],
                "description": "Default path if Lead/user does not answer.",
            },
        },
        "required": ["task_id", "question", "context", "options", "default"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        task_store: TaskStore,
        message_store: AgentMessageStore,
        *,
        wait_timeout_seconds: float = _REQUEST_INPUT_WAIT_TIMEOUT_SECONDS,
        poll_interval_seconds: float = _REQUEST_INPUT_POLL_SECONDS,
    ) -> None:
        self._task_store = task_store
        self._message_store = message_store
        if wait_timeout_seconds <= 0:
            raise ValueError("wait_timeout_seconds must be positive.")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive.")
        self._wait_timeout_seconds = wait_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def get_prompt(self) -> str:
        return (
            "- `request_input`: use when your assigned task is blocked on a "
            "decision or missing requirement. This asks Lead for input, marks "
            "the task `blocked_needs_input`, and waits for a correlated answer. "
            "If no answer arrives before timeout, continue with the supplied "
            "default or your best safe judgment; do not fake a final report."
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        task = await _resolve_task(
            self._task_store, context, _optional_str(arguments.get("task_id"))
        )
        question = _require_str(arguments, "question")
        message = await self._message_store.send(
            from_session_id=context.session_id,
            from_agent_name=context.agent_name,
            to_session_id=task.lead_session_id,
            to_agent_name="Lead",
            body=_render_input_request(task, arguments),
            expects_response=True,
        )
        await self._task_store.update_task(
            task.id,
            status=TaskStatus.BLOCKED_NEEDS_INPUT,
            blocked_question=question,
            blocked_correlation_id=message.correlation_id,
            error=None,
        )
        await self._task_store.add_event(
            task.id,
            event_type="input_requested",
            message=question,
            agent_name=context.agent_name,
            session_id=context.session_id,
        )
        assert message.correlation_id is not None
        reply = await self._wait_for_reply(
            session_id=context.session_id,
            agent_name=context.agent_name,
            correlation_id=message.correlation_id,
        )
        if reply is not None:
            await self._task_store.update_task(
                task.id,
                status=TaskStatus.RUNNING,
                blocked_question=None,
                blocked_correlation_id=None,
                error=None,
            )
            await self._task_store.add_event(
                task.id,
                event_type="input_received",
                message=f"Input received for: {question}",
                agent_name=context.agent_name,
                session_id=context.session_id,
            )
            return ToolExecutionResult(
                output=(
                    "Input response received; continue the task.\n"
                    f"task_id: {task.id}\n"
                    f"correlation_id: {message.correlation_id}\n"
                    f"from: {reply.from_agent_name} {reply.from_session_id}\n"
                    f"answer:\n{reply.body}"
                )
            )

        default = _optional_str(arguments.get("default"))
        await self._task_store.update_task(
            task.id,
            status=TaskStatus.RUNNING,
            blocked_question=None,
            blocked_correlation_id=None,
            error=None,
        )
        await self._task_store.add_event(
            task.id,
            event_type="input_timeout",
            message=f"Input timed out for: {question}",
            agent_name=context.agent_name,
            session_id=context.session_id,
        )
        return ToolExecutionResult(
            output=(
                "Input wait timed out; continue the task if safe.\n"
                f"task_id: {task.id}\n"
                f"correlation_id: {message.correlation_id}\n"
                f"wait_timeout_seconds: {self._wait_timeout_seconds:g}\n"
                f"question: {question}\n"
                f"default: {default or '-'}\n"
                "No Lead/user answer arrived before the timeout. Use the default "
                "if provided. If no default is safe, use your best judgment or "
                "record a final task_output explaining the blocker."
            )
        )

    async def _wait_for_reply(
        self,
        *,
        session_id: str,
        agent_name: str,
        correlation_id: str,
    ) -> AgentMessage | None:
        """Poll the message bus for one correlated answer."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._wait_timeout_seconds
        while True:
            reply = await self._message_store.claim_reply(
                to_session_id=session_id,
                to_agent_name=agent_name,
                in_reply_to=correlation_id,
            )
            if reply is not None:
                return reply
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            await asyncio.sleep(min(self._poll_interval_seconds, remaining))


class TaskStopTool(BaseTool):
    """Stop a durable task and terminate its live sub-agent if present."""

    name = "task_stop"
    description = "Stop a task and terminate the live responsible sub-agent process when present."
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task id to stop."},
            "reason": {"type": ["string", "null"], "description": "Stop reason."},
        },
        "required": ["task_id", "reason"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        task_store: TaskStore,
        registry: SubagentRegistry,
        message_store: AgentMessageStore,
    ) -> None:
        self._task_store = task_store
        self._registry = registry
        self._message_store = message_store

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        task = await _resolve_task(
            self._task_store, context, _require_str(arguments, "task_id")
        )
        note = "no live process"
        if task.responsible_session_id is not None:
            live = await self._registry.get(task.responsible_session_id)
            if live is not None and live.process.returncode is not None:
                return ToolExecutionResult(
                    output=(
                        f"Task `{task.id}` has an exited sub-agent waiting for "
                        "the reaper to deliver its final report. Try again shortly."
                    )
                )
            live = await self._registry.remove(task.responsible_session_id)
            if live is not None:
                note = await _terminate_live(live)
                if live.task_run_id is not None:
                    await self._task_store.finish_run(
                        live.task_run_id,
                        status=TaskRunStatus.KILLED,
                        exit_code=getattr(live.process, "returncode", None),
                        envelope_status=None,
                        error=_optional_str(arguments.get("reason")) or "stopped",
                    )
                await self._message_store.send(
                    from_session_id=live.session_id,
                    from_agent_name=live.agent_name,
                    to_session_id=live.parent_session_id,
                    to_agent_name=live.parent_agent_name,
                    body=(
                        f"Task `{task.id}` / sub-agent `{live.agent_name}` "
                        f"(session_id={live.session_id}) stopped by {context.agent_name}.\n"
                        f"reason: {_optional_str(arguments.get('reason')) or '-'}"
                    ),
                    in_reply_to=live.correlation_id,
                )
        updated = await self._task_store.update_task(
            task.id,
            status=TaskStatus.STOPPED,
            error=_optional_str(arguments.get("reason")),
        )
        return ToolExecutionResult(output=f"{_render_task('Stopped task', updated)}\nprocess: {note}")


class TaskResumeTool(BaseTool):
    """Resume a blocked or failed sub-agent task on the same session id."""

    name = "task_resume"
    description = "Resume a task by launching its responsible sub-agent again on the same session id."
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task id to resume."},
            "message": {
                "type": ["string", "null"],
                "description": "Optional answer or instruction to send before resuming.",
            },
        },
        "required": ["task_id", "message"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        root: Path,
        agent_catalog: AgentCatalog,
        registry: SubagentRegistry,
        task_store: TaskStore,
        message_store: AgentMessageStore,
        python_executable: str | None = None,
    ) -> None:
        self._root = root
        self._agent_catalog = agent_catalog
        self._registry = registry
        self._task_store = task_store
        self._message_store = message_store
        self._python = python_executable or sys.executable

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        task = await _resolve_task(
            self._task_store, context, _require_str(arguments, "task_id")
        )
        if task.responsible_agent_name is None or task.responsible_session_id is None:
            raise ValueError("Task has no responsible sub-agent session to resume.")
        if task.status in {
            TaskStatus.COMPLETED_WITH_REPORT,
            TaskStatus.COMPLETED_WITH_ARTIFACTS,
            TaskStatus.COMPLETED_WITHOUT_ARTIFACTS,
        }:
            raise ValueError(
                f"Task is already complete ({task.status.value}); create a follow-up task instead."
            )
        entry = self._agent_catalog.get(task.responsible_agent_name)
        if entry is None or not AgentCatalog.is_dispatchable(entry):
            raise ValueError(f"Task agent `{task.responsible_agent_name}` is not dispatchable.")
        live = await self._registry.get(task.responsible_session_id)
        if live is not None:
            if live.process.returncode is None:
                raise ValueError(
                    f"Task already has a live process for session {task.responsible_session_id}; "
                    "answer it with send_message using the task's blocked_correlation_id."
                )
            raise ValueError(
                "Task has an exited sub-agent that has not been reaped yet. "
                "Wait for the final report to arrive, then resume if needed."
            )

        message = _optional_str(arguments.get("message"))
        if message is not None:
            recipient_agent_name = load_agent_config(
                self._root, task.responsible_agent_name
            ).name
            await self._message_store.send(
                from_session_id=context.session_id,
                from_agent_name=context.agent_name,
                to_session_id=task.responsible_session_id,
                to_agent_name=recipient_agent_name,
                body=message,
                in_reply_to=task.blocked_correlation_id,
            )

        correlation_id = str(uuid4())
        resume_prompt = _resume_prompt(task, message)
        run = await self._task_store.create_run(
            task_id=task.id,
            session_id=task.responsible_session_id,
            agent_name=task.responsible_agent_name,
            pid=None,
        )
        try:
            launched = await launch_subagent_process(
                root=self._root,
                python_executable=self._python,
                agent_name=task.responsible_agent_name,
                task_text=resume_prompt,
                parent_session_id=task.lead_session_id,
                parent_agent_name=context.agent_name,
                session_id=task.responsible_session_id,
                correlation_id=correlation_id,
                task_id=task.id,
            )
        except Exception as exc:
            await self._task_store.finish_run(
                run.id,
                status=TaskRunStatus.CRASHED,
                exit_code=None,
                envelope_status=None,
                error=f"failed to resume task: {exc}",
            )
            await self._task_store.update_task(
                task.id,
                status=TaskStatus.FAILED,
                error=f"failed to resume task: {exc}",
            )
            raise
        live_entry = LiveSubagent(
            session_id=task.responsible_session_id,
            agent_name=task.responsible_agent_name,
            parent_session_id=task.lead_session_id,
            parent_agent_name=context.agent_name,
            process=launched.process,
            task_text=resume_prompt,
            correlation_id=correlation_id,
            task_id=task.id,
            task_run_id=run.id,
            stdout_buffer=launched.stdout_buffer,
            stderr_buffer=launched.stderr_buffer,
            drainers=launched.drainers,
        )
        try:
            await self._registry.register(live_entry)
        except BaseException:
            await _terminate_live(live_entry)
            await self._task_store.finish_run(
                run.id,
                status=TaskRunStatus.KILLED,
                exit_code=getattr(launched.process, "returncode", None),
                envelope_status=None,
                error="task resume cancelled before registry ownership",
            )
            await self._task_store.update_task(
                task.id,
                status=TaskStatus.FAILED,
                error="task resume cancelled before registry ownership",
            )
            raise
        try:
            await self._task_store.update_run_pid(run.id, launched.process.pid)
            await self._task_store.update_task(
                task.id,
                status=TaskStatus.RUNNING,
                blocked_question=None,
                blocked_correlation_id=None,
                error=None,
            )
        except Exception:
            logger.exception(
                "task_resume state update failed task_id=%s session_id=%s",
                task.id,
                task.responsible_session_id,
            )
        return ToolExecutionResult(
            output=(
                "Resumed task.\n"
                f"task_id: {task.id}\n"
                f"session_id: {task.responsible_session_id}\n"
                f"correlation_id: {correlation_id}\n"
                f"pid: {launched.process.pid}"
            )
        )


async def _resolve_task(
    task_store: TaskStore,
    context: ToolExecutionContext,
    task_id: str | None,
) -> TaskRecord:
    if task_id is None:
        task = await task_store.find_task_by_session(context.session_id)
        if task is None:
            raise ValueError("No current task is assigned to this session.")
    else:
        task = await task_store.get_task(task_id)
    if _is_lead(context):
        if task.lead_session_id != context.session_id:
            raise ValueError("Task does not belong to the current lead session.")
    elif task.responsible_session_id != context.session_id:
        raise ValueError("Task is not assigned to the current agent session.")
    return task


async def _terminate_live(live: LiveSubagent) -> str:
    proc = live.process
    note = "SIGTERM sent"
    try:
        proc.terminate()
    except ProcessLookupError:
        note = "process already gone"
    else:
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                note = "SIGKILL after SIGTERM timeout"
            except ProcessLookupError:
                note = "process disappeared before SIGKILL"
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                note = "process did not exit after SIGKILL"
    for drainer in live.drainers:
        if not drainer.done():
            drainer.cancel()
        try:
            await drainer
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    return note


def _render_input_request(task: TaskRecord, arguments: dict[str, Any]) -> str:
    lines = [
        f"Task `{task.title}` is blocked and needs input.",
        f"task_id: {task.id}",
        f"responsible_session_id: {task.responsible_session_id or '-'}",
        f"question: {_require_str(arguments, 'question')}",
        f"context: {str(arguments.get('context') or '').strip() or '-'}",
    ]
    options = arguments.get("options") or []
    if options:
        lines.append("options:")
        for index, option in enumerate(options, 1):
            lines.append(f"{index}. {option}")
    default = _optional_str(arguments.get("default"))
    if default is not None:
        lines.append(f"default: {default}")
    return "\n".join(lines)


def _resume_prompt(task: TaskRecord, message: str | None) -> str:
    lines = [
        "Resume this durable task from the existing session history and inbox.",
        f"task_id: {task.id}",
        f"title: {task.title}",
        f"status_before_resume: {task.status.value}",
        f"success_criteria: {task.success_criteria or '-'}",
        f"required_outputs: {', '.join(task.required_outputs) or '-'}",
    ]
    if message is not None:
        lines.append(f"new_instruction_or_answer: {message}")
    lines.append(
        "Continue the task. If blocked, call request_input. When finished, "
        "call task_output with is_final=true."
    )
    return "\n".join(lines)


def _validate_path_exists(root: Path, raw_path: str) -> None:
    path = Path(raw_path)
    resolved = (path if path.is_absolute() else (root / path)).resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("Task output path must stay inside the Feather root.") from exc
    if not resolved.exists():
        raise ValueError(f"Validated task output path does not exist: {raw_path}")


def _render_task(prefix: str, task: TaskRecord) -> str:
    return (
        f"{prefix}.\n"
        f"id: {task.id}\n"
        f"title: {task.title}\n"
        f"status: {task.status.value}\n"
        f"plan_id: {task.plan_id or '-'}\n"
        f"lead_session_id: {task.lead_session_id}\n"
        f"responsible: {task.responsible_agent_name or '-'} "
        f"{task.responsible_session_id or '-'}\n"
        f"required_outputs: {', '.join(task.required_outputs) or '-'}\n"
        f"blocked_question: {task.blocked_question or '-'}\n"
        f"error: {task.error or '-'}"
    )


def _task_line(task: TaskRecord) -> str:
    return (
        f"- {task.status.value} id={task.id} title={task.title} "
        f"agent={task.responsible_agent_name or '-'} "
        f"session={task.responsible_session_id or '-'}"
    )


def _require_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{key}` must be a non-empty string.")
    return value.strip()


_NULLISH_LITERALS: frozenset[str] = frozenset({"null", "none"})


def _optional_str(value: object) -> str | None:
    """Normalize an optional string field from tool arguments.

    Returns ``None`` for actual ``None``, non-strings, empty/whitespace-only
    strings, AND the literal JSON/Python null spellings ``"null"`` /
    ``"None"`` (case-insensitive). Models occasionally stringify "no value"
    as the literal word ``"null"`` after the openrouter_translator flattens
    ``["string", "null"]`` schemas to plain ``"string"`` for cross-provider
    compatibility — without this normalization the value reaches the task
    store as a real ID and lookups fail with ``Unknown plan: null`` /
    ``Unknown task: null`` (observed in user runtime).
    """

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.lower() in _NULLISH_LITERALS:
        return None
    return stripped


def _is_lead(context: ToolExecutionContext) -> bool:
    return context.agent_name.lower() == "lead"
