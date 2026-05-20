"""Launch a catalog-listed sub-agent as a detached subprocess.

``spawn_agent`` returns **immediately** after the subprocess has been
launched. It does not wait for the sub-agent to complete. The caller is
expected to watch its inbox on subsequent turns for a final report
message delivered by the runtime's subprocess reaper.

Why non-blocking: if the lead awaited the subprocess's completion here,
the lead's agent loop would be frozen for the entire duration of the
child's work. That would defeat the `send_message` tool — the lead could
never receive status updates or send mid-run instructions. With
non-blocking spawn, the lead can spawn multiple children in parallel,
exchange messages with them via the SQLite bus, and see each child's
final report arrive on the same bus when it finishes.

Child subprocess lifecycle:

  1. This tool launches `python -m feather.subagent_entry ...`.
  2. It registers a :class:`LiveSubagent` entry in the runtime registry
     so the reaper can find it.
  3. The runtime's subprocess reaper watches the child's PID; when it
     exits, the reaper parses stdout, extracts the envelope, and posts a
     final message to the parent session's inbox.
  4. Parent drains its inbox on its next turn and sees the report.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from feather.core.agent_catalog import AgentCatalog
from feather.core.subagent_registry import LiveSubagent, SubagentRegistry
from feather.models import (
    TaskRecord,
    TaskRunStatus,
    TaskStatus,
    ToolExecutionContext,
    ToolExecutionResult,
)
from feather.storage.task_store import TaskStore
from feather.subagent_protocol import RESULT_BEGIN, RESULT_END
from feather.tools.base import BaseTool

logger = logging.getLogger(__name__)

_ENVELOPE_PATTERN = re.compile(
    rf"{re.escape(RESULT_BEGIN)}\s*(.*?)\s*{re.escape(RESULT_END)}",
    re.DOTALL,
)


@dataclass(slots=True)
class LaunchedSubagent:
    """Subprocess launch result with stream-drainer state."""

    process: asyncio.subprocess.Process
    stdout_buffer: bytearray
    stderr_buffer: bytearray
    drainers: tuple[asyncio.Task[None], ...]


class SpawnAgentTool(BaseTool):
    """Launch a sub-agent subprocess; return immediately, report arrives via inbox."""

    name = "spawn_agent"
    description = (
        "Dispatch a focused task to a sub-agent running in a new process. "
        "Returns **immediately** with the sub-agent's session_id after the "
        "subprocess has been launched — it does NOT wait for completion. "
        "Use `send_message` with the returned `session_id` to interact with "
        "the sub-agent while it works. The sub-agent's final report arrives "
        "in your inbox as an agent message when it finishes. Consult "
        "`<dispatchable_agents>` in the prompt for valid `agent_name` values."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": (
                    "Catalog name of the sub-agent (matches its YAML filename "
                    "without the .yaml extension). See `<dispatchable_agents>`."
                ),
            },
            "task": {
                "type": "string",
                "description": (
                    "One self-contained instruction for the sub-agent. Include "
                    "the success criteria and any required context — the sub-agent "
                    "does not see the parent's conversation history."
                ),
            },
            "task_id": {
                "type": ["string", "null"],
                "description": (
                    "Optional existing durable task id to dispatch. Use this when "
                    "Lead has already created a planned task row; spawn_agent will "
                    "bind the subprocess to that task instead of creating a duplicate."
                ),
            },
        },
        "required": ["agent_name", "task", "task_id"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        root: Path,
        agent_catalog: AgentCatalog,
        registry: SubagentRegistry,
        parent_agent_name: str = "lead",
        python_executable: str | None = None,
        task_store: TaskStore | None = None,
    ) -> None:
        self._root = root
        self._agent_catalog = agent_catalog
        self._registry = registry
        self._parent_agent_name = parent_agent_name
        self._python = python_executable or sys.executable
        self._task_store = task_store

    def get_prompt(self) -> str:
        return (
            "- `spawn_agent`: launch a sub-agent as a new OS process and get "
            "its session_id back immediately. Delivery model is asynchronous: "
            "the sub-agent's final report arrives via your inbox as an "
            "agent_message (correlation_id returned by spawn_agent). While the "
            "sub-agent runs, you can keep working, spawn more sub-agents in "
            "parallel, or send it status-check messages with `send_message`. "
            "When dispatching a task created with `task_create`, pass its "
            "`task_id` so Feather does not create a duplicate task row."
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        raw_name = arguments.get("agent_name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("spawn_agent `agent_name` must be a non-empty string.")
        agent_name = raw_name.strip()
        if not AgentCatalog.is_valid_name(agent_name):
            raise ValueError(
                "spawn_agent `agent_name` may only contain letters, digits, "
                f"underscores, and hyphens. got {agent_name!r}."
            )
        entry = self._agent_catalog.get(agent_name)
        if entry is None:
            available = ", ".join(
                e.name for e in self._agent_catalog.list_dispatchable()
            ) or "(none)"
            raise ValueError(
                f"Unknown sub-agent `{agent_name}`. Available: {available}."
            )
        if not AgentCatalog.is_dispatchable(entry):
            raise ValueError(
                f"Agent `{agent_name}` (role={entry.role}) is not dispatchable."
            )

        raw_task = arguments.get("task")
        if not isinstance(raw_task, str) or not raw_task.strip():
            raise ValueError("spawn_agent `task` must be a non-empty string.")
        task_text = raw_task.strip()

        correlation_id = str(uuid4())
        child_session_id = str(uuid4())
        task_id: str | None = None
        task_run_id: str | None = None
        if self._task_store is not None:
            task = await self._resolve_or_create_task(
                context=context,
                task_id=_optional_str(arguments.get("task_id")),
                task_text=task_text,
                agent_name=agent_name,
                child_session_id=child_session_id,
            )
            task_id = task.id
            run = await self._task_store.create_run(
                task_id=task_id,
                session_id=child_session_id,
                agent_name=agent_name,
                pid=None,
            )
            task_run_id = run.id
        logger.info(
            "spawn_agent launching agent_name=%s parent_session_id=%s child_session_id=%s correlation_id=%s",
            agent_name,
            context.session_id,
            child_session_id,
            correlation_id,
        )

        try:
            launched = await launch_subagent_process(
                root=self._root,
                python_executable=self._python,
                agent_name=agent_name,
                task_text=task_text,
                parent_session_id=context.session_id,
                parent_agent_name=self._parent_agent_name,
                session_id=child_session_id,
                correlation_id=correlation_id,
                task_id=task_id,
            )
        except Exception as exc:
            if self._task_store is not None and task_id is not None:
                if task_run_id is not None:
                    await self._task_store.finish_run(
                        task_run_id,
                        status=TaskRunStatus.CRASHED,
                        exit_code=None,
                        envelope_status=None,
                        error=f"failed to launch sub-agent executable: {exc}",
                    )
                await self._task_store.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=f"failed to launch sub-agent executable: {exc}",
                )
            raise RuntimeError(
                f"failed to launch sub-agent executable: {exc}"
            ) from exc

        # Register BEFORE we release our reference to the process, so the
        # reaper can see it even for instant-exit children (before this
        # tool call returns, a reaper tick might fire).
        live = LiveSubagent(
            session_id=child_session_id,
            agent_name=agent_name,
            parent_session_id=context.session_id,
            parent_agent_name=self._parent_agent_name,
            process=launched.process,
            task_text=task_text,
            correlation_id=correlation_id,
            task_id=task_id,
            task_run_id=task_run_id,
            stdout_buffer=launched.stdout_buffer,
            stderr_buffer=launched.stderr_buffer,
            drainers=launched.drainers,
        )
        try:
            await self._registry.register(live)
        except BaseException:
            await _terminate_launched(launched)
            if self._task_store is not None and task_id is not None:
                if task_run_id is not None:
                    await self._task_store.finish_run(
                        task_run_id,
                        status=TaskRunStatus.KILLED,
                        exit_code=getattr(launched.process, "returncode", None),
                        envelope_status=None,
                        error="sub-agent launch cancelled before registry ownership",
                    )
                await self._task_store.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error="sub-agent launch cancelled before registry ownership",
                )
            raise

        if self._task_store is not None and task_id is not None:
            try:
                if task_run_id is not None:
                    await self._task_store.update_run_pid(task_run_id, launched.process.pid)
                await self._task_store.update_task(
                    task_id,
                    status=TaskStatus.RUNNING,
                    responsible_agent_name=agent_name,
                    responsible_session_id=child_session_id,
                    error=None,
                )
            except Exception:
                logger.exception(
                    "spawn_agent task state update failed task_id=%s session_id=%s",
                    task_id,
                    child_session_id,
                )

        # Task file is eagerly-removed by the child on read; if the child
        # never starts (e.g. bad python path) the reaper will pick up the
        # empty envelope and report failure.
        return ToolExecutionResult(
            output=(
                f"Sub-agent `{agent_name}` spawned.\n"
                f"session_id: {child_session_id}\n"
                f"task_id: {task_id or '-'}\n"
                f"correlation_id: {correlation_id}\n"
                f"pid: {launched.process.pid}\n"
                "Its final report will be delivered to your inbox as an agent "
                "message with `in_reply_to` set to this correlation_id. You can "
                "continue working — use `send_message` if you need to interact "
                "with it before then."
            )
        )

    async def _resolve_or_create_task(
        self,
        *,
        context: ToolExecutionContext,
        task_id: str | None,
        task_text: str,
        agent_name: str,
        child_session_id: str,
    ) -> TaskRecord:
        """Return an existing planned task or create one for this dispatch."""

        assert self._task_store is not None
        if task_id is None:
            return await self._task_store.create_task(
                lead_session_id=context.session_id,
                title=_task_title(task_text),
                description=task_text,
                responsible_agent_name=agent_name,
                responsible_session_id=child_session_id,
                status=TaskStatus.QUEUED,
            )
        task = await self._task_store.get_task(task_id)
        if task.lead_session_id != context.session_id:
            raise ValueError("Task does not belong to the current lead session.")
        if task.status != TaskStatus.QUEUED:
            raise ValueError(
                "spawn_agent can only dispatch queued planned tasks; "
                f"task {task.id} is {task.status.value}."
            )
        if (
            task.responsible_agent_name is not None
            and task.responsible_agent_name != agent_name
        ):
            raise ValueError(
                f"Task {task.id} is assigned to {task.responsible_agent_name}, "
                f"not {agent_name}."
            )
        return task


async def launch_subagent_process(
    *,
    root: Path,
    python_executable: str,
    agent_name: str,
    task_text: str,
    parent_session_id: str,
    parent_agent_name: str,
    session_id: str,
    correlation_id: str,
    task_id: str | None = None,
) -> LaunchedSubagent:
    """Launch a sub-agent subprocess and start stdout/stderr drainers."""

    task_file = _write_task_file(root, task_text)
    argv = [
        python_executable,
        "-m",
        "feather.subagent_entry",
        "--agent-name",
        agent_name,
        "--task-file",
        str(task_file),
        "--parent-session",
        parent_session_id,
        "--parent-agent-name",
        parent_agent_name,
        "--session-id",
        session_id,
        "--correlation-id",
        correlation_id,
        "--root",
        str(root),
    ]
    if task_id is not None:
        argv.extend(["--task-id", task_id])
    # Snapshot the parent env explicitly and patch in fallbacks for the
    # vars the sub-agent's code actually depends on. ``HOME`` is the
    # critical one — ``Path.expanduser()`` raises RuntimeError when it
    # is missing, and the attachment-drop parser calls it on every
    # inbound user-text token. With the default ``env=None`` asyncio
    # would inherit the parent env, but we've seen this crash happen in
    # the field, so we make the propagation explicit AND defend against
    # an empty value by re-deriving from ``pwd`` (the same source
    # ``os.path.expanduser`` consults when ``HOME`` is empty).
    subprocess_env = os.environ.copy()
    if not subprocess_env.get("HOME"):
        try:
            import pwd

            subprocess_env["HOME"] = pwd.getpwuid(os.getuid()).pw_dir
        except (ImportError, KeyError, OSError):
            pass

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(root),
            env=subprocess_env,
        )
    except Exception:
        _remove_task_file(task_file)
        raise
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    stdout_drainer = asyncio.create_task(
        _drain_stream(proc.stdout, stdout_buf),
        name=f"subagent-stdout-{session_id}",
    )
    stderr_drainer = asyncio.create_task(
        _drain_stream(proc.stderr, stderr_buf),
        name=f"subagent-stderr-{session_id}",
    )
    return LaunchedSubagent(
        process=proc,
        stdout_buffer=stdout_buf,
        stderr_buffer=stderr_buf,
        drainers=(stdout_drainer, stderr_drainer),
    )


def _write_task_file(root: Path, task_text: str) -> Path:
    """Stage the task prompt on disk so it survives subprocess argv limits."""

    staging_dir = (root / ".feather" / "tmp" / "subagent_tasks").resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        prefix="subagent_task_", suffix=".txt", dir=str(staging_dir)
    )
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(task_text)
    return Path(raw_path)


def _remove_task_file(task_file: Path) -> None:
    try:
        task_file.unlink()
    except FileNotFoundError:
        return


def _task_title(task_text: str) -> str:
    first = next((line.strip() for line in task_text.splitlines() if line.strip()), "")
    if first.lower().startswith("goal:"):
        first = first.split(":", 1)[1].strip()
    if len(first) > 120:
        first = first[:117] + "..."
    return first or "Sub-agent task"


_NULLISH_LITERALS: frozenset[str] = frozenset({"null", "none"})


def _optional_str(value: object) -> str | None:
    """Normalize an optional string field from tool arguments.

    See ``feather.tools.task_tools._optional_str`` for the full rationale
    — keep these two helpers in sync. The short version: models sometimes
    send the literal string ``"null"`` for absent optional IDs after the
    OpenRouter translator flattens nullable union types, and that string
    must coerce to ``None`` here or it reaches the task store as an ID
    lookup and crashes.
    """

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.lower() in _NULLISH_LITERALS:
        return None
    return stripped


async def _drain_stream(
    stream: asyncio.StreamReader | None, buffer: bytearray
) -> None:
    """Continuously read from ``stream`` into ``buffer`` until EOF.

    Failure isolation: any exception (closed stream, cancellation) is
    swallowed — this coroutine must never raise into the event loop.
    """

    if stream is None:
        return
    try:
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                return
            buffer.extend(chunk)
    except Exception:  # noqa: BLE001
        return


async def _terminate_launched(launched: LaunchedSubagent) -> None:
    """Best-effort cleanup for a launched process not owned by the registry."""

    proc = launched.process
    if proc.returncode is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
    for drainer in launched.drainers:
        if not drainer.done():
            drainer.cancel()
        try:
            await drainer
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


def extract_envelope(stdout_text: str) -> dict[str, Any] | None:
    """Parse the marker-wrapped JSON envelope emitted by the subprocess."""

    import json

    match = _ENVELOPE_PATTERN.search(stdout_text)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed
