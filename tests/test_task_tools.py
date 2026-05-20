"""Tests for durable task management tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from feather.core.agent_catalog import AgentCatalog, AgentCatalogEntry
from feather.core.subagent_registry import LiveSubagent, SubagentRegistry
from feather.models import TaskRunStatus, TaskStatus, ToolExecutionContext
from feather.storage.agent_message_store import AgentMessageStore
from feather.storage.task_store import TaskStore
from feather.tools.task_tools import (
    RequestInputTool,
    TaskCreateTool,
    TaskListTool,
    TaskOutputTool,
    TaskResumeTool,
    TaskStopTool,
    _optional_str,
)


def _lead() -> ToolExecutionContext:
    return ToolExecutionContext(session_id="lead-sess", agent_name="Lead")


def _research() -> ToolExecutionContext:
    return ToolExecutionContext(session_id="child-sess", agent_name="Research")


class _EofStream:
    async def read(self, n: int) -> bytes:
        return b""


class _FakeProc:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.pid = 123
        self.returncode = returncode
        self.stdout = _EofStream()
        self.stderr = _EofStream()

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class _FakeCatalog(AgentCatalog):
    def __init__(self) -> None:
        super().__init__(Path("/nonexistent"))

    def list_entries(self) -> list[AgentCatalogEntry]:
        return [
            AgentCatalogEntry(
                name="research",
                role="research",
                description="research",
                personality="",
                registered_tools=("web_search",),
                is_builtin=True,
            )
        ]


def _install_subprocess_stub(
    monkeypatch: pytest.MonkeyPatch, proc: _FakeProc
) -> list[list[str]]:
    captured: list[list[str]] = []

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured.append(list(args))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    return captured


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        (42, None),  # non-string
        ("real-id-123", "real-id-123"),
        ("  padded  ", "padded"),
        # The bug: model stringifies JSON null as the literal word "null"
        # after the openrouter_translator flattens ["string","null"] schemas.
        # Without coercion these reach the task store as ID lookups and
        # crash with ``Unknown plan: null`` / ``Unknown task: null``.
        ("null", None),
        ("NULL", None),
        ("Null", None),
        ("none", None),
        ("None", None),
        ("NONE", None),
        ("  null  ", None),
    ],
)
def test_optional_str_coerces_nullish_strings_to_none(
    value: object, expected: str | None
) -> None:
    assert _optional_str(value) == expected


async def test_task_create_treats_null_string_plan_id_as_absent(
    tmp_path: Path,
) -> None:
    """Regression: model sends ``"plan_id": "null"`` (literal stringified
    JSON null) after schema flattening; the tool must treat that as
    "no plan" rather than passing the string through and crashing in
    ``TaskStore.get_plan`` with ``Unknown plan: null``.
    """

    store = TaskStore(tmp_path / "feather.db")
    await store.initialize()
    try:
        created = await TaskCreateTool(store).execute(
            {
                "title": "Plan-less task",
                "description": "no plan id provided",
                "success_criteria": "done",
                "required_outputs": [],
                "plan_id": "null",       # the regression input
                "plan_filepath": None,
                "plan_title": None,
                "responsible_agent_name": "research",
                "responsible_session_id": "child-sess",
            },
            _lead(),
        )
        assert "Created task" in created.output
        # And nothing got persisted with a bogus plan_id reference.
        task_id = _field(created.output, "id")
        task = await store.get_task(task_id)
        assert task.plan_id is None
    finally:
        await store.close()


async def test_task_tools_create_list_and_final_output(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "feather.db")
    await store.initialize()
    try:
        created = await TaskCreateTool(store).execute(
            {
                "title": "Research prices",
                "description": "Find comps",
                "success_criteria": "final report",
                "required_outputs": ["report"],
                "plan_id": None,
                "plan_filepath": ".feather/artifacts/plan/research.md",
                "plan_title": "Research plan",
                "responsible_agent_name": "research",
                "responsible_session_id": "child-sess",
            },
            _lead(),
        )
        assert "Created task" in created.output
        task_id = _field(created.output, "id")

        listed = await TaskListTool(store).execute(
            {
                "status": None,
                "plan_id": None,
                "responsible_session_id": None,
                "limit": 10,
            },
            _lead(),
        )
        assert task_id in listed.output

        # Strict OpenAI tool calling sends empty strings instead of nulls
        # for optional filters. TaskListTool must treat "" identically to
        # None — TaskStatus("") would otherwise raise ValueError.
        listed_empty = await TaskListTool(store).execute(
            {
                "status": "",
                "plan_id": "",
                "responsible_session_id": "",
                "limit": 10,
            },
            _lead(),
        )
        assert task_id in listed_empty.output

        output = await TaskOutputTool(store, root=tmp_path).execute(
            {
                "task_id": task_id,
                "kind": "report",
                "path": None,
                "content": "final answer",
                "summary": "Final report",
                "validated": True,
                "is_final": True,
            },
            _lead(),
        )
        assert "Recorded task output" in output.output
        task = await store.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED_WITH_REPORT
    finally:
        await store.close()


async def test_request_input_waits_for_correlated_reply(tmp_path: Path) -> None:
    task_store = TaskStore(tmp_path / "feather.db")
    message_store = AgentMessageStore(tmp_path / "feather.db")
    await task_store.initialize()
    await message_store.initialize()
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Blocked research",
            responsible_agent_name="Research",
            responsible_session_id="child-sess",
            status=TaskStatus.RUNNING,
        )
        request_task = asyncio.create_task(
            RequestInputTool(
                task_store,
                message_store,
                wait_timeout_seconds=1.0,
                poll_interval_seconds=0.01,
            ).execute(
                {
                    "task_id": None,
                    "question": "Include rowhouses?",
                    "context": "This changes filters.",
                    "options": ["yes", "no"],
                    "default": "no",
                },
                _research(),
            )
        )
        await asyncio.sleep(0.02)

        inbox = await message_store.inbox(to_session_id="lead-sess", to_agent_name="Lead")
        assert len(inbox) == 1
        assert inbox[0].expects_response
        assert "Include rowhouses?" in inbox[0].body
        blocked = await task_store.get_task(task.id)
        assert blocked.status == TaskStatus.BLOCKED_NEEDS_INPUT
        assert blocked.blocked_correlation_id == inbox[0].correlation_id

        await message_store.send(
            from_session_id="lead-sess",
            from_agent_name="Lead",
            to_session_id="child-sess",
            to_agent_name="Research",
            body="Yes, include rowhouses.",
            in_reply_to=inbox[0].correlation_id,
        )

        result = await request_task
        assert "Input response received" in result.output
        assert "Yes, include rowhouses." in result.output
        updated = await task_store.get_task(task.id)
        assert updated.status == TaskStatus.RUNNING
        assert updated.blocked_correlation_id is None
        assert (
            await message_store.inbox(to_session_id="child-sess", to_agent_name="Research")
        ) == []
    finally:
        await message_store.close()
        await task_store.close()


async def test_request_input_timeout_returns_default_guidance(tmp_path: Path) -> None:
    task_store = TaskStore(tmp_path / "feather.db")
    message_store = AgentMessageStore(tmp_path / "feather.db")
    await task_store.initialize()
    await message_store.initialize()
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Blocked research",
            responsible_agent_name="Research",
            responsible_session_id="child-sess",
            status=TaskStatus.RUNNING,
        )
        result = await RequestInputTool(
            task_store,
            message_store,
            wait_timeout_seconds=0.01,
            poll_interval_seconds=0.001,
        ).execute(
            {
                "task_id": None,
                "question": "Include rowhouses?",
                "context": "This changes filters.",
                "options": ["yes", "no"],
                "default": "no",
            },
            _research(),
        )

        assert "Input wait timed out" in result.output
        assert "default: no" in result.output
        updated = await task_store.get_task(task.id)
        assert updated.status == TaskStatus.RUNNING
        assert updated.blocked_correlation_id is None
    finally:
        await message_store.close()
        await task_store.close()


async def test_task_resume_reuses_session_and_sends_correlated_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_store = TaskStore(tmp_path / "feather.db")
    message_store = AgentMessageStore(tmp_path / "feather.db")
    await task_store.initialize()
    await message_store.initialize()
    proc = _FakeProc()
    captured = _install_subprocess_stub(monkeypatch, proc)
    registry = SubagentRegistry()
    agents_dir = tmp_path / "config" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "research.yaml").write_text(
        """name: Research
role: research
personality: Direct
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
registered_tools: []
""",
        encoding="utf-8",
    )
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Resume me",
            responsible_agent_name="research",
            responsible_session_id="child-sess",
            status=TaskStatus.BLOCKED_NEEDS_INPUT,
        )
        await task_store.update_task(
            task.id,
            blocked_question="Need answer",
            blocked_correlation_id="corr-1",
        )

        result = await TaskResumeTool(
            root=tmp_path,
            agent_catalog=_FakeCatalog(),
            registry=registry,
            task_store=task_store,
            message_store=message_store,
        ).execute({"task_id": task.id, "message": "Use default no."}, _lead())

        assert "Resumed task" in result.output
        assert "--session-id" in captured[0]
        assert captured[0][captured[0].index("--session-id") + 1] == "child-sess"
        assert "--task-id" in captured[0]
        assert captured[0][captured[0].index("--task-id") + 1] == task.id
        live = await registry.snapshot()
        assert len(live) == 1
        assert live[0].session_id == "child-sess"
        updated = await task_store.get_task(task.id)
        assert updated.status == TaskStatus.RUNNING
        inbox = await message_store.inbox(
            to_session_id="child-sess", to_agent_name="Research"
        )
        assert len(inbox) == 1
        assert inbox[0].in_reply_to == "corr-1"
    finally:
        await message_store.close()
        await task_store.close()


async def test_task_resume_rejects_completed_task(tmp_path: Path) -> None:
    task_store = TaskStore(tmp_path / "feather.db")
    message_store = AgentMessageStore(tmp_path / "feather.db")
    await task_store.initialize()
    await message_store.initialize()
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Done",
            responsible_agent_name="research",
            responsible_session_id="child-sess",
            status=TaskStatus.COMPLETED_WITH_REPORT,
        )
        with pytest.raises(ValueError, match="already complete"):
            await TaskResumeTool(
                root=tmp_path,
                agent_catalog=_FakeCatalog(),
                registry=SubagentRegistry(),
                task_store=task_store,
                message_store=message_store,
            ).execute({"task_id": task.id, "message": None}, _lead())
    finally:
        await message_store.close()
        await task_store.close()


async def test_final_artifact_must_be_validated(tmp_path: Path) -> None:
    task_store = TaskStore(tmp_path / "feather.db")
    await task_store.initialize()
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Artifact",
            responsible_agent_name="research",
            responsible_session_id="child-sess",
            status=TaskStatus.RUNNING,
        )

        with pytest.raises(ValueError, match="Final artifact outputs must be validated"):
            await TaskOutputTool(task_store, root=tmp_path).execute(
                {
                    "task_id": task.id,
                    "kind": "artifact",
                    "path": "report.md",
                    "content": None,
                    "summary": "Report",
                    "validated": False,
                    "is_final": True,
                },
                _lead(),
            )
    finally:
        await task_store.close()


async def test_task_resume_rejects_exited_unreaped_child(tmp_path: Path) -> None:
    task_store = TaskStore(tmp_path / "feather.db")
    message_store = AgentMessageStore(tmp_path / "feather.db")
    await task_store.initialize()
    await message_store.initialize()
    registry = SubagentRegistry()
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Resume race",
            responsible_agent_name="research",
            responsible_session_id="child-sess",
            status=TaskStatus.FAILED,
        )
        await registry.register(
            _live_subagent(task.id, process=_FakeProc(returncode=0))
        )

        with pytest.raises(ValueError, match="not been reaped"):
            await TaskResumeTool(
                root=tmp_path,
                agent_catalog=_FakeCatalog(),
                registry=registry,
                task_store=task_store,
                message_store=message_store,
            ).execute({"task_id": task.id, "message": None}, _lead())
    finally:
        await message_store.close()
        await task_store.close()


async def test_task_resume_launch_failure_marks_run_and_task_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_store = TaskStore(tmp_path / "feather.db")
    message_store = AgentMessageStore(tmp_path / "feather.db")
    await task_store.initialize()
    await message_store.initialize()

    async def fail_create_subprocess_exec(*args: Any, **kwargs: Any) -> None:
        raise OSError("boom")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_create_subprocess_exec)
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Resume failure",
            responsible_agent_name="research",
            responsible_session_id="child-sess",
            status=TaskStatus.FAILED,
        )

        with pytest.raises(OSError, match="boom"):
            await TaskResumeTool(
                root=tmp_path,
                agent_catalog=_FakeCatalog(),
                registry=SubagentRegistry(),
                task_store=task_store,
                message_store=message_store,
            ).execute({"task_id": task.id, "message": None}, _lead())

        updated = await task_store.get_task(task.id)
        run = await task_store.latest_run_for_task(task.id)
        assert updated.status == TaskStatus.FAILED
        assert "failed to resume task" in (updated.error or "")
        assert run is not None
        assert run.status == TaskRunStatus.CRASHED
        assert "failed to resume task" in (run.error or "")
    finally:
        await message_store.close()
        await task_store.close()


async def test_task_stop_waits_for_reaper_when_child_already_exited(
    tmp_path: Path,
) -> None:
    task_store = TaskStore(tmp_path / "feather.db")
    message_store = AgentMessageStore(tmp_path / "feather.db")
    await task_store.initialize()
    await message_store.initialize()
    registry = SubagentRegistry()
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Stop race",
            responsible_agent_name="research",
            responsible_session_id="child-sess",
            status=TaskStatus.RUNNING,
        )
        await registry.register(
            _live_subagent(task.id, process=_FakeProc(returncode=0))
        )

        result = await TaskStopTool(task_store, registry, message_store).execute(
            {"task_id": task.id, "reason": "cancel"}, _lead()
        )

        assert "waiting for the reaper" in result.output
        updated = await task_store.get_task(task.id)
        assert updated.status == TaskStatus.RUNNING
        assert await registry.get("child-sess") is not None
    finally:
        await message_store.close()
        await task_store.close()


async def test_task_stop_kills_live_child_and_notifies_lead(tmp_path: Path) -> None:
    task_store = TaskStore(tmp_path / "feather.db")
    message_store = AgentMessageStore(tmp_path / "feather.db")
    await task_store.initialize()
    await message_store.initialize()
    registry = SubagentRegistry()
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Stop running",
            responsible_agent_name="research",
            responsible_session_id="child-sess",
            status=TaskStatus.RUNNING,
        )
        run = await task_store.create_run(
            task_id=task.id,
            session_id="child-sess",
            agent_name="research",
            pid=123,
        )
        await registry.register(
            _live_subagent(task.id, process=_FakeProc(), task_run_id=run.id)
        )

        result = await TaskStopTool(task_store, registry, message_store).execute(
            {"task_id": task.id, "reason": "user cancelled"}, _lead()
        )

        assert "Stopped task" in result.output
        updated = await task_store.get_task(task.id)
        updated_run = await task_store.get_run(run.id)
        inbox = await message_store.inbox(to_session_id="lead-sess", to_agent_name="Lead")
        assert updated.status == TaskStatus.STOPPED
        assert updated_run.status == TaskRunStatus.KILLED
        assert await registry.get("child-sess") is None
        assert len(inbox) == 1
        assert inbox[0].in_reply_to == "corr-1"
        assert "stopped by Lead" in inbox[0].body
    finally:
        await message_store.close()
        await task_store.close()


def _field(output: str, key: str) -> str:
    prefix = f"{key}: "
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise AssertionError(f"missing {key} in {output!r}")


def _live_subagent(
    task_id: str,
    *,
    process: _FakeProc,
    task_run_id: str | None = None,
) -> LiveSubagent:
    return LiveSubagent(
        session_id="child-sess",
        agent_name="research",
        parent_session_id="lead-sess",
        parent_agent_name="Lead",
        process=process,
        task_text="work",
        correlation_id="corr-1",
        task_id=task_id,
        task_run_id=task_run_id,
    )
