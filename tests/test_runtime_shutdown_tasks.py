"""Tests for durable task cleanup during runtime shutdown."""

from __future__ import annotations

from pathlib import Path

from feather.core.subagents.registry import LiveSubagent, SubagentRegistry
from feather.models import TaskRunStatus, TaskStatus
from feather.runtime import FeatherRuntime
from feather.storage.task_store import TaskStore


class _FakeProc:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.pid = 456
        self.returncode = returncode

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


async def test_shutdown_marks_live_task_stopped_and_run_killed(tmp_path: Path) -> None:
    task_store = TaskStore(tmp_path / "feather.db")
    await task_store.initialize()
    registry = SubagentRegistry()
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Running child",
            responsible_agent_name="research",
            responsible_session_id="child-sess",
            status=TaskStatus.RUNNING,
        )
        run = await task_store.create_run(
            task_id=task.id,
            session_id="child-sess",
            agent_name="research",
            pid=456,
        )
        await registry.register(
            _live_subagent(
                task.id,
                task_run_id=run.id,
                process=_FakeProc(),
            )
        )
        runtime = object.__new__(FeatherRuntime)
        runtime._task_store = task_store
        runtime._subagent_registry = registry

        await FeatherRuntime._terminate_live_subagents(runtime)

        updated = await task_store.get_task(task.id)
        updated_run = await task_store.get_run(run.id)
        assert updated.status == TaskStatus.STOPPED
        assert updated.error == "runtime shutdown terminated live sub-agent"
        assert updated_run.status == TaskRunStatus.KILLED
        assert await registry.snapshot() == []
    finally:
        await task_store.close()


async def test_shutdown_marks_exited_unreaped_task_failed(tmp_path: Path) -> None:
    task_store = TaskStore(tmp_path / "feather.db")
    await task_store.initialize()
    registry = SubagentRegistry()
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Exited child",
            responsible_agent_name="research",
            responsible_session_id="child-sess",
            status=TaskStatus.RUNNING,
        )
        run = await task_store.create_run(
            task_id=task.id,
            session_id="child-sess",
            agent_name="research",
            pid=456,
        )
        await registry.register(
            _live_subagent(
                task.id,
                task_run_id=run.id,
                process=_FakeProc(returncode=0),
            )
        )
        runtime = object.__new__(FeatherRuntime)
        runtime._task_store = task_store
        runtime._subagent_registry = registry

        await FeatherRuntime._terminate_live_subagents(runtime)

        updated = await task_store.get_task(task.id)
        updated_run = await task_store.get_run(run.id)
        assert updated.status == TaskStatus.FAILED
        assert updated.error == "runtime shutdown before reaper delivered final report"
        assert updated_run.status == TaskRunStatus.EXITED
        assert await registry.snapshot() == []
    finally:
        await task_store.close()


def _live_subagent(
    task_id: str,
    *,
    task_run_id: str,
    process: _FakeProc,
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
