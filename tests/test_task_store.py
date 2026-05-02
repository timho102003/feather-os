"""Tests for durable task storage."""

from __future__ import annotations

from pathlib import Path

from feather.models import TaskOutputKind, TaskRunStatus, TaskStatus
from feather.storage.task_store import TaskStore


async def test_task_store_tracks_plan_task_run_output_and_events(tmp_path: Path) -> None:
    """Task storage should persist the full task lifecycle."""

    store = TaskStore(tmp_path / "feather.db")
    await store.initialize()
    try:
        plan = await store.create_plan(
            filepath=".feather/artifacts/plan/demo.md",
            title="Demo plan",
            summary="test",
            lead_session_id="lead-sess",
        )
        task = await store.create_task(
            plan_id=plan.id,
            lead_session_id="lead-sess",
            title="Research prices",
            description="Find comps",
            success_criteria="Report with sources",
            required_outputs=["report.md"],
            responsible_agent_name="research",
            responsible_session_id="child-sess",
        )
        run = await store.create_run(
            task_id=task.id,
            session_id="child-sess",
            agent_name="research",
            pid=123,
        )
        output = await store.add_output(
            task_id=task.id,
            kind=TaskOutputKind.REPORT,
            path=".feather/artifacts/outputs/report.md",
            content=None,
            summary="Final report",
            created_by_session_id="child-sess",
            validated=True,
            is_final=True,
        )
        await store.finish_run(
            run.id,
            status=TaskRunStatus.EXITED,
            exit_code=0,
            envelope_status="completed",
        )
        updated = await store.update_task(
            task.id,
            status=TaskStatus.COMPLETED_WITH_ARTIFACTS,
        )

        assert updated.status == TaskStatus.COMPLETED_WITH_ARTIFACTS
        assert await store.has_final_output(task.id)
        assert (await store.list_outputs(task.id))[0].id == output.id
        listed = await store.list_tasks(lead_session_id="lead-sess")
        assert [item.id for item in listed] == [task.id]
        events = await store.list_events(task.id)
        assert any(event.event_type == "run_finished" for event in events)
    finally:
        await store.close()


async def test_find_task_by_session_prefers_non_terminal_task(tmp_path: Path) -> None:
    """Resumable/live lookups should prefer the newest non-terminal task."""

    store = TaskStore(tmp_path / "feather.db")
    await store.initialize()
    try:
        done = await store.create_task(
            lead_session_id="lead",
            title="done",
            responsible_session_id="same-session",
            status=TaskStatus.COMPLETED_WITH_REPORT,
        )
        blocked = await store.create_task(
            lead_session_id="lead",
            title="blocked",
            responsible_session_id="same-session",
            status=TaskStatus.BLOCKED_NEEDS_INPUT,
        )

        found = await store.find_task_by_session("same-session")

        assert found is not None
        assert found.id == blocked.id
        assert found.id != done.id
    finally:
        await store.close()
