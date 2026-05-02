"""Tests for cron management tools."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from feather.models import ToolExecutionContext
from feather.storage.cron_store import CronJobStore
from feather.storage.session_store import SessionStore
from feather.tools.cron_tools import CreateCronTool, DeleteCronTool, ListCronsTool, UpdateCronTool


async def test_cron_tools_manage_jobs_inside_the_active_session(tmp_path: Path) -> None:
    """Cron tools should create, inspect, update, and delete session-scoped jobs."""

    db_path = tmp_path / "feather.db"
    session_store = SessionStore(db_path)
    cron_store = CronJobStore(db_path)
    await session_store.initialize()
    await cron_store.initialize()

    try:
        session = await session_store.create_session("Lead")
        context = ToolExecutionContext(session_id=session.id, agent_name="Lead")
        create_tool = CreateCronTool(cron_store)
        update_tool = UpdateCronTool(cron_store)
        list_tool = ListCronsTool(cron_store)
        delete_tool = DeleteCronTool(cron_store)

        schedule_value = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
        created = await create_tool.execute(
            {
                "name": "Follow up",
                "schedule_type": "once",
                "schedule_value": schedule_value,
                "timezone": "UTC",
                "prompt": "Ask for the deployment status.",
            },
            context,
        )
        assert "Created cron job." in created.output

        listed = await list_tool.execute(
            {
                "status": "active",
                "limit": 20,
            },
            context,
        )
        assert "Follow up" in listed.output

        updated = await update_tool.execute(
            {
                "job_id": None,
                "job_name": "Follow up",
                "new_name": "Deployment follow up",
                "schedule_type": None,
                "schedule_value": None,
                "timezone": None,
                "prompt": "Ask whether the deployment finished successfully.",
                "status": "paused",
            },
            context,
        )
        assert "Updated cron job." in updated.output
        assert "Deployment follow up" in updated.output
        assert "paused" in updated.output

        jobs = await cron_store.list_jobs(session_id=session.id)
        job_id = jobs[0].id
        deleted = await delete_tool.execute(
            {
                "job_id": job_id,
                "job_name": None,
            },
            context,
        )
        assert "Deleted cron job." in deleted.output
        assert await cron_store.list_jobs(session_id=session.id) == []
    finally:
        await cron_store.close()
        await session_store.close()
