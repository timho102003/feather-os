"""Tests for background cron-job dispatch."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from feather.config import load_app_config
from feather.core.agent_factory import AgentFactory
from feather.core.cron_scheduler import CronScheduler
from feather.core.session_run_coordinator import SessionRunCoordinator
from feather.models import CronScheduleType, ModelTurn, ProviderRequestConfig, RuntimeEvent
from feather.providers.base import BaseLLMProvider
from feather.skills.catalog import SkillCatalog
from feather.storage.cron_store import CronJobStore
from feather.storage.session_store import SessionStore
from feather.storage.tool_output_store import ToolOutputStore


class FakeProvider(BaseLLMProvider):
    """Provider stub that records the scheduled input it receives."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        event_handler=None,
        request_config: ProviderRequestConfig | None = None,
    ) -> ModelTurn:
        self.calls.append(
            {
                "instructions": instructions,
                "input_items": input_items,
                "previous_response_id": previous_response_id,
            }
        )
        if event_handler is not None:
            for character in "Scheduled work handled.":
                event_handler(RuntimeEvent(kind="assistant_text_delta", text=character))
        return ModelTurn(response_id="resp-scheduled", output_text="Scheduled work handled.", tool_calls=[])


async def test_cron_scheduler_dispatches_due_jobs_into_the_agent_loop(tmp_path: Path) -> None:
    """Due cron jobs should be injected into the lead session and processed normally."""

    _write_app_config(tmp_path)
    _write_agent_config(tmp_path)
    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    cron_store = CronJobStore(tmp_path / "feather.db")
    await session_store.initialize()
    await cron_store.initialize()

    try:
        provider = FakeProvider()
        app_config = load_app_config(tmp_path)
        factory = AgentFactory(
            root=tmp_path,
            app_config=app_config,
            provider=provider,
            session_store=session_store,
            cron_store=cron_store,
            tool_output_store=ToolOutputStore(tmp_path, app_config.storage.temp_directory),
            skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
            run_coordinator=SessionRunCoordinator(),
        )
        lead_agent = factory.build("lead")
        session_id = await lead_agent.create_session()
        due_at = datetime.now(UTC) + timedelta(seconds=2)
        job = await cron_store.create_job(
            session_id=session_id,
            agent_key="lead",
            name="Reminder",
            schedule_type=CronScheduleType.ONCE,
            schedule_value=due_at.isoformat(),
            timezone="UTC",
            prompt="Tell the user the reminder fired.",
        )

        events: list[str] = []
        scheduler = CronScheduler(
            config=app_config.scheduler,
            cron_store=cron_store,
            agent_factory=factory,
            event_handler_resolver=lambda current_session_id: (
                (lambda event: events.append(f"{current_session_id}:{event.kind}:{event.text or ''}"))
                if current_session_id == session_id
                else None
            ),
        )

        dispatched = await scheduler.run_pending(now=due_at + timedelta(seconds=1))

        updated_job = await cron_store.get_job(job.id)
        messages = await session_store.list_messages(session_id)

        assert dispatched == 1
        assert updated_job.status.value == "completed"
        assert provider.calls
        assert "<scheduled_task_trigger>" in provider.calls[0]["input_items"][0]["content"][0]["text"]
        assert messages[0].content.startswith("<scheduled_task_trigger>")
        assert messages[1].content == "Scheduled work handled."
        assert any("scheduled_task_triggered" in event for event in events)
        assert any("scheduled_task_completed" in event for event in events)
    finally:
        await cron_store.close()
        await session_store.close()


def _write_app_config(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "app.yaml").write_text(
        """database:
  path: feather.db

storage:
  temp_directory: .feather/tmp

logging:
  path: .feather/logs/feather.log
  level: INFO

compaction:
  enabled: false
  trigger_ratio: 0.8
  context_window_tokens: 400000
  model:
  max_output_tokens: 2000
  temperature: 0.2

skills:
  directory: .feather/skills

scheduler:
  enabled: true
  poll_interval_seconds: 2
  failure_retry_seconds: 30
  max_due_jobs_per_tick: 10

openai:
  api_key_env: OPENAI_API_KEY
  model: gpt-5-mini
  max_output_tokens: 4000
  temperature: 1.0
  parallel_tool_calls: true
  prompt_cache_key:
  prompt_cache_retention:
  store: true
  reasoning:
    effort: low
    summary: auto
""",
        encoding="utf-8",
    )


def _write_agent_config(root: Path) -> None:
    (root / "config" / "agents").mkdir(parents=True, exist_ok=True)
    (root / "config" / "agents" / "lead.yaml").write_text(
        """name: Lead
role: lead
personality: Direct
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
  - feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT
registered_tools: []
""",
        encoding="utf-8",
    )
