"""Tests for the CustomAgent wiring + inline_prompt rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feather.config import load_app_config
from feather.core.agent.factory import AgentFactory
from feather.core.agent.base import BaseAgent
from feather.models import ModelTurn, ProviderRequestConfig
from feather.providers.base import BaseLLMProvider
from feather.skills.catalog import SkillCatalog
from feather.storage.cron_store import CronJobStore
from feather.storage.session_store import SessionStore
from feather.storage.tool_output_store import ToolOutputStore


class _FakeProvider(BaseLLMProvider):
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
        raise AssertionError("CustomAgent wiring tests should not call the provider.")


def _write_app_config(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "app.yaml").write_text(
        """database:
  path: .feather/db/feather.db
storage:
  temp_directory: .feather/tmp
logging:
  path: .feather/logs/feather.log
  level: INFO
compaction:
  enabled: true
  trigger_ratio: 0.8
  context_window_tokens: 400000
  model:
  max_output_tokens: 2000
  temperature: 0.2
skills:
  directory: .feather/skills
scheduler:
  enabled: false
  poll_interval_seconds: 2
  failure_retry_seconds: 30
  max_due_jobs_per_tick: 10
openai:
  api_key_env: OPENAI_API_KEY
  model: gpt-5-mini
  max_output_tokens: 4000
  temperature: 1.0
  parallel_tool_calls: true
  store: true
memory:
  enabled: false
""",
        encoding="utf-8",
    )


def _write_custom_yaml(tmp_path: Path, slug: str = "reviewer-custom") -> None:
    (tmp_path / "config" / "agents").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "agents" / f"{slug}.yaml").write_text(
        """name: Reviewer
role: custom
personality: Meticulous
description: Reviews code for mistakes.
memory_enabled: false
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
inline_prompt: |
  <reviewer_identity>
  You are Feather's Reviewer sub-agent.
  </reviewer_identity>
registered_tools:
  - read_file
  - grep
  - load_skill
""",
        encoding="utf-8",
    )


def _write_custom_yaml_with_tools(
    tmp_path: Path, slug: str, tools: list[str]
) -> None:
    (tmp_path / "config" / "agents").mkdir(parents=True, exist_ok=True)
    tools_yaml = "\n".join(f"  - {t}" for t in tools)
    (tmp_path / "config" / "agents" / f"{slug}.yaml").write_text(
        f"""name: Rogue
role: custom
personality: Sneaky
description: tries to escalate
memory_enabled: false
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
inline_prompt: |
  <rogue_identity>I try to do lead things.</rogue_identity>
registered_tools:
{tools_yaml}
""",
        encoding="utf-8",
    )


async def test_factory_strips_lead_only_tools_from_non_lead_agent(
    tmp_path: Path, caplog
) -> None:
    """A custom YAML that lists spawn_agent/cron/ask_user should have those stripped."""

    import logging

    _write_app_config(tmp_path)
    _write_custom_yaml_with_tools(
        tmp_path,
        "rogue-custom",
        [
            "read_file",
            "spawn_agent",
            "create_cron",
            "ask_user",
            "load_skill",
        ],
    )
    (tmp_path / ".feather" / "skills").mkdir(parents=True, exist_ok=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    cron_store = CronJobStore(tmp_path / "feather.db")
    await cron_store.initialize()

    caplog.set_level(logging.WARNING)
    try:
        app_config = load_app_config(tmp_path)
        factory = AgentFactory(
            root=tmp_path,
            app_config=app_config,
            provider=_FakeProvider(),
            session_store=session_store,
            cron_store=cron_store,
            tool_output_store=ToolOutputStore(tmp_path, app_config.storage.temp_directory),
            skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
        )
        agent = factory.build("rogue-custom")
        # Lead-only tools must not appear in the effective tool list.
        assert "spawn_agent" not in agent.config.registered_tools
        assert "create_cron" not in agent.config.registered_tools
        assert "ask_user" not in agent.config.registered_tools
        # Legitimate tools survived.
        assert agent.config.registered_tools == ["read_file", "load_skill"]
        # The openai_tools_for() path reads from the same filtered list, so no
        # KeyError for dropped tools.
        names = [t["name"] for t in agent._tool_registry.openai_tools_for(agent.config.registered_tools)]
        assert names == ["read_file", "load_skill"]
    finally:
        await cron_store.close()
        await session_store.close()

    warnings = [rec.message for rec in caplog.records]
    assert any("spawn_agent" in msg and "lacks capability can_spawn" in msg for msg in warnings)
    assert any("create_cron" in msg and "lacks capability can_schedule" in msg for msg in warnings)
    assert any("ask_user" in msg and "lacks capability can_message_user" in msg for msg in warnings)


async def test_factory_routes_role_custom_to_custom_agent(tmp_path: Path) -> None:
    _write_app_config(tmp_path)
    _write_custom_yaml(tmp_path)
    (tmp_path / ".feather" / "skills").mkdir(parents=True, exist_ok=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    cron_store = CronJobStore(tmp_path / "feather.db")
    await cron_store.initialize()

    try:
        app_config = load_app_config(tmp_path)
        factory = AgentFactory(
            root=tmp_path,
            app_config=app_config,
            provider=_FakeProvider(),
            session_store=session_store,
            cron_store=cron_store,
            tool_output_store=ToolOutputStore(tmp_path, app_config.storage.temp_directory),
            skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
        )
        agent = factory.build("reviewer-custom")
        assert isinstance(agent, BaseAgent) and agent.config.role == "custom"
        assert agent.config.role == "custom"
        assert agent.config.description == "Reviews code for mistakes."
        assert "<reviewer_identity>" in agent.config.inline_prompt

        # Prompt should contain the inline body.
        rendered = agent._prompt_builder.build(agent.config, [])
        assert "<reviewer_identity>" in rendered
    finally:
        await cron_store.close()
        await session_store.close()
