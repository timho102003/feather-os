"""Tests that the AgentFactory routes sub-agent roles to the right classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feather.config import load_app_config
from feather.core.agent_factory import AgentFactory
from feather.core.sub_agents import ExploreAgent, ResearchAgent, ValidateAgent
from feather.core.subagent_registry import SubagentRegistry
from feather.models import ModelTurn, ProviderRequestConfig
from feather.providers.base import BaseLLMProvider
from feather.skills.catalog import SkillCatalog
from feather.storage.agent_message_store import AgentMessageStore
from feather.storage.cron_store import CronJobStore
from feather.storage.session_store import SessionStore
from feather.storage.task_store import TaskStore
from feather.storage.tool_output_store import ToolOutputStore


class FakeProvider(BaseLLMProvider):
    """Minimal provider stub used for factory wiring tests."""

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
        raise AssertionError("Factory wiring tests should not call the provider.")


class _StubParallelClient:
    """Stand-in for ParallelClient that never touches the network."""

    async def search(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("Factory wiring tests should not call search().")

    async def extract(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("Factory wiring tests should not call extract().")


def _copy_agent_config(repo_root: Path, tmp_path: Path, name: str) -> None:
    """Copy a real packaged agent YAML into the tmp-path config fixture.

    Reads from ``feather._resources.config.agents`` so the fixture stays
    in sync with whatever the wheel ships, rather than trusting a path
    that no longer exists at the repo root after the package layout move.
    """

    from feather.resources import packaged_agent_yaml_text

    dest = tmp_path / "config" / "agents" / f"{name}.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(packaged_agent_yaml_text(name), encoding="utf-8")


def _write_minimal_app_config(tmp_path: Path) -> None:
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


async def _build_factory(
    tmp_path: Path, *, include_parallel: bool = False
) -> tuple[AgentFactory, SessionStore, CronJobStore, AgentMessageStore, TaskStore]:
    (tmp_path / ".feather" / "skills").mkdir(parents=True, exist_ok=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    cron_store = CronJobStore(tmp_path / "feather.db")
    await cron_store.initialize()
    message_store = AgentMessageStore(tmp_path / "feather.db")
    await message_store.initialize()
    task_store = TaskStore(tmp_path / "feather.db")
    await task_store.initialize()
    app_config = load_app_config(tmp_path)
    factory = AgentFactory(
        root=tmp_path,
        app_config=app_config,
        provider=FakeProvider(),
        session_store=session_store,
        cron_store=cron_store,
        tool_output_store=ToolOutputStore(tmp_path, app_config.storage.temp_directory),
        skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
        parallel_client=_StubParallelClient() if include_parallel else None,  # type: ignore[arg-type]
        agent_message_store=message_store,
        task_store=task_store,
        subagent_registry=SubagentRegistry(),
    )
    return factory, session_store, cron_store, message_store, task_store


async def test_factory_routes_explore_role_to_explore_agent(tmp_path: Path) -> None:
    """Explore role should resolve to ExploreAgent with the expected tool set."""

    _write_minimal_app_config(tmp_path)
    _copy_agent_config(Path.cwd(), tmp_path, "explore")
    # Explorer registers web tools so it can do quick external disambiguation.
    (tmp_path / "config" / "app.yaml").write_text(
        (tmp_path / "config" / "app.yaml").read_text(encoding="utf-8")
        + "\nparallel:\n  api_key_env: PARALLEL_API_KEY\n  default_search_mode: fast\n  max_results: 5\n",
        encoding="utf-8",
    )
    factory, session_store, cron_store, message_store, task_store = await _build_factory(tmp_path, include_parallel=True)
    try:
        agent = factory.build("explore")
        names = [tool["name"] for tool in agent._tool_registry.openai_tools_for(agent.config.registered_tools)]
        assert isinstance(agent, ExploreAgent)
        assert names == [
            "read_file",
            "read_pdf",
            "write_file",
            "grep",
            "bash",
            "web_search",
            "web_fetch",
            "load_skill",
            "send_message",
            "task_get",
            "task_update",
            "task_output",
            "request_input",
        ]
        assert agent.config.memory_enabled is False
    finally:
        await cron_store.close()
        await message_store.close()
        await task_store.close()
        await session_store.close()


async def test_factory_routes_research_role_to_research_agent(tmp_path: Path) -> None:
    """Research role should resolve to ResearchAgent when Parallel is available."""

    _write_minimal_app_config(tmp_path)
    _copy_agent_config(Path.cwd(), tmp_path, "research")
    # Research YAML registers web tools that only exist when Parallel config is present.
    (tmp_path / "config" / "app.yaml").write_text(
        (tmp_path / "config" / "app.yaml").read_text(encoding="utf-8")
        + "\nparallel:\n  api_key_env: PARALLEL_API_KEY\n  default_search_mode: fast\n  max_results: 5\n",
        encoding="utf-8",
    )
    factory, session_store, cron_store, message_store, task_store = await _build_factory(tmp_path, include_parallel=True)
    try:
        agent = factory.build("research")
        names = [tool["name"] for tool in agent._tool_registry.openai_tools_for(agent.config.registered_tools)]
        assert isinstance(agent, ResearchAgent)
        assert names == [
            "web_search",
            "web_fetch",
            "read_file",
            "read_pdf",
            "write_file",
            "load_skill",
            "send_message",
            "task_get",
            "task_update",
            "task_output",
            "request_input",
        ]
    finally:
        await cron_store.close()
        await message_store.close()
        await task_store.close()
        await session_store.close()


async def test_factory_routes_validate_role_to_validate_agent(tmp_path: Path) -> None:
    _write_minimal_app_config(tmp_path)
    _copy_agent_config(Path.cwd(), tmp_path, "validate")
    factory, session_store, cron_store, message_store, task_store = await _build_factory(tmp_path)
    try:
        agent = factory.build("validate")
        names = [tool["name"] for tool in agent._tool_registry.openai_tools_for(agent.config.registered_tools)]
        assert isinstance(agent, ValidateAgent)
        assert names == [
            "bash",
            "read_file",
            "read_pdf",
            "write_file",
            "grep",
            "load_skill",
            "send_message",
            "task_get",
            "task_update",
            "task_output",
            "request_input",
        ]
    finally:
        await cron_store.close()
        await message_store.close()
        await task_store.close()
        await session_store.close()


async def test_factory_registers_spawn_agent_tool_for_lead_when_listed(tmp_path: Path) -> None:
    """Lead YAML that lists `spawn_agent` should get the SpawnAgentTool wired in."""

    _write_minimal_app_config(tmp_path)
    (tmp_path / "config" / "agents").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "agents" / "lead.yaml").write_text(
        """name: Lead
role: lead
personality: Direct
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
  - feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT
registered_tools:
  - spawn_agent
""",
        encoding="utf-8",
    )
    factory, session_store, cron_store, message_store, task_store = await _build_factory(tmp_path)
    try:
        agent = factory.build("lead")
        names = [tool["name"] for tool in agent._tool_registry.openai_tools_for(agent.config.registered_tools)]
        assert names == ["spawn_agent"]
    finally:
        await cron_store.close()
        await message_store.close()
        await task_store.close()
        await session_store.close()
