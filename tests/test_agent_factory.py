"""Tests for shared agent runtime construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feather.config import load_app_config
from feather.core.agent_factory import AgentFactory
from feather.core.default_agent import DefaultAgent
from feather.core.lead_agent import LeadAgent
from feather.models import ModelTurn, ProviderRequestConfig
from feather.providers.base import BaseLLMProvider
from feather.runtime import FeatherRuntime
from feather.skills.catalog import SkillCatalog
from feather.storage.cron_store import CronJobStore
from feather.storage.session_store import SessionStore
from feather.storage.tool_output_store import ToolOutputStore


class FakeProvider(BaseLLMProvider):
    """Minimal provider stub used for runtime construction tests."""

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
        raise AssertionError("Construction tests should not call the provider.")


async def test_agent_factory_builds_lead_agent_with_shared_base_services(tmp_path: Path) -> None:
    """The shared factory should wire lead agents through the common runtime path."""

    _write_app_config(tmp_path)
    _write_agent_config(tmp_path, "lead", name="Lead", role="lead", registered_tools=["read_file", "grep", "ask_user"])
    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    cron_store = CronJobStore(tmp_path / "feather.db")
    await cron_store.initialize()

    try:
        app_config = load_app_config(tmp_path)
        tool_output_store = ToolOutputStore(tmp_path, app_config.storage.temp_directory)
        factory = AgentFactory(
            root=tmp_path,
            app_config=app_config,
            provider=FakeProvider(),
            session_store=session_store,
            cron_store=cron_store,
            tool_output_store=tool_output_store,
            skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
        )

        agent = factory.build("lead")
        tool_names = [tool["name"] for tool in agent._tool_registry.openai_tools_for(agent.config.registered_tools)]

        assert isinstance(agent, LeadAgent)
        assert agent.config.name == "Lead"
        assert agent._tool_output_store is tool_output_store
        assert agent._compactor is not None
        assert tool_names == ["read_file", "grep", "ask_user"]
    finally:
        await cron_store.close()
        await session_store.close()


async def test_agent_factory_registers_mcp_management_tools_and_hides_proxy_until_active(
    tmp_path: Path, monkeypatch
) -> None:
    """OpenRouter agents get tiny MCP controls without prompt-listing every server."""

    from feather.mcp_client import MCPProxyTool

    _write_app_config(tmp_path)
    cfg_path = tmp_path / "config" / "app.yaml"
    cfg_path.write_text(
        cfg_path.read_text()
        + """
active_provider: openrouter
openrouter:
  api_key_env: OPEN_ROUTER_API_KEY
  model: anthropic/claude-sonnet-4.6
  max_output_tokens: 16000
  temperature: 1.0
  parallel_tool_calls: true
mcp:
  enabled: true
  servers:
    docs:
      url: https://developers.openai.com/mcp
      description: OpenAI docs
      providers: [openrouter]
      agents: [Lead]
""",
        encoding="utf-8",
    )
    _write_agent_config(
        tmp_path,
        "lead",
        name="Lead",
        role="lead",
        registered_tools=["grep"],
    )
    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    runtime = await FeatherRuntime.create(tmp_path)
    try:
        agent = runtime.build_agent("lead")
        assert "list_mcp_servers" in agent.config.registered_tools
        assert "register_mcp_server" in agent.config.registered_tools
        assert "mcp_docs" not in agent.config.registered_tools
        assert [server.label for server in agent.config.mcp_servers] == ["docs"]
        # The proxy is registered for execution once the session activates it,
        # but it is not exposed in prompts or provider tool schemas up front.
        assert isinstance(agent._tool_registry.get("mcp_docs"), MCPProxyTool)
    finally:
        await runtime.close()


async def test_agent_factory_filters_approval_requiring_mcp_servers(
    tmp_path: Path,
) -> None:
    """Unsupported MCP approval flows should not create dead registration paths."""

    _write_app_config(tmp_path)
    cfg_path = tmp_path / "config" / "app.yaml"
    cfg_path.write_text(
        cfg_path.read_text()
        + """
active_provider: openrouter
openrouter:
  api_key_env: OPEN_ROUTER_API_KEY
  model: anthropic/claude-sonnet-4.6
  max_output_tokens: 16000
  temperature: 1.0
  parallel_tool_calls: true
mcp:
  enabled: true
  servers:
    docs:
      url: https://developers.openai.com/mcp
      description: OpenAI docs
      providers: [openrouter]
      agents: [Lead]
      require_approval: always
""",
        encoding="utf-8",
    )
    _write_agent_config(
        tmp_path,
        "lead",
        name="Lead",
        role="lead",
        registered_tools=["grep"],
    )
    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        app_config = load_app_config(tmp_path)
        factory = AgentFactory(
            root=tmp_path,
            app_config=app_config,
            provider=FakeProvider(),
            session_store=session_store,
            tool_output_store=ToolOutputStore(
                tmp_path, app_config.storage.temp_directory
            ),
            skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
        )

        agent = factory.build("lead")

        assert agent.config.registered_tools == ["grep"]
        assert agent._mcp_servers == ()
        try:
            agent._tool_registry.get("mcp_docs")
        except KeyError:
            pass
        else:
            raise AssertionError("Unsupported MCP proxy should not be registered")
    finally:
        await session_store.close()


async def test_manage_memory_tool_registers_when_memory_stack_live(tmp_path: Path) -> None:
    """Wiring proof: when memory_stack.enabled is True, the lead receives manage_memory."""

    from feather.memory.reader import NoOpMemoryReader
    from feather.memory.runtime import MemoryStack
    from feather.memory.trigger import NoOpMemoryTrigger

    _write_app_config(tmp_path)
    _write_agent_config(
        tmp_path,
        "lead",
        name="Lead",
        role="lead",
        registered_tools=["read_file", "manage_memory"],
        memory_enabled=True,
    )
    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()

    class _StubService:
        async def proactive_create(self, **_kw: Any) -> Any: ...
        async def proactive_update(self, **_kw: Any) -> Any: ...
        async def proactive_delete(self, **_kw: Any) -> Any: ...

    try:
        app_config = load_app_config(tmp_path)
        # Build a "live" stack with stub service — gating only checks `enabled`
        # plus the stack object's `service` attribute.
        live_stack = MemoryStack(
            reader=NoOpMemoryReader(),
            trigger=NoOpMemoryTrigger(),
            service=_StubService(),  # type: ignore[arg-type]
            enabled=True,
        )
        factory = AgentFactory(
            root=tmp_path,
            app_config=app_config,
            provider=FakeProvider(),
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, app_config.storage.temp_directory),
            skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
            memory_stack=live_stack,
        )

        agent = factory.build("lead")
        tool_names = [
            t["name"]
            for t in agent._tool_registry.openai_tools_for(agent.config.registered_tools)
        ]
        assert "manage_memory" in tool_names
    finally:
        await session_store.close()


async def test_manage_memory_tool_dropped_when_memory_stack_disabled(tmp_path: Path) -> None:
    """Without a live memory stack the lead silently loses manage_memory (no crash)."""

    _write_app_config(tmp_path)
    _write_agent_config(
        tmp_path,
        "lead",
        name="Lead",
        role="lead",
        registered_tools=["read_file", "manage_memory"],
    )
    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        app_config = load_app_config(tmp_path)
        factory = AgentFactory(
            root=tmp_path,
            app_config=app_config,
            provider=FakeProvider(),
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, app_config.storage.temp_directory),
            skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
            memory_stack=None,  # disabled
        )

        agent = factory.build("lead")
        tool_names = [
            t["name"]
            for t in agent._tool_registry.openai_tools_for(agent.config.registered_tools)
        ]
        assert "manage_memory" not in tool_names
        assert "read_file" in tool_names  # unaffected
    finally:
        await session_store.close()


async def test_manage_memory_tool_denied_to_non_lead_agents(tmp_path: Path) -> None:
    """manage_memory is in _LEAD_ONLY_TOOLS — sub-agents listing it must lose it."""

    from feather.memory.reader import NoOpMemoryReader
    from feather.memory.runtime import MemoryStack
    from feather.memory.trigger import NoOpMemoryTrigger

    _write_app_config(tmp_path)
    _write_agent_config(
        tmp_path,
        "researcher",
        name="Researcher",
        role="research",
        registered_tools=["read_file", "manage_memory"],
        memory_enabled=True,
    )
    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()

    class _StubService:
        async def proactive_create(self, **_kw: Any) -> Any: ...
        async def proactive_update(self, **_kw: Any) -> Any: ...
        async def proactive_delete(self, **_kw: Any) -> Any: ...

    try:
        app_config = load_app_config(tmp_path)
        live_stack = MemoryStack(
            reader=NoOpMemoryReader(),
            trigger=NoOpMemoryTrigger(),
            service=_StubService(),  # type: ignore[arg-type]
            enabled=True,
        )
        factory = AgentFactory(
            root=tmp_path,
            app_config=app_config,
            provider=FakeProvider(),
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, app_config.storage.temp_directory),
            skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
            memory_stack=live_stack,
        )

        agent = factory.build("researcher")
        tool_names = [
            t["name"]
            for t in agent._tool_registry.openai_tools_for(agent.config.registered_tools)
        ]
        # Lead-only enforcement: researcher must not be handed manage_memory.
        assert "manage_memory" not in tool_names
    finally:
        await session_store.close()


async def test_agent_factory_falls_back_to_default_agent_for_unknown_roles(tmp_path: Path) -> None:
    """Future agents without a specialized subclass should still inherit the shared base loop."""

    _write_app_config(tmp_path)
    _write_agent_config(tmp_path, "worker", name="Worker", role="worker", registered_tools=["grep"])
    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    cron_store = CronJobStore(tmp_path / "feather.db")
    await cron_store.initialize()

    try:
        app_config = load_app_config(tmp_path)
        factory = AgentFactory(
            root=tmp_path,
            app_config=app_config,
            provider=FakeProvider(),
            session_store=session_store,
            cron_store=cron_store,
            tool_output_store=ToolOutputStore(tmp_path, app_config.storage.temp_directory),
            skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
        )

        agent = factory.build("worker")

        assert isinstance(agent, DefaultAgent)
        assert agent.config.role == "worker"
        assert agent._compactor is not None
    finally:
        await cron_store.close()
        await session_store.close()


async def test_feather_runtime_builds_agents_through_the_factory(tmp_path: Path) -> None:
    """Runtime bootstrap should expose agent creation without CLI-side manual wiring."""

    _write_app_config(tmp_path)
    _write_agent_config(tmp_path, "lead", name="Lead", role="lead", registered_tools=["grep"])
    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    runtime = await FeatherRuntime.create(
        tmp_path,
        provider_factory=lambda _config: FakeProvider(),
    )

    try:
        agent = runtime.build_agent("lead")
        assert isinstance(agent, LeadAgent)
        assert agent.config.registered_tools == ["grep"]
    finally:
        await runtime.close()


async def test_runtime_uses_openrouter_provider_when_active_provider_is_openrouter(
    tmp_path: Path, monkeypatch
) -> None:
    """Setting active_provider=openrouter builds an OpenRouterChatProvider."""

    from feather.providers.openrouter_provider import OpenRouterChatProvider

    _write_app_config(tmp_path)
    # Append openrouter block + active_provider flip.
    cfg_path = tmp_path / "config" / "app.yaml"
    cfg_path.write_text(
        cfg_path.read_text()
        + """
active_provider: openrouter
openrouter:
  api_key_env: OPEN_ROUTER_API_KEY
  model: anthropic/claude-sonnet-4.6
  max_output_tokens: 16000
  temperature: 1.0
  parallel_tool_calls: true
""",
        encoding="utf-8",
    )
    _write_agent_config(tmp_path, "lead", name="Lead", role="lead", registered_tools=["grep"])
    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    runtime = await FeatherRuntime.create(tmp_path)
    try:
        agent = runtime.build_agent("lead")
        assert isinstance(agent._provider, OpenRouterChatProvider)
        # Model-name resolution picks openrouter.model, not openai.model.
        assert agent._model_name == "anthropic/claude-sonnet-4.6"
    finally:
        await runtime.close()


async def test_agent_factory_per_agent_openrouter_override(
    tmp_path: Path, monkeypatch
) -> None:
    """An agent YAML `provider: openrouter` lifts that single agent off the default."""

    from feather.providers.openrouter_provider import OpenRouterChatProvider

    _write_app_config(tmp_path)
    cfg_path = tmp_path / "config" / "app.yaml"
    cfg_path.write_text(
        cfg_path.read_text()
        + """
openrouter:
  api_key_env: OPEN_ROUTER_API_KEY
  model: anthropic/claude-sonnet-4.6
  max_output_tokens: 16000
  temperature: 1.0
  parallel_tool_calls: true
""",
        encoding="utf-8",
    )
    # App-level default stays openai; one agent opts into openrouter.
    _write_agent_config(tmp_path, "lead", name="Lead", role="lead", registered_tools=["grep"])
    _write_openrouter_agent(tmp_path, "router", name="Router", role="research")
    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    runtime = await FeatherRuntime.create(
        tmp_path, provider_factory=lambda _cfg: FakeProvider()
    )
    try:
        lead = runtime.build_agent("lead")
        router = runtime.build_agent("router")
        assert isinstance(lead._provider, FakeProvider)
        assert isinstance(router._provider, OpenRouterChatProvider)
        # Per-agent explicit model override takes precedence.
        assert router._model_name == "anthropic/claude-opus-4.7"
    finally:
        await runtime.close()


async def test_agent_factory_registers_user_info_only_for_lead(tmp_path: Path) -> None:
    """``user_info`` must be registered for the lead and skipped for sub-agents."""

    from feather.profile import UserProfileStore
    from feather.core.agent_factory import _LEAD_ONLY_TOOLS

    _write_app_config(tmp_path)
    _write_agent_config(
        tmp_path,
        "lead",
        name="Lead",
        role="lead",
        registered_tools=["read_file", "user_info"],
    )
    _write_agent_config(
        tmp_path,
        "explore",
        name="Explore",
        role="explore",
        registered_tools=["read_file", "user_info"],
    )
    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    cron_store = CronJobStore(tmp_path / "feather.db")
    await cron_store.initialize()
    profile_store = UserProfileStore(tmp_path / ".feather" / "user.md")

    try:
        app_config = load_app_config(tmp_path)
        tool_output_store = ToolOutputStore(tmp_path, app_config.storage.temp_directory)
        factory = AgentFactory(
            root=tmp_path,
            app_config=app_config,
            provider=FakeProvider(),
            session_store=session_store,
            cron_store=cron_store,
            tool_output_store=tool_output_store,
            skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
            profile_store=profile_store,
        )

        assert "user_info" in _LEAD_ONLY_TOOLS
        assert "user_info" in factory._tool_builders

        lead = factory.build("lead")
        assert "user_info" in lead.config.registered_tools
        # The tool must resolve to a UserInfoTool instance bound to our store.
        from feather.tools.user_info_tool import UserInfoTool

        tool = lead._tool_registry.get("user_info")
        assert isinstance(tool, UserInfoTool)
        assert tool._store is profile_store

        explore = factory.build("explore")
        assert "user_info" not in explore.config.registered_tools
    finally:
        await cron_store.close()
        await session_store.close()


def _write_app_config(root: Path) -> None:
    """Write a minimal app config for runtime construction tests."""

    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "app.yaml").write_text(
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
  prompt_cache_key: feather-lead
  prompt_cache_retention: in_memory
  store: true
  reasoning:
    effort: low
    summary: auto
""",
        encoding="utf-8",
    )


def _write_openrouter_agent(
    root: Path, agent_file_name: str, *, name: str, role: str
) -> None:
    """Write an agent config that opts into OpenRouter with an explicit model override."""

    (root / "config" / "agents").mkdir(parents=True, exist_ok=True)
    (root / "config" / "agents" / f"{agent_file_name}.yaml").write_text(
        f"""name: {name}
role: {role}
personality: Direct
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
registered_tools:
  - grep
provider: openrouter
model: anthropic/claude-opus-4.7
""",
        encoding="utf-8",
    )


def _write_agent_config(
    root: Path,
    agent_file_name: str,
    *,
    name: str,
    role: str,
    registered_tools: list[str],
    memory_enabled: bool = False,
) -> None:
    """Write one agent config file for tests."""

    tools_yaml = "\n".join(f"  - {tool_name}" for tool_name in registered_tools)
    (root / "config" / "agents").mkdir(parents=True, exist_ok=True)
    (root / "config" / "agents" / f"{agent_file_name}.yaml").write_text(
        f"""name: {name}
role: {role}
personality: Direct
memory_enabled: {str(memory_enabled).lower()}
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
  - feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT
registered_tools:
{tools_yaml}
""",
        encoding="utf-8",
    )
