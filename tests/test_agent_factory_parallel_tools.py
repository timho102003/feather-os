"""Agent factory wiring tests for the Parallel AI web tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feather.config import load_app_config
from feather.core.agent_factory import AgentFactory
from feather.models import ModelTurn, ParallelConfig, ProviderRequestConfig
from feather.providers.base import BaseLLMProvider
from feather.providers.parallel_client import ParallelClient
from feather.skills.catalog import SkillCatalog
from feather.storage.session_store import SessionStore
from feather.storage.tool_output_store import ToolOutputStore
from feather.tools.parallel_extract_tool import ParallelExtractTool
from feather.tools.parallel_search_tool import ParallelSearchTool


class _StubProvider(BaseLLMProvider):
    """Provider stub that fails loudly if ever called."""

    async def complete(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        event_handler: Any = None,
        request_config: ProviderRequestConfig | None = None,
    ) -> ModelTurn:
        raise AssertionError("Construction tests should not call the provider.")


class _FakeBeta:
    def search(self, **kwargs: Any) -> Any: ...

    def extract(self, **kwargs: Any) -> Any: ...


class _FakeParallel:
    def __init__(self) -> None:
        self.beta = _FakeBeta()


def _write_configs(root: Path, *, include_parallel: bool) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    parallel_block = (
        "\nparallel:\n  api_key_env: UNUSED_PARALLEL_KEY\n  default_search_mode: fast\n  max_results: 5\n"
        if include_parallel
        else ""
    )
    (root / "config" / "app.yaml").write_text(
        f"""database:
  path: .feather/db/feather.db

storage:
  temp_directory: .feather/tmp

logging:
  path: .feather/logs/feather.log

compaction:
  enabled: true
  trigger_ratio: 0.8
  context_window_tokens: 400000
  max_output_tokens: 2000
  temperature: 0.2

skills:
  directory: .feather/skills

openai:
  api_key_env: OPENAI_API_KEY
  model: gpt-5-mini
  max_output_tokens: 4000
  temperature: 1.0
  parallel_tool_calls: true
{parallel_block}""",
        encoding="utf-8",
    )
    (root / "config" / "agents").mkdir(parents=True, exist_ok=True)
    (root / "config" / "agents" / "lead.yaml").write_text(
        """name: Lead
role: lead
personality: Direct
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
  - feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT
registered_tools:
  - web_search
  - web_fetch
""",
        encoding="utf-8",
    )
    (root / ".feather" / "skills").mkdir(parents=True, exist_ok=True)


async def test_agent_factory_registers_parallel_tools_when_client_and_config_present(
    tmp_path: Path,
) -> None:
    """With config + client wired in, lead agent picks up web_search/web_fetch."""

    _write_configs(tmp_path, include_parallel=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        app_config = load_app_config(tmp_path)
        assert app_config.parallel is not None
        parallel_client = ParallelClient(app_config.parallel, client=_FakeParallel())

        factory = AgentFactory(
            root=tmp_path,
            app_config=app_config,
            provider=_StubProvider(),
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, app_config.storage.temp_directory),
            skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
            parallel_client=parallel_client,
        )

        agent = factory.build("lead")
        registered_tool_objects = [
            agent._tool_registry.get(name) for name in agent.config.registered_tools
        ]
        registered_types = {type(obj) for obj in registered_tool_objects}
        assert ParallelSearchTool in registered_types
        assert ParallelExtractTool in registered_types
    finally:
        await session_store.close()


async def test_agent_factory_skips_parallel_tools_when_config_missing(
    tmp_path: Path,
) -> None:
    """Without a `parallel` block and no client, the web tools must be unknown."""

    _write_configs(tmp_path, include_parallel=False)
    # Strip the parallel-dependent tools so tool build doesn't fail.
    (tmp_path / "config" / "agents" / "lead.yaml").write_text(
        """name: Lead
role: lead
personality: Direct
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
  - feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT
registered_tools:
  - read_file
""",
        encoding="utf-8",
    )

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        app_config = load_app_config(tmp_path)
        assert app_config.parallel is None
        factory = AgentFactory(
            root=tmp_path,
            app_config=app_config,
            provider=_StubProvider(),
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, app_config.storage.temp_directory),
            skill_catalog=SkillCatalog((tmp_path / ".feather" / "skills").resolve()),
            parallel_client=None,
        )

        builders = factory._default_tool_builders()
        assert "web_search" not in builders
        assert "web_fetch" not in builders
    finally:
        await session_store.close()


def test_parallel_config_default_values_are_sane() -> None:
    """Ensure the default ParallelConfig mirrors the approved design."""

    config = ParallelConfig(api_key_env="X")
    assert config.default_search_mode == "fast"
    assert config.max_results == 5
    assert config.inline_full_content_threshold == 4000
