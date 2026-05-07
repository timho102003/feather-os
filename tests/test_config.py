"""Tests for YAML config loading."""

from __future__ import annotations

from pathlib import Path

from feather.config import load_agent_config, load_app_config


_MIN_APP_YAML = """database: {path: .feather/db/feather.db}
storage: {temp_directory: .feather/tmp}
logging: {path: .feather/logs/feather.log, level: INFO}
compaction: {}
skills: {directory: .feather/skills}
scheduler: {}
openai:
  api_key_env: OPENAI_API_KEY
  model: gpt-5-mini
  max_output_tokens: 4000
  temperature: 1.0
  parallel_tool_calls: true
"""


def test_app_config_defaults_active_provider_to_openai(tmp_path: Path) -> None:
    """Omitted ``active_provider`` stays on OpenAI; ``openrouter`` block stays ``None``."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "app.yaml").write_text(_MIN_APP_YAML, encoding="utf-8")

    cfg = load_app_config(tmp_path)
    assert cfg.active_provider == "openai"
    assert cfg.openrouter is None


def test_app_config_parses_openrouter_block(tmp_path: Path) -> None:
    """active_provider=openrouter + openrouter block populates OpenRouterConfig."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "app.yaml").write_text(
        _MIN_APP_YAML + """active_provider: openrouter
openrouter:
  api_key_env: OPEN_ROUTER_API_KEY
  http_referer: https://example
  app_title: Tester
  model: anthropic/claude-sonnet-4.6
  max_output_tokens: 32000
  temperature: 1.0
  parallel_tool_calls: true
  reasoning: {effort: medium}
  provider_preferences: {require_parameters: true, allow_fallbacks: true}
  fallback_models: ["openai/gpt-5.2-mini"]
  cache_strategy: anthropic_breakpoint
  stream_idle_timeout_seconds: 90
  request_timeout_seconds: 120
  max_attempts: 3
  supports_multimodal: false
""",
        encoding="utf-8",
    )

    cfg = load_app_config(tmp_path)
    assert cfg.active_provider == "openrouter"
    assert cfg.openrouter is not None
    or_cfg = cfg.openrouter
    assert or_cfg.api_key_env == "OPEN_ROUTER_API_KEY"
    assert or_cfg.http_referer == "https://example"
    assert or_cfg.app_title == "Tester"
    assert or_cfg.model == "anthropic/claude-sonnet-4.6"
    assert or_cfg.max_output_tokens == 32_000
    assert or_cfg.reasoning is not None and or_cfg.reasoning.effort == "medium"
    assert or_cfg.provider_preferences == {
        "require_parameters": True,
        "allow_fallbacks": True,
    }
    assert or_cfg.fallback_models == ["openai/gpt-5.2-mini"]
    assert or_cfg.cache_strategy == "anthropic_breakpoint"
    assert or_cfg.max_attempts == 3
    assert or_cfg.supports_multimodal is False


def test_app_config_parses_openrouter_tracing_block(tmp_path: Path) -> None:
    """The optional ``openrouter.tracing`` block populates OpenRouterTracingConfig."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "app.yaml").write_text(
        _MIN_APP_YAML + """active_provider: openrouter
openrouter:
  api_key_env: OPEN_ROUTER_API_KEY
  model: anthropic/claude-sonnet-4.6
  max_output_tokens: 32000
  temperature: 1.0
  parallel_tool_calls: true
  tracing:
    enabled: true
    user: ops@example.com
    metadata:
      deployment: prod
      build_sha: abc123
""",
        encoding="utf-8",
    )

    cfg = load_app_config(tmp_path)
    assert cfg.openrouter is not None
    tracing = cfg.openrouter.tracing
    assert tracing is not None
    assert tracing.enabled is True
    assert tracing.user == "ops@example.com"
    assert tracing.metadata == {"deployment": "prod", "build_sha": "abc123"}


def test_app_config_openrouter_tracing_omitted_defaults_to_none(tmp_path: Path) -> None:
    """When ``openrouter.tracing`` is absent the field stays None (back-compat)."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "app.yaml").write_text(
        _MIN_APP_YAML + """active_provider: openrouter
openrouter:
  api_key_env: OPEN_ROUTER_API_KEY
  model: anthropic/claude-sonnet-4.6
  max_output_tokens: 32000
  temperature: 1.0
  parallel_tool_calls: true
""",
        encoding="utf-8",
    )

    cfg = load_app_config(tmp_path)
    assert cfg.openrouter is not None
    assert cfg.openrouter.tracing is None


def test_app_config_parses_claude_block(tmp_path: Path) -> None:
    """active_provider=claude + claude block populates ClaudeConfig."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "app.yaml").write_text(
        _MIN_APP_YAML + """active_provider: claude
claude:
  api_key_env: ANTHROPIC_API_KEY
  base_url: https://api.anthropic.com
  anthropic_version: "2023-06-01"
  anthropic_beta:
    - extended-cache-ttl-2025-04-11
  model: claude-opus-4-7
  max_output_tokens: 32000
  temperature: 1.0
  parallel_tool_calls: true
  thinking: {type: enabled, budget_tokens: 4000}
  cache_strategy: anthropic_breakpoint
  stream_idle_timeout_seconds: 90
  request_timeout_seconds: 120
  max_attempts: 3
  supports_multimodal: true
  max_stream_wall_seconds: 600
""",
        encoding="utf-8",
    )

    cfg = load_app_config(tmp_path)
    assert cfg.active_provider == "claude"
    assert cfg.claude is not None
    cl = cfg.claude
    assert cl.api_key_env == "ANTHROPIC_API_KEY"
    assert cl.anthropic_version == "2023-06-01"
    assert cl.anthropic_beta == ("extended-cache-ttl-2025-04-11",)
    assert cl.model == "claude-opus-4-7"
    assert cl.max_output_tokens == 32_000
    assert cl.thinking is not None
    assert cl.thinking.type == "enabled"
    assert cl.thinking.budget_tokens == 4000
    assert cl.cache_strategy == "anthropic_breakpoint"
    assert cl.max_attempts == 3
    assert cl.supports_multimodal is True


def test_app_config_claude_block_omitted_defaults_to_none(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "app.yaml").write_text(_MIN_APP_YAML, encoding="utf-8")
    cfg = load_app_config(tmp_path)
    assert cfg.claude is None


def test_app_config_claude_anthropic_beta_accepts_string_or_list(tmp_path: Path) -> None:
    """``anthropic_beta`` should normalize a single string into a one-tuple."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "app.yaml").write_text(
        _MIN_APP_YAML + """active_provider: claude
claude:
  api_key_env: ANTHROPIC_API_KEY
  model: claude-opus-4-7
  max_output_tokens: 32000
  temperature: 1.0
  parallel_tool_calls: true
  anthropic_beta: extended-cache-ttl-2025-04-11
""",
        encoding="utf-8",
    )
    cfg = load_app_config(tmp_path)
    assert cfg.claude is not None
    assert cfg.claude.anthropic_beta == ("extended-cache-ttl-2025-04-11",)


def test_agent_config_parses_provider_override(tmp_path: Path) -> None:
    """Agent YAML ``provider`` and ``model`` fields land on AgentConfig."""

    (tmp_path / "config" / "agents").mkdir(parents=True)
    (tmp_path / "config" / "agents" / "override.yaml").write_text(
        """name: Override
role: lead
personality: terse
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
registered_tools: [read_file]
provider: openrouter
model: anthropic/claude-opus-4.7
""",
        encoding="utf-8",
    )

    cfg = load_agent_config(tmp_path, "override")
    assert cfg.provider == "openrouter"
    assert cfg.model == "anthropic/claude-opus-4.7"


def test_agent_config_provider_omitted_defaults_to_none(tmp_path: Path) -> None:
    (tmp_path / "config" / "agents").mkdir(parents=True)
    (tmp_path / "config" / "agents" / "plain.yaml").write_text(
        """name: Plain
role: lead
personality: terse
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
registered_tools: [read_file]
""",
        encoding="utf-8",
    )

    cfg = load_agent_config(tmp_path, "plain")
    assert cfg.provider is None
    assert cfg.model is None
    assert cfg.reasoning is None


def test_agent_config_reads_reasoning_block(tmp_path: Path) -> None:
    (tmp_path / "config" / "agents").mkdir(parents=True)
    (tmp_path / "config" / "agents" / "thinker.yaml").write_text(
        """name: Thinker
role: lead
personality: deliberate
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
registered_tools: [read_file]
reasoning:
  effort: high
  summary: detailed
""",
        encoding="utf-8",
    )

    cfg = load_agent_config(tmp_path, "thinker")
    assert cfg.reasoning is not None
    assert cfg.reasoning.effort == "high"
    assert cfg.reasoning.summary == "detailed"


def test_load_app_config_reads_openai_model_and_reasoning(tmp_path: Path) -> None:
    """App config should load the model and reasoning settings from YAML."""

    (tmp_path / "config").mkdir(parents=True)
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

    config = load_app_config(tmp_path)

    assert config.openai.model == "gpt-5-mini"
    assert config.openai.prompt_cache_retention == "in_memory"
    assert config.storage.temp_directory == ".feather/tmp"
    assert config.compaction.enabled is True
    assert config.compaction.trigger_ratio == 0.8
    assert config.compaction.context_window_tokens == 400000
    assert config.scheduler.enabled is True
    assert config.scheduler.poll_interval_seconds == 2
    assert config.scheduler.failure_retry_seconds == 30
    assert config.openai.reasoning is not None
    assert config.openai.reasoning.effort == "low"
    assert config.openai.reasoning.summary == "auto"
    assert config.parallel is None


def test_load_app_config_reads_parallel_block(tmp_path: Path) -> None:
    """App config should surface the optional Parallel AI web-tools block."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "app.yaml").write_text(
        """database:
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

parallel:
  api_key_env: PARALLEL_API_KEY
  default_search_mode: fast
  max_results: 7
  inline_full_content_threshold: 2048
""",
        encoding="utf-8",
    )

    config = load_app_config(tmp_path)

    assert config.parallel is not None
    assert config.parallel.api_key_env == "PARALLEL_API_KEY"
    assert config.parallel.default_search_mode == "fast"
    assert config.parallel.max_results == 7
    assert config.parallel.inline_full_content_threshold == 2048


def test_load_agent_config_reads_prompt_modules_and_supports_legacy_fallback(tmp_path: Path) -> None:
    """Agent config loading should prefer `prompt_modules` and still support the old single-module key."""

    (tmp_path / "config" / "agents").mkdir(parents=True)
    (tmp_path / "config" / "agents" / "lead.yaml").write_text(
        """name: Lead
role: lead
personality: Direct
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
  - feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT
registered_tools:
  - grep
""",
        encoding="utf-8",
    )
    (tmp_path / "config" / "agents" / "legacy.yaml").write_text(
        """name: Legacy
role: worker
personality: Direct
system_prompt_module: feather.core.prompts.default_agent_prompt:DEFAULT_AGENT_PROMPT
registered_tools:
  - grep
""",
        encoding="utf-8",
    )

    lead = load_agent_config(tmp_path, "lead")
    legacy = load_agent_config(tmp_path, "legacy")

    assert lead.prompt_modules == [
        "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
        "feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT",
    ]
    assert legacy.prompt_modules == [
        "feather.core.prompts.default_agent_prompt:DEFAULT_AGENT_PROMPT",
    ]
