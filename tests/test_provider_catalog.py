"""Tests for the declarative provider spec catalog."""

from __future__ import annotations

from feather.models.config_models import (
    ClaudeConfig,
    CompactionConfig,
    DatabaseConfig,
    LoggingConfig,
    OpenAIConfig,
    OpenRouterConfig,
    SchedulerConfig,
    SkillsConfig,
    StorageConfig,
    AppConfig,
)
from feather.providers.catalog import PROVIDER_NAMES, ProviderSpec, provider_spec
from feather.providers.claude_provider import ClaudeMessagesProvider
from feather.providers.openai_provider import OpenAIResponsesProvider
from feather.providers.openrouter_provider import OpenRouterChatProvider


# ---------------------------------------------------------------------------
# Minimal AppConfig factory helpers
# ---------------------------------------------------------------------------

def _openai_config() -> OpenAIConfig:
    return OpenAIConfig(
        api_key_env="OPENAI_API_KEY",
        model="gpt-5-mini",
        max_output_tokens=4000,
        temperature=1.0,
        parallel_tool_calls=True,
    )


def _app_config(
    *,
    openrouter: OpenRouterConfig | None = None,
    claude: ClaudeConfig | None = None,
) -> AppConfig:
    return AppConfig(
        database=DatabaseConfig(path=".feather/db/feather.db"),
        storage=StorageConfig(temp_directory=".feather/tmp"),
        logging=LoggingConfig(path=".feather/logs/feather.log", level="INFO"),
        compaction=CompactionConfig(
            enabled=True,
            trigger_ratio=0.8,
            context_window_tokens=400_000,
        ),
        skills=SkillsConfig(directory=".feather/skills"),
        scheduler=SchedulerConfig(
            enabled=True,
            poll_interval_seconds=2.0,
            failure_retry_seconds=30.0,
            max_due_jobs_per_tick=10,
        ),
        openai=_openai_config(),
        openrouter=openrouter,
        claude=claude,
    )


# ---------------------------------------------------------------------------
# Catalog shape
# ---------------------------------------------------------------------------

def test_specs_cover_expected_names() -> None:
    """PROVIDER_NAMES must be exactly the three supported providers in order."""

    assert PROVIDER_NAMES == ("openai", "openrouter", "claude")


def test_unknown_name_returns_none() -> None:
    """Names not in the registry return None."""

    assert provider_spec("anthropic") is None
    assert provider_spec("") is None
    assert provider_spec("openai2") is None


def test_all_specs_are_provider_spec_instances() -> None:
    """Each entry is a ProviderSpec with the correct name field."""

    for name in PROVIDER_NAMES:
        spec = provider_spec(name)
        assert isinstance(spec, ProviderSpec)
        assert spec.name == name


# ---------------------------------------------------------------------------
# default_model
# ---------------------------------------------------------------------------

def test_default_model_prefers_provider_block() -> None:
    """When the provider block is present its model is returned."""

    cfg = _app_config(
        openrouter=OpenRouterConfig(model="anthropic/custom-model"),
        claude=ClaudeConfig(model="claude-custom"),
    )

    assert provider_spec("openai").default_model(cfg) == "gpt-5-mini"
    assert provider_spec("openrouter").default_model(cfg) == "anthropic/custom-model"
    assert provider_spec("claude").default_model(cfg) == "claude-custom"


def test_default_model_falls_back_to_openai_when_block_missing() -> None:
    """openrouter/claude fall back to openai.model when their block is None."""

    cfg = _app_config()  # openrouter=None, claude=None

    assert provider_spec("openrouter").default_model(cfg) == "gpt-5-mini"
    assert provider_spec("claude").default_model(cfg) == "gpt-5-mini"


# ---------------------------------------------------------------------------
# supports_multimodal
# ---------------------------------------------------------------------------

def test_supports_multimodal_matrix() -> None:
    """openai always True; openrouter/claude mirror block flag; False when block absent."""

    cfg_with_blocks = _app_config(
        openrouter=OpenRouterConfig(supports_multimodal=True),
        claude=ClaudeConfig(supports_multimodal=True),
    )
    cfg_without_blocks = _app_config()  # openrouter=None, claude=None
    cfg_disabled = _app_config(
        openrouter=OpenRouterConfig(supports_multimodal=False),
        claude=ClaudeConfig(supports_multimodal=False),
    )

    # openai always True regardless
    assert provider_spec("openai").supports_multimodal(cfg_with_blocks) is True
    assert provider_spec("openai").supports_multimodal(cfg_without_blocks) is True

    # with blocks present, mirrors the flag
    assert provider_spec("openrouter").supports_multimodal(cfg_with_blocks) is True
    assert provider_spec("claude").supports_multimodal(cfg_with_blocks) is True

    assert provider_spec("openrouter").supports_multimodal(cfg_disabled) is False
    assert provider_spec("claude").supports_multimodal(cfg_disabled) is False

    # block absent → False
    assert provider_spec("openrouter").supports_multimodal(cfg_without_blocks) is False
    assert provider_spec("claude").supports_multimodal(cfg_without_blocks) is False


# ---------------------------------------------------------------------------
# config_block
# ---------------------------------------------------------------------------

def test_config_block_returns_correct_section() -> None:
    """config_block returns the config section or None when absent."""

    or_cfg = OpenRouterConfig()
    cl_cfg = ClaudeConfig()
    cfg = _app_config(openrouter=or_cfg, claude=cl_cfg)

    assert provider_spec("openai").config_block(cfg) is cfg.openai
    assert provider_spec("openrouter").config_block(cfg) is or_cfg
    assert provider_spec("claude").config_block(cfg) is cl_cfg


def test_config_block_none_when_optional_sections_missing() -> None:
    """openrouter and claude config blocks are None when not configured."""

    cfg = _app_config()

    assert provider_spec("openai").config_block(cfg) is not None  # always present
    assert provider_spec("openrouter").config_block(cfg) is None
    assert provider_spec("claude").config_block(cfg) is None


# ---------------------------------------------------------------------------
# build — type checks with fake API keys
# ---------------------------------------------------------------------------

def test_build_returns_working_provider_instances(monkeypatch) -> None:
    """build() returns the correct provider class for each name."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-or-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-claude-key")

    cfg = _app_config(
        openrouter=OpenRouterConfig(api_key_env="OPEN_ROUTER_API_KEY"),
        claude=ClaudeConfig(api_key_env="ANTHROPIC_API_KEY"),
    )

    openai_provider = provider_spec("openai").build(cfg)
    assert isinstance(openai_provider, OpenAIResponsesProvider)

    openrouter_provider = provider_spec("openrouter").build(cfg)
    assert isinstance(openrouter_provider, OpenRouterChatProvider)

    claude_provider = provider_spec("claude").build(cfg)
    assert isinstance(claude_provider, ClaudeMessagesProvider)
