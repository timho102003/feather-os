"""Per-operation provider override for memory ops.

Each of `extraction`, `classification`, and `query_builder` accepts a
`provider:` key whose semantics mirror the agent-level override:

    provider: ~          → use the app-default provider (built once,
                            shared with the agent loop)
    provider: openai     → build & cache an OpenAI client for this op
    provider: openrouter → build & cache an OpenRouter client (requires
                            `app.openrouter` to be present)

When `provider` is overridden, `model: ~` falls back to that provider's
configured default model rather than the calling agent's conversation
model — that prevents an OpenRouter slug from being routed to OpenAI.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from feather.memory.config import MemoryConfig, MemoryOperationModelConfig, MemoryQdrantConfig
from feather.memory.runtime import build_memory_stack
from feather.models import (
    AppConfig,
    CompactionConfig,
    DatabaseConfig,
    LoggingConfig,
    OpenAIConfig,
    OpenRouterConfig,
    SchedulerConfig,
    SkillsConfig,
    StorageConfig,
)
from feather.providers.openai_provider import OpenAIResponsesProvider
from feather.providers.openrouter_provider import OpenRouterChatProvider
from feather.storage.session_store import SessionStore


def _app_config(*, active: str = "openai", with_openrouter: bool = True) -> AppConfig:
    """Minimal AppConfig sufficient for build_memory_stack to honour overrides."""

    openai = OpenAIConfig(
        api_key_env="OPENAI_API_KEY",
        model="gpt-5-mini",
        max_output_tokens=8000,
        temperature=1.0,
        parallel_tool_calls=True,
    )
    openrouter = (
        OpenRouterConfig(model="deepseek/deepseek-v3.2") if with_openrouter else None
    )
    return AppConfig(
        database=DatabaseConfig(path=":memory:"),
        storage=StorageConfig(temp_directory="/tmp"),
        logging=LoggingConfig(path="/tmp/feather.log"),
        compaction=CompactionConfig(
            enabled=False, trigger_ratio=0.8, context_window_tokens=400_000
        ),
        skills=SkillsConfig(directory=".feather/skills"),
        scheduler=SchedulerConfig(
            enabled=False,
            poll_interval_seconds=2.0,
            failure_retry_seconds=30.0,
            max_due_jobs_per_tick=10,
        ),
        openai=openai,
        active_provider=active,
        openrouter=openrouter,
    )


def _live_memory_cfg() -> MemoryConfig:
    cfg = MemoryConfig()
    cfg.enabled = True
    cfg.qdrant = MemoryQdrantConfig(url="http://qdrant:6333", embedding_dims=3072)
    return cfg


async def _session_store(tmp_path: Path) -> SessionStore:
    store = SessionStore(tmp_path / "db.sqlite")
    await store.initialize()
    return store


# ---------------------------------------------------------------------------


async def test_op_inherits_app_default_provider_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`extraction.provider: ~` reuses the app-default provider instance."""

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "stub")
    sess = await _session_store(tmp_path)
    try:
        cfg = _live_memory_cfg()
        cfg.extraction = MemoryOperationModelConfig(provider=None, model=None)
        default_provider = OpenAIResponsesProvider(_app_config().openai)
        stack = build_memory_stack(
            cfg=cfg,
            default_provider=default_provider,
            app_config=_app_config(),
            session_store=sess,
        )
        assert stack.enabled is True
        # Extractor must hold the same provider instance — no extra client built.
        assert stack.service is not None
        assert stack.service._extractor._provider is default_provider  # type: ignore[attr-defined]
        assert stack.owned_providers == []
    finally:
        await sess.close()


async def test_op_builds_openai_alternate_when_active_is_openrouter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`extraction.provider: openai` while active=openrouter builds an alternate."""

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "stub")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    sess = await _session_store(tmp_path)
    try:
        cfg = _live_memory_cfg()
        cfg.extraction = MemoryOperationModelConfig(
            provider="openai", model="gpt-5.4-nano"
        )
        app_cfg = _app_config(active="openrouter")
        default_provider = OpenRouterChatProvider(app_cfg.openrouter)  # type: ignore[arg-type]
        try:
            stack = build_memory_stack(
                cfg=cfg,
                default_provider=default_provider,
                app_config=app_cfg,
                session_store=sess,
            )
            extractor_provider = stack.service._extractor._provider  # type: ignore[attr-defined]
            assert isinstance(extractor_provider, OpenAIResponsesProvider)
            assert extractor_provider is not default_provider
            # The alternate is owned by the stack so shutdown can close it.
            assert extractor_provider in stack.owned_providers
        finally:
            await stack.aclose()
    finally:
        await sess.close()


async def test_multiple_ops_share_one_alternate_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If extraction and classification both override to openai, only one client is built."""

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "stub")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    sess = await _session_store(tmp_path)
    try:
        cfg = _live_memory_cfg()
        cfg.extraction = MemoryOperationModelConfig(provider="openai")
        cfg.classification = MemoryOperationModelConfig(provider="openai")
        app_cfg = _app_config(active="openrouter")
        default_provider = OpenRouterChatProvider(app_cfg.openrouter)  # type: ignore[arg-type]
        try:
            stack = build_memory_stack(
                cfg=cfg,
                default_provider=default_provider,
                app_config=app_cfg,
                session_store=sess,
            )
            extr = stack.service._extractor._provider  # type: ignore[attr-defined]
            clf = stack.service._classifier._provider  # type: ignore[attr-defined]
            assert extr is clf
            assert len(stack.owned_providers) == 1
        finally:
            await stack.aclose()
    finally:
        await sess.close()


async def test_op_falls_back_to_alternate_provider_default_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When provider is overridden and model is `~`, the op uses that provider's default model."""

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "stub")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    sess = await _session_store(tmp_path)
    try:
        cfg = _live_memory_cfg()
        cfg.extraction = MemoryOperationModelConfig(provider="openai", model=None)
        app_cfg = _app_config(active="openrouter")
        default_provider = OpenRouterChatProvider(app_cfg.openrouter)  # type: ignore[arg-type]
        try:
            stack = build_memory_stack(
                cfg=cfg,
                default_provider=default_provider,
                app_config=app_cfg,
                session_store=sess,
            )
            extractor = stack.service._extractor  # type: ignore[attr-defined]
            # Cross-provider default model is exposed for use when cfg.model is None.
            assert extractor._default_model == "gpt-5-mini"
        finally:
            await stack.aclose()
    finally:
        await sess.close()


async def test_op_with_no_provider_override_keeps_default_model_as_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a provider override, default_model stays None so agent_model is used at call time."""

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "stub")
    sess = await _session_store(tmp_path)
    try:
        cfg = _live_memory_cfg()
        cfg.extraction = MemoryOperationModelConfig(provider=None, model=None)
        default_provider = OpenAIResponsesProvider(_app_config().openai)
        stack = build_memory_stack(
            cfg=cfg,
            default_provider=default_provider,
            app_config=_app_config(),
            session_store=sess,
        )
        assert stack.service._extractor._default_model is None  # type: ignore[attr-defined]
    finally:
        await sess.close()


async def test_op_provider_openrouter_without_block_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`extraction.provider: openrouter` with no `app.openrouter` block must fail fast."""

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "stub")
    sess = await _session_store(tmp_path)
    try:
        cfg = _live_memory_cfg()
        cfg.extraction = MemoryOperationModelConfig(provider="openrouter")
        app_cfg = _app_config(with_openrouter=False)
        default_provider = OpenAIResponsesProvider(app_cfg.openai)
        with pytest.raises(ValueError, match="extraction"):
            build_memory_stack(
                cfg=cfg,
                default_provider=default_provider,
                app_config=app_cfg,
                session_store=sess,
            )
    finally:
        await sess.close()


async def test_op_provider_unknown_value_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown provider names must fail at startup, not silently."""

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "stub")
    sess = await _session_store(tmp_path)
    try:
        cfg = _live_memory_cfg()
        cfg.classification = MemoryOperationModelConfig(provider="not-a-provider")
        app_cfg = _app_config()
        default_provider = OpenAIResponsesProvider(app_cfg.openai)
        with pytest.raises(ValueError, match="classification"):
            build_memory_stack(
                cfg=cfg,
                default_provider=default_provider,
                app_config=app_cfg,
                session_store=sess,
            )
    finally:
        await sess.close()


async def test_op_provider_matching_active_reuses_default_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`extraction.provider: openai` when active=openai should reuse the default, not build twice."""

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "stub")
    sess = await _session_store(tmp_path)
    try:
        cfg = _live_memory_cfg()
        cfg.extraction = MemoryOperationModelConfig(provider="openai")
        app_cfg = _app_config(active="openai")
        default_provider = OpenAIResponsesProvider(app_cfg.openai)
        stack = build_memory_stack(
            cfg=cfg,
            default_provider=default_provider,
            app_config=app_cfg,
            session_store=sess,
        )
        assert stack.service._extractor._provider is default_provider  # type: ignore[attr-defined]
        assert stack.owned_providers == []
    finally:
        await sess.close()


def test_memory_operation_model_config_yaml_parses_provider_field(tmp_path: Path) -> None:
    """`provider:` key must round-trip through load_app_config."""

    from feather.config import load_app_config

    app_yaml = """
database:
  path: db.sqlite
storage:
  temp_directory: tmp
logging:
  path: log
compaction:
  enabled: false
skills:
  directory: .feather/skills
scheduler:
  enabled: false
active_provider: openrouter
openai:
  api_key_env: OPENAI_API_KEY
  model: gpt-5-mini
  max_output_tokens: 1000
  temperature: 0.5
  parallel_tool_calls: true
openrouter:
  model: deepseek/deepseek-v3.2
memory:
  enabled: true
  qdrant:
    url: http://qdrant:6333
  extraction:
    provider: openai
    model: gpt-5.4-nano
    max_output_tokens: 2000
    temperature: 0.1
  classification:
    provider: ~
    model: ~
  query_builder:
    provider: openai
    model: gpt-5.4-nano
"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "app.yaml").write_text(app_yaml, encoding="utf-8")
    cfg = load_app_config(tmp_path)
    assert cfg.memory.extraction.provider == "openai"
    assert cfg.memory.extraction.model == "gpt-5.4-nano"
    assert cfg.memory.classification.provider is None
    assert cfg.memory.classification.model is None
    assert cfg.memory.query_builder.provider == "openai"
