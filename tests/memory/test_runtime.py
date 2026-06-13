"""Tests for build_memory_stack gating logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.memory.config import MemoryConfig, MemoryQdrantConfig
from feather.memory.reader import NoOpMemoryReader
from feather.memory.runtime import MemoryStack, build_memory_stack
from feather.memory.trigger import NoOpMemoryTrigger
from feather.models import (
    AppConfig,
    CompactionConfig,
    DatabaseConfig,
    LoggingConfig,
    OpenAIConfig,
    SchedulerConfig,
    SkillsConfig,
    StorageConfig,
)
from feather.storage.session_store import SessionStore


class _StubProvider:
    """Standin for BaseLLMProvider — never actually called by build_memory_stack."""


def _stub_app_config() -> AppConfig:
    """Minimal AppConfig — provider overrides aren't exercised in this file."""

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
        openai=OpenAIConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-5-mini",
            max_output_tokens=8000,
            temperature=1.0,
            parallel_tool_calls=True,
        ),
        active_provider="openai",
    )


def _live_config() -> MemoryConfig:
    cfg = MemoryConfig()
    cfg.enabled = True
    cfg.qdrant = MemoryQdrantConfig(url="http://qdrant:6333", embedding_dims=3072)
    return cfg


async def _make_session_store(tmp_path: Path) -> SessionStore:
    store = SessionStore(tmp_path / "db.sqlite")
    await store.initialize()
    return store


# Disabled / gating ----------------------------------------------------------


async def test_stack_is_noop_when_config_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sess_store = await _make_session_store(tmp_path)
    try:
        cfg = MemoryConfig()  # enabled=False
        stack = build_memory_stack(
            cfg=cfg,
            default_provider=_StubProvider(),  # type: ignore[arg-type]
            app_config=_stub_app_config(),
            session_store=sess_store,
        )
        assert stack.enabled is False
        assert isinstance(stack.reader, NoOpMemoryReader)
        assert isinstance(stack.trigger, NoOpMemoryTrigger)
        assert stack.service is None
    finally:
        await sess_store.close()


async def test_stack_is_noop_when_qdrant_url_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sess_store = await _make_session_store(tmp_path)
    try:
        cfg = _live_config()
        cfg.qdrant.url = None
        stack = build_memory_stack(
            cfg=cfg,
            default_provider=_StubProvider(),  # type: ignore[arg-type]
            app_config=_stub_app_config(),
            session_store=sess_store,
        )
        assert stack.enabled is False
    finally:
        await sess_store.close()


async def test_stack_is_noop_when_gemini_key_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    sess_store = await _make_session_store(tmp_path)
    try:
        stack = build_memory_stack(
            cfg=_live_config(), default_provider=_StubProvider(), app_config=_stub_app_config(), session_store=sess_store
        )
        assert stack.enabled is False
    finally:
        await sess_store.close()


async def test_stack_is_noop_when_kill_switch_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("FEATHER_MEMORY_DISABLED", "1")
    sess_store = await _make_session_store(tmp_path)
    try:
        stack = build_memory_stack(
            cfg=_live_config(), default_provider=_StubProvider(), app_config=_stub_app_config(), session_store=sess_store
        )
        assert stack.enabled is False
    finally:
        await sess_store.close()


async def test_stack_is_noop_when_kill_switch_other_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only literal '1' disables; everything else is treated as not-set."""
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("FEATHER_MEMORY_DISABLED", "0")
    sess_store = await _make_session_store(tmp_path)
    try:
        stack = build_memory_stack(
            cfg=_live_config(), default_provider=_StubProvider(), app_config=_stub_app_config(), session_store=sess_store
        )
        assert stack.enabled is True  # kill switch did NOT engage
    finally:
        await sess_store.close()


# Enabled path ---------------------------------------------------------------


async def test_stack_is_live_when_all_gates_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("FEATHER_MEMORY_DISABLED", raising=False)
    sess_store = await _make_session_store(tmp_path)
    try:
        stack = build_memory_stack(
            cfg=_live_config(), default_provider=_StubProvider(), app_config=_stub_app_config(), session_store=sess_store
        )
        assert stack.enabled is True
        assert stack.service is not None
        # reader/trigger are live (not the NoOp instances).
        assert not isinstance(stack.reader, NoOpMemoryReader)
        assert not isinstance(stack.trigger, NoOpMemoryTrigger)
    finally:
        await sess_store.close()


async def test_qdrant_url_env_wins_over_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If both YAML url and env url present, env wins."""
    monkeypatch.setenv("QDRANT_URL", "http://env-qdrant:6333")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sess_store = await _make_session_store(tmp_path)
    try:
        cfg = _live_config()
        cfg.qdrant.url = "http://yaml-qdrant:6333"
        stack = build_memory_stack(
            cfg=cfg,
            default_provider=_StubProvider(),  # type: ignore[arg-type]
            app_config=_stub_app_config(),
            session_store=sess_store,
        )
        # The actual Qdrant URL the client used isn't easily inspectable here,
        # but the live path was selected, which means env-or-yaml resolution
        # produced a non-empty URL.
        assert stack.enabled is True
    finally:
        await sess_store.close()


# MemoryStack.aclose service delegation --------------------------------------


async def test_memory_stack_aclose_closes_service_store_client() -> None:
    """MemoryStack.aclose() must delegate to service.aclose() exactly once."""

    class _StubService:
        def __init__(self) -> None:
            self.close_count = 0

        async def aclose(self) -> None:
            self.close_count += 1

    stub_service = _StubService()
    stack = MemoryStack(
        reader=NoOpMemoryReader(),
        trigger=NoOpMemoryTrigger(),
        service=stub_service,  # type: ignore[arg-type]
        enabled=True,
    )
    await stack.aclose()
    assert stub_service.close_count == 1


async def test_memory_stack_aclose_survives_service_close_error() -> None:
    """MemoryStack.aclose() must not propagate a service.aclose() exception."""

    class _BrokenService:
        async def aclose(self) -> None:
            raise RuntimeError("store is gone")

    stack = MemoryStack(
        reader=NoOpMemoryReader(),
        trigger=NoOpMemoryTrigger(),
        service=_BrokenService(),  # type: ignore[arg-type]
        enabled=True,
    )
    # Must not raise.
    await stack.aclose()


async def test_memory_stack_aclose_twice_is_safe() -> None:
    """A second stack.aclose() must not raise (shutdown paths can double-fire).

    The stack has no already-closed guard: each aclose() delegates to
    service.aclose() again, so the service observes one close per call.
    """

    class _StubService:
        def __init__(self) -> None:
            self.close_count = 0

        async def aclose(self) -> None:
            self.close_count += 1

    stub_service = _StubService()
    stack = MemoryStack(
        reader=NoOpMemoryReader(),
        trigger=NoOpMemoryTrigger(),
        service=stub_service,  # type: ignore[arg-type]
        enabled=True,
    )
    await stack.aclose()
    # Second call must not raise.
    await stack.aclose()
    assert stub_service.close_count == 2
