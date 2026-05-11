"""Tests for FeatherRuntime config reload primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from feather.core.lead_supervisor import ConfigReloadAckResult
from feather.models import ModelTurn, ProviderRequestConfig
from feather.providers.base import BaseLLMProvider
from feather.runtime import FeatherRuntime

# Minimal app.yaml that boots the runtime without Qdrant or any messaging
# service.  memory.enabled=false avoids the Qdrant stack; active_provider=openai
# so the tests can swap it to claude to observe provider reconstruction.
# A claude: block is included so provider-switch tests can flip active_provider
# to claude without hitting "no claude: block" validation.
_MINIMAL_YAML = """\
database: { path: feather.db }
storage: { temp_directory: tmp }
logging: { path: log, level: INFO }
compaction: { enabled: true, trigger_ratio: 0.8, context_window_tokens: 100, model: null, max_output_tokens: 100, temperature: 0.2 }
skills: { directory: skills }
scheduler: { enabled: false, poll_interval_seconds: 2, failure_retry_seconds: 30, max_due_jobs_per_tick: 10 }
self_repair: { enabled: false }
active_provider: openai
openai:
  api_key_env: OPENAI_API_KEY
  model: gpt-5-mini
  max_output_tokens: 100
  temperature: 1.0
  parallel_tool_calls: true
claude:
  api_key_env: ANTHROPIC_API_KEY
  model: claude-opus-4-7
  max_output_tokens: 100
memory:
  enabled: false
"""


class _FakeProvider(BaseLLMProvider):
    """Provider stub so FeatherRuntime.create doesn't demand an API key."""

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
        raise AssertionError("This test should not invoke the provider.")


def _fake_provider_factory(_config: Any) -> _FakeProvider:
    """Return a fresh _FakeProvider so the factory tracks each construction."""

    return _FakeProvider()


# ---------------------------------------------------------------------------
# Task 18 — reload_config
# ---------------------------------------------------------------------------


async def test_reload_config_swaps_app_config(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(
        project, provider_factory=_fake_provider_factory
    )
    try:
        assert runtime.config.active_provider == "openai"

        (project / "config" / "app.yaml").write_text(
            _MINIMAL_YAML.replace("active_provider: openai", "active_provider: claude"),
            encoding="utf-8",
        )

        await runtime.reload_config()
        assert runtime.config.active_provider == "claude"
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# Task 19 — rebuild_agent
# ---------------------------------------------------------------------------

# Minimal agent YAML for the test lead.  The packaged lead.yaml lists
# ``web_search`` which requires a ParallelClient the test runtime doesn't
# provision.  By staging a project-local override we avoid the unknown-tool
# error without touching production configs.
_MINIMAL_LEAD_YAML = """\
name: Lead
role: lead
personality: Test stub.
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
memory_enabled: false
registered_tools:
  - bash
  - ask_user
"""


def _write_minimal_agent_yaml(project: Path) -> None:
    """Stage a minimal lead agent config inside the test project directory."""

    agents_dir = project / "config" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "lead.yaml").write_text(_MINIMAL_LEAD_YAML, encoding="utf-8")


async def test_rebuild_agent_uses_new_provider_after_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rebuild_agent() installs a fresh provider instance from the new config."""

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")
    _write_minimal_agent_yaml(project)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-before")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-after")

    runtime = await FeatherRuntime.create(project)
    try:
        agent_before = runtime.build_agent("lead")
        provider_before = id(agent_before._provider)

        (project / "config" / "app.yaml").write_text(
            _MINIMAL_YAML.replace("active_provider: openai", "active_provider: claude"),
            encoding="utf-8",
        )

        await runtime.reload_config()
        runtime.rebuild_agent("lead")

        agent_after = runtime.get_agent("lead")
        assert id(agent_after._provider) != provider_before
    finally:
        await runtime.close()


async def test_get_agent_raises_if_not_built(tmp_path: Path) -> None:
    """get_agent() raises KeyError when no agent with that name is cached."""

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")
    _write_minimal_agent_yaml(project)

    runtime = await FeatherRuntime.create(
        project, provider_factory=_fake_provider_factory
    )
    try:
        with pytest.raises(KeyError, match="lead"):
            runtime.get_agent("lead")
    finally:
        await runtime.close()


async def test_build_agent_populates_cache(tmp_path: Path) -> None:
    """build_agent() stores the result so get_agent() returns the same instance."""

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")
    _write_minimal_agent_yaml(project)

    runtime = await FeatherRuntime.create(
        project, provider_factory=_fake_provider_factory
    )
    try:
        agent = runtime.build_agent("lead")
        assert runtime.get_agent("lead") is agent
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# Task 20 — apply_config_change
# ---------------------------------------------------------------------------


async def test_apply_config_change_live_reload_only(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(
        project, provider_factory=_fake_provider_factory
    )
    try:
        (project / "config" / "app.yaml").write_text(
            _MINIMAL_YAML.replace("trigger_ratio: 0.8", "trigger_ratio: 0.5"),
            encoding="utf-8",
        )

        result = await runtime.apply_config_change(
            ["app.compaction.trigger_ratio"]
        )

        assert result.applied == ["app.compaction.trigger_ratio"]
        assert result.needs_restart_lead == []
        assert result.needs_restart_app == []
        assert runtime.config.compaction.trigger_ratio == 0.5
    finally:
        await runtime.close()


async def test_apply_config_change_next_turn_rebuilds_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")
    _write_minimal_agent_yaml(project)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-before")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-after")

    runtime = await FeatherRuntime.create(project)
    try:
        agent_before = runtime.build_agent("lead")
        before_id = id(agent_before)

        (project / "config" / "app.yaml").write_text(
            _MINIMAL_YAML.replace("active_provider: openai", "active_provider: claude"),
            encoding="utf-8",
        )

        result = await runtime.apply_config_change(["app.active_provider"])

        assert "app.active_provider" in result.applied
        assert id(runtime.get_agent("lead")) != before_id
    finally:
        await runtime.close()


async def test_apply_config_change_flags_restart_lead(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(
        project, provider_factory=_fake_provider_factory
    )
    try:
        result = await runtime.apply_config_change(
            ["app.claude.request_timeout_seconds"]
        )

        assert "app.claude.request_timeout_seconds" in result.needs_restart_lead
    finally:
        await runtime.close()


async def test_apply_config_change_flags_restart_app(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(
        project, provider_factory=_fake_provider_factory
    )
    try:
        result = await runtime.apply_config_change(["app.database.path"])

        assert "app.database.path" in result.needs_restart_app
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# Task 24 — apply_config_change worker-mode fanout
# ---------------------------------------------------------------------------


class _FakeSupervisor:
    """Minimal stand-in for LeadSupervisor that records reload requests."""

    def __init__(self, *, ok: bool = True, error: str | None = None) -> None:
        self._ok = ok
        self._error = error
        self.calls: list[tuple[list[str], str]] = []

    async def request_config_reload(
        self,
        changed_paths: list[str],
        reload_class: str,
        *,
        timeout: float = 10.0,
    ) -> ConfigReloadAckResult:
        self.calls.append((list(changed_paths), reload_class))
        return ConfigReloadAckResult(
            ok=self._ok,
            applied_paths=list(changed_paths) if self._ok else [],
            error=self._error,
            correlation_id="fake-corr",
        )


async def test_attach_supervisor_is_called_on_live_apply(tmp_path: Path) -> None:
    """When a supervisor is attached, apply_config_change calls request_config_reload."""

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(
        project, provider_factory=_fake_provider_factory
    )
    try:
        supervisor = _FakeSupervisor(ok=True)
        runtime.attach_supervisor(supervisor)

        (project / "config" / "app.yaml").write_text(
            _MINIMAL_YAML.replace("trigger_ratio: 0.8", "trigger_ratio: 0.5"),
            encoding="utf-8",
        )

        result = await runtime.apply_config_change(["app.compaction.trigger_ratio"])

        assert result.applied == ["app.compaction.trigger_ratio"]
        assert len(supervisor.calls) == 1
        paths, reload_class = supervisor.calls[0]
        assert "app.compaction.trigger_ratio" in paths
        assert reload_class == "live"
    finally:
        runtime.detach_supervisor()
        await runtime.close()


async def test_attach_supervisor_not_called_for_restart_class(tmp_path: Path) -> None:
    """RESTART_LEAD paths do not trigger the supervisor fanout (nothing to reload)."""

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(
        project, provider_factory=_fake_provider_factory
    )
    try:
        supervisor = _FakeSupervisor(ok=True)
        runtime.attach_supervisor(supervisor)

        result = await runtime.apply_config_change(["app.database.path"])

        assert "app.database.path" in result.needs_restart_app
        # Supervisor fanout not called — no live/next_turn paths.
        assert supervisor.calls == []
    finally:
        runtime.detach_supervisor()
        await runtime.close()


async def test_supervisor_error_ack_returns_empty_applied(tmp_path: Path) -> None:
    """When the worker ack reports ok=False, applied list is empty."""

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(
        project, provider_factory=_fake_provider_factory
    )
    try:
        supervisor = _FakeSupervisor(ok=False, error="worker blown up")
        runtime.attach_supervisor(supervisor)

        (project / "config" / "app.yaml").write_text(
            _MINIMAL_YAML.replace("trigger_ratio: 0.8", "trigger_ratio: 0.5"),
            encoding="utf-8",
        )

        result = await runtime.apply_config_change(["app.compaction.trigger_ratio"])

        assert result.applied == []
    finally:
        runtime.detach_supervisor()
        await runtime.close()


async def test_detach_supervisor_stops_fanout(tmp_path: Path) -> None:
    """After detach_supervisor, apply_config_change no longer fans out to the worker."""

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(
        project, provider_factory=_fake_provider_factory
    )
    try:
        supervisor = _FakeSupervisor(ok=True)
        runtime.attach_supervisor(supervisor)
        runtime.detach_supervisor()

        (project / "config" / "app.yaml").write_text(
            _MINIMAL_YAML.replace("trigger_ratio: 0.8", "trigger_ratio: 0.5"),
            encoding="utf-8",
        )

        result = await runtime.apply_config_change(["app.compaction.trigger_ratio"])

        assert result.applied == ["app.compaction.trigger_ratio"]
        assert supervisor.calls == []  # no fanout after detach
    finally:
        await runtime.close()
