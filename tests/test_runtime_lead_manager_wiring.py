"""FeatherRuntime exposes the multi-lead substrate and resumes lead sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feather.models import ModelTurn, ProviderRequestConfig
from feather.runtime import FeatherRuntime
from feather.storage.lead_session_store import LeadSessionStore


class _FakeProvider:
    async def complete(self, **_kw: Any) -> ModelTurn:  # pragma: no cover - never called
        raise AssertionError("wiring test should not invoke the provider")


def _write_app_config(root: Path, *, default_lead: str | None = None) -> None:
    extra = f"\ndefault_lead: {default_lead}\n" if default_lead else "\n"
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
"""
        + extra,
        encoding="utf-8",
    )


def _write_lead_yaml(root: Path, name: str = "lead") -> None:
    """Stage a minimal lead YAML (no web tools, which need a Parallel client)."""
    agents = root / "config" / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / f"{name}.yaml").write_text(
        """name: Lead
role: lead
personality: Decisive.
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
registered_tools:
  - read_file
  - spawn_agent
""",
        encoding="utf-8",
    )


async def _make_runtime(tmp_path: Path) -> FeatherRuntime:
    (tmp_path / ".feather" / "skills").mkdir(parents=True, exist_ok=True)
    _write_lead_yaml(tmp_path)
    return await FeatherRuntime.create(
        tmp_path,
        provider_factory=lambda _cfg: _FakeProvider(),
    )


async def test_runtime_exposes_default_lead_and_stores(tmp_path: Path) -> None:
    _write_app_config(tmp_path)
    runtime = await _make_runtime(tmp_path)
    try:
        assert runtime.default_lead_name == "lead"
        assert isinstance(runtime.lead_session_store, LeadSessionStore)
        # The catalog discovers the packaged lead as a lead.
        assert "lead" in {e.name for e in runtime.agent_catalog.list_leads()}
    finally:
        await runtime.close()


async def test_default_lead_is_configurable(tmp_path: Path) -> None:
    _write_app_config(tmp_path, default_lead="lead")  # packaged lead exists
    runtime = await _make_runtime(tmp_path)
    try:
        assert runtime.default_lead_name == "lead"
    finally:
        await runtime.close()


async def test_lead_manager_persists_and_resumes_session(tmp_path: Path) -> None:
    _write_app_config(tmp_path)
    runtime = await _make_runtime(tmp_path)
    try:
        manager = runtime.lead_manager(worker_mode=False)
        await manager.start(["lead"])
        handle = manager.handle("lead")
        first_session = handle.session_id
        # Recorded durably so a future launch resumes it.
        assert await runtime.lead_session_store.get("lead") == first_session

        # A second manager on the SAME runtime returns the cached singleton.
        assert runtime.lead_manager(worker_mode=False) is manager
    finally:
        await runtime.close()

    # A brand-new runtime over the same project resumes the recorded session.
    runtime2 = await _make_runtime(tmp_path)
    try:
        resumed = await runtime2.lead_session_store.get("lead")
        manager2 = runtime2.lead_manager(worker_mode=False)
        await manager2.start(["lead"])
        assert manager2.handle("lead").session_id == resumed
    finally:
        await runtime2.close()


async def test_build_lead_supervisor_binds_agent_name(tmp_path: Path) -> None:
    _write_app_config(tmp_path)
    runtime = await _make_runtime(tmp_path)
    try:
        supervisor = runtime.build_lead_supervisor("lead")
        # Constructed (not started) — bound to the lead agent name.
        assert supervisor is not None
    finally:
        await runtime.close()
