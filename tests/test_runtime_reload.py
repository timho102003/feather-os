"""Tests for FeatherRuntime config reload primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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
