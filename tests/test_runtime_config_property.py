"""Regression tests for ``FeatherRuntime.config`` exposure.

The Textual TUI's ``on_mount`` reads ``self._runtime.config.self_repair.enabled``
to decide whether to launch the lead in a worker subprocess. Without an
exposed ``config`` property, mounting the TUI raises
``AttributeError: 'FeatherRuntime' object has no attribute 'config'`` and
the app fails to start. These tests pin both the property and the exact
attribute chain the TUI relies on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feather.models import AppConfig, ModelTurn, ProviderRequestConfig
from feather.providers.base import BaseLLMProvider
from feather.runtime import FeatherRuntime


class _FakeProvider(BaseLLMProvider):
    """Provider stub so FeatherRuntime.create doesn't demand an API key."""

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
        raise AssertionError("This test should not invoke the provider.")


def _write_app_config(root: Path) -> None:
    """Write a minimal app config sufficient for FeatherRuntime.create."""

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


async def test_runtime_exposes_loaded_app_config_via_config_property(
    tmp_path: Path,
) -> None:
    """The TUI dereferences ``runtime.config.self_repair.enabled`` on mount."""

    _write_app_config(tmp_path)
    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    runtime = await FeatherRuntime.create(
        tmp_path,
        provider_factory=lambda _config: _FakeProvider(),
    )
    try:
        assert isinstance(runtime.config, AppConfig)
        # The exact attribute chain the Textual TUI relies on at
        # textual_tui.py:680 — keep this pinned so future refactors
        # of AppConfig don't silently break TUI startup.
        assert runtime.config.self_repair.enabled is False
    finally:
        await runtime.close()
