"""Console endpoints: config get/set/fields + mid-turn input injection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from feather.api.server import create_app
from feather.models import ModelTurn, RuntimeEvent
from feather.providers.base import BaseLLMProvider


class _StreamingProvider(BaseLLMProvider):
    async def complete(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        event_handler=None,
        request_config=None,
    ) -> ModelTurn:
        if event_handler is not None:
            event_handler(RuntimeEvent(kind="assistant_text_delta", text="ok"))
        return ModelTurn(response_id="r1", output_text="ok", tool_calls=[], usage=None)


def _write_project(root: Path) -> None:
    (root / "config" / "agents").mkdir(parents=True, exist_ok=True)
    (root / ".feather" / "skills").mkdir(parents=True, exist_ok=True)
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
""",
        encoding="utf-8",
    )
    (root / "config" / "agents" / "lead.yaml").write_text(
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


@pytest.fixture
def client(tmp_path):
    _write_project(tmp_path)
    root_logger = logging.getLogger()
    saved_handlers = root_logger.handlers[:]
    saved_level = root_logger.level
    try:
        app = create_app(tmp_path, provider_factory=lambda _cfg: _StreamingProvider())
        with TestClient(app) as c:
            yield c
    finally:
        root_logger.handlers[:] = saved_handlers
        root_logger.setLevel(saved_level)


def test_config_fields_listed(client) -> None:
    fields = client.get("/api/config/fields").json()
    assert len(fields) > 20
    row = next(f for f in fields if f["path"] == "app.compaction.trigger_ratio")
    assert {"path", "value", "type", "reload", "source", "description"} <= set(row)
    assert row["reload"]  # reload class string present


def test_config_set_live_field_applies(client) -> None:
    resp = client.post(
        "/api/config", json={"path": "app.compaction.trigger_ratio", "value": 0.75}
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["ok"] is True, out
    assert "app.compaction.trigger_ratio" in out["applied"]
    # The new value is reflected back through the live config.
    cfg = client.get("/api/config").json()
    assert cfg["values"]["compaction_trigger_ratio"] == 0.75


def test_config_set_unknown_path_is_rejected(client) -> None:
    out = client.post("/api/config", json={"path": "app.nope.field", "value": 1}).json()
    assert out["ok"] is False
    assert out["error"]


def test_inject_input_accepted(client) -> None:
    resp = client.post("/api/leads/lead/input", json={"text": "steer the turn"})
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] in ("injected", "no_input_queue")


def test_inject_unknown_lead_404(client) -> None:
    assert client.post("/api/leads/ghost/input", json={"text": "x"}).status_code == 404
