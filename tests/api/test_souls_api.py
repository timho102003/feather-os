"""API: soul library listing + lead creation from a soul preset."""

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


def test_list_souls_returns_packaged_library(client) -> None:
    resp = client.get("/api/souls")
    assert resp.status_code == 200
    souls = resp.json()
    assert len(souls) >= 20
    ids = {s["id"] for s in souls}
    assert "systems-thinker" in ids
    soul = next(s for s in souls if s["id"] == "systems-thinker")
    assert soul["title"] == "The Systems Thinker"
    assert soul["color"].startswith("#")
    assert soul["emoji"] and soul["tags"]


def test_souls_sorted_by_title(client) -> None:
    titles = [s["title"] for s in client.get("/api/souls").json()]
    assert titles == sorted(titles, key=str.lower)


def test_create_lead_from_soul_preset(client) -> None:
    resp = client.post(
        "/api/leads", json={"name": "backend", "soul_id": "systems-thinker"}
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "backend"            # slug stays the given name
    assert body["display_name"] == "Backend"    # lead keeps its OWN name, not the soul's
    assert body["color"] == "#5B8DEF"           # temperament color from the soul
    assert body["emoji"] == "🏛️"
    assert len(body["soul"]) > 50               # working-character prose applied
    # And it now shows up in the lead list.
    names = {lead["name"] for lead in client.get("/api/leads").json()}
    assert "backend" in names


def test_create_lead_unknown_soul_id_is_400(client) -> None:
    resp = client.post("/api/leads", json={"name": "ghosty", "soul_id": "no-such-soul"})
    assert resp.status_code == 400
    assert "no-such-soul" in resp.text
