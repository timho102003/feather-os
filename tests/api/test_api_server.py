"""API parity layer: REST endpoints + WebSocket event streaming."""

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
    """Emits two assistant deltas then completes — deterministic for tests."""

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
            event_handler(RuntimeEvent(kind="assistant_text_delta", text="Hello "))
            event_handler(RuntimeEvent(kind="assistant_text_delta", text="world"))
        return ModelTurn(
            response_id="r1", output_text="Hello world", tool_calls=[], usage=None
        )


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
    # FeatherRuntime.create() calls configure_logging(), which clears the root
    # logger's handlers and sets its level globally. Snapshot + restore so this
    # test's runtimes don't leak logging state into later tests (e.g. the
    # telegram token-scrub test, which is sensitive to the root level).
    root_logger = logging.getLogger()
    saved_handlers = root_logger.handlers[:]
    saved_level = root_logger.level
    try:
        app = create_app(tmp_path, provider_factory=lambda _cfg: _StreamingProvider())
        with TestClient(app) as c:  # runs lifespan (builds runtime + channels)
            yield c
    finally:
        root_logger.handlers[:] = saved_handlers
        root_logger.setLevel(saved_level)


def test_list_leads_includes_default(client) -> None:
    resp = client.get("/api/leads")
    assert resp.status_code == 200
    names = {lead["name"] for lead in resp.json()}
    assert "lead" in names


def test_config_endpoint(client) -> None:
    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["default_lead"] == "lead"
    assert body["model"] == "gpt-5-mini"
    assert "values" in body


def test_index_html_served(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Feather" in resp.text
    assert 'data-testid="lead-list"' in resp.text


def test_subagents_empty(client) -> None:
    resp = client.get("/api/leads/lead/subagents")
    assert resp.status_code == 200
    assert resp.json() == []


def test_unknown_lead_404(client) -> None:
    assert client.get("/api/leads/ghost/subagents").status_code == 404
    assert client.post("/api/leads/ghost/messages", json={"text": "hi"}).status_code == 404


def test_create_lead(client) -> None:
    resp = client.post("/api/leads", json={"name": "sophia", "soul": "You are Sophia."})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "sophia"
    assert body["soul"] == "You are Sophia."
    # Now appears in the list.
    names = {lead["name"] for lead in client.get("/api/leads").json()}
    assert "sophia" in names


def test_create_lead_rejects_bad_name(client) -> None:
    resp = client.post("/api/leads", json={"name": "bad/name", "soul": ""})
    assert resp.status_code == 400


def test_websocket_streams_assistant_deltas(client) -> None:
    with client.websocket_connect("/api/leads/lead/ws") as ws:
        first = ws.receive_json()
        assert first["kind"] == "connected"
        ws.send_json({"text": "hi there"})
        deltas: list[str] = []
        saw_idle = False
        for _ in range(40):
            ev = ws.receive_json()
            if ev["kind"] == "assistant_text_delta":
                deltas.append(ev["text"])
            if ev["kind"] == "status" and ev.get("payload", {}).get("status") == "idle":
                saw_idle = True
                break
        assert "".join(deltas) == "Hello world"
        assert saw_idle


def test_lead_transcript_after_turn(client) -> None:
    with client.websocket_connect("/api/leads/lead/ws") as ws:
        assert ws.receive_json()["kind"] == "connected"
        ws.send_json({"text": "remember this"})
        for _ in range(40):
            ev = ws.receive_json()
            if ev["kind"] == "status" and ev.get("payload", {}).get("status") == "idle":
                break
    transcript = client.get("/api/leads/lead/transcript").json()
    roles = [m["role"] for m in transcript["messages"]]
    assert "user" in roles and "assistant" in roles
