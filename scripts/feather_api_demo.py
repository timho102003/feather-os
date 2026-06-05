"""Launch the Feather API with a fake streaming provider + two demo leads.

For local/manual testing and Playwright browser tests — no API keys needed.
Run:  uv run python scripts/feather_api_demo.py [port]
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import uvicorn

from feather.api.server import create_app
from feather.models import ModelTurn, RuntimeEvent
from feather.providers.base import BaseLLMProvider

_APP_YAML = """database:
  path: .feather/db/feather.db
storage:
  temp_directory: .feather/tmp
logging:
  path: .feather/logs/feather.log
  level: WARNING
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
default_lead: lead
"""


def _lead_yaml(name: str, soul: str) -> str:
    return f"""name: {name}
role: lead
personality: {soul}
soul: |
  {soul}
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
registered_tools:
  - read_file
  - spawn_agent
"""


class _DemoProvider(BaseLLMProvider):
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
            for chunk in (
                "On it. ",
                "Here's a quick plan:\n",
                "1. Scan the repo\n",
                "2. Summarize findings\n",
                "3. Report back.",
            ):
                event_handler(RuntimeEvent(kind="assistant_text_delta", text=chunk))
        return ModelTurn(response_id="r", output_text="", tool_calls=[], usage=None)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    root = Path(tempfile.mkdtemp(prefix="feather-demo-"))
    # Isolate global config so the demo doesn't pick up a real ~/.feather
    # (which may enable memory/qdrant). Mirrors the test conftest.
    os.environ["FEATHER_HOME"] = str(root / "home")
    (root / "config" / "agents").mkdir(parents=True)
    (root / ".feather" / "skills").mkdir(parents=True)
    (root / "config" / "app.yaml").write_text(_APP_YAML, encoding="utf-8")
    (root / "config" / "agents" / "lead.yaml").write_text(
        _lead_yaml("Tim", "Decisive, pragmatic operator."), encoding="utf-8"
    )
    (root / "config" / "agents" / "sophia.yaml").write_text(
        _lead_yaml("Sophia", "Meticulous, curious researcher."), encoding="utf-8"
    )
    app = create_app(root, provider_factory=lambda _cfg: _DemoProvider())
    print(f"feather demo on http://127.0.0.1:{port}  (root={root})", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
