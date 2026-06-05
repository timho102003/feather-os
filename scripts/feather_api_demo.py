"""Launch the Feather web console for manual / browser end-to-end testing.

Two modes:

    uv run python scripts/feather_api_demo.py [port]
        Fake streaming provider, two demo leads. No API keys, fully offline —
        for Playwright/dev. Responses are canned.

    uv run python scripts/feather_api_demo.py [port] --real
        REAL provider + REAL tools. Picks whichever API key is in your
        environment (OPENAI_API_KEY / OPEN_ROUTER_API_KEY / ANTHROPIC_API_KEY),
        seeds two leads from the soul library, and serves the full agent so you
        can test everything end to end with genuine model output. Override the
        model with FEATHER_DEMO_MODEL=<id>.

In both modes the web console at http://127.0.0.1:<port> exercises every API
(leads, souls, messages, mid-turn input, sub-agents, transcript, config get/set,
live event stream over WebSocket).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import uvicorn

from feather.api.server import create_app
from feather.core.leads.scaffold import scaffold_lead_yaml
from feather.core.leads.soul_library import SoulLibrary
from feather.models import ModelTurn, RuntimeEvent
from feather.providers.base import BaseLLMProvider

_BASE_YAML = """database:
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
memory:
  enabled: false
default_lead: lead
"""

def _openai_block(model: str) -> str:
    """A complete openai block — load_app_config always parses it, even when
    another provider is active, so it must be present and valid in every mode."""
    return (
        "openai:\n"
        "  api_key_env: OPENAI_API_KEY\n"
        f"  model: {model}\n"
        "  max_output_tokens: 16000\n"
        "  temperature: 1.0\n"
        "  parallel_tool_calls: true\n"
        "  store: true\n"
        "  reasoning:\n"
        "    effort: low\n"
        "    summary: auto\n"
    )


# (key env var, active_provider, extra provider block or None, default model).
# Order = priority when several keys are present. The openai block is always
# emitted separately; ``extra`` is the *additional* block for non-openai providers.
_PROVIDERS = [
    ("OPENAI_API_KEY", "openai", None, "gpt-5-mini"),
    (
        "OPEN_ROUTER_API_KEY",
        "openrouter",
        "openrouter:\n"
        "  api_key_env: OPEN_ROUTER_API_KEY\n"
        "  base_url: https://openrouter.ai/api/v1\n"
        "  model: {model}\n"
        "  max_output_tokens: 32000\n"
        "  temperature: 0.7\n"
        "  parallel_tool_calls: false\n",
        "qwen/qwen3.6-plus",
    ),
    (
        "ANTHROPIC_API_KEY",
        "claude",
        "claude:\n"
        "  api_key_env: ANTHROPIC_API_KEY\n"
        "  model: {model}\n"
        "  max_output_tokens: 16000\n",
        "claude-sonnet-4-6",
    ),
]

_FAKE_LEAD = """name: {name}
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
    """Canned streaming response — offline, no keys."""

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


def _detect_provider() -> tuple[str, str, str] | None:
    """Return (active_provider, full_provider_yaml, model) for the first key in env."""
    override = os.environ.get("FEATHER_DEMO_MODEL")
    for key_env, active, extra, default_model in _PROVIDERS:
        if os.environ.get(key_env):
            model = override or default_model
            openai_model = model if active == "openai" else "gpt-5-mini"
            block = f"active_provider: {active}\n" + _openai_block(openai_model)
            if extra is not None:
                block += extra.format(model=model)
            return active, block, model
    return None


def _setup_fake(root: Path) -> None:
    (root / "config" / "agents").mkdir(parents=True)
    (root / ".feather" / "skills").mkdir(parents=True)
    (root / "config" / "app.yaml").write_text(
        _BASE_YAML + "active_provider: openai\n" + _openai_block("gpt-5-mini"),
        encoding="utf-8",
    )
    (root / "config" / "agents" / "lead.yaml").write_text(
        _FAKE_LEAD.format(name="Tim", soul="Decisive, pragmatic operator."), encoding="utf-8"
    )
    (root / "config" / "agents" / "sophia.yaml").write_text(
        _FAKE_LEAD.format(name="Sophia", soul="Meticulous, curious researcher."), encoding="utf-8"
    )


_PARALLEL_BLOCK = (
    "parallel:\n"
    "  api_key_env: PARALLEL_API_KEY\n"
    "  default_search_mode: fast\n"
    "  max_results: 5\n"
)


def _add_web_tools(path: Path) -> None:
    """Append web_search/web_fetch to a scaffolded lead's registered_tools."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write("  - web_search\n  - web_fetch\n")


def _setup_real(root: Path, provider_yaml: str, *, web: bool) -> None:
    (root / "config" / "agents").mkdir(parents=True)
    (root / ".feather" / "skills").mkdir(parents=True)
    app_yaml = _BASE_YAML + provider_yaml
    if web:
        # Feather's web tools are backed by Parallel (parallel.ai), not the LLM
        # provider — they need their own PARALLEL_API_KEY + this config block.
        app_yaml += _PARALLEL_BLOCK
    (root / "config" / "app.yaml").write_text(app_yaml, encoding="utf-8")
    # Seed two leads from the soul library. Add web tools when Parallel is on so
    # the leads (and the research/explore sub-agents they spawn) can search.
    library = SoulLibrary(root)
    lead_path = scaffold_lead_yaml(root, "lead", soul_preset=library.get("systems-thinker"))
    skeptic_path = scaffold_lead_yaml(root, "skeptic", soul_preset=library.get("the-skeptic"))
    if web:
        _add_web_tools(lead_path)
        _add_web_tools(skeptic_path)


def main() -> None:
    args = sys.argv[1:]
    real = "--real" in args
    ports = [a for a in args if a.isdigit()]
    port = int(ports[0]) if ports else 8765

    root = Path(tempfile.mkdtemp(prefix="feather-demo-"))
    # Isolate global config (keeps memory/qdrant off); API keys still come from
    # the real process environment, not ~/.feather.
    os.environ["FEATHER_HOME"] = str(root / "home")

    if real:
        detected = _detect_provider()
        if detected is None:
            keys = ", ".join(p[0] for p in _PROVIDERS)
            sys.exit(f"--real needs one of these API keys in your environment: {keys}")
        active, provider_yaml, model = detected
        web = bool(os.environ.get("PARALLEL_API_KEY"))
        _setup_real(root, provider_yaml, web=web)
        app = create_app(root)  # provider_factory=None → real provider
        web_status = "on" if web else "off — set PARALLEL_API_KEY to enable"
        print(
            f"feather REAL console on http://127.0.0.1:{port}  "
            f"(provider={active}, model={model}, web_search={web_status}, root={root})",
            flush=True,
        )
    else:
        _setup_fake(root)
        app = create_app(root, provider_factory=lambda _cfg: _DemoProvider())
        print(
            f"feather demo (fake provider) on http://127.0.0.1:{port}  "
            f"(root={root})  — add --real to use your provider",
            flush=True,
        )

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
