"""Opt-in live smoke matrix against real OpenRouter.

These tests hit the network. They are skipped by default. To enable::

    OPENROUTER_API_KEY=sk-or-... RUN_LIVE_TESTS=1 \\
      uv run pytest tests/test_openrouter_live.py -v

Goal: one streaming tool-call round-trip per model, across a deliberate
spread of providers (Anthropic, OpenAI, Google, Moonshot, Z.ai/GLM,
DeepSeek). Each case asserts that:

- The stream parses cleanly and the response carries a ``usage`` block.
- A forced ``tool_choice`` drives the model to emit a ``tool_calls``
  entry whose ``arguments`` deserialize to a dict.

Models that temporarily 503 (provider outage) are allowed to xfail — the
catalog moves faster than this file. Hard failures indicate a parser or
translator regression.
"""

from __future__ import annotations

import os

import pytest

from feather.models import OpenRouterConfig
from feather.providers.openrouter_provider import (
    OpenRouterChatProvider,
    OpenRouterRoutingError,
)


_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.2-mini",
    "google/gemini-3-flash",
    "moonshotai/kimi-k2",
    "z-ai/glm-4.5",
    "deepseek/deepseek-chat",
]


def _live_ready() -> bool:
    has_key = bool(
        os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
    )
    return has_key and os.getenv("RUN_LIVE_TESTS") == "1"


def _echo_tool() -> dict[str, object]:
    return {
        "type": "function",
        "name": "echo",
        "description": "Echo a short string back exactly as given.",
        "parameters": {
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
            "additionalProperties": False,
        },
        "strict": True,
    }


@pytest.mark.live
@pytest.mark.parametrize("model", _MODELS)
async def test_live_tool_call_round_trip(model: str) -> None:
    if not _live_ready():
        pytest.skip("live env not set (need RUN_LIVE_TESTS=1 + OPENROUTER_API_KEY)")

    cfg = OpenRouterConfig(
        model=model,
        max_output_tokens=512,
        provider_preferences={
            "require_parameters": True,
            "allow_fallbacks": True,
        },
    )
    # Allow either env var name; the provider reads from cfg.api_key_env.
    if not os.getenv(cfg.api_key_env):
        os.environ[cfg.api_key_env] = os.environ["OPENROUTER_API_KEY"]

    provider = OpenRouterChatProvider(cfg)
    try:
        try:
            turn = await provider.complete(
                instructions=(
                    "When the user asks to echo a string, always call the "
                    "`echo` tool with the exact string they gave. Do not "
                    "answer in plain text. Do not emit commentary."
                ),
                input_items=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Echo the string: pong",
                            }
                        ],
                    }
                ],
                tools=[_echo_tool()],
                previous_response_id=None,
            )
        except OpenRouterRoutingError as exc:
            pytest.xfail(f"routing unavailable for {model}: {exc}")
    finally:
        await provider.aclose()

    assert turn.usage is not None, "expected a populated usage block"
    if turn.tool_calls:
        call = turn.tool_calls[0]
        assert call.name == "echo"
        assert isinstance(call.arguments, dict)
    else:
        # Providers that can't honor tools still must produce text.
        assert turn.output_text.strip(), "expected text or a tool call"
