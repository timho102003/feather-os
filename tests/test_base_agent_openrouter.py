"""End-to-end BaseAgent tool loop against a MockTransport-backed OpenRouter provider.

This test exists to verify that the real :class:`OpenRouterChatProvider`
integrates cleanly with :class:`BaseAgent`:

- ``run_loop`` drives a two-turn conversation (assistant asks for a tool
  call → tool executes → assistant produces the final answer).
- The tool call arrives via the SSE delta reconstructor on turn 1.
- On turn 2 the provider sees the original user message *and* the
  ``function_call_output`` item produced by the tool, translates both
  into Chat Completions shape, and returns a final text.
- SessionStore ends up with the expected USER / ASSISTANT / TOOL rows.

Unit-level shape guarantees (translate_request output, tool-call
reconstruction, error paths) live in ``test_openrouter_provider.py`` and
``test_openrouter_translator.py``; this file is the only place where the
full loop runs with BaseAgent touching the real provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from feather.core.base_agent import BaseAgent
from feather.core.prompt_builder import PromptBuilder
from feather.models import (
    AgentConfig,
    AgentOutcome,
    MessageRole,
    OpenRouterConfig,
    OpenRouterTracingConfig,
)
from feather.providers.openrouter_provider import OpenRouterChatProvider
from feather.skills.catalog import SkillCatalog
from feather.storage.session_store import SessionStore
from feather.storage.tool_output_store import ToolOutputStore
from feather.tools.read_file_tool import ReadFileTool
from feather.tools.registry import ToolRegistry


def _sse(chunks: list[dict[str, Any]]) -> bytes:
    out = b": OPENROUTER PROCESSING\n\n"
    for chunk in chunks:
        out += b"data: " + json.dumps(chunk).encode() + b"\n\n"
    out += b"data: [DONE]\n\n"
    return out


def _stream_response(chunks: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(
        200,
        content=_sse(chunks),
        headers={
            "Content-Type": "text/event-stream",
            "X-Generation-Id": chunks[0].get("id", "gen"),
        },
    )


def _tool_call_turn(call_id: str, path: str) -> list[dict[str, Any]]:
    args = json.dumps({"path": path})
    return [
        {
            "id": "gen-1",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": args[: len(args) // 2],
                                },
                            }
                        ]
                    }
                }
            ],
        },
        {
            "id": "gen-1",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": args[len(args) // 2 :]},
                            }
                        ]
                    }
                }
            ],
        },
        {
            "id": "gen-1",
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    ]


def _final_text_turn(text: str) -> list[dict[str, Any]]:
    return [
        {"id": "gen-2", "choices": [{"delta": {"content": text}}]},
        {
            "id": "gen-2",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 7},
        },
    ]


@pytest.mark.asyncio
async def test_base_agent_completes_tool_loop_under_openrouter_provider(
    tmp_path: Path, monkeypatch
) -> None:
    """A two-turn read_file tool loop completes end-to-end under OpenRouterChatProvider."""

    # Workspace: a pyproject-like file the agent is asked to read.
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = \"integration\"\nrequires-python = \">=3.12\"\n",
        encoding="utf-8",
    )
    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    turn_state = {"turn": 0, "captured_bodies": []}

    def handler(request: httpx.Request) -> httpx.Response:
        turn_state["captured_bodies"].append(json.loads(request.content.decode()))
        turn_state["turn"] += 1
        if turn_state["turn"] == 1:
            return _stream_response(_tool_call_turn("call_abc", "pyproject.toml"))
        return _stream_response(_final_text_turn("Python >=3.12."))

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    provider = OpenRouterChatProvider(
        OpenRouterConfig(max_output_tokens=4000),
        transport=httpx.MockTransport(handler),
    )

    read_tool = ReadFileTool(tmp_path)
    tool_registry = ToolRegistry([read_tool])
    prompt_builder = PromptBuilder(
        SkillCatalog(tmp_path / ".feather" / "skills"), tool_registry
    )

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=["read_file"],
            ),
            prompt_builder=prompt_builder,
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=tool_registry,
        )

        session_id = await agent.create_session()
        result = await agent.run(
            session_id,
            "Read pyproject.toml and tell me the Python version.",
        )
        # Snapshot while the connection is still open.
        roles_recorded = await _role_sequence(session_store, session_id)
    finally:
        await provider.aclose()
        await session_store.close()

    assert result.status == AgentOutcome.COMPLETED
    assert "Python >=3.12." in result.assistant_text
    assert MessageRole.USER in roles_recorded
    assert MessageRole.TOOL in roles_recorded
    assert MessageRole.ASSISTANT in roles_recorded

    # Turn-1 body: system + user message. Turn-2 body: under a stateless
    # provider we replay the full prior conversation — otherwise the
    # model has no way to ground the tool output it's being asked to
    # answer over.
    bodies = turn_state["captured_bodies"]
    assert len(bodies) == 2
    assert any(msg["role"] == "user" for msg in bodies[0]["messages"])
    turn2 = bodies[1]["messages"]

    def _flatten(msg: dict[str, Any]) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        parts: list[str] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
        return " ".join(parts)

    turn2_concat = " ".join(_flatten(m) for m in turn2 if m.get("role") != "system")
    # The replay must carry the user's original question AND reference to
    # the tool result. Either shape is acceptable: structural tool-role
    # messages *or* a single rendered user message containing both.
    assert "Read pyproject.toml" in turn2_concat, (
        f"turn-2 body lost the original user question; got: {turn2_concat[:300]}"
    )
    assert "requires-python" in turn2_concat or "python" in turn2_concat.lower(), (
        f"turn-2 body lost the tool result content; got: {turn2_concat[:300]}"
    )
    assert any(
        msg.get("role") == "assistant" and msg.get("tool_calls") for msg in turn2
    ), f"turn-2 body lost structural assistant tool_calls; got: {turn2}"
    assert any(
        msg.get("role") == "tool" and msg.get("tool_call_id") == "call_abc"
        for msg in turn2
    ), f"turn-2 body lost structural tool result; got: {turn2}"
    # cache_control breakpoint should sit on the system message in both turns.
    for body in bodies:
        sys_msg = body["messages"][0]
        assert sys_msg["role"] == "system"
        assert sys_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_base_agent_emits_opik_tracing_metadata_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    """When OpenRouter tracing is enabled, every turn body carries the trace bundle.

    This is the end-to-end proof that BaseAgent threads its session/agent
    identity into the OpenRouter wire payload, where Opik (and any other
    OpenRouter broadcast destination) can pick it up.
    """

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode()))
        return _stream_response(_final_text_turn("ok"))

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    provider = OpenRouterChatProvider(
        OpenRouterConfig(
            max_output_tokens=4000,
            tracing=OpenRouterTracingConfig(
                enabled=True,
                user="ops@example.com",
                metadata={"deployment": "prod", "build_sha": "abc123"},
            ),
        ),
        transport=httpx.MockTransport(handler),
    )
    tool_registry = ToolRegistry([])
    prompt_builder = PromptBuilder(
        SkillCatalog(tmp_path / ".feather" / "skills"), tool_registry
    )

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="primary lead agent",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
            ),
            prompt_builder=prompt_builder,
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=tool_registry,
        )
        session_id = await agent.create_session()
        await agent.run(session_id, "ping")
    finally:
        await provider.aclose()
        await session_store.close()

    assert len(captured_bodies) == 1
    body = captured_bodies[0]
    assert body["session_id"] == session_id
    assert body["user"] == "ops@example.com"
    trace = body["trace"]
    assert trace["trace_name"] == "feather/Lead"
    assert trace["generation_name"] == "anthropic/claude-sonnet-4.6"
    assert trace["feather_app"] == "feather-agent-os"
    assert trace["feather_agent_name"] == "Lead"
    assert trace["feather_agent_role"] == "primary lead agent"
    assert trace["feather_session_id"] == session_id
    assert trace["deployment"] == "prod"
    assert trace["build_sha"] == "abc123"


@pytest.mark.asyncio
async def test_base_agent_emits_no_tracing_fields_when_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    """Default config (no tracing block) ⇒ wire body free of trace fields.

    Pins backwards-compatibility: anyone not opting into tracing keeps
    the exact byte-on-the-wire they have today.
    """

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode()))
        return _stream_response(_final_text_turn("ok"))

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    provider = OpenRouterChatProvider(
        OpenRouterConfig(max_output_tokens=4000),  # tracing field defaults to None
        transport=httpx.MockTransport(handler),
    )
    tool_registry = ToolRegistry([])
    prompt_builder = PromptBuilder(
        SkillCatalog(tmp_path / ".feather" / "skills"), tool_registry
    )

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
            ),
            prompt_builder=prompt_builder,
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=tool_registry,
        )
        session_id = await agent.create_session()
        await agent.run(session_id, "ping")
    finally:
        await provider.aclose()
        await session_store.close()

    body = captured_bodies[0]
    assert "user" not in body
    assert "session_id" not in body
    assert "trace" not in body


async def _role_sequence(store: SessionStore, session_id: str) -> list[MessageRole]:
    """Return the ordered sequence of message roles written during the session."""

    rows = await store.list_messages(session_id)
    return [row.role for row in rows]
