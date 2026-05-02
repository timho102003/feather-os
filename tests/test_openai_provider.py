"""Tests for OpenAI request assembly."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from feather.models import MCPServerConfig, OpenAIConfig, ProviderRequestConfig, ReasoningConfig
from feather.providers.openai_provider import (
    OpenAIResponsesProvider,
    OpenAIStreamError,
    OpenAIStreamIdleTimeoutError,
)


def test_openai_provider_builds_request_kwargs(monkeypatch) -> None:
    """The provider should pass through the configured OpenAI features."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIResponsesProvider(
        OpenAIConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-5-mini",
            max_output_tokens=1234,
            temperature=0.2,
            parallel_tool_calls=True,
            prompt_cache_key="feather-key",
            prompt_cache_retention="in-memory",
            store=True,
            reasoning=ReasoningConfig(effort="low", summary="auto"),
        )
    )

    request = provider._build_request_kwargs(
        instructions="Be useful.",
        input_items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        tools=[{"type": "function", "name": "ask_user", "parameters": {}, "strict": True}],
        previous_response_id="resp-123",
    )

    assert request["model"] == "gpt-5-mini"
    assert request["parallel_tool_calls"] is True
    assert request["prompt_cache_key"] == "feather-key"
    assert request["prompt_cache_retention"] == "in_memory"
    assert request["previous_response_id"] == "resp-123"
    assert request["store"] is True
    assert request["reasoning"] == {"effort": "low", "summary": "auto"}
    assert "temperature" not in request


def test_openai_provider_passes_multimodal_input_blocks(monkeypatch) -> None:
    """Responses API input blocks should reach OpenAI unchanged."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIResponsesProvider(
        OpenAIConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-5-mini",
            max_output_tokens=1234,
            temperature=0.2,
            parallel_tool_calls=True,
        )
    )
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "inspect"},
                {"type": "input_image", "image_url": "data:image/png;base64,aQ=="},
                {
                    "type": "input_file",
                    "filename": "notes.txt",
                    "file_data": "data:text/plain;base64,aGk=",
                },
            ],
        }
    ]

    request = provider._build_request_kwargs(
        instructions="Be useful.",
        input_items=input_items,
        tools=[],
        previous_response_id=None,
    )

    assert request["input"] == input_items


def test_openai_provider_appends_enabled_mcp_servers(monkeypatch) -> None:
    """OpenAI Responses should receive configured MCP servers as native tools."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MCP_AUTH_TOKEN", "secret")
    provider = OpenAIResponsesProvider(
        OpenAIConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-5-mini",
            max_output_tokens=1234,
            temperature=0.2,
            parallel_tool_calls=True,
            reasoning=ReasoningConfig(effort="low", summary="auto"),
        ),
    )
    server = MCPServerConfig(
        label="docs",
        server_url="https://developers.openai.com/mcp",
        allowed_tools=("search_openai_docs", "fetch_openai_doc"),
        require_approval="never",
        providers=("openai",),
        headers={"X-Static": "static"},
        header_envs={"Authorization": "MCP_AUTH_TOKEN"},
    )

    request = provider._build_request_kwargs(
        instructions="Be useful.",
        input_items=[
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ],
        tools=[{"type": "function", "name": "ask_user", "parameters": {}, "strict": True}],
        previous_response_id=None,
        request_config=ProviderRequestConfig(mcp_servers=(server,)),
    )

    assert request["tools"] == [
        {"type": "function", "name": "ask_user", "parameters": {}, "strict": True},
        {
            "type": "mcp",
            "server_label": "docs",
            "server_url": "https://developers.openai.com/mcp",
            "allowed_tools": ["search_openai_docs", "fetch_openai_doc"],
            "require_approval": "never",
            "headers": {"X-Static": "static", "Authorization": "secret"},
        },
    ]


def test_openai_provider_rejects_unsupported_prompt_cache_retention(monkeypatch) -> None:
    """Invalid retention values should fail locally before the API request."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIResponsesProvider(
        OpenAIConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-5-mini",
            max_output_tokens=1234,
            temperature=0.2,
            parallel_tool_calls=True,
            prompt_cache_key="feather-key",
            prompt_cache_retention="forever",
            store=True,
            reasoning=ReasoningConfig(effort="low", summary="auto"),
        )
    )

    try:
        provider._build_request_kwargs(
            instructions="Be useful.",
            input_items=[
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ],
            tools=[{"type": "function", "name": "ask_user", "parameters": {}, "strict": True}],
            previous_response_id=None,
        )
    except ValueError as exc:
        assert "Unsupported prompt_cache_retention" in str(exc)
    else:
        raise AssertionError("Expected unsupported prompt_cache_retention to raise ValueError.")


def test_openai_provider_omits_tool_knobs_when_no_tools(monkeypatch) -> None:
    """Tool-only request knobs should not be sent on tool-less calls."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIResponsesProvider(
        OpenAIConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-5-mini",
            max_output_tokens=1234,
            temperature=0.2,
            parallel_tool_calls=True,
            prompt_cache_key=None,
            prompt_cache_retention=None,
            store=True,
            reasoning=ReasoningConfig(effort="low", summary="auto"),
        )
    )

    request = provider._build_request_kwargs(
        instructions="Be useful.",
        input_items=[
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ],
        tools=[],
        previous_response_id=None,
    )

    assert "tools" not in request
    assert "parallel_tool_calls" not in request


def test_openai_provider_applies_request_overrides(monkeypatch) -> None:
    """Per-request generation overrides should replace the default model settings."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIResponsesProvider(
        OpenAIConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-4.1-mini",
            max_output_tokens=1234,
            temperature=0.2,
            parallel_tool_calls=True,
            prompt_cache_key="feather-key",
            prompt_cache_retention="in-memory",
            store=True,
            reasoning=ReasoningConfig(effort="low", summary="auto"),
        )
    )

    request = provider._build_request_kwargs(
        instructions="Compact this.",
        input_items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "history"}]}],
        tools=[],
        previous_response_id=None,
        request_config=ProviderRequestConfig(
            model="gpt-4.1-mini-alt",
            max_output_tokens=2222,
            temperature=0.0,
            reasoning=ReasoningConfig(effort="minimal", summary="concise"),
        ),
    )

    assert request["model"] == "gpt-4.1-mini-alt"
    assert request["max_output_tokens"] == 2222
    assert request["temperature"] == 0.0
    assert request["reasoning"] == {"effort": "minimal", "summary": "concise"}


class _StallingStream:
    """Async context manager whose event iterator never yields."""

    def __init__(self, stall_event: asyncio.Event) -> None:
        self._stall_event = stall_event

    async def __aenter__(self) -> "_StallingStream":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        self._stall_event.set()

    def __aiter__(self) -> "_StallingStream":
        return self

    async def __anext__(self) -> Any:
        await asyncio.Event().wait()  # never completes
        raise AssertionError("unreachable")

    async def get_final_response(self) -> Any:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _StallingResponses:
    def __init__(self, stream: _StallingStream) -> None:
        self._stream = stream

    def stream(self, **_kwargs: Any) -> _StallingStream:
        return self._stream


class _StallingClient:
    def __init__(self, stream: _StallingStream) -> None:
        self.responses = _StallingResponses(stream)


async def test_openai_provider_raises_idle_timeout_on_stalled_stream(monkeypatch) -> None:
    """A silent stream stall must surface as OpenAIStreamIdleTimeoutError, not hang."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIResponsesProvider(
        OpenAIConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-5-mini",
            max_output_tokens=1234,
            temperature=0.2,
            parallel_tool_calls=True,
            prompt_cache_key=None,
            prompt_cache_retention=None,
            store=True,
            reasoning=ReasoningConfig(effort="low", summary="auto"),
            stream_idle_timeout_seconds=0.05,
        )
    )
    stall_event = asyncio.Event()
    provider._client = _StallingClient(_StallingStream(stall_event))  # type: ignore[assignment]

    with pytest.raises(OpenAIStreamIdleTimeoutError, match="idle >0s"):
        await provider.complete(
            instructions="x",
            input_items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            tools=[],
            previous_response_id=None,
        )
    # the async-with must have exited cleanly even though the iterator was cancelled
    assert stall_event.is_set()


class _StubEvent:
    """Minimal stand-in for an openai SDK stream event."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _StubResponse:
    """Minimal stand-in for the openai ``Response`` object returned by terminal events."""

    def __init__(
        self,
        *,
        id: str = "resp-stub",
        output_text: str = "",
        output: list[Any] | None = None,
        usage: Any = None,
        error: Any = None,
        incomplete_details: Any = None,
    ) -> None:
        self.id = id
        self.output_text = output_text
        self.output = output or []
        self.usage = usage
        self.error = error
        self.incomplete_details = incomplete_details


class _ScriptedStream:
    """Async context manager that replays a scripted sequence of events."""

    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)

    async def __aenter__(self) -> "_ScriptedStream":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    def __aiter__(self) -> "_ScriptedStream":
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def get_final_response(self) -> Any:  # pragma: no cover - should not be called
        raise AssertionError(
            "get_final_response must not be invoked; terminal events are tracked inline"
        )


class _ScriptedResponses:
    def __init__(self, stream: _ScriptedStream) -> None:
        self._stream = stream

    def stream(self, **_kwargs: Any) -> _ScriptedStream:
        return self._stream


class _ScriptedClient:
    def __init__(self, stream: _ScriptedStream) -> None:
        self.responses = _ScriptedResponses(stream)


def _make_provider(monkeypatch) -> OpenAIResponsesProvider:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return OpenAIResponsesProvider(
        OpenAIConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-5-mini",
            max_output_tokens=1234,
            temperature=0.2,
            parallel_tool_calls=True,
            prompt_cache_key=None,
            prompt_cache_retention=None,
            store=True,
            reasoning=ReasoningConfig(effort="low", summary="auto"),
            stream_idle_timeout_seconds=30.0,
        )
    )


async def test_openai_provider_raises_stream_error_on_response_incomplete(monkeypatch) -> None:
    """``response.incomplete`` must surface as OpenAIStreamError with the real reason,
    not the SDK's generic ``RuntimeError('Didn't receive a response.completed event.')``."""

    provider = _make_provider(monkeypatch)
    incomplete_details = _StubEvent(reason="max_output_tokens")
    response = _StubResponse(id="resp-incomplete", incomplete_details=incomplete_details)
    scripted = _ScriptedStream(
        [
            _StubEvent(type="response.output_text.delta", delta="partial "),
            _StubEvent(type="response.incomplete", response=response),
        ]
    )
    provider._client = _ScriptedClient(scripted)  # type: ignore[assignment]

    with pytest.raises(OpenAIStreamError, match="response incomplete"):
        await provider.complete(
            instructions="x",
            input_items=[
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ],
            tools=[],
            previous_response_id=None,
        )


async def test_openai_provider_raises_stream_error_on_response_failed(monkeypatch) -> None:
    """``response.failed`` must surface as OpenAIStreamError with the error code."""

    provider = _make_provider(monkeypatch)
    error = _StubEvent(code="server_error", message="upstream exploded")
    response = _StubResponse(id="resp-failed", error=error)
    scripted = _ScriptedStream(
        [_StubEvent(type="response.failed", response=response)]
    )
    provider._client = _ScriptedClient(scripted)  # type: ignore[assignment]

    with pytest.raises(OpenAIStreamError, match="response failed.*server_error"):
        await provider.complete(
            instructions="x",
            input_items=[
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ],
            tools=[],
            previous_response_id=None,
        )


async def test_openai_provider_raises_stream_error_on_top_level_error_event(monkeypatch) -> None:
    """A top-level ``error`` event mid-stream must surface immediately with its code/message."""

    provider = _make_provider(monkeypatch)
    scripted = _ScriptedStream(
        [_StubEvent(type="error", code="rate_limited", message="too many requests")]
    )
    provider._client = _ScriptedClient(scripted)  # type: ignore[assignment]

    with pytest.raises(OpenAIStreamError, match="stream error.*rate_limited"):
        await provider.complete(
            instructions="x",
            input_items=[
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ],
            tools=[],
            previous_response_id=None,
        )


async def test_openai_provider_raises_stream_error_when_no_terminal_event(monkeypatch) -> None:
    """A stream that closes without any terminal event must surface a clean error."""

    provider = _make_provider(monkeypatch)
    scripted = _ScriptedStream(
        [_StubEvent(type="response.output_text.delta", delta="partial ")]
    )
    provider._client = _ScriptedClient(scripted)  # type: ignore[assignment]

    with pytest.raises(OpenAIStreamError, match="without a terminal event"):
        await provider.complete(
            instructions="x",
            input_items=[
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ],
            tools=[],
            previous_response_id=None,
        )


def test_openai_provider_omits_temperature_for_gpt5_family(monkeypatch) -> None:
    """GPT-5 family requests should omit `temperature` unless explicitly supported."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIResponsesProvider(
        OpenAIConfig(
            api_key_env="OPENAI_API_KEY",
            model="gpt-5-mini",
            max_output_tokens=1234,
            temperature=0.7,
            parallel_tool_calls=True,
            prompt_cache_key=None,
            prompt_cache_retention=None,
            store=True,
            reasoning=ReasoningConfig(effort="low", summary="auto"),
        )
    )

    request = provider._build_request_kwargs(
        instructions="Be useful.",
        input_items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        tools=[],
        previous_response_id=None,
    )

    assert "temperature" not in request
