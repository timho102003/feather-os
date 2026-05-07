"""Tests for the Anthropic Claude (Messages API) provider."""

from __future__ import annotations

import asyncio
import json as _json
from typing import Any

import httpx
import pytest

from feather.models import ClaudeConfig, RuntimeEvent
from feather.providers.claude_provider import (
    ClaudeBillingError,
    ClaudeMessagesProvider,
    ClaudeOverloadedError,
    ClaudeStreamError,
    ClaudeStreamIdleTimeoutError,
    ClaudeStreamWallClockError,
    SSEParser,
    _retry_sleep_seconds,
    parse_sse_events,
)


# -------------------------------------------------------------- SSE parser


def test_parse_sse_events_recognizes_named_events_and_payloads() -> None:
    raw = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":5}}}\n\n'
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n\n'
    )
    events = list(parse_sse_events(raw))
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "message_stop",
    ]
    assert events[0][1]["message"]["id"] == "msg_1"


def test_sse_parser_handles_split_chunks() -> None:
    parser = SSEParser()
    a = list(parser.feed(b"event: content_block_delta\ndata: {\"type\":"))
    b = list(parser.feed(b'"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}\n\n'))
    assert a == []
    assert len(b) == 1
    assert b[0][1]["delta"]["text"] == "x"


def test_sse_parser_skips_colon_comments_and_unknown_fields() -> None:
    """``: heartbeat`` lines and ``id:``/``retry:`` fields must be ignored."""

    raw = (
        b": ping\n"
        b"id: 1\n"
        b"retry: 5000\n"
        b"event: ping\n"
        b'data: {"type":"ping"}\n\n'
    )
    events = list(parse_sse_events(raw))
    assert events == [("ping", {"type": "ping"})]


def test_sse_parser_handles_crlf_line_endings() -> None:
    raw = b'event: ping\r\ndata: {"type":"ping"}\r\n\r\n'
    assert list(parse_sse_events(raw)) == [("ping", {"type": "ping"})]


def test_sse_parser_preserves_multibyte_utf8_split_across_chunks() -> None:
    parser = SSEParser()
    a = list(parser.feed(b"event: content_block_delta\ndata: {\"delta\":{\"text\":\"h\xe2\x82"))
    b = list(parser.feed(b'\xac\"}}\n\n'))
    events = a + b
    assert len(events) == 1
    assert events[0][1]["delta"]["text"] == "h€"


def test_sse_parser_skips_event_with_no_data_lines() -> None:
    """Comment-only events have no ``data:`` payload — must be silently ignored."""

    parser = SSEParser()
    events = list(parser.feed(b"event: ping\n\n"))
    assert events == []


def test_sse_parser_drops_malformed_json_payload() -> None:
    parser = SSEParser()
    events = list(parser.feed(b"event: x\ndata: not-json\n\n"))
    assert events == []


# -------------------------------------------------------- _retry_sleep_seconds


def test_retry_sleep_seconds_honors_integer_retry_after() -> None:
    sleep = _retry_sleep_seconds(429, 0, 0.5, "3")
    assert sleep == 3.0


def test_retry_sleep_seconds_clamps_huge_retry_after() -> None:
    sleep = _retry_sleep_seconds(429, 0, 0.5, "9999")
    assert sleep <= 60.0


def test_retry_sleep_seconds_falls_back_to_jittered_backoff_when_no_header() -> None:
    sleep = _retry_sleep_seconds(500, 2, 0.5, None)
    # base * 2^2 = 2.0; with jitter up to base it's <= 2.5.
    assert 2.0 <= sleep <= 2.5 + 1e-6


def test_retry_sleep_seconds_ignores_malformed_retry_after() -> None:
    """Anything that's not pure digits and not an HTTP date drops back to backoff."""

    sleep = _retry_sleep_seconds(429, 0, 0.5, "not-a-date")
    assert 0.5 <= sleep <= 1.0 + 1e-6


# ------------------------------------------------------- provider.complete


def _sse_bytes(events: list[tuple[str, dict[str, Any]]]) -> bytes:
    out = b""
    for name, payload in events:
        out += f"event: {name}\n".encode()
        out += b"data: " + _json.dumps(payload).encode() + b"\n\n"
    return out


def _sse_response(
    events: list[tuple[str, dict[str, Any]]], status: int = 200
) -> httpx.Response:
    return httpx.Response(
        status,
        content=_sse_bytes(events),
        headers={"content-type": "text/event-stream"},
    )


def _happy_path_events() -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_01ABC",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 12, "cache_read_input_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hel"},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "lo"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 7},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


@pytest.mark.asyncio
async def test_complete_returns_model_turn_and_emits_text_deltas(monkeypatch) -> None:
    events: list[RuntimeEvent] = []
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _json.loads(request.content.decode())
        captured["headers"] = dict(request.headers)
        return _sse_response(_happy_path_events())

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cfg = ClaudeConfig()
    provider = ClaudeMessagesProvider(cfg, transport=httpx.MockTransport(handler))
    try:
        turn = await provider.complete(
            instructions="sys",
            input_items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ],
            tools=[],
            previous_response_id=None,
            event_handler=events.append,
        )
    finally:
        await provider.aclose()

    assert turn.response_id == "msg_01ABC"
    assert turn.output_text == "hello"
    # Final usage merges message_start and message_delta payloads.
    assert turn.usage == {
        "input_tokens": 12,
        "cache_read_input_tokens": 0,
        "output_tokens": 7,
    }
    # Required Anthropic headers were sent.
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == cfg.anthropic_version
    # Two text deltas streamed to the event handler.
    text_events = [e for e in events if e.kind == "assistant_text_delta"]
    assert [e.text for e in text_events] == ["hel", "lo"]


@pytest.mark.asyncio
async def test_complete_includes_anthropic_beta_header_when_configured(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return _sse_response(_happy_path_events())

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    cfg = ClaudeConfig(
        anthropic_beta=("extended-cache-ttl-2025-04-11", "interleaved-thinking-2025-05-14")
    )
    provider = ClaudeMessagesProvider(cfg, transport=httpx.MockTransport(handler))
    try:
        await provider.complete(
            instructions="s",
            input_items=[{"role": "user", "content": "hi"}],
            tools=[],
            previous_response_id=None,
        )
    finally:
        await provider.aclose()

    assert (
        captured["headers"]["anthropic-beta"]
        == "extended-cache-ttl-2025-04-11,interleaved-thinking-2025-05-14"
    )


@pytest.mark.asyncio
async def test_complete_assembles_streamed_tool_use_with_input_json_delta(
    monkeypatch,
) -> None:
    """The provider must accumulate ``input_json_delta`` partial_json fragments,
    parse the final string at ``content_block_stop``, and surface the call."""

    def handler(_req: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_2",
                            "role": "assistant",
                            "content": [],
                            "usage": {"input_tokens": 1},
                        },
                    },
                ),
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "weather",
                            "input": {},
                        },
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": '{"city":',
                        },
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": '"SF"}',
                        },
                    },
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use"},
                        "usage": {"output_tokens": 5},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    provider = ClaudeMessagesProvider(
        ClaudeConfig(), transport=httpx.MockTransport(handler)
    )
    try:
        turn = await provider.complete(
            instructions="s",
            input_items=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "name": "weather",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            previous_response_id=None,
        )
    finally:
        await provider.aclose()

    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].call_id == "toolu_1"
    assert turn.tool_calls[0].name == "weather"
    assert turn.tool_calls[0].arguments == {"city": "SF"}


@pytest.mark.asyncio
async def test_complete_unparseable_input_json_preserved_as_raw(monkeypatch) -> None:
    """Malformed accumulated tool input must not crash; raw string is preserved."""

    def handler(_req: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {"id": "m", "content": [], "usage": {}},
                    },
                ),
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "f",
                            "input": {},
                        },
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": "{garbled",
                        },
                    },
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                ("message_stop", {"type": "message_stop"}),
            ]
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    provider = ClaudeMessagesProvider(
        ClaudeConfig(), transport=httpx.MockTransport(handler)
    )
    try:
        turn = await provider.complete(
            instructions="s",
            input_items=[{"role": "user", "content": "hi"}],
            tools=[],
            previous_response_id=None,
        )
    finally:
        await provider.aclose()

    assert turn.tool_calls[0].arguments == {"raw_arguments": "{garbled"}


@pytest.mark.asyncio
async def test_complete_emits_402_as_billing_error(monkeypatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            json={"type": "error", "error": {"type": "billing_error", "message": "no funds"}},
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    provider = ClaudeMessagesProvider(
        ClaudeConfig(max_attempts=1), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ClaudeBillingError):
            await provider.complete(
                instructions="s",
                input_items=[{"role": "user", "content": "hi"}],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_complete_400_raises_immediately_no_retry(monkeypatch) -> None:
    attempts = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}},
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    provider = ClaudeMessagesProvider(
        ClaudeConfig(max_attempts=3), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await provider.complete(
                instructions="s",
                input_items=[{"role": "user", "content": "hi"}],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()
    assert attempts["n"] == 1


@pytest.mark.asyncio
async def test_complete_429_retries_then_succeeds(monkeypatch) -> None:
    attempts = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "slow"},
                },
            )
        return _sse_response(_happy_path_events())

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    provider = ClaudeMessagesProvider(
        ClaudeConfig(max_attempts=3),
        transport=httpx.MockTransport(handler),
    )
    try:
        turn = await provider.complete(
            instructions="s",
            input_items=[{"role": "user", "content": "hi"}],
            tools=[],
            previous_response_id=None,
        )
    finally:
        await provider.aclose()
    assert attempts["n"] == 2
    assert turn.output_text == "hello"


@pytest.mark.asyncio
async def test_complete_529_overloaded_retries_then_surfaces_terminal_error(
    monkeypatch,
) -> None:
    """529 is the Anthropic-specific overloaded code; persistent 529 must
    surface as :class:`ClaudeOverloadedError`."""

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            529,
            headers={"retry-after": "0"},
            json={
                "type": "error",
                "error": {"type": "overloaded_error", "message": "overloaded"},
            },
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    provider = ClaudeMessagesProvider(
        ClaudeConfig(max_attempts=2), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ClaudeOverloadedError):
            await provider.complete(
                instructions="s",
                input_items=[{"role": "user", "content": "hi"}],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_complete_mid_stream_error_event_terminates(monkeypatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {"id": "m", "content": [], "usage": {}},
                    },
                ),
                (
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "overloaded_error",
                            "message": "Overloaded",
                        },
                    },
                ),
            ]
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    provider = ClaudeMessagesProvider(
        ClaudeConfig(max_attempts=1), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ClaudeStreamError):
            await provider.complete(
                instructions="s",
                input_items=[{"role": "user", "content": "hi"}],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_complete_idle_timeout_when_stream_silent(monkeypatch) -> None:
    """A stream that opens 200 OK then never emits a chunk must trip the idle timer."""

    async def handler(_req: httpx.Request) -> httpx.Response:
        async def slow_iter():
            await asyncio.sleep(5)
            yield b""

        return httpx.Response(
            200,
            content=slow_iter(),
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    provider = ClaudeMessagesProvider(
        ClaudeConfig(
            stream_idle_timeout_seconds=0.1,
            max_stream_wall_seconds=10,
            max_attempts=1,
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ClaudeStreamIdleTimeoutError):
            await provider.complete(
                instructions="s",
                input_items=[{"role": "user", "content": "hi"}],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_complete_wall_clock_timeout_caps_endless_pings(monkeypatch) -> None:
    """A stream that drips ``ping`` events faster than the idle threshold must
    still terminate at the wall-clock cap."""

    async def handler(_req: httpx.Request) -> httpx.Response:
        async def drip():
            event = b'event: ping\ndata: {"type":"ping"}\n\n'
            for _ in range(1000):
                yield event
                await asyncio.sleep(0.01)

        return httpx.Response(
            200,
            content=drip(),
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    provider = ClaudeMessagesProvider(
        ClaudeConfig(
            stream_idle_timeout_seconds=10,
            max_stream_wall_seconds=0.2,
            max_attempts=1,
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ClaudeStreamWallClockError):
            await provider.complete(
                instructions="s",
                input_items=[{"role": "user", "content": "hi"}],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_complete_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        ClaudeMessagesProvider(ClaudeConfig())


@pytest.mark.asyncio
async def test_complete_replays_tool_call_history_correctly(monkeypatch) -> None:
    """When replaying a function_call → function_call_output pair, the
    on-the-wire body must show the assistant tool_use → user tool_result
    alternation Anthropic requires."""

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _json.loads(request.content.decode())
        return _sse_response(_happy_path_events())

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    provider = ClaudeMessagesProvider(
        ClaudeConfig(), transport=httpx.MockTransport(handler)
    )
    try:
        await provider.complete(
            instructions="sys",
            input_items=[
                {"role": "user", "content": "ask"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "weather",
                    "arguments": {"q": "SF"},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny",
                },
                {"role": "user", "content": "thanks"},
            ],
            tools=[],
            previous_response_id=None,
        )
    finally:
        await provider.aclose()

    msgs = captured["body"]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    # Assistant turn carries one tool_use block.
    assert msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[1]["content"][0]["id"] == "call_1"
    # Trailing user turn merges the tool_result with the follow-up text.
    user_blocks = msgs[2]["content"]
    assert user_blocks[0]["type"] == "tool_result"
    assert user_blocks[0]["tool_use_id"] == "call_1"
    assert user_blocks[1]["type"] == "text"
    assert user_blocks[1]["text"] == "thanks"
