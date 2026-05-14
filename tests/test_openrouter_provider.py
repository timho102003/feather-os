"""Tests for the OpenRouter Chat Completions provider."""

from __future__ import annotations

import asyncio
import json as _json
from typing import Any

import httpx
import pytest

from feather.models import OpenRouterConfig, RuntimeEvent
from feather.providers.openrouter_provider import (
    OpenRouterChatProvider,
    OpenRouterCreditsExhausted,
    OpenRouterStreamError,
    OpenRouterStreamIdleTimeoutError,
    OpenRouterStreamWallClockError,
    SSEParser,
    parse_sse_events,
    retryable_post,
)


# -------------------------------------------------------------- SSE parser


def test_parse_sse_events_skips_heartbeats_and_stops_on_done() -> None:
    raw = (
        ": OPENROUTER PROCESSING\n\n"
        'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    events = list(parse_sse_events(raw.encode()))
    assert len(events) == 2
    assert events[0]["choices"][0]["delta"]["content"] == "a"
    assert events[1]["choices"][0]["delta"]["content"] == "b"


def test_sse_parser_handles_split_chunks() -> None:
    parser = SSEParser()
    a = list(parser.feed(b'data: {"choices":[{"delta":{"con'))
    b = list(parser.feed(b'tent":"x"}}]}\n\n'))
    c = list(parser.feed(b"data: [DONE]\n\n"))
    assert a == []
    assert c == []
    assert b == [{"choices": [{"delta": {"content": "x"}}]}]
    assert parser.stopped


def test_sse_parser_ignores_malformed_json_line() -> None:
    parser = SSEParser()
    events = list(parser.feed(b"data: not-json\n\n"))
    assert events == []


def test_sse_parser_handles_crlf_line_endings() -> None:
    # Some intermediate proxies normalize to CRLF; we should handle it too.
    parser = SSEParser()
    events = list(
        parser.feed(b'data: {"choices":[{"delta":{"content":"x"}}]}\r\n\r\n')
    )
    assert events == [{"choices": [{"delta": {"content": "x"}}]}]


def test_sse_parser_preserves_multibyte_utf8_split_across_chunks() -> None:
    """A codepoint straddling two network chunks must decode cleanly — never U+FFFD.

    OpenRouter streams non-ASCII content for international models all the
    time; decoding early at the byte-chunk layer corrupts it. The parser
    must buffer bytes, split on ``\\n\\n``, then decode.
    """

    parser = SSEParser()
    # The Euro sign U+20AC is 0xE2 0x82 0xAC in UTF-8. Split inside it.
    ev1 = list(parser.feed(b'data: {"choices":[{"delta":{"content":"h\xe2\x82'))
    ev2 = list(parser.feed(b'\xac""}}]}\n\n'.replace(b'""', b'"')))
    events = ev1 + ev2
    assert len(events) == 1
    assert events[0]["choices"][0]["delta"]["content"] == "h\u20ac"


def test_sse_parser_ignores_empty_data_lines_without_warning() -> None:
    parser = SSEParser()
    events = list(parser.feed(b"data: \n\n"))
    assert events == []


# ---------------------------------------------------------- retryable_post


@pytest.mark.asyncio
async def test_retryable_post_retries_429_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 2:
            return httpx.Response(
                429,
                headers={"X-RateLimit-Reset": "1"},
                json={"error": {"code": 429, "message": "slow"}},
            )
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        resp = await retryable_post(
            client,
            "http://test/ok",
            json_body={"x": 1},
            headers={},
            max_attempts=3,
            base_delay=0.001,
        )
    assert resp.status_code == 200
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_retryable_post_402_surfaces_credits_exhausted() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402, json={"error": {"code": 402, "message": "no credit"}}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        with pytest.raises(OpenRouterCreditsExhausted):
            await retryable_post(
                client,
                "http://test/x",
                json_body={},
                headers={},
                max_attempts=3,
                base_delay=0.001,
            )


@pytest.mark.asyncio
async def test_retryable_post_400_raises_no_retry() -> None:
    attempts = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(
            400, json={"error": {"code": 400, "message": "bad"}}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        with pytest.raises(httpx.HTTPStatusError):
            await retryable_post(
                client,
                "http://test/bad",
                json_body={},
                headers={},
                max_attempts=3,
                base_delay=0.001,
            )
    assert attempts["n"] == 1


@pytest.mark.asyncio
async def test_retryable_post_503_widens_routing_once() -> None:
    attempts = {"n": 0, "last_body": None}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        attempts["last_body"] = _json.loads(request.content.decode())
        if attempts["n"] == 1:
            return httpx.Response(
                503,
                json={"error": {"code": 503, "message": "no provider"}},
            )
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        resp = await retryable_post(
            client,
            "http://test/x",
            json_body={
                "provider": {"zdr": True, "quantizations": ["fp8"], "sort": "throughput"},
            },
            headers={},
            max_attempts=3,
            base_delay=0.001,
            widen_routing_on_503=True,
        )
    assert resp.status_code == 200
    body = attempts["last_body"]
    assert body is not None
    provider = body.get("provider") or {}
    assert "zdr" not in provider
    assert "quantizations" not in provider
    # Non-restrictive knobs like `sort` are preserved.
    assert provider.get("sort") == "throughput"


# ---------------------------------------------------------- provider.complete


def _sse_lines(chunks: list[dict[str, Any]]) -> bytes:
    out = b": OPENROUTER PROCESSING\n\n"
    for chunk in chunks:
        out += b"data: " + _json.dumps(chunk).encode() + b"\n\n"
    out += b"data: [DONE]\n\n"
    return out


def _sse_response(chunks: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(
        200,
        content=_sse_lines(chunks),
        headers={
            "Content-Type": "text/event-stream",
            "X-Generation-Id": "gen-1",
        },
    )


@pytest.mark.asyncio
async def test_complete_returns_model_turn_and_emits_deltas(monkeypatch) -> None:
    events: list[RuntimeEvent] = []
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(_json.loads(request.content.decode()))
        return _sse_response(
            [
                {"id": "gen-1", "choices": [{"delta": {"content": "hel"}}]},
                {"id": "gen-1", "choices": [{"delta": {"content": "lo"}}]},
                {
                    "id": "gen-1",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
            ]
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    cfg = OpenRouterConfig()
    provider = OpenRouterChatProvider(cfg, transport=httpx.MockTransport(handler))
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

    assert turn.response_id == "gen-1"
    assert turn.output_text == "hello"
    assert turn.usage == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "input_tokens": 5,
        "output_tokens": 3,
    }
    assert captured["model"] == cfg.model
    # Two delta events with content were emitted (heartbeat not counted):
    text_events = [e for e in events if e.kind == "assistant_text_delta"]
    assert [e.text for e in text_events] == ["hel", "lo"]


@pytest.mark.asyncio
async def test_complete_warns_when_resolved_model_differs_from_configured(
    monkeypatch, caplog
) -> None:
    """OpenRouter's ``models`` field (built from
    ``app.openrouter.fallback_models``) lets the upstream silently route
    away from the primary when the primary is unavailable. The provider
    must emit a WARNING that names both the configured primary and the
    resolved fallback so the substitution is visible in feather.log
    instead of only surfacing through Opik / billing dashboards.

    Regression for: user changed ``app.openrouter.model`` to
    ``deepseek/deepseek-v4-pro``, save persisted, but every request kept
    landing on ``qwen/qwen3.5-plus-02-15`` (the first slug in the
    packaged ``fallback_models``). The save was correct; the routing
    was silent.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                # OpenRouter's response ``model`` field is the SUBSTITUTED
                # one — not what we asked for in the request body.
                {
                    "id": "gen-fb",
                    "model": "qwen/qwen3.5-plus-02-15",
                    "choices": [{"delta": {"content": "hi"}}],
                },
                {
                    "id": "gen-fb",
                    "model": "qwen/qwen3.5-plus-02-15",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ]
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    cfg = OpenRouterConfig(
        model="deepseek/deepseek-v4-pro",
        fallback_models=(
            "qwen/qwen3.5-plus-02-15",
            "qwen/qwen3.5-397b-a17b",
        ),
    )
    provider = OpenRouterChatProvider(cfg, transport=httpx.MockTransport(handler))
    try:
        import logging

        with caplog.at_level(
            logging.WARNING, logger="feather.providers.openrouter_provider"
        ):
            await provider.complete(
                instructions="sys",
                input_items=[],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()

    warnings = [
        r for r in caplog.records if "model_substituted" in r.getMessage()
    ]
    assert warnings, (
        "expected a model_substituted warning when OpenRouter routes "
        "to a fallback model"
    )
    msg = warnings[0].getMessage()
    assert "deepseek/deepseek-v4-pro" in msg
    assert "qwen/qwen3.5-plus-02-15" in msg


@pytest.mark.asyncio
async def test_complete_does_not_warn_when_resolved_matches_configured(
    monkeypatch, caplog
) -> None:
    """No spurious warning when OpenRouter actually serves what we asked
    for — otherwise users get warning fatigue on every healthy request."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {
                    "id": "gen-ok",
                    "model": "deepseek/deepseek-v4-pro",
                    "choices": [{"delta": {"content": "hi"}}],
                },
                {
                    "id": "gen-ok",
                    "model": "deepseek/deepseek-v4-pro",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ]
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    cfg = OpenRouterConfig(
        model="deepseek/deepseek-v4-pro",
        fallback_models=("qwen/qwen3.5-plus-02-15",),
    )
    provider = OpenRouterChatProvider(cfg, transport=httpx.MockTransport(handler))
    try:
        import logging

        with caplog.at_level(
            logging.WARNING, logger="feather.providers.openrouter_provider"
        ):
            await provider.complete(
                instructions="sys",
                input_items=[],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()

    substitutions = [
        r for r in caplog.records if "model_substituted" in r.getMessage()
    ]
    assert not substitutions, (
        f"unexpected substitution warning when models matched: {substitutions}"
    )


@pytest.mark.asyncio
async def test_complete_no_warning_when_resolved_is_dated_snapshot(
    monkeypatch, caplog
) -> None:
    """OpenRouter pins rolling aliases (``deepseek/deepseek-v4-pro``) to dated
    snapshots (``...-20260423``) in the response model field. That's expected
    aliasing — not the silent fallback re-routing the substitution warning
    is meant to flag. Suppress the warning in this case so the real signal
    (qwen ← deepseek and friends) isn't drowned in benign noise.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {
                    "id": "gen-snap",
                    "model": "deepseek/deepseek-v4-pro-20260423",
                    "choices": [{"delta": {"content": "ok"}}],
                },
                {
                    "id": "gen-snap",
                    "model": "deepseek/deepseek-v4-pro-20260423",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ]
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    cfg = OpenRouterConfig(model="deepseek/deepseek-v4-pro")
    provider = OpenRouterChatProvider(cfg, transport=httpx.MockTransport(handler))
    try:
        import logging

        with caplog.at_level(
            logging.WARNING, logger="feather.providers.openrouter_provider"
        ):
            await provider.complete(
                instructions="sys",
                input_items=[],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()

    subs = [r for r in caplog.records if "model_substituted" in r.getMessage()]
    assert not subs, (
        f"dated-snapshot alias should not trip the substitution warning: {subs}"
    )


@pytest.mark.asyncio
async def test_complete_tolerates_missing_finish_reason_when_content_streamed(
    monkeypatch, caplog
) -> None:
    """DeepSeek reasoning models routed through OpenRouter occasionally
    send ``[DONE]`` without a preceding chunk that carries ``finish_reason``
    (an upstream protocol violation per DeepSeek's own spec). When the
    stream produced valid content and no in-flight tool calls, synthesize
    a turn with implicit ``finish_reason=stop`` and WARN — losing the
    streamed content because of a missing trailer is worse UX than the
    log line.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        # Content chunks, then [DONE] — NO finish_reason chunk at all.
        return _sse_response(
            [
                {
                    "id": "gen-noend",
                    "model": "deepseek/deepseek-v4-pro",
                    "choices": [{"delta": {"content": "answer "}}],
                },
                {
                    "id": "gen-noend",
                    "model": "deepseek/deepseek-v4-pro",
                    "choices": [{"delta": {"content": "is 42"}}],
                },
                # No finish_reason chunk, no usage-only chunk — straight to [DONE].
            ]
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    cfg = OpenRouterConfig(model="deepseek/deepseek-v4-pro")
    provider = OpenRouterChatProvider(cfg, transport=httpx.MockTransport(handler))
    try:
        import logging

        with caplog.at_level(
            logging.WARNING, logger="feather.providers.openrouter_provider"
        ):
            turn = await provider.complete(
                instructions="sys",
                input_items=[],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()

    # The streamed content survives.
    assert turn.output_text == "answer is 42"
    # And we logged that we synthesized the trailer.
    warnings = [
        r for r in caplog.records if "stream.no_final_chunk" in r.getMessage()
    ]
    assert warnings, "expected a no_final_chunk warning when synthesizing"


@pytest.mark.asyncio
async def test_complete_still_raises_when_stream_ends_with_nothing_useful(
    monkeypatch,
) -> None:
    """An empty stream with no final chunk is a real failure — synthesizing
    ``stop`` would invent a turn from nothing. Keep raising in that case
    so the caller can retry or surface the error."""

    def handler(_request: httpx.Request) -> httpx.Response:
        # [DONE] with no preceding data chunks at all.
        return _sse_response([])

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    provider = OpenRouterChatProvider(
        OpenRouterConfig(), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(OpenRouterStreamError):
            await provider.complete(
                instructions="sys",
                input_items=[],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_complete_still_raises_when_tool_calls_in_flight_with_no_final_chunk(
    monkeypatch,
) -> None:
    """If we have partial tool-call deltas but no finish_reason, synthesizing
    ``stop`` would commit a half-built tool call. Raise instead."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {
                    "id": "gen-tc",
                    "model": "deepseek/deepseek-v4-pro",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": "",
                                        },
                                    }
                                ]
                            }
                        }
                    ],
                },
                # No finish_reason — and a tool call left mid-build.
            ]
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    provider = OpenRouterChatProvider(
        OpenRouterConfig(), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(OpenRouterStreamError):
            await provider.complete(
                instructions="sys",
                input_items=[],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_complete_uses_generation_header_when_chunks_are_idless(
    monkeypatch,
) -> None:
    """OpenRouter can expose generation lookup via X-Generation-Id."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {"choices": [{"delta": {"content": "ok"}}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ]
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    provider = OpenRouterChatProvider(
        OpenRouterConfig(), transport=httpx.MockTransport(handler)
    )
    try:
        turn = await provider.complete(
            instructions="sys",
            input_items=[],
            tools=[],
            previous_response_id=None,
        )
    finally:
        await provider.aclose()

    assert turn.response_id == "gen-1"


@pytest.mark.asyncio
async def test_complete_reconstructs_tool_calls_across_chunks(monkeypatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {
                    "id": "g",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"p',
                                        },
                                    }
                                ]
                            }
                        }
                    ],
                },
                {
                    "id": "g",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": 'ath":"x"}'},
                                    }
                                ]
                            }
                        }
                    ],
                },
                {
                    "id": "g",
                    "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ]
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "x")
    provider = OpenRouterChatProvider(
        OpenRouterConfig(), transport=httpx.MockTransport(handler)
    )
    try:
        turn = await provider.complete(
            instructions="",
            input_items=[],
            tools=[],
            previous_response_id=None,
        )
    finally:
        await provider.aclose()
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "read_file"
    assert turn.tool_calls[0].arguments == {"path": "x"}


@pytest.mark.asyncio
async def test_complete_raises_on_mid_stream_error(monkeypatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        body = b"data: " + _json.dumps(
            {
                "id": "gen-1",
                "choices": [{"delta": {}, "finish_reason": "error"}],
                "error": {"code": 502, "message": "upstream blew up"},
            }
        ).encode() + b"\n\n"
        return httpx.Response(
            200,
            content=body,
            headers={"Content-Type": "text/event-stream"},
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "x")
    provider = OpenRouterChatProvider(
        OpenRouterConfig(), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(OpenRouterStreamError):
            await provider.complete(
                instructions="",
                input_items=[],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_complete_400_raises_before_streaming(monkeypatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"code": 400, "message": "bad request"}}
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "x")
    provider = OpenRouterChatProvider(
        OpenRouterConfig(), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await provider.complete(
                instructions="",
                input_items=[],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_complete_402_raises_credits_exhausted(monkeypatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402, json={"error": {"code": 402, "message": "no credit"}}
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "x")
    provider = OpenRouterChatProvider(
        OpenRouterConfig(), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(OpenRouterCreditsExhausted):
            await provider.complete(
                instructions="",
                input_items=[],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_complete_sends_auth_and_attribution_headers(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update({k: v for k, v in request.headers.items()})
        return _sse_response(
            [
                {
                    "id": "g",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ]
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "secret-key")
    cfg = OpenRouterConfig(http_referer="https://example", app_title="Tester")
    provider = OpenRouterChatProvider(cfg, transport=httpx.MockTransport(handler))
    try:
        await provider.complete(
            instructions="",
            input_items=[],
            tools=[],
            previous_response_id=None,
        )
    finally:
        await provider.aclose()
    assert captured.get("authorization") == "Bearer secret-key"
    assert captured.get("http-referer") == "https://example"
    assert captured.get("x-title") == "Tester"


def test_provider_raises_if_api_key_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPEN_ROUTER_API_KEY"):
        OpenRouterChatProvider(OpenRouterConfig())


@pytest.mark.asyncio
async def test_complete_retries_pre_stream_429_then_succeeds(monkeypatch) -> None:
    """Pre-stream 429 must route through the retry helper, not surface raw."""

    attempts = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(
                429,
                headers={"X-RateLimit-Reset": "0"},
                json={"error": {"code": 429, "message": "slow down"}},
            )
        return _sse_response(
            [
                {
                    "id": "g",
                    "choices": [{"delta": {"content": "ok"}}],
                },
                {
                    "id": "g",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ]
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "x")
    cfg = OpenRouterConfig(max_attempts=3)
    provider = OpenRouterChatProvider(cfg, transport=httpx.MockTransport(handler))
    try:
        turn = await provider.complete(
            instructions="",
            input_items=[],
            tools=[],
            previous_response_id=None,
        )
    finally:
        await provider.aclose()
    assert attempts["n"] == 2
    assert turn.output_text == "ok"


@pytest.mark.asyncio
async def test_complete_pre_stream_502_retries_until_success(monkeypatch) -> None:
    attempts = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 2:
            return httpx.Response(
                502, json={"error": {"code": 502, "message": "upstream"}}
            )
        return _sse_response(
            [
                {
                    "id": "g",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 0},
                }
            ]
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "x")
    provider = OpenRouterChatProvider(
        OpenRouterConfig(max_attempts=3), transport=httpx.MockTransport(handler)
    )
    try:
        turn = await provider.complete(
            instructions="",
            input_items=[],
            tools=[],
            previous_response_id=None,
        )
    finally:
        await provider.aclose()
    assert attempts["n"] == 2
    assert turn.response_id == "g"


@pytest.mark.asyncio
async def test_complete_idle_stall_raises_idle_timeout(monkeypatch) -> None:
    """A stream that goes silent after the first chunk must trip the idle timer,
    not hang the event loop forever."""

    async def slow_body_iter():
        # One initial chunk, then indefinite silence.
        yield b": OPENROUTER PROCESSING\n\n"
        await asyncio.sleep(5.0)  # longer than the idle budget below

    # Build a response whose body is an async iterator that stalls.
    # We use a transport that returns a response backed by slow_body_iter.
    from httpx import Response as _Resp

    async def handler_async(_req: httpx.Request) -> _Resp:
        return _Resp(
            200,
            content=slow_body_iter(),
            headers={"Content-Type": "text/event-stream"},
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "x")
    cfg = OpenRouterConfig(stream_idle_timeout_seconds=0.1)
    provider = OpenRouterChatProvider(
        cfg, transport=httpx.MockTransport(handler_async)
    )
    try:
        with pytest.raises(OpenRouterStreamIdleTimeoutError):
            await provider.complete(
                instructions="",
                input_items=[],
                tools=[],
                previous_response_id=None,
            )
    finally:
        await provider.aclose()


async def test_stream_wall_clock_cap_fires_when_keepalives_starve_idle_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misbehaving upstream that drips keep-alives faster than the idle
    threshold must still be bounded by the overall wall-clock cap."""

    async def keep_alive_body() -> Any:
        # Each chunk lands inside the 0.2s idle window, so the idle timer
        # never trips — but no actual content is ever produced. Without a
        # wall-clock cap this loop is unbounded.
        for _ in range(200):
            yield b": OPENROUTER PROCESSING\n\n"
            await asyncio.sleep(0.05)

    from httpx import Response as _Resp

    async def handler_async(_req: httpx.Request) -> _Resp:
        return _Resp(
            200,
            content=keep_alive_body(),
            headers={"Content-Type": "text/event-stream"},
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "x")
    cfg = OpenRouterConfig(
        stream_idle_timeout_seconds=0.2,    # idle never trips — keep-alives are 50ms
        max_stream_wall_seconds=0.4,        # wall-clock cap MUST fire after ~0.4s
        max_attempts=1,                     # don't retry — surface the error
    )
    provider = OpenRouterChatProvider(
        cfg, transport=httpx.MockTransport(handler_async)
    )
    try:
        loop = asyncio.get_event_loop()
        started = loop.time()
        with pytest.raises(OpenRouterStreamWallClockError):
            await provider.complete(
                instructions="",
                input_items=[],
                tools=[],
                previous_response_id=None,
            )
        elapsed = loop.time() - started
        assert elapsed < 1.5, (
            f"wall-clock cap should fire promptly; took {elapsed:.2f}s "
            "(if this exceeds 1.5s, the bound isn't actually applied)"
        )
    finally:
        await provider.aclose()


async def test_stream_wall_clock_cap_does_not_fire_for_fast_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normally-completing stream must NOT be killed by the wall-clock cap."""

    async def fast_body() -> Any:
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b'data: {"id":"gen-1","model":"qwen/q","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
        yield b"data: [DONE]\n\n"

    from httpx import Response as _Resp

    async def handler_async(_req: httpx.Request) -> _Resp:
        return _Resp(
            200,
            content=fast_body(),
            headers={"Content-Type": "text/event-stream"},
        )

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "x")
    cfg = OpenRouterConfig(
        stream_idle_timeout_seconds=2.0,
        max_stream_wall_seconds=2.0,        # tight but enough for a normal stream
    )
    provider = OpenRouterChatProvider(
        cfg, transport=httpx.MockTransport(handler_async)
    )
    try:
        turn = await provider.complete(
            instructions="",
            input_items=[],
            tools=[],
            previous_response_id=None,
        )
        assert turn.output_text == "hi"
    finally:
        await provider.aclose()
