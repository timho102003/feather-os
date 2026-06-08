"""Anthropic Claude (Messages API) provider.

This provider implements :class:`~feather.providers.base.BaseLLMProvider`
against Anthropic's ``POST /v1/messages`` endpoint. It owns three
concerns — HTTP transport, retry semantics, and SSE parsing — and
delegates every request/response shape decision to
:mod:`feather.providers.claude_translator`.

Design notes:

- One :class:`httpx.AsyncClient` per provider instance. It is reused
  across turns so the HTTP/2 connection and keep-alive state are
  preserved.
- ``stream=True`` is always set on the request body. Anthropic emits
  named SSE events (``event: <name>\\ndata: {...}\\n\\n``) — no
  ``[DONE]`` sentinel; the stream ends when the HTTP body closes after
  ``message_stop``.
- Pre-stream errors (status ≥ 400 before any SSE event) are classified
  once: ``402`` becomes :class:`ClaudeBillingError`; ``400/401/403/404``
  raise immediately; ``408/409/429/500/503/504/529`` route through the
  retry loop and honor ``retry-after`` on 429 / 529.
- Once the stream is open, we never retry that stream. A mid-stream
  ``event: error`` (or a top-level ``error`` field on any chunk) raises
  :class:`ClaudeStreamError` as a terminal signal.
- Idle-timeout budget is measured between events. A stall longer than
  ``cfg.stream_idle_timeout_seconds`` raises
  :class:`ClaudeStreamIdleTimeoutError`. A second-line wall-clock cap
  catches misbehaving upstreams that drip ``ping`` events forever.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Iterable

import httpx

from feather.models import (
    ClaudeConfig,
    EventHandler,
    ModelTurn,
    ProviderRequestConfig,
    RuntimeEvent,
)
from feather.providers.base import BaseLLMProvider
from feather.providers.claude_translator import translate_request, translate_response

logger = logging.getLogger(__name__)


# -------------------------------------------------------------- exceptions


class ClaudeBillingError(RuntimeError):
    """Raised on HTTP 402 (billing). Never retried."""


class ClaudeOverloadedError(RuntimeError):
    """Raised on a persistent 529 (overloaded) after retries."""


class ClaudeStreamIdleTimeoutError(TimeoutError):
    """Raised when an SSE stream stalls longer than the configured idle budget."""


class ClaudeStreamWallClockError(TimeoutError):
    """Raised when an SSE stream exceeds the configured wall-clock cap.

    The idle-timeout check only bounds the gap between consecutive bytes.
    A misbehaving upstream that drips ``ping`` events faster than the
    idle threshold can keep a stream alive indefinitely while never
    producing actual content. This wall-clock cap is the second-line
    defence — once the stream exceeds
    ``ClaudeConfig.max_stream_wall_seconds`` it raises regardless of
    inter-byte gap.
    """


class ClaudeStreamError(RuntimeError):
    """Raised on a mid-stream terminal error (``event: error`` payload)."""


# -------------------------------------------------------------- SSE parser


class SSEParser:
    """Incremental SSE parser for Anthropic Messages streams.

    Maintains a **byte** buffer across ``feed()`` calls (an incoming HTTP
    chunk can split a multi-byte UTF-8 codepoint, and decoding too early
    corrupts the content with U+FFFD). Splits on blank-line event
    boundaries (``\\n\\n``, with CRLF normalization), discards
    colon-prefixed comment lines, and yields ``(event_name, payload)``
    tuples where ``event_name`` is the SSE ``event:`` field (defaulting
    to ``"message"`` when omitted) and ``payload`` is the decoded JSON
    body. Lines other than ``event:`` and ``data:`` are ignored.

    Anthropic's wire format always sets ``event:``, so callers can rely
    on that name; the type-discriminator on the payload (``data.type``)
    matches it and is exposed for redundancy.
    """

    def __init__(self) -> None:
        self._buffer: bytes = b""

    def feed(self, chunk: bytes) -> Iterable[tuple[str, dict[str, Any]]]:
        """Consume raw bytes, yield any complete events."""

        self._buffer += chunk
        if b"\r\n" in self._buffer:
            self._buffer = self._buffer.replace(b"\r\n", b"\n")
        while b"\n\n" in self._buffer:
            raw_event, self._buffer = self._buffer.split(b"\n\n", 1)
            event = raw_event.decode("utf-8", errors="replace")
            event_name = "message"
            data_lines: list[str] = []
            for line in event.splitlines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[len("data:"):].lstrip())
                    continue
                # Other SSE fields (``id:``, ``retry:``) are ignored.
            if not data_lines:
                continue
            payload_str = "\n".join(data_lines)
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                logger.warning(
                    "claude_sse_parse_failed event=%s len=%d head=%r",
                    event_name,
                    len(payload_str),
                    payload_str[:60],
                )
                continue
            if not isinstance(payload, dict):
                continue
            yield event_name, payload


def parse_sse_events(raw: bytes) -> Iterable[tuple[str, dict[str, Any]]]:
    """One-shot parse of a complete SSE byte blob."""

    parser = SSEParser()
    yield from parser.feed(raw)


# ------------------------------------------------------------ retry helper


# Anthropic-specific retryable codes. 408/409 are general-purpose
# transient codes; 429 is rate-limited; 500 is generic server error;
# 503 is unavailable; 504 is gateway timeout; 529 is the Anthropic-
# specific "overloaded" code that warrants polite back-off and retry.
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 429, 500, 503, 504, 529})
_MAX_RETRY_AFTER_SLEEP: float = 60.0


class _PreStreamRetryable(Exception):
    """Internal signal that a pre-stream response is retryable."""

    def __init__(
        self, status: int, retry_after: str | None, inner: Exception
    ) -> None:
        super().__init__(f"pre-stream {status}")
        self.status = status
        self.retry_after = retry_after
        self.inner = inner


def _retry_sleep_seconds(
    status: int,
    attempt: int,
    base_delay: float,
    retry_after_header: str | None,
) -> float:
    """Compute backoff; honors ``retry-after`` (seconds OR HTTP date) but clamps to a sane ceiling.

    ``retry-after`` is the canonical Anthropic backoff hint and arrives
    on both 429 and 529 responses. RFC 7231 lets it be either an integer
    delta-seconds or an HTTP date — we honour the integer form (the
    common case Anthropic actually emits) and parse the HTTP date as a
    fallback. Anything malformed falls back to exponential backoff with
    jitter.
    """

    if retry_after_header:
        hint = retry_after_header.strip()
        if hint.isdigit():
            return min(float(hint), _MAX_RETRY_AFTER_SLEEP)
        try:
            target = datetime.strptime(hint, "%a, %d %b %Y %H:%M:%S GMT")
            delta = max(0.0, target.timestamp() - time.time())
            return min(delta, _MAX_RETRY_AFTER_SLEEP)
        except ValueError:
            pass
    sleep = base_delay * (2 ** attempt)
    sleep += random.uniform(0, base_delay)
    return sleep


# ---------------------------------------------------- ClaudeMessagesProvider


class ClaudeMessagesProvider(BaseLLMProvider):
    """Anthropic Claude (Messages API) provider.

    One instance owns one :class:`httpx.AsyncClient` and is safe to call
    ``complete()`` on concurrently from multiple tasks — each turn opens
    its own response stream via ``client.stream(...)``.

    Args:
        cfg: Provider configuration, typically loaded from the
            ``claude:`` block of ``config/app.yaml``.
        transport: Optional httpx transport override for tests.
    """

    stateful = False

    def __init__(
        self,
        cfg: ClaudeConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._cfg = cfg
        api_key = os.getenv(cfg.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing required environment variable: {cfg.api_key_env}"
            )
        headers = {
            "x-api-key": api_key,
            "anthropic-version": cfg.anthropic_version,
            "content-type": "application/json",
        }
        if cfg.anthropic_beta:
            headers["anthropic-beta"] = ",".join(cfg.anthropic_beta)
        self._headers = headers
        # ``read=None`` is intentional — a streaming response can exceed
        # any per-read deadline; idle protection lives in the event loop.
        timeout = httpx.Timeout(
            cfg.request_timeout_seconds,
            connect=30.0,
            read=None,
        )
        client_kwargs: dict[str, Any] = {
            "base_url": cfg.base_url,
            "timeout": timeout,
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        self._client = httpx.AsyncClient(**client_kwargs)

    async def complete(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        event_handler: EventHandler | None = None,
        request_config: ProviderRequestConfig | None = None,
    ) -> ModelTurn:
        """Run one Anthropic Messages turn.

        ``previous_response_id`` is accepted for interface compatibility;
        the Messages API is stateless, so it is used only for log
        correlation. Feather's compaction + history replay guarantees
        that ``input_items`` already carries everything the model needs.
        """

        active_config = request_config or ProviderRequestConfig()
        body = translate_request(
            instructions=instructions,
            input_items=input_items,
            tools=tools,
            request_config=active_config,
            cfg=self._cfg,
            cache_prefix=active_config.cache_prefix,
        )
        logger.info(
            "claude request model=%s tools=%s previous_response_id=%s",
            body.get("model"),
            [t["name"] for t in body.get("tools", [])],
            previous_response_id,
        )
        return await self._stream_one_turn(body=body, event_handler=event_handler)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""

        await self._client.aclose()

    # ---------------------------------------------------------- internals

    async def _stream_one_turn(
        self,
        *,
        body: dict[str, Any],
        event_handler: EventHandler | None,
    ) -> ModelTurn:
        """Open the stream (with pre-stream retries), parse SSE events, return a turn.

        Pre-stream retries: on ``408 / 409 / 429 / 500 / 503 / 504 / 529``
        we close the stream and re-issue, honoring ``retry-after`` on 429
        and 529 (Anthropic's "overloaded" signal). ``402`` surfaces as
        :class:`ClaudeBillingError` with no retry; ``400 / 401 / 403 / 404``
        surface as raw :class:`httpx.HTTPStatusError`. Once the first
        byte of the 200 stream is emitted, this method commits and never
        retries that stream — mid-stream errors raise
        :class:`ClaudeStreamError`.
        """

        max_attempts = max(1, int(self._cfg.max_attempts))
        for attempt in range(max_attempts):
            try:
                return await self._consume_stream(body, event_handler=event_handler)
            except _PreStreamRetryable as exc:
                if attempt == max_attempts - 1:
                    raise exc.inner
                await asyncio.sleep(
                    _retry_sleep_seconds(
                        exc.status, attempt, 0.5, exc.retry_after
                    )
                )
        raise RuntimeError("claude pre-stream retry loop exited unexpectedly")

    async def _consume_stream(
        self,
        body: dict[str, Any],
        *,
        event_handler: EventHandler | None,
    ) -> ModelTurn:
        """Open one stream attempt and consume it end-to-end."""

        idle = self._cfg.stream_idle_timeout_seconds
        wall_cap = self._cfg.max_stream_wall_seconds
        stream_started = time.monotonic()

        def _wall_clock_error() -> ClaudeStreamWallClockError:
            return ClaudeStreamWallClockError(
                f"claude stream wall-clock >{wall_cap:.0f}s model={body.get('model')}"
            )

        async with self._client.stream(
            "POST",
            "/v1/messages",
            json=body,
            headers=self._headers,
        ) as resp:
            if resp.status_code >= 400:
                await self._classify_pre_stream_error(resp)

            parser = SSEParser()
            output_text = ""
            blocks: list[dict[str, Any]] = []
            tool_input_buffers: dict[int, str] = {}
            message_id: str | None = None
            usage: dict[str, Any] = {}
            iterator = resp.aiter_bytes().__aiter__()
            stream_done = False

            while not stream_done:
                remaining_wall = wall_cap - (time.monotonic() - stream_started)
                if remaining_wall <= 0:
                    raise _wall_clock_error()
                # ``read_budget`` is the smaller of the per-event idle
                # cap and what's left of the wall-clock budget — whichever
                # bound trips first wins.
                try:
                    raw_chunk = await asyncio.wait_for(
                        anext(iterator), timeout=min(idle, remaining_wall)
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    if (time.monotonic() - stream_started) >= wall_cap:
                        raise _wall_clock_error() from exc
                    raise ClaudeStreamIdleTimeoutError(
                        f"claude stream idle >{idle:.0f}s "
                        f"model={body.get('model')}"
                    ) from exc

                for event_name, payload in parser.feed(raw_chunk):
                    event_type = payload.get("type") or event_name
                    if event_type == "error":
                        err = payload.get("error") or {}
                        raise ClaudeStreamError(
                            f"claude stream error type={err.get('type')!r} "
                            f"message={err.get('message')!r}"
                        )
                    if event_type == "message_start":
                        message = payload.get("message") or {}
                        message_id = message.get("id") or message_id
                        msg_usage = message.get("usage")
                        if isinstance(msg_usage, dict):
                            usage.update(msg_usage)
                        continue
                    if event_type == "content_block_start":
                        index = payload.get("index")
                        block = dict(payload.get("content_block") or {})
                        if isinstance(index, int):
                            while len(blocks) <= index:
                                blocks.append({})
                            blocks[index] = block
                            if block.get("type") == "tool_use":
                                tool_input_buffers[index] = ""
                        continue
                    if event_type == "content_block_delta":
                        index = payload.get("index")
                        delta = payload.get("delta") or {}
                        if not isinstance(index, int) or index >= len(blocks):
                            continue
                        delta_type = delta.get("type")
                        if delta_type == "text_delta":
                            text = delta.get("text") or ""
                            output_text += text
                            existing = blocks[index].get("text") or ""
                            blocks[index]["text"] = existing + text
                            if event_handler is not None and text:
                                event_handler(
                                    RuntimeEvent(
                                        kind="assistant_text_delta",
                                        text=text,
                                    )
                                )
                        elif delta_type == "input_json_delta":
                            tool_input_buffers[index] = (
                                tool_input_buffers.get(index, "")
                                + (delta.get("partial_json") or "")
                            )
                        elif delta_type == "thinking_delta":
                            existing = blocks[index].get("thinking") or ""
                            blocks[index]["thinking"] = (
                                existing + (delta.get("thinking") or "")
                            )
                        elif delta_type == "signature_delta":
                            blocks[index]["signature"] = delta.get("signature")
                        # Other delta types (citations) are ignored — they
                        # carry no Feather-visible payload today.
                        continue
                    if event_type == "content_block_stop":
                        index = payload.get("index")
                        if (
                            isinstance(index, int)
                            and index < len(blocks)
                            and blocks[index].get("type") == "tool_use"
                        ):
                            raw = tool_input_buffers.pop(index, "")
                            try:
                                parsed = json.loads(raw) if raw else {}
                            except json.JSONDecodeError:
                                parsed = {"raw_arguments": raw}
                            blocks[index]["input"] = parsed
                        continue
                    if event_type == "message_delta":
                        msg_usage = payload.get("usage")
                        if isinstance(msg_usage, dict):
                            usage.update(msg_usage)
                        continue
                    if event_type == "message_stop":
                        stream_done = True
                        break
                    # ``ping`` and other events fall through.

        if message_id is None and not blocks:
            raise ClaudeStreamError(
                "claude stream ended without a message_start event "
                f"model={body.get('model')}"
            )
        turn = translate_response(
            message_id=message_id,
            blocks=blocks,
            output_text=output_text,
            usage=usage or None,
        )
        logger.info(
            "claude response id=%s tool_calls=%s output_chars=%s",
            turn.response_id,
            len(turn.tool_calls),
            len(turn.output_text or ""),
        )
        return turn

    async def _classify_pre_stream_error(self, resp: httpx.Response) -> None:
        """Raise the right exception for a pre-stream non-2xx response.

        Retryable codes raise :class:`_PreStreamRetryable`, which the
        outer retry loop catches and handles; non-retryable codes raise
        the final exception directly.
        """

        text = (await resp.aread()).decode("utf-8", errors="replace")
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {"error": {"type": str(resp.status_code), "message": text[:500]}}
        message = str(payload.get("error", {}).get("message") or text[:500] or "")
        code = resp.status_code

        def _http_error() -> httpx.HTTPStatusError:
            return httpx.HTTPStatusError(
                f"claude pre-stream {code}: {message[:400]}",
                request=resp.request,
                response=resp,
            )

        if code == 402:
            raise ClaudeBillingError(message or "billing error")
        if code in _RETRYABLE_STATUS:
            # 529 is Anthropic's "overloaded" signal — surface it as the
            # dedicated exception when retries are exhausted; everything
            # else collapses to the generic HTTPStatusError.
            terminal: Exception = (
                ClaudeOverloadedError(message or "overloaded")
                if code == 529
                else _http_error()
            )
            raise _PreStreamRetryable(code, resp.headers.get("retry-after"), terminal)
        # 400/401/403/404/413/422 and any other non-retryable status all
        # surface as a raw HTTPStatusError.
        raise _http_error()
