"""OpenRouter Chat Completions provider.

This provider implements :class:`~feather.providers.base.BaseLLMProvider`
against OpenRouter's OpenAI-compatible ``/chat/completions`` endpoint. It
owns three concerns — HTTP transport, retry semantics, and SSE parsing —
and delegates every request/response shape decision to
:mod:`feather.providers.openrouter_translator`.

Design notes:

- One :class:`httpx.AsyncClient` per provider instance. It is reused
  across turns so the HTTP/2 connection and keep-alive state are preserved.
- ``stream=True`` is always set on the request body. OpenRouter's SSE
  format is parsed by :class:`SSEParser`, which (a) skips colon-prefixed
  heartbeat comments, (b) stops cleanly on ``data: [DONE]``, and (c)
  tolerates JSON that straddles byte-chunk boundaries.
- Pre-stream errors (status ≥ 400 before any SSE event) are classified
  once: ``402`` becomes :class:`OpenRouterCreditsExhausted`;
  ``400/401/403`` raise immediately;
  ``408/429/500/502/503/504`` route through :func:`retryable_post`'s
  retry loop; 503 optionally widens routing (strips restrictive
  ``provider`` keys) once before retrying.
- Once the stream is open, we never retry that stream. A mid-stream
  ``finish_reason: "error"`` (or a top-level ``error`` field on any
  chunk) raises :class:`OpenRouterStreamError` as a terminal signal, to
  match the skill's research and the spec.
- Idle-timeout budget is measured between events. A stall longer than
  ``cfg.stream_idle_timeout_seconds`` raises
  :class:`OpenRouterStreamIdleTimeoutError`, mirroring the OpenAI
  provider's defense against silent mid-stream stalls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from typing import Any, Iterable

import httpx

from feather.models import (
    EventHandler,
    ModelTurn,
    OpenRouterConfig,
    ProviderRequestConfig,
    RuntimeEvent,
)
from feather.providers.base import BaseLLMProvider
from feather.providers.openrouter_translator import (
    reconstruct_tool_calls,
    translate_request,
    translate_response,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------- exceptions


class OpenRouterCreditsExhausted(RuntimeError):
    """Raised on HTTP 402 (insufficient credits). Never retried."""


class OpenRouterRoutingError(RuntimeError):
    """Raised when OpenRouter cannot find a provider that satisfies routing constraints."""


class OpenRouterStreamIdleTimeoutError(TimeoutError):
    """Raised when an SSE stream stalls longer than the configured idle budget."""


class OpenRouterStreamWallClockError(TimeoutError):
    """Raised when an SSE stream exceeds the configured wall-clock cap.

    The idle-timeout check (:class:`OpenRouterStreamIdleTimeoutError`) only
    bounds the gap between consecutive bytes. A misbehaving upstream that
    drips keep-alive frames (``: OPENROUTER PROCESSING\\n\\n``) faster than
    the idle threshold can keep a stream alive indefinitely while never
    producing actual content. This wall-clock cap is the second-line
    defence — once the stream exceeds
    ``OpenRouterConfig.max_stream_wall_seconds`` it raises regardless of
    inter-byte gap.
    """


class OpenRouterStreamError(RuntimeError):
    """Raised on a mid-stream terminal error (``finish_reason="error"``)."""


# -------------------------------------------------------------- SSE parser


class SSEParser:
    """Incremental Server-Sent Events parser for OpenRouter streams.

    Maintains a **byte** buffer across ``feed()`` calls (critical — an
    incoming HTTP chunk can split a multi-byte UTF-8 codepoint, and
    decoding too early corrupts the content with U+FFFD). Splits on
    blank-line event boundaries (``\\n\\n``, with CRLF normalization),
    skips colon-prefixed comment lines (OpenRouter heartbeats like
    ``: OPENROUTER PROCESSING``), and yields decoded JSON payloads. Stops
    yielding after the ``data: [DONE]`` sentinel — subsequent ``feed()``
    calls become no-ops.
    """

    def __init__(self) -> None:
        self._buffer: bytes = b""
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    def feed(self, chunk: bytes) -> Iterable[dict[str, Any]]:
        """Consume raw bytes, yield any complete events."""

        if self._stopped:
            return
        self._buffer += chunk
        # Normalize CRLF to LF so \n\n is the sole event terminator.
        if b"\r\n" in self._buffer:
            self._buffer = self._buffer.replace(b"\r\n", b"\n")
        while b"\n\n" in self._buffer:
            raw_event, self._buffer = self._buffer.split(b"\n\n", 1)
            # Only decode complete events — every codepoint straddling a
            # chunk boundary now lives on one side of the split.
            event = raw_event.decode("utf-8", errors="replace")
            for line in event.splitlines():
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    self._stopped = True
                    return
                if not payload:
                    # Empty data frames are legal SSE — skip silently.
                    continue
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    # Log only length + a short literal prefix — full
                    # payloads may carry user-visible content.
                    logger.warning(
                        "openrouter_sse_parse_failed len=%d head=%r",
                        len(payload),
                        payload[:20],
                    )


def parse_sse_events(raw: bytes) -> Iterable[dict[str, Any]]:
    """One-shot parse of a complete SSE byte blob. Thin wrapper over :class:`SSEParser`."""

    parser = SSEParser()
    yield from parser.feed(raw)


# ------------------------------------------------------------ retry helper


_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
_RESTRICTIVE_PROVIDER_KEYS: tuple[str, ...] = ("zdr", "quantizations", "only")

#: OpenRouter resolves rolling model aliases (e.g. ``deepseek/deepseek-v4-pro``)
#: to dated snapshots (``deepseek/deepseek-v4-pro-20260423``) in the response's
#: ``model`` field. We don't want to fire the ``model_substituted`` warning
#: for this benign aliasing — only for actual fallback re-routing where the
#: family changes (qwen ← deepseek, etc.). Match an 8-digit ``YYYYMMDD``
#: suffix or a ``-YYYY-MM-DD`` form, optionally followed by a small qualifier
#: like ``-preview`` / ``-experimental``.
_DATED_SNAPSHOT_RE = re.compile(
    r"-(\d{8}|\d{4}-\d{2}-\d{2})(-[a-z0-9]+)?$"
)


def _is_dated_snapshot_of(requested: str, resolved: str) -> bool:
    """Return ``True`` iff ``resolved`` is ``requested`` plus a date suffix.

    Used to suppress the ``model_substituted`` warning when OpenRouter
    pins a rolling alias to a dated snapshot — that's expected aliasing,
    not silent re-routing to a different model family.
    """

    if not resolved.startswith(requested):
        return False
    tail = resolved[len(requested):]
    return bool(_DATED_SNAPSHOT_RE.fullmatch(tail))
_MAX_RATE_LIMIT_RESET_SLEEP: float = 60.0


class _PreStreamRetryable(Exception):
    """Internal signal that a pre-stream response is retryable.

    Carries the HTTP status, the optional ``X-RateLimit-Reset`` header,
    and the terminal exception to raise if retries are exhausted.
    """

    def __init__(
        self, status: int, rate_limit_reset: str | None, inner: Exception
    ) -> None:
        super().__init__(f"pre-stream {status}")
        self.status = status
        self.rate_limit_reset = rate_limit_reset
        self.inner = inner


#: Mid-stream injected error codes documented as transient by OpenRouter.
#: See https://openrouter.ai/docs/api/reference/errors-and-debugging — the
#: 502 ``provider_unavailable`` (and friends 503/504/524) class is delivered
#: as an SSE event with an ``error`` object and is recommended to be handled
#: with exponential backoff retry. We extend our pre-stream retry policy to
#: cover this mid-stream variant when, and only when, no observable output
#: has been committed yet (otherwise retrying would duplicate streamed
#: content or tool calls).
_MID_STREAM_RETRYABLE_STATUS: frozenset[int] = frozenset({502, 503, 504, 524})


class _MidStreamRetryable(Exception):
    """Internal signal: OpenRouter injected a transient error into the SSE
    stream BEFORE any output_text or tool_call deltas were committed. Caught
    by :meth:`OpenRouterChatProvider._stream_one_turn` to retry the whole
    turn from scratch (idempotent because no caller-visible state was emitted)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"mid-stream {status}: {message[:200]}")
        self.status = status
        self.message = message


def _retry_sleep_seconds_from_headers(
    status: int, attempt: int, base_delay: float, reset_header: str | None
) -> float:
    """Compute backoff; honors X-RateLimit-Reset on 429 but clamps to a sane ceiling."""

    if status == 429 and reset_header and reset_header.isdigit():
        hinted = max(0.0, int(reset_header) - time.time())
        return min(hinted, _MAX_RATE_LIMIT_RESET_SLEEP)
    sleep = base_delay * (2 ** attempt)
    sleep += random.uniform(0, base_delay)
    return sleep


async def retryable_post(
    client: httpx.AsyncClient,
    url: str,
    *,
    json_body: dict[str, Any],
    headers: dict[str, str],
    max_attempts: int,
    base_delay: float = 0.5,
    widen_routing_on_503: bool = True,
) -> httpx.Response:
    """POST with status-code-aware retries.

    Non-streaming callers use this directly. The streaming path in
    :meth:`OpenRouterChatProvider.complete` duplicates the classification
    logic because we have to open the stream and then inspect status
    code in-stream to preserve the connection for the happy path.

    Args:
        client: An open :class:`httpx.AsyncClient`.
        url: Absolute or base-URL-relative endpoint.
        json_body: Request body. Mutated in-place on 503 widen-routing.
        headers: Request headers.
        max_attempts: Total attempts before surfacing the last error.
        base_delay: Full-jitter exponential backoff base in seconds.
        widen_routing_on_503: When ``True`` and the first attempt 503s,
            strip restrictive ``provider`` keys (``zdr``,
            ``quantizations``, ``only``) on exactly one retry and log
            the degradation. Non-restrictive keys like ``sort`` /
            ``allow_fallbacks`` are preserved.

    Returns:
        A successful :class:`httpx.Response`.

    Raises:
        OpenRouterCreditsExhausted: on 402.
        OpenRouterRoutingError: on a persistent 503.
        httpx.HTTPStatusError: on 400/401/403 or after exhausting retries.
    """

    body = dict(json_body)
    widened = False
    last_error: dict[str, Any] | None = None

    for attempt in range(max_attempts):
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code < 400:
            return resp
        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001
            payload = {"error": {"code": resp.status_code, "message": resp.text[:500]}}
        last_error = payload
        code = resp.status_code
        if code == 402:
            raise OpenRouterCreditsExhausted(
                str(payload.get("error", {}).get("message", "insufficient credits"))
            )
        if code in (400, 401, 403):
            resp.raise_for_status()
        if code not in _RETRYABLE_STATUS or attempt == max_attempts - 1:
            if code == 503:
                raise OpenRouterRoutingError(
                    str(payload.get("error", {}).get("message", "no provider available"))
                )
            resp.raise_for_status()

        if code == 503 and widen_routing_on_503 and not widened:
            prefs = dict(body.get("provider") or {})
            for key in _RESTRICTIVE_PROVIDER_KEYS:
                prefs.pop(key, None)
            if prefs:
                body["provider"] = prefs
            else:
                body.pop("provider", None)
            widened = True
            logger.warning(
                "openrouter_503_widened_routing attempt=%s removed_keys=%s",
                attempt,
                _RESTRICTIVE_PROVIDER_KEYS,
            )
        sleep = _retry_sleep_seconds(code, attempt, base_delay, resp)
        await asyncio.sleep(sleep)

    # Exhausted retries without returning a 2xx.
    raise OpenRouterRoutingError(str(last_error))


def _retry_sleep_seconds(
    status: int, attempt: int, base_delay: float, resp: httpx.Response
) -> float:
    """Compute backoff for one retry attempt. Honors X-RateLimit-Reset on 429."""

    if status == 429:
        reset = resp.headers.get("X-RateLimit-Reset")
        if reset and reset.isdigit():
            return max(0.0, int(reset) - time.time())
    sleep = base_delay * (2 ** attempt)
    sleep += random.uniform(0, base_delay)
    return sleep


# ---------------------------------------------------- OpenRouterChatProvider


class OpenRouterChatProvider(BaseLLMProvider):
    """Chat Completions provider routed through OpenRouter.

    One instance owns one :class:`httpx.AsyncClient` and is safe to call
    ``complete()`` on concurrently from multiple tasks — each turn opens
    its own response stream via ``client.stream(...)``.

    Args:
        cfg: Provider configuration, typically loaded from the
            ``openrouter:`` block of ``config/app.yaml``.
        transport: Optional httpx transport override for tests.
    """

    stateful = False

    def __init__(
        self,
        cfg: OpenRouterConfig,
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
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if cfg.http_referer:
            headers["HTTP-Referer"] = cfg.http_referer
        if cfg.app_title:
            headers["X-Title"] = cfg.app_title
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
        """Run one OpenRouter turn.

        ``previous_response_id`` is accepted for interface compatibility
        with :class:`BaseLLMProvider`; OpenRouter is stateless, so it is
        used only for log correlation. Feather's compaction + history
        replay guarantees that ``input_items`` already carries everything
        the model needs.
        """

        body = translate_request(
            instructions=instructions,
            input_items=input_items,
            tools=tools,
            request_config=request_config or ProviderRequestConfig(),
            cfg=self._cfg,
            model_limits=None,
        )
        logger.info(
            "openrouter request model=%s tools=%s previous_response_id=%s",
            body.get("model"),
            [t["function"]["name"] for t in body.get("tools", [])],
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

        Pre-stream retries: on ``408 / 429 / 500 / 502 / 503 / 504`` we
        close the stream and re-issue, honoring ``X-RateLimit-Reset``
        (bounded) on 429. 503 widens restrictive routing knobs once before
        the next attempt. ``402`` surfaces as :class:`OpenRouterCreditsExhausted`
        with no retry; ``400 / 401 / 403`` surface as raw
        :class:`httpx.HTTPStatusError` with no retry. Once the first byte
        of the 200 stream is emitted, this method commits and never
        retries that stream — mid-stream errors raise :class:`OpenRouterStreamError`.
        """

        max_attempts = max(1, int(self._cfg.max_attempts))
        widen_attempted = False
        active_body = dict(body)

        for attempt in range(max_attempts):
            try:
                return await self._consume_stream(
                    active_body, event_handler=event_handler
                )
            except _PreStreamRetryable as exc:
                if attempt == max_attempts - 1:
                    raise exc.inner
                if exc.status == 503 and not widen_attempted:
                    prefs = dict(active_body.get("provider") or {})
                    for key in _RESTRICTIVE_PROVIDER_KEYS:
                        prefs.pop(key, None)
                    if prefs:
                        active_body["provider"] = prefs
                    else:
                        active_body.pop("provider", None)
                    widen_attempted = True
                    logger.warning(
                        "openrouter_503_widened_routing attempt=%s removed_keys=%s",
                        attempt,
                        _RESTRICTIVE_PROVIDER_KEYS,
                    )
                await asyncio.sleep(
                    _retry_sleep_seconds_from_headers(
                        exc.status, attempt, 0.25, exc.rate_limit_reset
                    )
                )
            except _MidStreamRetryable as exc:
                if attempt == max_attempts - 1:
                    # Convert to terminal so the caller sees a stable
                    # exception type — same as the pre-stream branch.
                    raise OpenRouterStreamError(
                        f"openrouter mid-stream {exc.status} after "
                        f"{max_attempts} attempts: {exc.message[:400]}"
                    )
                logger.warning(
                    "openrouter.midstream.retry status=%s attempt=%s "
                    "model=%s message=%s",
                    exc.status,
                    attempt,
                    active_body.get("model"),
                    exc.message[:200],
                )
                await asyncio.sleep(
                    _retry_sleep_seconds_from_headers(
                        exc.status, attempt, 0.5, None
                    )
                )
        # Unreachable: the loop either returns or raises.
        raise RuntimeError("openrouter pre-stream retry loop exited unexpectedly")

    async def _consume_stream(
        self,
        body: dict[str, Any],
        *,
        event_handler: EventHandler | None,
    ) -> ModelTurn:
        """Open one stream attempt and consume it end-to-end."""

        idle = self._cfg.stream_idle_timeout_seconds
        wall_cap = self._cfg.max_stream_wall_seconds
        loop = asyncio.get_event_loop()
        stream_started = loop.time()
        async with self._client.stream(
            "POST",
            "/chat/completions",
            json=body,
            headers=self._headers,
        ) as resp:
            if resp.status_code >= 400:
                await self._classify_pre_stream_error(resp)

            parser = SSEParser()
            output_text = ""
            deltas: list[dict[str, Any]] = []
            final_chunk: dict[str, Any] | None = None
            last_chunk_id: str | None = resp.headers.get("X-Generation-Id")
            iterator = resp.aiter_bytes().__aiter__()
            while True:
                # Wall-clock cap: a misbehaving upstream that drips
                # keep-alive bytes inside the idle window forever cannot
                # be caught by the idle timer. Cap the per-read wait at
                # whichever is smaller, so the wall-clock bound is always
                # honoured even if `idle > wall_cap` after partial elapse.
                remaining_wall = wall_cap - (loop.time() - stream_started)
                if remaining_wall <= 0:
                    raise OpenRouterStreamWallClockError(
                        f"openrouter stream wall-clock >{wall_cap:.0f}s "
                        f"model={body.get('model')}"
                    )
                read_budget = min(idle, remaining_wall)
                try:
                    # Wrap the chunk-read so neither a silently stalled
                    # upstream nor a never-completing one can hang the
                    # loop.
                    raw_chunk = await asyncio.wait_for(
                        iterator.__anext__(), timeout=read_budget
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    if (loop.time() - stream_started) >= wall_cap:
                        raise OpenRouterStreamWallClockError(
                            f"openrouter stream wall-clock >{wall_cap:.0f}s "
                            f"model={body.get('model')}"
                        ) from exc
                    raise OpenRouterStreamIdleTimeoutError(
                        f"openrouter stream idle >{idle:.0f}s "
                        f"model={body.get('model')}"
                    ) from exc
                for event in parser.feed(raw_chunk):
                    if event.get("id"):
                        last_chunk_id = event["id"]
                    if event.get("error"):
                        err = event["error"]
                        message = str(err.get("message") or err)
                        self._raise_stream_error(
                            status=err.get("code"),
                            message=message,
                            output_text=output_text,
                            deltas=deltas,
                        )
                    choices = event.get("choices") or []
                    if not choices:
                        final_chunk = event
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if delta:
                        content_frag = delta.get("content")
                        if content_frag:
                            output_text += content_frag
                            if event_handler is not None:
                                event_handler(
                                    RuntimeEvent(
                                        kind="assistant_text_delta",
                                        text=content_frag,
                                    )
                                )
                        deltas.append(delta)
                    if choice.get("finish_reason") == "error":
                        err = event.get("error") or {}
                        self._raise_stream_error(
                            status=err.get("code"),
                            message=str(err.get("message") or "mid-stream error"),
                            output_text=output_text,
                            deltas=deltas,
                        )
                    if choice.get("finish_reason") is not None:
                        final_chunk = event

        if final_chunk is None:
            # Upstream protocol violation: per DeepSeek's own API spec
            # (https://api-docs.deepseek.com/api/create-chat-completion)
            # ``finish_reason`` is REQUIRED on the last chunk before
            # ``[DONE]``. OpenRouter's docs are silent on the same
            # guarantee but their SSE format mirrors OpenAI's, which
            # carries the same requirement. Reasoning models routed
            # through OpenRouter — DeepSeek-V4-Pro / R1 specifically —
            # have been observed to send ``[DONE]`` with no preceding
            # finish_reason chunk after long reasoning sequences (text
            # output) AND after long tool-call planning sequences (the
            # 34-tool_call_deltas symptom). Vercel AI SDK and the OpenAI
            # Python SDK both ship pragmatic tolerance for this; do the
            # same here, but only when the deltas already form a valid
            # turn that we can hand back without inventing state.
            have_streamed_text = bool(output_text)
            have_tool_calls = any(d.get("tool_calls") for d in deltas)
            tool_call_delta_count = sum(
                1 for d in deltas if d.get("tool_calls")
            )

            tool_calls_complete = False
            if have_tool_calls:
                # ``reconstruct_tool_calls`` is the deterministic answer to
                # "are these deltas a valid tool-call set?". A truncated
                # mid-args stream surfaces as ``{"raw_arguments": <partial>}``;
                # a stream that dropped the chunk carrying ``function.name``
                # surfaces as an empty ``name``. Either signal means we MUST
                # raise rather than commit a half-built call. Anything else
                # is bit-for-bit equivalent to a stream that DID terminate
                # cleanly with ``finish_reason="tool_calls"``.
                reconstructed = reconstruct_tool_calls(deltas)
                tool_calls_complete = bool(reconstructed) and all(
                    bool(tc.name) and "raw_arguments" not in tc.arguments
                    for tc in reconstructed
                )

            if have_tool_calls and tool_calls_complete:
                # OpenRouter normalizes the tool-using terminator to
                # ``tool_calls`` (per their streaming docs), not ``stop``.
                logger.warning(
                    "openrouter.stream.no_final_chunk model=%s "
                    "synthesizing finish_reason=tool_calls after %d tool_call "
                    "deltas reconstruct to a complete tool-call set "
                    "(upstream sent [DONE] without a finish_reason chunk — "
                    "known pattern for DeepSeek reasoning models routed "
                    "through OpenRouter)",
                    body.get("model"),
                    tool_call_delta_count,
                )
                final_chunk = {
                    "id": last_chunk_id,
                    "model": body.get("model"),
                    "choices": [
                        {
                            "delta": {},
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            elif have_streamed_text and not have_tool_calls:
                logger.warning(
                    "openrouter.stream.no_final_chunk model=%s "
                    "synthesizing finish_reason=stop after %d output chars "
                    "(upstream sent [DONE] without a finish_reason chunk — "
                    "known pattern for DeepSeek reasoning models routed "
                    "through OpenRouter)",
                    body.get("model"),
                    len(output_text),
                )
                final_chunk = {
                    "id": last_chunk_id,
                    "model": body.get("model"),
                    "choices": [
                        {
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
            else:
                # Empty stream OR truncated tool calls (incomplete name /
                # mid-args JSON): synthesizing here would either invent a
                # turn from nothing or commit a half-built tool call.
                # Surface the real error so the caller can retry.
                raise OpenRouterStreamError(
                    "openrouter stream ended without a final chunk "
                    f"(output_chars={len(output_text)}, "
                    f"tool_call_deltas={tool_call_delta_count}, "
                    f"model={body.get('model')})"
                )
        if last_chunk_id and not final_chunk.get("id"):
            final_chunk = dict(final_chunk)
            final_chunk["id"] = last_chunk_id
        turn = translate_response(
            final_chunk=final_chunk,
            output_text=output_text,
            deltas=deltas,
        )
        requested_model = body.get("model")
        resolved_model = final_chunk.get("model")
        logger.info(
            "openrouter response id=%s model_requested=%s model_resolved=%s "
            "tool_calls=%s output_chars=%s",
            turn.response_id,
            requested_model,
            resolved_model,
            len(turn.tool_calls),
            len(turn.output_text or ""),
        )
        # OpenRouter silently routes to a fallback in ``models`` (our
        # ``fallback_models``) when the primary model is unavailable.
        # Without this warning the substitution is invisible — the
        # symptom is "I picked model A in /config, but Opik logs and
        # billing show model B was used", which is exactly the bug
        # report this log line was added for. Compare against the
        # configured primary (``self._cfg.model``), not ``body["model"]``,
        # so an attempt fanned out across fallback chains via
        # ``models`` still flags a non-primary as a deviation.
        configured_primary = getattr(self._cfg, "model", None)
        if (
            resolved_model
            and configured_primary
            and resolved_model != configured_primary
            and not _is_dated_snapshot_of(configured_primary, resolved_model)
        ):
            fallback_list = list(getattr(self._cfg, "fallback_models", ()) or ())
            logger.warning(
                "openrouter.model_substituted requested=%s resolved=%s "
                "fallback_models=%s — your /config selection was overridden "
                "by OpenRouter's automatic fallback. Either remove the "
                "primary's slug from app.openrouter.fallback_models or "
                "verify the primary is actually served by OpenRouter.",
                configured_primary,
                resolved_model,
                fallback_list,
            )
        return turn

    @staticmethod
    def _raise_stream_error(
        *,
        status: Any,
        message: str,
        output_text: str,
        deltas: list[dict[str, Any]],
    ) -> None:
        """Raise the right exception for an in-stream error event.

        OpenRouter's docs document mid-stream injected errors with codes in
        ``_MID_STREAM_RETRYABLE_STATUS`` (502 ``provider_unavailable`` and
        friends) as transient, recommending exponential-backoff retry. We
        extend that policy to streams that have not yet committed observable
        state — if any ``output_text`` or tool-call delta has been emitted,
        retrying would duplicate caller-visible side effects, so we surface
        as terminal :class:`OpenRouterStreamError` instead.
        """

        code: int | None = status if isinstance(status, int) else None
        clean_state = not output_text and not any(
            d.get("tool_calls") for d in deltas
        )
        if code in _MID_STREAM_RETRYABLE_STATUS and clean_state:
            raise _MidStreamRetryable(code, message)
        raise OpenRouterStreamError(message)

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
            payload = {"error": {"code": resp.status_code, "message": text[:500]}}
        message = str(payload.get("error", {}).get("message") or text[:500] or "")
        code = resp.status_code
        if code == 402:
            raise OpenRouterCreditsExhausted(message or "insufficient credits")
        if code in (400, 401, 403):
            raise httpx.HTTPStatusError(
                f"openrouter pre-stream {code}: {message[:400]}",
                request=resp.request,
                response=resp,
            )
        if code in _RETRYABLE_STATUS:
            reset = resp.headers.get("X-RateLimit-Reset")
            terminal = OpenRouterRoutingError(message) if code == 503 else (
                httpx.HTTPStatusError(
                    f"openrouter pre-stream {code}: {message[:400]}",
                    request=resp.request,
                    response=resp,
                )
            )
            raise _PreStreamRetryable(code, reset, terminal)
        # Other status codes surface verbatim.
        raise httpx.HTTPStatusError(
            f"openrouter pre-stream {code}: {message[:400]}",
            request=resp.request,
            response=resp,
        )
