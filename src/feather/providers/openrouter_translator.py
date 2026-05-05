"""Pure translation between Feather's Responses-API-shaped state and OpenRouter Chat Completions.

Isolating translation in one pure module means every branch is
unit-testable without touching HTTP, and lets the provider orchestrator
focus on streaming/retry without also owning shape concerns.

Three entry points mirror the turn lifecycle:

- :func:`translate_request` — builds the POST body for one turn.
- :func:`reconstruct_tool_calls` — merges indexed-cumulative tool-call
  deltas into :class:`feather.models.ToolCall` instances.
- :func:`translate_response` — combines the final SSE chunk, accumulated
  content, and tool-call deltas into a normalized :class:`ModelTurn`.

``translate_tools`` and ``translate_input_items`` are the two sub-helpers
``translate_request`` depends on; they are exposed so tests can pin their
behavior independently.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

from feather.models import (
    ModelTurn,
    OpenRouterConfig,
    OpenRouterTracingConfig,
    ProviderRequestConfig,
    ToolCall,
    TraceContext,
)


# OpenRouter trace metadata limits, lifted from the chat-completions API
# spec at openrouter.ai/docs/api/api-reference/chat:
#   ``user`` ≤ 128 chars, ``session_id`` ≤ 256 chars,
#   ``metadata``: ≤ 16 kv pairs, 64-char keys, 512-char values.
# We treat the ``trace`` object as the same kind of bag for clamping
# purposes — the spec doesn't formally bound it but mirroring the
# ``metadata`` limits keeps Feather safely under any sane upstream cap
# and prevents an operator typo from triggering a 400.
_OR_USER_MAX_LEN = 128
_OR_SESSION_ID_MAX_LEN = 256
_OR_TRACE_MAX_KEYS = 16
_OR_TRACE_KEY_MAX_LEN = 64
_OR_TRACE_VALUE_MAX_LEN = 512

# Reserved trace keys Feather always populates. Operator metadata that
# collides with one of these is silently shadowed: the per-turn identity
# bundle is the source of truth, otherwise an operator misconfiguration
# could disguise a sub-agent's traces as the lead's.
_RESERVED_TRACE_KEYS = frozenset(
    {
        "trace_name",
        "generation_name",
        "trace_id",
        "span_name",
        "parent_span_id",
        "feather_app",
        "feather_agent_name",
        "feather_agent_role",
        "feather_session_id",
    }
)


# ---------------------------------------------------------------------- tools


def translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrap Responses-API flat tool definitions into Chat Completions nested form.

    Feather's :meth:`feather.tools.base.BaseTool.to_openai_tool` returns a
    flat dict ``{type, name, description, parameters, strict}`` suitable
    for the OpenAI Responses API. Chat Completions wants the
    ``{type: "function", function: {name, description, parameters, strict}}``
    shape instead. This rewraps; entries already in Chat Completions shape
    (identified by the presence of a ``function`` subdict) pass through.

    Args:
        tools: Tool definitions in either flat or nested form.

    Returns:
        Tools in Chat Completions nested form.
    """

    out: list[dict[str, Any]] = []
    for tool in tools:
        if "function" in tool and "name" not in tool:
            out.append(tool)
            continue
        inner: dict[str, Any] = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        }
        if tool.get("strict"):
            inner["strict"] = True
        out.append({"type": "function", "function": inner})
    return out


# ----------------------------------------------------------------- input_items


_TEXT_BLOCK_TYPES = {"input_text", "output_text", "text"}


def translate_input_items(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Feather ``input_items`` (Responses API shape) into Chat Completions messages.

    Handles the three concrete cases observed in Feather:

    1. ``{"type": "message", "role": ..., "content": [{"type": "input_text", "text": ...}, ...]}``
       — agent-loop user/assistant messages. Text-only content blocks are
       concatenated into a plain string; image/file blocks become
       OpenRouter multimodal content parts.
    2. ``{"type": "function_call", "call_id": ..., "name": ..., "arguments": ...}``
       — assistant tool-call requests replayed for stateless providers.
       Consecutive function-call items are grouped into one assistant
       ``tool_calls`` message.
    3. ``{"type": "function_call_output", "call_id": ..., "output": ...}``
       — tool-result rows emitted after a tool call.
    4. Flat ``{"role": ..., "content": ...}`` — memory extractor /
       classifier / query-builder callers hand-shape this directly.

    Unknown item types (e.g. ``type == "reasoning"``) are skipped: the
    Responses API persists reasoning items that have no Chat Completions
    equivalent, and in Feather's compaction-reset-and-replay model these
    rarely appear as input items for live turns.

    Args:
        input_items: Feather-shaped input items.

    Returns:
        Chat Completions ``messages[]`` entries.
    """

    messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    pending_tool_content: str | None = None

    def flush_pending_tool_calls() -> None:
        nonlocal pending_tool_calls, pending_tool_content
        if not pending_tool_calls:
            return
        messages.append(
            {
                "role": "assistant",
                "content": pending_tool_content,
                "tool_calls": pending_tool_calls,
            }
        )
        pending_tool_calls = []
        pending_tool_content = None

    for item in input_items:
        item_type = item.get("type")
        if item_type == "message":
            flush_pending_tool_calls()
            messages.append(
                {
                    "role": item.get("role", "user"),
                    "content": _translate_content_blocks(item.get("content")),
                }
            )
            continue
        if item_type == "function_call":
            if not pending_tool_calls and item.get("content") is not None:
                pending_tool_content = str(item.get("content") or "")
            pending_tool_calls.append(
                {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": _stringify_tool_arguments(
                            item.get("arguments")
                        ),
                    },
                }
            )
            continue
        if item_type == "function_call_output":
            flush_pending_tool_calls()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "content": item.get("output", ""),
                }
            )
            continue
        if item_type is None and "role" in item and "content" in item:
            flush_pending_tool_calls()
            messages.append({"role": item["role"], "content": item["content"]})
            continue
        # Unsupported shapes (reasoning items etc.) are intentionally ignored.
    flush_pending_tool_calls()
    return messages


def _translate_content_blocks(content: Any) -> str | list[dict[str, Any]]:
    """Translate Responses-API content blocks into OpenRouter message content."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        translated: list[dict[str, Any]] = []
        has_multimodal = False
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in _TEXT_BLOCK_TYPES:
                text = str(block.get("text", ""))
                text_parts.append(text)
                if text:
                    translated.append({"type": "text", "text": text})
                continue
            if block_type == "input_image":
                image_url = block.get("image_url") or block.get("url")
                if isinstance(image_url, str) and image_url:
                    translated.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )
                    has_multimodal = True
                continue
            if block_type == "input_file":
                file_part = _translate_file_block(block)
                if file_part is not None:
                    translated.append(file_part)
                    has_multimodal = True
        if has_multimodal:
            return translated
        return "".join(text_parts)
    return ""


def _translate_file_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """Translate a Responses ``input_file`` block into OpenRouter's file part."""

    file_data = block.get("file_data")
    file_url = block.get("file_url")
    file_id = block.get("file_id")
    file_obj: dict[str, Any] = {
        "filename": block.get("filename") or "attachment",
    }
    if isinstance(file_data, str) and file_data:
        file_obj["file_data"] = file_data
    elif isinstance(file_url, str) and file_url:
        file_obj["file_url"] = file_url
    elif isinstance(file_id, str) and file_id:
        file_obj["file_id"] = file_id
    else:
        return None
    return {"type": "file", "file": file_obj}


def _stringify_tool_arguments(arguments: Any) -> str:
    """Return Chat Completions' JSON-encoded ``function.arguments`` string."""

    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments or {}, separators=(",", ":"), sort_keys=True)


# -------------------------------------------------------------- translate_request


def translate_request(
    *,
    instructions: str,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    request_config: ProviderRequestConfig,
    cfg: OpenRouterConfig,
    model_limits: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the OpenRouter Chat Completions POST body for one turn.

    Key behaviors:

    - When ``cfg.cache_strategy == "anthropic_breakpoint"`` the system
      message is emitted as a list with one content block carrying
      ``cache_control: {type: "ephemeral"}``. Non-caching providers
      ignore the hint; Anthropic/Gemini use it to cache the stable
      instructions prefix.
    - When ``model_limits`` exposes ``top_provider.max_completion_tokens``
      (or a raw ``context_length``), ``max_tokens`` is capped at that
      ceiling. This avoids a class of 400s when the configured value
      exceeds what the chosen upstream provider accepts.
    - ``reasoning``, ``provider`` (preferences), and ``models[]``
      (fallbacks) are forwarded only when set on the config.

    Args:
        instructions: Full system instructions.
        input_items: Feather-shaped input items for the new turn.
        tools: Tool definitions in Feather's Responses-API shape.
        request_config: Per-request overrides from the agent loop.
        cfg: Provider-level defaults.
        model_limits: Optional cached ``/api/v1/models`` entry used to
            cap ``max_tokens`` at the model's completion ceiling.

    Returns:
        JSON-serializable POST body for ``/chat/completions``.
    """

    model = request_config.model or cfg.model
    desired_max = (
        cfg.max_output_tokens
        if request_config.max_output_tokens is None
        else request_config.max_output_tokens
    )
    if model_limits:
        # Cap only on ``max_completion_tokens`` — ``context_length`` is
        # total (input + output) and does not bound the completion alone.
        ceiling = (model_limits.get("top_provider") or {}).get("max_completion_tokens")
        if isinstance(ceiling, int) and ceiling > 0:
            desired_max = min(desired_max, ceiling)
    temperature = (
        request_config.temperature
        if request_config.temperature is not None
        else cfg.temperature
    )

    if cfg.cache_strategy == "anthropic_breakpoint":
        system_msg: dict[str, Any] = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": instructions,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    else:
        system_msg = {"role": "system", "content": instructions}

    body: dict[str, Any] = {
        "model": model,
        "stream": True,
        "temperature": temperature,
        "max_tokens": desired_max,
        "messages": [system_msg, *translate_input_items(input_items)],
    }
    chat_tools = translate_tools(tools)
    if chat_tools:
        body["tools"] = chat_tools
        # ``parallel_tool_calls`` narrows routing under
        # ``provider.require_parameters: true`` because many providers
        # don't advertise it in ``supported_parameters`` even when they
        # functionally support parallel calls (e.g. Parasail). Emit it
        # only when the operator explicitly opts in (``True``); when
        # ``False`` or the default, omit so the provider's own default
        # applies and routing stays open.
        if cfg.parallel_tool_calls:
            body["parallel_tool_calls"] = True

    if request_config.response_schema is not None:
        schema_cls = request_config.response_schema
        schema = schema_cls.model_json_schema()
        _harden_strict_schema(schema)
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request_config.response_schema_name or schema_cls.__name__,
                "schema": schema,
                "strict": True,
            },
        }

    reasoning = request_config.reasoning or cfg.reasoning
    if reasoning is not None:
        reasoning_obj: dict[str, Any] = {}
        if reasoning.effort is not None:
            reasoning_obj["effort"] = reasoning.effort
        if reasoning.summary is not None:
            reasoning_obj["summary"] = reasoning.summary
        if reasoning_obj:
            body["reasoning"] = reasoning_obj

    if cfg.provider_preferences:
        body["provider"] = dict(cfg.provider_preferences)
    if cfg.fallback_models:
        body["models"] = list(cfg.fallback_models)
    _maybe_apply_tracing(
        body=body,
        tracing=cfg.tracing,
        trace_context=request_config.trace_context,
        model=model,
    )
    return body


# ----------------------------------------------------------------- tracing


def _maybe_apply_tracing(
    *,
    body: dict[str, Any],
    tracing: OpenRouterTracingConfig | None,
    trace_context: TraceContext | None,
    model: str,
) -> None:
    """Mutate ``body`` in place to add OpenRouter trace-broadcast fields.

    The function is no-op safe in four ways:

    1. If tracing config is absent or ``enabled=False`` it returns
       immediately, so the wire body stays byte-identical to the
       pre-tracing behaviour.
    2. If the per-turn trace context is missing it also returns: emitting
       the operator-static ``user`` field alone (without session/agent
       grouping) would land traces in Opik with no useful aggregation
       and is more confusing than helpful.
    3. Operator-supplied ``metadata`` values that violate OpenRouter's
       limits (>16 keys, oversize key/value, non-primitive value) are
       silently coerced or dropped rather than raised — the agent loop
       must never crash because of an observability misconfig.
    4. Any unanticipated exception is caught at the outer boundary,
       logged at WARNING, and the body is left untouched. The agent loop
       must never die because of an observability bug.
    """

    if tracing is None or not tracing.enabled:
        return
    if trace_context is None:
        return

    try:
        body["session_id"] = trace_context.session_id[:_OR_SESSION_ID_MAX_LEN]
        if tracing.user:
            body["user"] = tracing.user[:_OR_USER_MAX_LEN]

        # Reserved keys are written last so they always win over operator
        # metadata in case of collision.
        trace: dict[str, Any] = {}
        if tracing.metadata:
            for key, value in tracing.metadata.items():
                coerced = _coerce_trace_value(value)
                if coerced is None:
                    continue
                if not isinstance(key, str) or not key:
                    continue
                if key.lower() in _RESERVED_TRACE_KEYS:
                    # Will be overwritten anyway; skip to preserve budget.
                    # Lowercase compare so ``Feather_App`` can't sneak in
                    # alongside ``feather_app`` and confuse trace search.
                    continue
                trace[key[:_OR_TRACE_KEY_MAX_LEN]] = coerced

        # Reserved values are clamped to the same per-value cap so an
        # oversize ``--session-id`` flag or a long ``model`` slug can
        # never trigger a 400 from OpenRouter.
        reserved: dict[str, Any] = {
            "trace_name": _clamp_trace_value(
                f"feather/{trace_context.agent_name}"
            ),
            "generation_name": _clamp_trace_value(model),
            "feather_app": "feather-agent-os",
            "feather_agent_name": _clamp_trace_value(trace_context.agent_name),
            "feather_session_id": _clamp_trace_value(trace_context.session_id),
        }
        if trace_context.agent_role:
            reserved["feather_agent_role"] = _clamp_trace_value(
                trace_context.agent_role
            )

        # Reserved keys take priority over operator extras when the
        # 16-key budget is tight.
        budget = _OR_TRACE_MAX_KEYS - len(reserved)
        if budget < 0:
            # Defensive: if reserved ever grows past 16 (it's 6 today)
            # this branch keeps us under the cap by trimming reserved
            # rather than ignoring the overflow.
            reserved = dict(list(reserved.items())[:_OR_TRACE_MAX_KEYS])
            budget = 0
        if len(trace) > budget:
            # Stable order: dict insertion order in Python 3.7+;
            # truncating the tail keeps the first N operator entries.
            trace = dict(list(trace.items())[:budget])
        trace.update(reserved)
        body["trace"] = trace
    except Exception:  # noqa: BLE001
        # Tracing is observability — never let it take down the loop.
        logger.warning(
            "openrouter_tracing_apply_failed agent=%s session=%s",
            getattr(trace_context, "agent_name", None),
            getattr(trace_context, "session_id", None),
            exc_info=True,
        )
        # Strip any partial fields so the wire body stays well-formed.
        body.pop("session_id", None)
        body.pop("user", None)
        body.pop("trace", None)


def _clamp_trace_value(value: str) -> str:
    """Char-clamp a string to OpenRouter's per-value metadata cap."""

    return value[:_OR_TRACE_VALUE_MAX_LEN]


def _coerce_trace_value(value: Any) -> str | None:
    """Coerce a metadata value to a string-typed payload, or drop it.

    OpenRouter's metadata schema is string-typed. We accept primitives
    (str/int/float/bool) and stringify them; nested dicts/lists are
    dropped because they would either trigger a 400 or get serialized
    into something the operator didn't intend.
    """

    if isinstance(value, bool):
        # Check bool before int — ``isinstance(True, int)`` is ``True``.
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        out = str(value)
        if len(out) > _OR_TRACE_VALUE_MAX_LEN:
            out = out[:_OR_TRACE_VALUE_MAX_LEN]
        return out
    return None


def _harden_strict_schema(schema: dict[str, Any]) -> None:
    """Enforce OpenAI-style strict-mode invariants on a JSON schema.

    OpenAI and OpenRouter-routed OpenAI-compatible backends reject
    ``response_format=json_schema`` with ``strict=True`` unless every
    ``type: "object"`` node sets ``additionalProperties: false`` and
    lists every property name in ``required``. Pydantic emits
    ``additionalProperties`` at the root but not always on nested
    objects, and it drops fields with defaults from ``required``. We
    walk the schema in place to fix both.
    """

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
                props = node.get("properties")
                if isinstance(props, dict) and props:
                    node["required"] = list(props.keys())
            for key in ("properties", "$defs", "definitions", "patternProperties"):
                sub = node.get(key)
                if isinstance(sub, dict):
                    for child in sub.values():
                        _walk(child)
            for key in ("items", "additionalItems", "contains"):
                sub = node.get(key)
                if isinstance(sub, (dict, list)):
                    _walk(sub)
            for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
                sub = node.get(key)
                if isinstance(sub, list):
                    for child in sub:
                        _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(schema)


# ------------------------------------------------------ reconstruct_tool_calls


def reconstruct_tool_calls(deltas: list[dict[str, Any]]) -> list[ToolCall]:
    """Merge indexed-cumulative tool-call deltas into complete :class:`ToolCall`\\ s.

    OpenRouter (like OpenAI Chat Completions) streams tool calls as deltas
    indexed by position, with arguments arriving as cumulative string
    fragments. A single chunk may touch multiple indices at once.

    Reconstruction rules:

    - Merge by ``index``; concatenate ``function.arguments`` fragments
      into one string.
    - ``id`` and ``function.name`` latch on the first non-empty value seen
      for each index.
    - JSON-parse the final ``arguments`` string once; malformed arguments
      are preserved as ``{"raw_arguments": <string>}`` to mirror the
      existing OpenAI provider's posture.
    - Empty arguments default to ``{}`` (rather than raising), matching
      providers that emit a no-arg tool call with a missing ``arguments``
      field.

    Args:
        deltas: Raw ``delta`` objects collected from streamed chunks.

    Returns:
        Complete :class:`ToolCall` instances in index order.
    """

    slots: dict[int, dict[str, Any]] = {}
    for delta in deltas:
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index")
            if idx is None:
                continue
            slot = slots.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.get("id") and not slot["id"]:
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name") and not slot["name"]:
                slot["name"] = fn["name"]
            args_frag = fn.get("arguments")
            if args_frag is not None:
                slot["arguments"] += args_frag

    out: list[ToolCall] = []
    for idx in sorted(slots):
        slot = slots[idx]
        raw = slot["arguments"]
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"raw_arguments": raw}
        call_id = slot["id"]
        if not call_id:
            # Some providers (e.g. certain MoonshotAI / GLM variants) stream
            # tool_calls without ever populating ``id``. Synthesize one so
            # downstream ``function_call_output.call_id`` matching still
            # works — and keep it obviously client-side so logs are clear.
            call_id = f"call_{uuid.uuid4().hex[:12]}"
        out.append(ToolCall(call_id=call_id, name=slot["name"], arguments=parsed))
    return out


# ---------------------------------------------------------- translate_response


def translate_response(
    *,
    final_chunk: dict[str, Any],
    output_text: str,
    deltas: list[dict[str, Any]],
) -> ModelTurn:
    """Build a :class:`ModelTurn` from the final SSE chunk plus accumulated deltas.

    ``response_id`` is the OpenRouter generation id (also surfaced as the
    ``X-Generation-Id`` response header). Feather stores it as an opaque
    cursor; OpenRouter itself is stateless, so it is never sent back.

    Args:
        final_chunk: Last non-``[DONE]`` SSE chunk — the one carrying
            ``usage``.
        output_text: Concatenated ``delta.content`` text seen across the
            stream.
        deltas: Every raw ``delta`` object observed, used for tool-call
            reconstruction.

    Returns:
        Normalized ``ModelTurn``.
    """

    return ModelTurn(
        response_id=final_chunk.get("id"),
        output_text=output_text,
        tool_calls=reconstruct_tool_calls(deltas),
        usage=_normalize_usage(final_chunk.get("usage")),
    )


def _normalize_usage(usage: Any) -> dict[str, Any] | None:
    """Add Responses-style token keys to OpenRouter's Chat-style usage dict."""

    if not isinstance(usage, dict):
        return None
    normalized = dict(usage)
    prompt_tokens = normalized.get("prompt_tokens")
    completion_tokens = normalized.get("completion_tokens")
    if "input_tokens" not in normalized and isinstance(prompt_tokens, int):
        normalized["input_tokens"] = prompt_tokens
    if "output_tokens" not in normalized and isinstance(completion_tokens, int):
        normalized["output_tokens"] = completion_tokens
    return normalized
