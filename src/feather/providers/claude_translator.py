"""Pure translation between Feather's Responses-API-shaped state and Anthropic Messages API.

Mirrors the layout of :mod:`feather.providers.openrouter_translator` —
isolating shape concerns in one pure module keeps every branch
unit-testable without touching HTTP, and lets the provider orchestrator
focus on streaming/retry without owning shape.

Three entry points mirror the turn lifecycle:

- :func:`translate_request` — builds the POST body for one Messages turn.
- :func:`reconstruct_tool_calls` — assembles streamed ``input_json_delta``
  fragments into :class:`feather.models.ToolCall` instances.
- :func:`translate_response` — combines the final accumulated content
  blocks, output text, and message metadata into a normalized
  :class:`ModelTurn`.

``translate_tools`` and ``translate_input_items`` are the sub-helpers that
``translate_request`` depends on; they are exposed so tests can pin their
behavior independently.

Anthropic-specific quirks the translator handles:

- The Messages API requires strictly alternating user/assistant turns. We
  merge consecutive same-role items into one message with concatenated
  content blocks (e.g. multiple ``function_call_output`` items in a row
  become one user message with multiple ``tool_result`` blocks).
- ``system`` is a *separate* top-level field (not a message). When
  ``cache_strategy == "anthropic_breakpoint"`` it is emitted as a list of
  ``text`` blocks with ``cache_control: {type: "ephemeral"}`` on the
  trailing one to anchor a prompt-cache breakpoint.
- ``tool_use`` arguments arrive as accumulated ``input_json_delta``
  ``partial_json`` strings; we parse once at ``content_block_stop``.
- Tool definitions use ``input_schema`` (not OpenAI's ``parameters``).
- Extended thinking forbids ``tool_choice`` other than ``auto`` / ``none``;
  the translator silently downgrades a forced choice to ``auto`` rather
  than raising, so a structured-output schema doesn't break a
  thinking-enabled agent.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any

from feather.models import (
    ClaudeConfig,
    ModelTurn,
    ProviderRequestConfig,
    ToolCall,
)

logger = logging.getLogger(__name__)


_ANTHROPIC_REJECTED_INTEGER_KEYWORDS: frozenset[str] = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
)


def sanitize_anthropic_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` safe for Anthropic's tool validator.

    Anthropic's Messages API rejects ``minimum``/``maximum``/
    ``exclusiveMinimum``/``exclusiveMaximum`` on ``integer``-typed
    properties (and on ``number`` for that matter) and does not honour
    ``"type": [..., "null"]`` unions inside tool input schemas. This
    helper strips the unsupported keywords and normalises type unions,
    walking ``properties``, ``items``, ``anyOf``, ``oneOf``, and
    ``allOf`` recursively.

    The original ``schema`` is not mutated. Calling the sanitizer twice
    is a no-op (idempotent).

    Args:
        schema: A JSON Schema fragment (typically a tool's
            ``parameters`` / ``input_schema`` body).

    Returns:
        A deep-copied, sanitized schema fragment.
    """
    return _sanitize_node(copy.deepcopy(schema))


def _sanitize_node(node: Any) -> Any:
    if isinstance(node, dict):
        if "type" in node and isinstance(node["type"], list):
            non_null = [t for t in node["type"] if t != "null"]
            if len(non_null) == 1:
                node["type"] = non_null[0]
            elif len(non_null) == 0:
                # Pathological ``["null"]`` — fall back to "string"
                # rather than crash; logged so authors notice.
                logger.warning(
                    "claude tool schema: type list reduced to empty; "
                    "defaulting to string"
                )
                node["type"] = "string"
            else:
                node["type"] = non_null
                logger.warning(
                    "claude tool schema: multi-type union %s reached "
                    "the wire untouched (Anthropic may reject)",
                    non_null,
                )
        for keyword in _ANTHROPIC_REJECTED_INTEGER_KEYWORDS:
            node.pop(keyword, None)
        for key, value in list(node.items()):
            node[key] = _sanitize_node(value)
        return node
    if isinstance(node, list):
        return [_sanitize_node(item) for item in node]
    return node


# Cache-breakpoint placement is a single ephemeral marker on the trailing
# text block of the system prompt — Anthropic allows up to 4 breakpoints
# total, but one breakpoint anchored on the stable instructions prefix
# captures the dominant cache benefit and leaves headroom for callers
# that want to add tool/message-level breakpoints later.
_CACHE_STRATEGY_BREAKPOINT = "anthropic_breakpoint"

# Anthropic's image-block ``source`` accepts a base64 inline payload or a
# remote URL. Feather hands every image in as a single URL string; we
# split ``data:`` URLs into the base64 form because the Messages API
# requires both ``media_type`` and ``data`` to be set explicitly.
_DATA_URL_RE = re.compile(
    r"^data:(?P<media>[\w/+.\-]+)(;[\w-]+=[\w-]+)*;base64,(?P<data>.+)$",
    re.DOTALL,
)


# ---------------------------------------------------------------------- tools


def translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrap Responses-API flat tool definitions into Anthropic Messages form.

    Feather's :meth:`feather.tools.base.BaseTool.to_openai_tool` returns a
    flat ``{type, name, description, parameters, strict}`` dict suitable
    for the OpenAI Responses API. Anthropic uses
    ``{name, description, input_schema}`` (and optionally ``strict``) at
    the top level — no ``type`` wrapping. Entries already in Anthropic
    shape (identified by an ``input_schema`` key) pass through.

    Args:
        tools: Tool definitions in either flat-Responses or
            Anthropic-native form.

    Returns:
        Tools in Anthropic Messages API form.
    """

    out: list[dict[str, Any]] = []
    for tool in tools:
        if "input_schema" in tool:
            out.append(tool)
            continue
        translated: dict[str, Any] = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get(
                "parameters", {"type": "object", "properties": {}}
            ),
        }
        if tool.get("strict"):
            translated["strict"] = True
        out.append(translated)
    return out


# ----------------------------------------------------------------- input_items


_TEXT_BLOCK_TYPES = frozenset({"input_text", "output_text", "text"})


def translate_input_items(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Feather ``input_items`` (Responses API shape) into Anthropic messages.

    Handles the same four item shapes as the OpenRouter translator:

    1. ``{"type": "message", "role": ..., "content": [...]}`` — text /
       image / file blocks turn into Anthropic ``text`` / ``image`` /
       ``document`` content blocks.
    2. ``{"type": "function_call", ...}`` — assistant tool-call request
       replayed for stateless providers. Becomes a ``tool_use`` content
       block on the *current* assistant message; a run of consecutive
       ``function_call`` items collapses into one assistant message with
       multiple ``tool_use`` blocks (matching how Anthropic emits them).
    3. ``{"type": "function_call_output", ...}`` — tool-result rows become
       ``tool_result`` content blocks on a user message.
    4. Flat ``{"role": ..., "content": ...}`` — direct shape from memory
       extractor / classifier callers.

    Anthropic strictly requires alternating user/assistant turns, so any
    consecutive same-role messages are merged into one. Unknown item
    types (e.g. ``type == "reasoning"``) are skipped — they have no
    Messages-API equivalent and never appear as live input items in
    Feather's compaction-and-replay flow.

    Args:
        input_items: Feather-shaped input items.

    Returns:
        Anthropic Messages API ``messages[]`` entries — each is
        ``{"role": "user"|"assistant", "content": [<blocks>]}``.
    """

    messages: list[dict[str, Any]] = []

    def append_blocks(role: str, blocks: list[dict[str, Any]]) -> None:
        """Append ``blocks`` under ``role``, merging into the trailing message
        when the role matches so the wire body keeps strict alternation."""

        if not blocks:
            return
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
            return
        messages.append({"role": role, "content": list(blocks)})

    for item in input_items:
        item_type = item.get("type")
        if item_type == "message":
            role = item.get("role", "user")
            blocks = _content_blocks_from_message(item.get("content"))
            if blocks:
                append_blocks(role, blocks)
            continue
        if item_type == "function_call":
            tool_use = _function_call_to_tool_use(item)
            if tool_use is not None:
                append_blocks("assistant", [tool_use])
            continue
        if item_type == "function_call_output":
            tool_result = _function_call_output_to_tool_result(item)
            if tool_result is not None:
                append_blocks("user", [tool_result])
            continue
        if item_type is None and "role" in item and "content" in item:
            blocks = _content_blocks_from_message(item["content"])
            if blocks:
                append_blocks(item["role"], blocks)
            continue
        # Unsupported item types (e.g. reasoning) are silently skipped.

    return messages


def _content_blocks_from_message(content: Any) -> list[dict[str, Any]]:
    """Translate a Responses ``content`` field into Anthropic content blocks.

    String content is wrapped in a single ``text`` block. Block lists are
    walked, with ``input_text``/``output_text``/``text`` becoming
    ``text`` blocks, ``input_image`` becoming ``image`` blocks, and
    ``input_file`` becoming ``document`` blocks. Empty / unparseable
    blocks are dropped silently — the message-level loop handles the
    "no usable content" case by skipping the message entirely.

    Args:
        content: Raw content field from a Responses message item.

    Returns:
        List of Anthropic content blocks (possibly empty).
    """

    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return []
    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in _TEXT_BLOCK_TYPES:
            text = str(block.get("text", ""))
            if text:
                blocks.append({"type": "text", "text": text})
            continue
        if block_type == "input_image":
            image_block = _translate_image_block(block)
            if image_block is not None:
                blocks.append(image_block)
            continue
        if block_type == "input_file":
            file_block = _translate_file_block(block)
            if file_block is not None:
                blocks.append(file_block)
    return blocks


def _translate_image_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """Translate a Responses ``input_image`` block into an Anthropic ``image`` block.

    Accepts both raw URL strings and ``data:`` URLs. ``data:`` URLs are
    split into Anthropic's required ``{type: "base64", media_type, data}``
    shape; everything else flows through as ``{type: "url", url}``. Blocks
    without any usable URL field return ``None`` so the caller can drop
    them.
    """

    url = block.get("image_url") or block.get("url")
    if not isinstance(url, str) or not url:
        return None
    match = _DATA_URL_RE.match(url)
    if match:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": match.group("media"),
                "data": match.group("data"),
            },
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _translate_file_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """Translate a Responses ``input_file`` block into an Anthropic ``document`` block.

    Supports the three Feather attachment shapes — base64 ``file_data``,
    remote ``file_url``, and Anthropic Files-API ``file_id`` — by
    forwarding each into the matching Anthropic ``source`` form. Unknown
    shapes return ``None`` so the caller can drop them.
    """

    file_data = block.get("file_data")
    file_url = block.get("file_url")
    file_id = block.get("file_id")
    media_type = block.get("media_type") or "application/pdf"
    if isinstance(file_data, str) and file_data:
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": file_data,
            },
        }
    if isinstance(file_url, str) and file_url:
        return {"type": "document", "source": {"type": "url", "url": file_url}}
    if isinstance(file_id, str) and file_id:
        return {
            "type": "document",
            "source": {"type": "file", "file_id": file_id},
        }
    return None


def _function_call_to_tool_use(item: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a Responses ``function_call`` item into an Anthropic ``tool_use`` block."""

    call_id = item.get("call_id")
    name = item.get("name")
    if not call_id or not name:
        return None
    return {
        "type": "tool_use",
        "id": call_id,
        "name": name,
        "input": _coerce_tool_input(item.get("arguments")),
    }


def _function_call_output_to_tool_result(
    item: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert a Responses ``function_call_output`` item into a ``tool_result`` block."""

    call_id = item.get("call_id")
    if not call_id:
        return None
    output = item.get("output")
    if isinstance(output, str):
        content: list[dict[str, Any]] | str = output
    elif isinstance(output, list):
        # Already shaped as Anthropic content blocks (rare). Pass through.
        content = output
    else:
        content = "" if output is None else json.dumps(output)
    return {"type": "tool_result", "tool_use_id": call_id, "content": content}


def _coerce_tool_input(arguments: Any) -> dict[str, Any]:
    """Return ``arguments`` as a JSON object suitable for ``tool_use.input``.

    Anthropic's ``tool_use.input`` is a JSON object. Most callers hand us a
    pre-parsed dict already; the JSON-string fallback exists because the
    OpenRouter / OpenAI translators store it as a string and the same
    input items can flow through either path during compaction replay.
    """

    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        if not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw_arguments": arguments}
        if isinstance(parsed, dict):
            return parsed
        return {"raw_arguments": arguments}
    return {}


# -------------------------------------------------------------- translate_request


def translate_request(
    *,
    instructions: str,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    request_config: ProviderRequestConfig,
    cfg: ClaudeConfig,
) -> dict[str, Any]:
    """Build the Anthropic Messages POST body for one turn.

    Behaviors:

    - ``system`` is emitted as a list of ``text`` blocks when
      ``cfg.cache_strategy == "anthropic_breakpoint"`` so the trailing
      block carries ``cache_control: {type: "ephemeral"}``. Otherwise it
      flows through as a plain string.
    - ``thinking`` (extended thinking) is forwarded only when configured
      and ``type != "disabled"``. When thinking is on, ``temperature``
      is dropped (Anthropic ignores it for the thinking phase, and
      sending it together with low budget tokens has been observed to
      trip 400s on some models).
    - ``response_schema`` is implemented as a forced single-tool: a
      synthetic tool with the schema as ``input_schema`` is appended and
      ``tool_choice`` is pinned to it. This works on every Claude model
      that supports tool use and matches OpenAI's strict-JSON contract.
      The provider extracts the structured response from the tool call.
    - ``parallel_tool_calls=False`` translates to
      ``tool_choice.disable_parallel_tool_use=True``.

    Args:
        instructions: Full system instructions.
        input_items: Feather-shaped input items for the new turn.
        tools: Tool definitions in Feather's Responses-API shape.
        request_config: Per-request overrides from the agent loop.
        cfg: Provider-level defaults.

    Returns:
        JSON-serializable POST body for ``/v1/messages``.
    """

    model = request_config.model or cfg.model
    desired_max = (
        cfg.max_output_tokens
        if request_config.max_output_tokens is None
        else request_config.max_output_tokens
    )
    temperature = (
        request_config.temperature
        if request_config.temperature is not None
        else cfg.temperature
    )

    if cfg.cache_strategy == _CACHE_STRATEGY_BREAKPOINT:
        system_field: Any = [
            {
                "type": "text",
                "text": instructions,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_field = instructions

    body: dict[str, Any] = {
        "model": model,
        "stream": True,
        "max_tokens": desired_max,
        "system": system_field,
        "messages": translate_input_items(input_items),
    }

    thinking_on = False
    if cfg.thinking is not None and cfg.thinking.type != "disabled":
        thinking_block: dict[str, Any] = {"type": cfg.thinking.type}
        if cfg.thinking.type == "enabled" and cfg.thinking.budget_tokens is not None:
            thinking_block["budget_tokens"] = cfg.thinking.budget_tokens
        body["thinking"] = thinking_block
        thinking_on = True

    # Anthropic rejects ``temperature`` in two cases:
    #   1. Extended thinking is enabled — Anthropic ignores it for the
    #      thinking phase, and sending both has been observed to 400 on
    #      Sonnet 4.5.
    #   2. Newer model families (Opus 4.7+, Mythos) deprecated the field
    #      entirely; sending it returns ``400: temperature is deprecated
    #      for this model``. Same posture as gpt-5 in the OpenAI provider.
    if not thinking_on and _supports_temperature(model):
        body["temperature"] = temperature

    schema_tool: dict[str, Any] | None = None
    forced_tool_name: str | None = None
    if request_config.response_schema is not None:
        schema_cls = request_config.response_schema
        schema = schema_cls.model_json_schema()
        _harden_strict_schema(schema)
        forced_tool_name = (
            request_config.response_schema_name or schema_cls.__name__
        )
        schema_tool = {
            "name": forced_tool_name,
            "description": (
                f"Return a structured response matching the {forced_tool_name} schema."
            ),
            "input_schema": schema,
        }

    claude_tools = translate_tools(tools)
    if schema_tool is not None:
        claude_tools.append(schema_tool)

    if claude_tools:
        body["tools"] = claude_tools
        if forced_tool_name is not None and not thinking_on:
            # Forced single-tool is the canonical pattern for structured
            # outputs. Anthropic forbids ``any``/``tool`` choices when
            # extended thinking is on — fall back to ``auto`` in that
            # case so the agent doesn't crash; the model will usually
            # pick the schema tool anyway because it's the only one.
            body["tool_choice"] = {"type": "tool", "name": forced_tool_name}
        elif not cfg.parallel_tool_calls:
            body["tool_choice"] = {
                "type": "auto",
                "disable_parallel_tool_use": True,
            }

    return body


_TEMPERATURE_UNSUPPORTED_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-7",
    "claude-mythos",
)


def _supports_temperature(model: str) -> bool:
    """Return whether the configured model accepts a ``temperature`` field.

    Anthropic's newest model families (Opus 4.7, Mythos) reject
    ``temperature`` outright — they manage sampling via internal
    adaptive thinking and surface ``400: temperature is deprecated for
    this model`` when the field is sent. This helper is the
    centralized allow-list so callers stay readable; mirrors the
    OpenAI provider's ``_supports_temperature`` for ``gpt-5`` reasoning
    models.
    """

    normalized = model.strip().lower()
    return not any(normalized.startswith(p) for p in _TEMPERATURE_UNSUPPORTED_PREFIXES)


def _harden_strict_schema(schema: dict[str, Any]) -> None:
    """Enforce strict-mode invariants on a JSON schema in place.

    Anthropic's ``strict`` flag (and OpenAI's strict mode) require every
    ``type: "object"`` node to set ``additionalProperties: false`` and
    list every property name in ``required``. Pydantic emits the root
    ``additionalProperties`` correctly but not always on nested objects,
    and it drops fields with defaults from ``required``. This walker
    fixes both throughout the schema graph (``$defs``, ``properties``,
    ``items``, ``anyOf`` / ``oneOf`` / ``allOf``).
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


def reconstruct_tool_calls(blocks: list[dict[str, Any]]) -> list[ToolCall]:
    """Convert a list of accumulated content blocks into :class:`ToolCall`\\ s.

    The provider streams each ``tool_use`` block as a series of
    ``input_json_delta`` ``partial_json`` fragments and finalizes them at
    the ``content_block_stop`` event. The streamer hands us the merged
    blocks (one per index, ordered by ``index``); this helper just
    JSON-parses each finalized ``input`` and wraps the result in a
    :class:`ToolCall`.

    Malformed JSON is preserved as ``{"raw_arguments": <string>}`` to
    mirror the OpenAI / OpenRouter providers' posture and let the agent
    loop log the failure rather than crashing the run.
    """

    out: list[ToolCall] = []
    for block in blocks:
        if block.get("type") != "tool_use":
            continue
        out.append(
            ToolCall(
                call_id=str(block.get("id") or ""),
                name=str(block.get("name") or ""),
                arguments=_coerce_tool_input(block.get("input")),
            )
        )
    return out


# ---------------------------------------------------------- translate_response


def translate_response(
    *,
    message_id: str | None,
    blocks: list[dict[str, Any]],
    output_text: str,
    usage: dict[str, Any] | None,
) -> ModelTurn:
    """Build a :class:`ModelTurn` from accumulated message state.

    Args:
        message_id: ``message_start.message.id`` if seen, else ``None``.
            Stored as ``response_id`` for log correlation; the Messages
            API is stateless, so it is never replayed.
        blocks: Final content blocks (text, tool_use, thinking, etc.) in
            order, fully assembled by the streamer.
        output_text: Concatenated assistant text seen across the stream
            — passed in by the streamer so we don't recompute from
            ``blocks`` (and so streaming chunks can be emitted to the CLI
            in real time without buffering).
        usage: Anthropic ``usage`` dict (``input_tokens``,
            ``output_tokens``, optional ``cache_creation_input_tokens``,
            ``cache_read_input_tokens``).

    Returns:
        Normalized ``ModelTurn``.
    """

    return ModelTurn(
        response_id=message_id,
        output_text=output_text,
        tool_calls=reconstruct_tool_calls(blocks),
        usage=_normalize_usage(usage),
    )


def _normalize_usage(usage: Any) -> dict[str, Any] | None:
    """Add a Responses-API-friendly aliases to Anthropic usage.

    Feather's compaction trigger reads ``usage.input_tokens`` /
    ``usage.output_tokens`` directly. Anthropic already uses those
    names, so the dict passes through; we just preserve the original
    shape and forward the cache fields untouched so observability code
    can surface them.
    """

    if not isinstance(usage, dict):
        return None
    return dict(usage)
