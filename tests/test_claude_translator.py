"""Tests for the Anthropic Messages API translator."""

from __future__ import annotations

from pydantic import BaseModel

from feather.models import (
    ClaudeConfig,
    ClaudeThinkingConfig,
    ProviderRequestConfig,
)
from feather.providers.claude_translator import (
    reconstruct_tool_calls,
    translate_input_items,
    translate_request,
    translate_response,
    translate_tools,
)


# ---------------------------------------------------------------- translate_tools


def test_translate_tools_rewraps_responses_flat_into_anthropic_shape() -> None:
    tools = [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "strict": True,
        }
    ]
    out = translate_tools(tools)
    assert len(out) == 1
    assert out[0]["name"] == "read_file"
    assert out[0]["description"] == "Read a file from disk."
    assert out[0]["input_schema"]["properties"]["path"]["type"] == "string"
    assert out[0]["strict"] is True
    # No ``type`` wrapping in the Anthropic shape.
    assert "type" not in out[0]


def test_translate_tools_passes_through_anthropic_native_shape() -> None:
    """A tool already in Anthropic shape (with ``input_schema``) flows through unchanged."""

    native = {
        "name": "weather",
        "description": "Get the weather.",
        "input_schema": {"type": "object", "properties": {}},
    }
    out = translate_tools([native])
    assert out == [native]


def test_translate_tools_omits_strict_when_falsy() -> None:
    out = translate_tools(
        [{"type": "function", "name": "x", "parameters": {"type": "object"}}]
    )
    assert "strict" not in out[0]


def test_translate_tools_falls_back_to_empty_input_schema_when_parameters_missing() -> None:
    out = translate_tools([{"name": "no_schema"}])
    assert out[0]["input_schema"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------- translate_input_items


def test_translate_input_items_text_message_becomes_text_block() -> None:
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]
    out = translate_input_items(items)
    assert out == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_translate_input_items_strips_empty_text_blocks() -> None:
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": ""}],
        }
    ]
    assert translate_input_items(items) == []


def test_translate_input_items_string_content_becomes_text_block() -> None:
    out = translate_input_items([{"role": "user", "content": "hi"}])
    assert out == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_translate_input_items_function_call_becomes_tool_use_in_assistant_msg() -> None:
    items = [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": {"q": "weather"},
        }
    ]
    out = translate_input_items(items)
    assert out == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "lookup",
                    "input": {"q": "weather"},
                }
            ],
        }
    ]


def test_translate_input_items_function_call_with_string_arguments_parses_json() -> None:
    items = [
        {
            "type": "function_call",
            "call_id": "c",
            "name": "f",
            "arguments": '{"a": 1}',
        }
    ]
    out = translate_input_items(items)
    assert out[0]["content"][0]["input"] == {"a": 1}


def test_translate_input_items_function_call_with_unparseable_args_preserves_raw() -> None:
    items = [
        {
            "type": "function_call",
            "call_id": "c",
            "name": "f",
            "arguments": "not-json",
        }
    ]
    out = translate_input_items(items)
    assert out[0]["content"][0]["input"] == {"raw_arguments": "not-json"}


def test_translate_input_items_consecutive_function_calls_merge_into_one_assistant_msg() -> None:
    """Anthropic emits multiple tool_use blocks under one assistant message;
    the translator must replay them the same way."""

    items = [
        {"type": "function_call", "call_id": "1", "name": "a", "arguments": {}},
        {"type": "function_call", "call_id": "2", "name": "b", "arguments": {"x": 1}},
    ]
    out = translate_input_items(items)
    assert len(out) == 1
    assert out[0]["role"] == "assistant"
    assert [b["id"] for b in out[0]["content"]] == ["1", "2"]


def test_translate_input_items_function_call_output_becomes_tool_result_in_user_msg() -> None:
    items = [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "the result",
        }
    ]
    out = translate_input_items(items)
    assert out == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "the result",
                }
            ],
        }
    ]


def test_translate_input_items_consecutive_tool_results_merge_into_one_user_msg() -> None:
    items = [
        {"type": "function_call_output", "call_id": "1", "output": "a"},
        {"type": "function_call_output", "call_id": "2", "output": "b"},
    ]
    out = translate_input_items(items)
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert [b["tool_use_id"] for b in out[0]["content"]] == ["1", "2"]


def test_translate_input_items_alternation_assistant_then_user() -> None:
    """Tool call followed by tool result should produce alternating assistant/user."""

    items = [
        {"type": "function_call", "call_id": "1", "name": "f", "arguments": {}},
        {"type": "function_call_output", "call_id": "1", "output": "ok"},
    ]
    out = translate_input_items(items)
    assert [m["role"] for m in out] == ["assistant", "user"]


def test_translate_input_items_image_url_becomes_image_block_with_url_source() -> None:
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "https://example.com/cat.png"},
            ],
        }
    ]
    out = translate_input_items(items)
    assert out[0]["content"] == [
        {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/cat.png"},
        }
    ]


def test_translate_input_items_image_data_url_becomes_base64_source() -> None:
    """A ``data:`` URL must split into Anthropic's base64 source — Anthropic
    rejects unsplit ``data:`` URLs."""

    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAA=",
                },
            ],
        }
    ]
    out = translate_input_items(items)
    assert out[0]["content"] == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "AAA=",
            },
        }
    ]


def test_translate_input_items_input_file_base64_becomes_document_block() -> None:
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_file", "file_data": "PDFDATA", "filename": "x.pdf"},
            ],
        }
    ]
    out = translate_input_items(items)
    assert out[0]["content"] == [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "PDFDATA",
            },
        }
    ]


def test_translate_input_items_input_file_url_becomes_document_url() -> None:
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_file", "file_url": "https://example.com/x.pdf"},
            ],
        }
    ]
    out = translate_input_items(items)
    assert out[0]["content"][0]["source"] == {
        "type": "url",
        "url": "https://example.com/x.pdf",
    }


def test_translate_input_items_skips_unknown_block_types() -> None:
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "ok"},
                {"type": "weird_unknown_thing", "value": 1},
            ],
        }
    ]
    out = translate_input_items(items)
    assert out[0]["content"] == [{"type": "text", "text": "ok"}]


def test_translate_input_items_skips_reasoning_items() -> None:
    """Reasoning items have no Messages-API equivalent and must be dropped."""

    items = [
        {"type": "reasoning", "summary": "thinking..."},
        {"role": "user", "content": "hi"},
    ]
    out = translate_input_items(items)
    assert out == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


# -------------------------------------------------------------- translate_request


def test_translate_request_emits_cache_control_breakpoint_on_system_when_enabled() -> None:
    cfg = ClaudeConfig(cache_strategy="anthropic_breakpoint")
    body = translate_request(
        instructions="you are a bot",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
    )
    assert body["system"] == [
        {
            "type": "text",
            "text": "you are a bot",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_translate_request_emits_plain_string_system_when_cache_disabled() -> None:
    cfg = ClaudeConfig(cache_strategy="off")
    body = translate_request(
        instructions="sys",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
    )
    assert body["system"] == "sys"


def test_translate_request_splits_system_at_cache_prefix() -> None:
    """The breakpoint anchors the static prefix; the dynamic suffix is uncached."""

    prefix = "STATIC PREFIX"
    instructions = f"{prefix}\n\nDYNAMIC SUFFIX"
    cfg = ClaudeConfig(cache_strategy="anthropic_breakpoint")
    body = translate_request(
        instructions=instructions,
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
        cache_prefix=prefix,
    )

    blocks = body["system"]
    # Exactly one breakpoint, and it marks the static prefix block only.
    marked = [b for b in blocks if b.get("cache_control")]
    assert len(marked) == 1
    assert marked[0]["text"] == prefix
    assert marked[0]["cache_control"] == {"type": "ephemeral"}
    # The dynamic remainder is a separate, uncached block.
    assert blocks[-1].get("cache_control") is None
    assert "DYNAMIC SUFFIX" in blocks[-1]["text"]
    assert "DYNAMIC SUFFIX" not in marked[0]["text"]
    # The concatenated blocks reproduce the original instructions byte-for-byte.
    assert "".join(b["text"] for b in blocks) == instructions


def test_translate_request_falls_back_to_single_block_without_prefix() -> None:
    """No cache_prefix → legacy single cached block (un-threaded callers safe)."""

    cfg = ClaudeConfig(cache_strategy="anthropic_breakpoint")
    body = translate_request(
        instructions="whole prompt",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
        cache_prefix=None,
    )
    assert body["system"] == [
        {"type": "text", "text": "whole prompt", "cache_control": {"type": "ephemeral"}}
    ]


def test_translate_request_marks_rolling_breakpoint_on_last_message() -> None:
    """In breakpoint mode the last message block carries a rolling breakpoint.

    Plus the static-prefix breakpoint, that is at most 2 of the 4 allowed.
    """

    cfg = ClaudeConfig(cache_strategy="anthropic_breakpoint")
    body = translate_request(
        instructions="STATIC\n\nDYNAMIC",
        input_items=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
        cache_prefix="STATIC",
    )

    messages = body["messages"]
    # The rolling breakpoint is on the LAST block of the LAST message only.
    assert messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in messages[0]["content"][-1]
    # Total breakpoints across the request stay within Anthropic's limit of 4.
    total = sum(
        1 for b in body["system"] if b.get("cache_control")
    ) + sum(
        1
        for m in messages
        for b in m["content"]
        if isinstance(b, dict) and b.get("cache_control")
    )
    assert total <= 4


def test_translate_request_no_message_breakpoint_when_cache_off() -> None:
    """Cache-off mode adds no message-level breakpoints."""

    cfg = ClaudeConfig(cache_strategy="off")
    body = translate_request(
        instructions="sys",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
    )
    assert body["messages"][-1]["content"][-1] == {"type": "text", "text": "hi"}


def test_translate_request_single_block_when_prefix_equals_instructions() -> None:
    """A fully-stable prompt (prefix == instructions) stays one cached block."""

    cfg = ClaudeConfig(cache_strategy="anthropic_breakpoint")
    body = translate_request(
        instructions="all static",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
        cache_prefix="all static",
    )
    assert len([b for b in body["system"] if b.get("cache_control")]) == 1
    assert len(body["system"]) == 1


def test_translate_request_sets_required_fields() -> None:
    cfg = ClaudeConfig(model="claude-sonnet-4-6")
    body = translate_request(
        instructions="s",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
    )
    assert body["model"] == cfg.model
    assert body["stream"] is True
    assert body["max_tokens"] == cfg.max_output_tokens
    # Default config is breakpoint mode, so the last message block carries the
    # rolling conversation-history breakpoint.
    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    assert body["temperature"] == cfg.temperature


def test_translate_request_request_config_overrides_take_precedence() -> None:
    cfg = ClaudeConfig(model="cfg-model", max_output_tokens=1, temperature=0.1)
    body = translate_request(
        instructions="s",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(
            model="override-model", max_output_tokens=99, temperature=0.9
        ),
        cfg=cfg,
    )
    assert body["model"] == "override-model"
    assert body["max_tokens"] == 99
    assert body["temperature"] == 0.9


def test_translate_request_thinking_config_emits_thinking_block_and_drops_temperature() -> None:
    cfg = ClaudeConfig(
        thinking=ClaudeThinkingConfig(type="enabled", budget_tokens=4000)
    )
    body = translate_request(
        instructions="s",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
    )
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4000}
    assert "temperature" not in body


def test_translate_request_omits_temperature_for_opus_4_7() -> None:
    """Opus 4.7 rejects ``temperature`` outright — translator must drop it."""

    cfg = ClaudeConfig(model="claude-opus-4-7")
    body = translate_request(
        instructions="s",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
    )
    assert "temperature" not in body


def test_translate_request_keeps_temperature_for_sonnet() -> None:
    cfg = ClaudeConfig(model="claude-sonnet-4-6", temperature=0.5)
    body = translate_request(
        instructions="s",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
    )
    assert body["temperature"] == 0.5


def test_translate_request_thinking_disabled_omits_thinking_block() -> None:
    cfg = ClaudeConfig(model="claude-sonnet-4-6", thinking=ClaudeThinkingConfig(type="disabled"))
    body = translate_request(
        instructions="s",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
    )
    assert "thinking" not in body
    assert "temperature" in body


def test_translate_request_thinking_adaptive_omits_budget_tokens() -> None:
    cfg = ClaudeConfig(thinking=ClaudeThinkingConfig(type="adaptive"))
    body = translate_request(
        instructions="s",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
    )
    assert body["thinking"] == {"type": "adaptive"}


def test_translate_request_disable_parallel_tool_use_when_parallel_false() -> None:
    cfg = ClaudeConfig(parallel_tool_calls=False)
    body = translate_request(
        instructions="s",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "name": "f",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
    )
    assert body["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


def test_translate_request_no_tool_choice_when_parallel_true_no_schema() -> None:
    cfg = ClaudeConfig(parallel_tool_calls=True)
    body = translate_request(
        instructions="s",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "name": "f",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
    )
    assert "tool_choice" not in body


class _SchemaForTest(BaseModel):
    answer: str
    confidence: float


def test_translate_request_response_schema_emits_forced_single_tool() -> None:
    body = translate_request(
        instructions="s",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(
            response_schema=_SchemaForTest, response_schema_name="MyAnswer"
        ),
        cfg=ClaudeConfig(),
    )
    tools = body["tools"]
    assert any(t["name"] == "MyAnswer" for t in tools)
    assert body["tool_choice"] == {"type": "tool", "name": "MyAnswer"}
    schema_tool = next(t for t in tools if t["name"] == "MyAnswer")
    schema = schema_tool["input_schema"]
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["answer", "confidence"]


def test_translate_request_thinking_blocks_forced_choice_drops_to_auto() -> None:
    """Anthropic forbids tool_choice ``tool``/``any`` when extended thinking is on.
    The translator falls back to leaving ``tool_choice`` unset rather than
    failing — the model is still pointed at the schema tool because it's
    the only one defined."""

    cfg = ClaudeConfig(thinking=ClaudeThinkingConfig(type="adaptive"))
    body = translate_request(
        instructions="s",
        input_items=[{"role": "user", "content": "hi"}],
        tools=[],
        request_config=ProviderRequestConfig(response_schema=_SchemaForTest),
        cfg=cfg,
    )
    assert "tool_choice" not in body


# --------------------------------------------------------- reconstruct_tool_calls


def test_reconstruct_tool_calls_parses_finalized_input_dict() -> None:
    blocks = [
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "f",
            "input": {"a": 1, "b": "x"},
        }
    ]
    calls = reconstruct_tool_calls(blocks)
    assert len(calls) == 1
    assert calls[0].call_id == "call_1"
    assert calls[0].name == "f"
    assert calls[0].arguments == {"a": 1, "b": "x"}


def test_reconstruct_tool_calls_parses_string_input_as_json() -> None:
    blocks = [
        {"type": "tool_use", "id": "c", "name": "f", "input": '{"a": 1}'}
    ]
    calls = reconstruct_tool_calls(blocks)
    assert calls[0].arguments == {"a": 1}


def test_reconstruct_tool_calls_preserves_unparseable_input() -> None:
    blocks = [
        {"type": "tool_use", "id": "c", "name": "f", "input": "not-json"}
    ]
    calls = reconstruct_tool_calls(blocks)
    assert calls[0].arguments == {"raw_arguments": "not-json"}


def test_reconstruct_tool_calls_skips_text_and_thinking_blocks() -> None:
    blocks = [
        {"type": "text", "text": "hi"},
        {"type": "thinking", "thinking": "...", "signature": "sig"},
        {"type": "tool_use", "id": "c", "name": "f", "input": {}},
    ]
    calls = reconstruct_tool_calls(blocks)
    assert [c.call_id for c in calls] == ["c"]


# ----------------------------------------------------------- translate_response


def test_translate_response_builds_model_turn() -> None:
    turn = translate_response(
        message_id="msg_01",
        blocks=[
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "c", "name": "f", "input": {"x": 1}},
        ],
        output_text="ok",
        usage={"input_tokens": 10, "output_tokens": 4},
    )
    assert turn.response_id == "msg_01"
    assert turn.output_text == "ok"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].arguments == {"x": 1}
    assert turn.usage == {"input_tokens": 10, "output_tokens": 4}


def test_translate_response_handles_empty_usage() -> None:
    turn = translate_response(
        message_id=None, blocks=[], output_text="", usage=None
    )
    assert turn.usage is None


# -------------------------------------------------------- translate_tools sanitizer


def test_translate_tools_strips_anthropic_rejected_keywords() -> None:
    tools = [
        {
            "type": "function",
            "name": "grep",
            "description": "search files",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {"type": ["integer", "null"], "minimum": 1},
                },
                "required": ["max_results"],
            },
        },
    ]

    translated = translate_tools(tools)

    schema = translated[0]["input_schema"]
    assert schema["properties"]["max_results"]["type"] == "integer"
    assert "minimum" not in schema["properties"]["max_results"]


def test_translate_tools_sanitizes_passthrough_anthropic_native() -> None:
    tools = [
        {
            "name": "grep",
            "description": "search files",
            "input_schema": {
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "minimum": 1},
                },
            },
        },
    ]

    translated = translate_tools(tools)

    schema = translated[0]["input_schema"]
    assert "minimum" not in schema["properties"]["max_results"]
