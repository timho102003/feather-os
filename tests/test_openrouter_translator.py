"""Unit tests for openrouter_translator."""

from __future__ import annotations

from feather.models import (
    OpenRouterConfig,
    ProviderRequestConfig,
    ReasoningConfig,
)
from feather.providers.openrouter_translator import (
    reconstruct_tool_calls,
    translate_input_items,
    translate_request,
    translate_response,
    translate_tools,
)


# --------------------------------------------------------------------- tools


def test_translate_tools_rewraps_responses_shape_to_chat_shape() -> None:
    responses_tools = [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]
    chat_tools = translate_tools(responses_tools)
    assert chat_tools == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents.",
                "parameters": responses_tools[0]["parameters"],
                "strict": True,
            },
        }
    ]


def test_translate_tools_preserves_openrouter_proxy_tool_shape() -> None:
    """MCP proxy tools are normal function tools once they reach OpenRouter."""

    translated = translate_tools(
        [
            {
                "type": "function",
                "name": "mcp_docs",
                "description": "Call MCP server docs.",
                "parameters": {
                    "type": "object",
                    "properties": {"action": {"type": "string"}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]
    )

    assert translated[0]["function"]["name"] == "mcp_docs"
    assert translated[0]["function"]["strict"] is True


def test_translate_tools_passes_through_already_chat_shape() -> None:
    chat_style = [
        {
            "type": "function",
            "function": {"name": "x", "parameters": {"type": "object"}},
        }
    ]
    assert translate_tools(chat_style) == chat_style


def test_translate_tools_handles_empty_list() -> None:
    assert translate_tools([]) == []


def test_translate_tools_omits_strict_when_false_or_missing() -> None:
    result = translate_tools(
        [
            {
                "type": "function",
                "name": "no_strict",
                "description": "",
                "parameters": {"type": "object"},
            }
        ]
    )
    assert "strict" not in result[0]["function"]


# ------------------------------------------------------------- input_items


def test_translate_input_items_message_with_input_text_becomes_user() -> None:
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]
    assert translate_input_items(items) == [{"role": "user", "content": "hello"}]


def test_translate_input_items_concats_multiple_text_blocks() -> None:
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "part1 "},
                {"type": "input_text", "text": "part2"},
            ],
        }
    ]
    assert translate_input_items(items) == [{"role": "user", "content": "part1 part2"}]


def test_translate_input_items_preserves_image_and_file_blocks() -> None:
    """OpenRouter multimodal messages need content parts, not flattened text."""

    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "inspect these"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,aW1hZ2U=",
                },
                {
                    "type": "input_file",
                    "filename": "report.pdf",
                    "file_data": "data:application/pdf;base64,cGRm",
                },
            ],
        }
    ]

    assert translate_input_items(items) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect these"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                },
                {
                    "type": "file",
                    "file": {
                        "filename": "report.pdf",
                        "file_data": "data:application/pdf;base64,cGRm",
                    },
                },
            ],
        }
    ]


def test_translate_input_items_function_call_output_becomes_tool_message() -> None:
    items = [
        {
            "type": "function_call_output",
            "call_id": "call_01",
            "output": '{"ok":true}',
        }
    ]
    assert translate_input_items(items) == [
        {"role": "tool", "tool_call_id": "call_01", "content": '{"ok":true}'}
    ]


def test_translate_input_items_groups_function_calls_before_outputs() -> None:
    """Stateless Chat Completions replay must preserve the assistant tool turn.

    A bare ``tool`` role message without the immediately preceding assistant
    ``tool_calls`` request is not the OpenAI/OpenRouter tool-calling protocol.
    """

    items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "read two files"}],
        },
        {
            "type": "function_call",
            "call_id": "call_a",
            "name": "read_file",
            "arguments": {"path": "a.txt"},
        },
        {
            "type": "function_call",
            "call_id": "call_b",
            "name": "read_file",
            "arguments": '{"path":"b.txt"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_a",
            "output": "A",
        },
        {
            "type": "function_call_output",
            "call_id": "call_b",
            "output": "B",
        },
    ]

    messages = translate_input_items(items)

    assert messages[0] == {"role": "user", "content": "read two files"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] is None
    tool_calls = messages[1]["tool_calls"]
    assert [call["id"] for call in tool_calls] == ["call_a", "call_b"]
    assert [call["function"]["name"] for call in tool_calls] == [
        "read_file",
        "read_file",
    ]
    assert [call["function"]["arguments"] for call in tool_calls] == [
        '{"path":"a.txt"}',
        '{"path":"b.txt"}',
    ]
    assert messages[2:] == [
        {"role": "tool", "tool_call_id": "call_a", "content": "A"},
        {"role": "tool", "tool_call_id": "call_b", "content": "B"},
    ]


def test_translate_input_items_accepts_flat_role_content_shape() -> None:
    # Memory-subsystem callers pass this shape directly.
    items = [{"role": "user", "content": "plain string"}]
    assert translate_input_items(items) == [{"role": "user", "content": "plain string"}]


def test_translate_input_items_assistant_with_output_text_blocks() -> None:
    items = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer"}],
        }
    ]
    assert translate_input_items(items) == [
        {"role": "assistant", "content": "answer"}
    ]


def test_translate_input_items_skips_unknown_reasoning_items() -> None:
    items = [{"type": "reasoning", "some": "thing"}]
    assert translate_input_items(items) == []


def test_translate_input_items_preserves_ordering_across_types() -> None:
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "q"}],
        },
        {"type": "function_call_output", "call_id": "c1", "output": "r"},
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "q2"}],
        },
    ]
    translated = translate_input_items(items)
    assert [m["role"] for m in translated] == ["user", "tool", "user"]


# ---------------------------------------------------------- translate_request


def _min_cfg(**overrides: object) -> OpenRouterConfig:
    return OpenRouterConfig(
        model=overrides.get("model", "anthropic/claude-sonnet-4.6"),  # type: ignore[arg-type]
        max_output_tokens=overrides.get("max_output_tokens", 32_000),  # type: ignore[arg-type]
        temperature=overrides.get("temperature", 1.0),  # type: ignore[arg-type]
        parallel_tool_calls=overrides.get("parallel_tool_calls", True),  # type: ignore[arg-type]
        reasoning=overrides.get("reasoning"),  # type: ignore[arg-type]
        provider_preferences=overrides.get("provider_preferences"),  # type: ignore[arg-type]
        fallback_models=overrides.get("fallback_models"),  # type: ignore[arg-type]
        cache_strategy=overrides.get("cache_strategy", "anthropic_breakpoint"),  # type: ignore[arg-type]
    )


def test_translate_request_builds_minimal_body() -> None:
    cfg = _min_cfg()
    body = translate_request(
        instructions="sys prompt",
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
        model_limits=None,
    )
    assert body["model"] == "anthropic/claude-sonnet-4.6"
    assert body["stream"] is True
    # parallel_tool_calls rides along with tools[]; not emitted when tool-less.
    assert "parallel_tool_calls" not in body
    assert body["max_tokens"] == 32_000
    # System message: list content with cache_control on the last block
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}


def test_translate_request_cache_off_emits_plain_system_string() -> None:
    cfg = _min_cfg(cache_strategy="off")
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
        model_limits=None,
    )
    assert body["messages"][0] == {"role": "system", "content": "sys"}


def test_translate_request_caps_max_tokens_at_model_limit() -> None:
    cfg = _min_cfg(max_output_tokens=32_000)
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
        model_limits={"top_provider": {"max_completion_tokens": 8000}},
    )
    assert body["max_tokens"] == 8000


def test_translate_request_honours_request_config_model_and_temperature() -> None:
    cfg = _min_cfg()
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(
            model="openai/gpt-5.2-mini", temperature=0.2
        ),
        cfg=cfg,
        model_limits=None,
    )
    assert body["model"] == "openai/gpt-5.2-mini"
    assert body["temperature"] == 0.2


def test_translate_request_applies_provider_preferences_and_fallback_models() -> None:
    cfg = _min_cfg(
        provider_preferences={"require_parameters": True, "sort": "throughput"},
        fallback_models=["openai/gpt-5.2-mini", "google/gemini-3-pro"],
    )
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
        model_limits=None,
    )
    assert body["provider"] == {
        "require_parameters": True,
        "sort": "throughput",
    }
    assert body["models"] == ["openai/gpt-5.2-mini", "google/gemini-3-pro"]


def test_translate_request_applies_reasoning_config() -> None:
    cfg = _min_cfg(reasoning=ReasoningConfig(effort="high"))
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
        model_limits=None,
    )
    assert body["reasoning"] == {"effort": "high"}


def test_translate_request_includes_tools_when_provided() -> None:
    cfg = _min_cfg()
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[
            {
                "type": "function",
                "name": "t",
                "description": "",
                "parameters": {"type": "object"},
                "strict": True,
            }
        ],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
        model_limits=None,
    )
    assert body["tools"][0]["function"]["name"] == "t"
    # parallel_tool_calls now rides with tools presence.
    assert body["parallel_tool_calls"] is True


def test_translate_request_applies_response_schema_as_json_schema() -> None:
    """When request_config carries a Pydantic response_schema, the body must
    emit Chat Completions response_format=json_schema with strict=True and
    a hardened schema (additionalProperties=False, full required list)."""

    from pydantic import BaseModel

    class Example(BaseModel):
        name: str
        count: int

    cfg = _min_cfg()
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(
            response_schema=Example, response_schema_name="ExampleOut"
        ),
        cfg=cfg,
        model_limits=None,
    )
    rf = body.get("response_format")
    assert rf is not None and rf["type"] == "json_schema"
    js = rf["json_schema"]
    assert js["name"] == "ExampleOut"
    assert js["strict"] is True
    schema = js["schema"]
    assert schema.get("additionalProperties") is False
    assert set(schema.get("required", [])) == {"name", "count"}


def test_translate_request_omits_parallel_tool_calls_when_no_tools() -> None:
    """Some upstream providers reject `parallel_tool_calls` when there are no
    tools. Omitting the knob unless tools are present plays nicer with
    ``require_parameters: true`` routing."""

    cfg = _min_cfg()
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
        model_limits=None,
    )
    assert "parallel_tool_calls" not in body


def test_translate_request_max_tokens_cap_ignores_context_length_alone() -> None:
    """Falling back to `context_length` as a max-tokens cap is wrong —
    context_length is input+output. When only context_length is known we
    must leave the caller's desired_max alone."""

    cfg = _min_cfg(max_output_tokens=32000)
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(),
        cfg=cfg,
        model_limits={"context_length": 200000},  # no top_provider.max_completion_tokens
    )
    assert body["max_tokens"] == 32000


# ------------------------------------------------------ reconstruct_tool_calls


def test_reconstruct_tool_calls_single_across_two_deltas() -> None:
    deltas = [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"p'},
                }
            ]
        },
        {"tool_calls": [{"index": 0, "function": {"arguments": 'ath":"x"}'}}]},
    ]
    calls = reconstruct_tool_calls(deltas)
    assert len(calls) == 1
    assert calls[0].call_id == "call_1"
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "x"}


def test_reconstruct_tool_calls_parallel_interleaved() -> None:
    deltas = [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"a"}',
                    },
                },
                {
                    "index": 1,
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"pat'},
                },
            ]
        },
        {"tool_calls": [{"index": 1, "function": {"arguments": 'h":"b"}'}}]},
    ]
    calls = reconstruct_tool_calls(deltas)
    assert [c.call_id for c in calls] == ["call_1", "call_2"]
    assert [c.arguments for c in calls] == [{"path": "a"}, {"path": "b"}]


def test_reconstruct_tool_calls_malformed_arguments_preserve_raw() -> None:
    deltas = [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c",
                    "type": "function",
                    "function": {"name": "n", "arguments": "not-json"},
                }
            ]
        }
    ]
    calls = reconstruct_tool_calls(deltas)
    assert calls[0].arguments == {"raw_arguments": "not-json"}


def test_reconstruct_tool_calls_synthesizes_id_when_upstream_omits() -> None:
    """Some providers (MoonshotAI/GLM variants) occasionally omit the call id.
    We must never emit a ToolCall with ``call_id=""``; downstream matching
    against ``function_call_output.call_id`` would fail."""

    deltas = [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"x"}'},
                }
            ]
        }
    ]
    calls = reconstruct_tool_calls(deltas)
    assert len(calls) == 1
    assert calls[0].call_id, "expected a synthesized call_id, got empty string"
    # Synthesized ids should be stable-looking so logs/correlation work.
    assert calls[0].call_id.startswith("call_")


def test_reconstruct_tool_calls_empty_arguments_defaults_to_empty_dict() -> None:
    deltas = [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c",
                    "type": "function",
                    "function": {"name": "n"},
                }
            ]
        }
    ]
    calls = reconstruct_tool_calls(deltas)
    assert calls[0].arguments == {}


# --------------------------------------------------------- translate_response


def test_translate_response_builds_model_turn() -> None:
    final = {
        "id": "gen-123",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    turn = translate_response(final_chunk=final, output_text="hello", deltas=[])
    assert turn.response_id == "gen-123"
    assert turn.output_text == "hello"
    assert turn.tool_calls == []
    assert turn.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "input_tokens": 10,
        "output_tokens": 20,
    }


def test_translate_response_normalizes_cache_usage_for_compaction() -> None:
    """OpenRouter reports Chat-style usage; Feather also needs Responses-style keys."""

    turn = translate_response(
        final_chunk={
            "id": "gen-cache",
            "choices": [],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 1024},
            },
        },
        output_text="",
        deltas=[],
    )

    assert turn.usage is not None
    assert turn.usage["input_tokens"] == 1200
    assert turn.usage["output_tokens"] == 50
    assert turn.usage["prompt_tokens_details"]["cached_tokens"] == 1024


def test_translate_response_with_tool_calls() -> None:
    deltas = [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "n", "arguments": '{"x":1}'},
                }
            ]
        }
    ]
    final = {
        "id": "gen-1",
        "choices": [{"finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    turn = translate_response(final_chunk=final, output_text="", deltas=deltas)
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "n"
    assert turn.tool_calls[0].arguments == {"x": 1}
