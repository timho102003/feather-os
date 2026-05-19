"""Unit tests for openrouter_translator."""

from __future__ import annotations

from feather.models import (
    OpenRouterConfig,
    OpenRouterTracingConfig,
    ProviderRequestConfig,
    ReasoningConfig,
    TraceContext,
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


def test_translate_tools_flattens_nullable_union_to_single_type() -> None:
    """Several first-party tools declare optional params as
    ``"type": ["string", "null"]`` — canonical JSON Schema accepted by
    OpenAI/Anthropic/native DeepSeek, but rejected by Alibaba/DashScope's
    strict validator with ``InternalError.Algo.InvalidParameter: The tool
    parameter type must be a string`` (observed in user runtime when
    OpenRouter routed deepseek-v4-pro through Alibaba). Flatten the union
    by dropping the ``"null"`` member — optionality is already expressed
    via the parameter being absent from ``required``.
    """

    translated = translate_tools(
        [
            {
                "type": "function",
                "name": "grep",
                "description": "Search files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {
                            "type": ["string", "null"],
                            "description": "Optional base path.",
                        },
                        "ignore_case": {"type": ["boolean", "null"]},
                        "max_results": {"type": ["integer", "null"]},
                    },
                    "required": ["pattern"],
                },
            }
        ]
    )

    props = translated[0]["function"]["parameters"]["properties"]
    assert props["pattern"]["type"] == "string"  # untouched single type
    assert props["path"]["type"] == "string"
    assert props["path"]["description"] == "Optional base path."  # description preserved
    assert props["ignore_case"]["type"] == "boolean"
    assert props["max_results"]["type"] == "integer"


def test_translate_tools_flattens_nested_object_property_types() -> None:
    """Nested ``properties`` (object inside object) must also be flattened."""

    translated = translate_tools(
        [
            {
                "type": "function",
                "name": "task_create",
                "description": "Create a task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "owner": {"type": ["string", "null"]},
                                "priority": {"type": ["integer", "null"]},
                            },
                        },
                    },
                },
            }
        ]
    )

    nested = translated[0]["function"]["parameters"]["properties"]["metadata"][
        "properties"
    ]
    assert nested["owner"]["type"] == "string"
    assert nested["priority"]["type"] == "integer"


def test_translate_tools_flattens_array_items_with_nullable_union() -> None:
    """``items`` of an array property is itself a schema and must be flattened."""

    translated = translate_tools(
        [
            {
                "type": "function",
                "name": "task_update",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": ["array", "null"],
                            "items": {"type": ["string", "null"]},
                        },
                    },
                },
            }
        ]
    )

    tags = translated[0]["function"]["parameters"]["properties"]["tags"]
    assert tags["type"] == "array"
    assert tags["items"]["type"] == "string"


def test_translate_tools_does_not_mutate_input_schema() -> None:
    """Flattening must operate on a copy — the caller's tool dict is shared
    state (the BaseAgent rebuilds tools once per session) and silent mutation
    would leak the OpenRouter-only sanitization into the OpenAI Responses
    provider's request body."""

    original = {
        "type": "function",
        "name": "x",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {"p": {"type": ["string", "null"]}},
        },
    }
    before = original["parameters"]["properties"]["p"]["type"]
    translate_tools([original])
    after = original["parameters"]["properties"]["p"]["type"]
    assert before == after == ["string", "null"]


def test_translate_tools_preserves_single_string_type_unchanged() -> None:
    """Tools that already declare a single-string type must pass through
    bit-for-bit (regression guard for the flattening helper)."""

    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    translated = translate_tools(
        [
            {
                "type": "function",
                "name": "read_file",
                "description": "",
                "parameters": schema,
            }
        ]
    )
    assert translated[0]["function"]["parameters"] == schema


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
        tracing=overrides.get("tracing"),  # type: ignore[arg-type]
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


# ----------------------------------------------------------- tracing metadata


def _trace_ctx(**overrides: object) -> TraceContext:
    return TraceContext(
        session_id=str(overrides.get("session_id", "sess-uuid-001")),
        agent_name=str(overrides.get("agent_name", "lead")),
        agent_role=overrides.get("agent_role", "primary lead agent"),  # type: ignore[arg-type]
    )


def test_translate_request_omits_tracing_when_config_absent() -> None:
    """No tracing block ⇒ wire body must be byte-identical to today's behaviour."""

    cfg = _min_cfg()  # tracing field defaults to None
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=_trace_ctx()),
        cfg=cfg,
        model_limits=None,
    )
    assert "user" not in body
    assert "session_id" not in body
    assert "trace" not in body


def test_translate_request_omits_tracing_when_disabled() -> None:
    """Tracing block present but disabled ⇒ no fields emitted."""

    cfg = _min_cfg(tracing=OpenRouterTracingConfig(enabled=False, user="alice"))
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=_trace_ctx()),
        cfg=cfg,
        model_limits=None,
    )
    assert "user" not in body
    assert "session_id" not in body
    assert "trace" not in body


def test_translate_request_emits_session_and_trace_when_enabled() -> None:
    cfg = _min_cfg(tracing=OpenRouterTracingConfig(enabled=True))
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(
            trace_context=_trace_ctx(
                session_id="sess-42",
                agent_name="lead",
                agent_role="primary lead agent",
            )
        ),
        cfg=cfg,
        model_limits=None,
    )
    assert body["session_id"] == "sess-42"
    trace = body["trace"]
    assert trace["trace_name"] == "feather/lead"
    assert trace["generation_name"] == "anthropic/claude-sonnet-4.6"
    assert trace["feather_app"] == "feather-agent-os"
    assert trace["feather_agent_name"] == "lead"
    assert trace["feather_agent_role"] == "primary lead agent"
    assert trace["feather_session_id"] == "sess-42"
    # ``user`` is opt-in; absent unless configured.
    assert "user" not in body


def test_translate_request_includes_user_when_configured() -> None:
    cfg = _min_cfg(
        tracing=OpenRouterTracingConfig(enabled=True, user="ops@example.com")
    )
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=_trace_ctx()),
        cfg=cfg,
        model_limits=None,
    )
    assert body["user"] == "ops@example.com"


def test_translate_request_merges_static_metadata_into_trace() -> None:
    cfg = _min_cfg(
        tracing=OpenRouterTracingConfig(
            enabled=True,
            metadata={"deployment": "prod", "build_sha": "abc123"},
        )
    )
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=_trace_ctx()),
        cfg=cfg,
        model_limits=None,
    )
    assert body["trace"]["deployment"] == "prod"
    assert body["trace"]["build_sha"] == "abc123"
    # Reserved feather_* keys still present alongside operator extras.
    assert body["trace"]["feather_agent_name"] == "lead"


def test_translate_request_static_metadata_does_not_overwrite_reserved_keys() -> None:
    """Operator metadata must not silently shadow the per-turn identity bundle."""

    cfg = _min_cfg(
        tracing=OpenRouterTracingConfig(
            enabled=True,
            metadata={
                "feather_agent_name": "spoofed",
                "trace_name": "spoofed-trace",
                "deployment": "prod",
            },
        )
    )
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=_trace_ctx()),
        cfg=cfg,
        model_limits=None,
    )
    assert body["trace"]["feather_agent_name"] == "lead"
    assert body["trace"]["trace_name"] == "feather/lead"
    assert body["trace"]["deployment"] == "prod"


def test_translate_request_clamps_oversized_metadata_value() -> None:
    cfg = _min_cfg(
        tracing=OpenRouterTracingConfig(
            enabled=True,
            metadata={"big": "x" * 1000},
        )
    )
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=_trace_ctx()),
        cfg=cfg,
        model_limits=None,
    )
    assert len(body["trace"]["big"]) == 512


def test_translate_request_drops_excess_metadata_keys_beyond_limit() -> None:
    cfg = _min_cfg(
        tracing=OpenRouterTracingConfig(
            enabled=True,
            metadata={f"k{i}": str(i) for i in range(40)},
        )
    )
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=_trace_ctx()),
        cfg=cfg,
        model_limits=None,
    )
    # OpenRouter limit: 16 total kv pairs. Reserved feather_* keys must
    # always survive; operator extras get truncated to fit.
    trace = body["trace"]
    assert len(trace) <= 16
    # Reserved keys present.
    for reserved in (
        "trace_name",
        "generation_name",
        "feather_app",
        "feather_agent_name",
        "feather_session_id",
    ):
        assert reserved in trace


def test_translate_request_clamps_user_to_128_chars() -> None:
    cfg = _min_cfg(
        tracing=OpenRouterTracingConfig(enabled=True, user="u" * 500)
    )
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=_trace_ctx()),
        cfg=cfg,
        model_limits=None,
    )
    assert len(body["user"]) == 128


def test_translate_request_skips_tracing_when_no_trace_context_supplied() -> None:
    """Tracing enabled at config level but no per-turn context ⇒ no field emission.

    Avoids accidentally sending the ``user`` field in isolation, which
    would still hit Opik but with no session/agent grouping — confusing
    rather than helpful.
    """

    cfg = _min_cfg(tracing=OpenRouterTracingConfig(enabled=True, user="alice"))
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=None),
        cfg=cfg,
        model_limits=None,
    )
    assert "user" not in body
    assert "session_id" not in body
    assert "trace" not in body


def test_translate_request_uses_request_config_model_in_generation_name() -> None:
    """Per-call model override should reflect in trace.generation_name."""

    cfg = _min_cfg(tracing=OpenRouterTracingConfig(enabled=True))
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(
            model="openai/gpt-5.2-mini",
            trace_context=_trace_ctx(),
        ),
        cfg=cfg,
        model_limits=None,
    )
    assert body["trace"]["generation_name"] == "openai/gpt-5.2-mini"


def test_translate_request_clamps_oversize_reserved_trace_values() -> None:
    """Reserved trace values must respect OpenRouter's 512-char per-value cap.

    A long ``--session-id`` or a hostile model slug should never propagate
    through to the wire body and trigger a 400.
    """

    cfg = _min_cfg(
        model="x" * 700,  # absurd model slug → generation_name must clamp
        tracing=OpenRouterTracingConfig(enabled=True),
    )
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(
            trace_context=TraceContext(
                session_id="s" * 700,
                agent_name="a" * 700,
                agent_role="r" * 700,
            )
        ),
        cfg=cfg,
        model_limits=None,
    )
    trace = body["trace"]
    assert len(trace["trace_name"]) == 512
    assert len(trace["generation_name"]) == 512
    assert len(trace["feather_agent_name"]) == 512
    assert len(trace["feather_session_id"]) == 512
    assert len(trace["feather_agent_role"]) == 512
    # session_id top-level field uses the more generous 256-char cap.
    assert len(body["session_id"]) == 256


def test_translate_request_tracing_never_raises_on_unexpected_metadata_shape() -> None:
    """Defense-in-depth: a contrived metadata structure must not crash the loop.

    Today the coercer drops everything weird, but a future contributor
    might add a code path that assumes shape. The outer try/except is
    the last line of defense — verify it actually catches.
    """

    class _Bomb:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    cfg = _min_cfg(
        tracing=OpenRouterTracingConfig(
            enabled=True,
            # Mixed safe + dangerous values; a future change might call
            # str() on whatever passes the primitive whitelist.
            metadata={"deployment": "prod", "danger": _Bomb()},
        )
    )
    # Should not raise. Trace fields may be present (with safe entries)
    # or stripped entirely if the exception path fired — both are fine.
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=_trace_ctx()),
        cfg=cfg,
        model_limits=None,
    )
    # Body must remain well-formed.
    assert body["model"] == "anthropic/claude-sonnet-4.6"
    assert body["messages"][0]["role"] == "system"


def test_translate_request_tracing_strips_partial_fields_on_exception(
    monkeypatch,
) -> None:
    """If the tracing block raises, no half-populated fields leak to the wire."""

    from feather.providers import openrouter_translator as t

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic boom")

    monkeypatch.setattr(t, "_clamp_trace_value", _boom)

    cfg = _min_cfg(tracing=OpenRouterTracingConfig(enabled=True, user="a"))
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=_trace_ctx()),
        cfg=cfg,
        model_limits=None,
    )
    assert "session_id" not in body
    assert "user" not in body
    assert "trace" not in body


def test_translate_request_normalises_reserved_key_collision_case_insensitively() -> None:  # noqa: E501
    """``Feather_App`` (operator) must not bypass the ``feather_app`` reserved filter."""

    cfg = _min_cfg(
        tracing=OpenRouterTracingConfig(
            enabled=True,
            metadata={"Feather_App": "spoof", "deployment": "prod"},
        )
    )
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=_trace_ctx()),
        cfg=cfg,
        model_limits=None,
    )
    trace = body["trace"]
    assert trace["feather_app"] == "feather-agent-os"
    assert "Feather_App" not in trace
    assert trace["deployment"] == "prod"


def test_translate_request_tolerates_non_string_metadata_values() -> None:
    """Numbers and bools are allowed; nested objects/arrays must be dropped.

    OpenRouter's metadata-value schema is string-typed; sending a dict
    triggers a 400. Coerce primitives, drop the rest.
    """

    cfg = _min_cfg(
        tracing=OpenRouterTracingConfig(
            enabled=True,
            metadata={
                "version": 7,
                "active": True,
                "nested": {"bad": 1},
                "arr": [1, 2],
            },
        )
    )
    body = translate_request(
        instructions="sys",
        input_items=[],
        tools=[],
        request_config=ProviderRequestConfig(trace_context=_trace_ctx()),
        cfg=cfg,
        model_limits=None,
    )
    trace = body["trace"]
    assert trace["version"] == "7"
    assert trace["active"] == "true"
    assert "nested" not in trace
    assert "arr" not in trace


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
