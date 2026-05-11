"""Tests for the Anthropic tool-schema sanitizer.

The sanitizer strips JSON-Schema keywords Anthropic's tool validator
rejects (``minimum``/``maximum``/``exclusiveMinimum``/``exclusiveMaximum``)
and normalises ``"type": ["integer", "null"]`` unions, recursively.
"""

from __future__ import annotations

import copy

from feather.providers.claude_translator import sanitize_anthropic_tool_schema


def test_sanitizer_strips_minimum_at_top_level() -> None:
    schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "required": ["limit"],
    }
    snapshot = copy.deepcopy(schema)

    cleaned = sanitize_anthropic_tool_schema(schema)

    assert "minimum" not in cleaned["properties"]["limit"]
    assert "maximum" not in cleaned["properties"]["limit"]
    assert cleaned["properties"]["limit"]["type"] == "integer"
    assert cleaned["required"] == ["limit"]
    # Non-mutation: input dict unchanged.
    assert schema == snapshot
    # Identity: result is a distinct object.
    assert cleaned is not schema


def test_sanitizer_does_not_recurse_into_default_values() -> None:
    """JSON Schema treats `default` contents as instance data, not a schema."""

    schema = {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "default": {"minimum": 0, "maximum": 100},
            },
        },
    }

    cleaned = sanitize_anthropic_tool_schema(schema)

    assert cleaned["properties"]["config"]["default"] == {"minimum": 0, "maximum": 100}


def test_sanitizer_does_not_recurse_into_const_examples_enum() -> None:
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "object", "const": {"minimum": 1}},
            "b": {"type": "object", "examples": [{"minimum": 2}]},
            "c": {"type": "string", "enum": ["minimum", "maximum"]},
        },
    }

    cleaned = sanitize_anthropic_tool_schema(schema)

    assert cleaned["properties"]["a"]["const"] == {"minimum": 1}
    assert cleaned["properties"]["b"]["examples"] == [{"minimum": 2}]
    assert cleaned["properties"]["c"]["enum"] == ["minimum", "maximum"]


def test_sanitizer_strips_minimum_inside_items() -> None:
    schema = {
        "type": "object",
        "properties": {
            "pages": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
            }
        },
    }

    cleaned = sanitize_anthropic_tool_schema(schema)

    assert "minimum" not in cleaned["properties"]["pages"]["items"]


def test_sanitizer_strips_minimum_inside_anyof() -> None:
    schema = {
        "type": "object",
        "properties": {
            "n": {
                "anyOf": [
                    {"type": "integer", "minimum": 1, "maximum": 10},
                    {"type": "null"},
                ]
            }
        },
    }

    cleaned = sanitize_anthropic_tool_schema(schema)

    int_branch = cleaned["properties"]["n"]["anyOf"][0]
    assert "minimum" not in int_branch
    assert "maximum" not in int_branch


def test_sanitizer_normalises_integer_null_union() -> None:
    schema = {
        "type": "object",
        "properties": {
            "limit": {"type": ["integer", "null"], "minimum": 1},
        },
    }

    cleaned = sanitize_anthropic_tool_schema(schema)

    assert cleaned["properties"]["limit"]["type"] == "integer"
    assert "minimum" not in cleaned["properties"]["limit"]


def test_sanitizer_warns_on_multi_type_non_null_union(caplog) -> None:
    schema = {
        "type": "object",
        "properties": {
            "weird": {"type": ["integer", "string"]},
        },
    }

    with caplog.at_level("WARNING"):
        cleaned = sanitize_anthropic_tool_schema(schema)

    assert cleaned["properties"]["weird"]["type"] == ["integer", "string"]
    assert any("multi-type union" in r.message for r in caplog.records)


def test_sanitizer_handles_exclusive_minimum_maximum() -> None:
    schema = {
        "type": "object",
        "properties": {
            "ratio": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
        },
    }

    cleaned = sanitize_anthropic_tool_schema(schema)

    assert "exclusiveMinimum" not in cleaned["properties"]["ratio"]
    assert "exclusiveMaximum" not in cleaned["properties"]["ratio"]


def test_sanitizer_is_idempotent() -> None:
    schema = {
        "type": "object",
        "properties": {
            "limit": {"type": ["integer", "null"], "minimum": 1},
            "pages": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        },
    }

    once = sanitize_anthropic_tool_schema(schema)
    twice = sanitize_anthropic_tool_schema(once)

    assert once == twice


def test_sanitizer_does_not_mutate_input() -> None:
    schema = {
        "type": "object",
        "properties": {"limit": {"type": "integer", "minimum": 1}},
    }
    snapshot = copy.deepcopy(schema)

    sanitize_anthropic_tool_schema(schema)

    assert schema == snapshot
