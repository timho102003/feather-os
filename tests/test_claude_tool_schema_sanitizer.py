"""Tests for the Anthropic tool-schema sanitizer.

The sanitizer strips JSON-Schema keywords Anthropic's tool validator
rejects (``minimum``/``maximum``/``exclusiveMinimum``/``exclusiveMaximum``)
and normalises ``"type": ["integer", "null"]`` unions, recursively.
"""

from __future__ import annotations

from feather.providers.claude_translator import sanitize_anthropic_tool_schema


def test_sanitizer_strips_minimum_at_top_level() -> None:
    schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "required": ["limit"],
    }

    cleaned = sanitize_anthropic_tool_schema(schema)

    assert "minimum" not in cleaned["properties"]["limit"]
    assert "maximum" not in cleaned["properties"]["limit"]
    assert cleaned["properties"]["limit"]["type"] == "integer"
    assert cleaned["required"] == ["limit"]
