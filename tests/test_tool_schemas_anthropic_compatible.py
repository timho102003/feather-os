"""Regression: every shipped tool's schema is Anthropic-compatible
after the sanitizer pass.

If a future tool author introduces a JSON-Schema keyword Anthropic
rejects, this test fails with the offending tool name + path so the
fix is to either update the sanitizer or change the tool's schema.

Uses class-attribute access (``parameters_schema`` is declared at
class level on every tool subclass of :class:`feather.tools.base.BaseTool`)
so no constructor dependency injection is required.
"""

from __future__ import annotations

from typing import Any

import pytest

from feather.providers.claude_translator import (
    sanitize_anthropic_tool_schema,
    translate_tools,
)
from feather.tools.ask_user_tool import AskUserTool
from feather.tools.bash_tool import BashTool
from feather.tools.cron_tools import (
    CreateCronTool,
    DeleteCronTool,
    ListCronsTool,
    UpdateCronTool,
)
from feather.tools.grep_tool import GrepTool
from feather.tools.manage_memory_tool import ManageMemoryTool
from feather.tools.parallel_search_tool import ParallelSearchTool
from feather.tools.pdf_tool import ReadPdfTool
from feather.tools.read_file_tool import ReadFileTool
from feather.tools.recall_memory_tool import RecallMemoryTool
from feather.tools.skill_tool import LoadSkillTool
from feather.tools.write_file_tool import WriteFileTool

_REJECTED_KEYS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}


def _walk(node: Any) -> list[tuple[str, Any]]:
    """Yield (key, value) pairs for every dict node, recursively."""

    pairs: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            pairs.append((key, value))
            pairs.extend(_walk(value))
    elif isinstance(node, list):
        for item in node:
            pairs.extend(_walk(item))
    return pairs


def _tool_param_schemas() -> list[tuple[str, dict[str, Any]]]:
    """Pairs of (tool name, raw parameters_schema) for every shipped tool."""

    tools: list[type] = [
        AskUserTool,
        BashTool,
        GrepTool,
        ReadFileTool,
        WriteFileTool,
        ReadPdfTool,
        LoadSkillTool,
        ParallelSearchTool,
        RecallMemoryTool,
        ManageMemoryTool,
        CreateCronTool,
        UpdateCronTool,
        ListCronsTool,
        DeleteCronTool,
    ]
    out: list[tuple[str, dict[str, Any]]] = []
    for cls in tools:
        name = getattr(cls, "name", cls.__name__)
        schema = cls.parameters_schema  # class attribute; no instantiation needed
        out.append((name, schema))
    return out


@pytest.mark.parametrize("name,raw", _tool_param_schemas())
def test_tool_schema_post_sanitize_has_no_rejected_keywords(
    name: str, raw: dict[str, Any]
) -> None:
    cleaned = sanitize_anthropic_tool_schema(raw)

    for key, _ in _walk(cleaned):
        assert key not in _REJECTED_KEYS, (
            f"tool={name!r} schema still contains rejected keyword {key!r} after sanitize"
        )


@pytest.mark.parametrize("name,raw", _tool_param_schemas())
def test_tool_schema_post_sanitize_no_null_in_type_union(
    name: str, raw: dict[str, Any]
) -> None:
    cleaned = sanitize_anthropic_tool_schema(raw)

    for key, value in _walk(cleaned):
        if key == "type" and isinstance(value, list):
            assert "null" not in value, (
                f"tool={name!r} schema still has 'null' in type union after sanitize"
            )


def test_translate_tools_end_to_end_on_real_tool_set() -> None:
    """Final round-trip: every shipped tool, post-translate_tools, is
    clean. Catches the case where a future translator change skips the
    sanitizer for some code path."""

    inputs = [
        {
            "type": "function",
            "name": name,
            "description": "",
            "parameters": schema,
        }
        for name, schema in _tool_param_schemas()
    ]

    translated = translate_tools(inputs)

    for entry in translated:
        for key, value in _walk(entry["input_schema"]):
            assert key not in _REJECTED_KEYS, (
                f"tool={entry['name']!r} post-translate still has {key!r}"
            )
            if key == "type" and isinstance(value, list):
                assert "null" not in value
