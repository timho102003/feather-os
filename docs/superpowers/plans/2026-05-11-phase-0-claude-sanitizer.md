# Phase 0 — Claude Tool-Schema Sanitizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unblock the Claude (Anthropic Messages API) provider by stripping JSON-Schema keywords Anthropic's tool validator rejects (`minimum`/`maximum`/etc.) and normalising `["integer", "null"]` type unions, applied in the Anthropic translator immediately before the wire body is built.

**Architecture:** One pure-Python helper added to `src/feather/providers/claude_translator.py`. Called from `translate_tools()` on every tool, regardless of whether the input is in flat-Responses or Anthropic-native shape. Recursive, idempotent, non-mutating.

**Tech Stack:** Python 3.12+, pytest (auto-mode async), respx for the integration test.

**Worktree:** This plan executes inside `/home/dev/feather_v2/.worktrees/config-tui` on branch `feature/config-tui`. The design spec lives at `docs/superpowers/specs/2026-05-11-config-tui-design.md`.

**Workflow reminder:** Each task follows Explore → Plan → Implement → /simplify → test (happy + test-to-fail) → red-team review. Phase 0 is small enough that simplify + red-team are the last two tasks rather than woven through every implementation task.

---

### Task 1: Failing test — strip `minimum` recursively

**Files:**
- Create: `tests/test_claude_tool_schema_sanitizer.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_claude_tool_schema_sanitizer.py::test_sanitizer_strips_minimum_at_top_level -v`
Expected: FAIL with `ImportError: cannot import name 'sanitize_anthropic_tool_schema'`

- [ ] **Step 3: Implement the minimal sanitizer**

Modify: `src/feather/providers/claude_translator.py` — add after the existing imports and before the `_CACHE_STRATEGY_BREAKPOINT` constant.

```python
import copy
import logging

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_claude_tool_schema_sanitizer.py::test_sanitizer_strips_minimum_at_top_level -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_claude_tool_schema_sanitizer.py src/feather/providers/claude_translator.py
git commit -m "Add Anthropic tool-schema sanitizer with top-level minimum/maximum strip"
```

---

### Task 2: Test-to-fail cases — recursion into nested shapes

**Files:**
- Modify: `tests/test_claude_tool_schema_sanitizer.py`

- [ ] **Step 1: Append nested-shape failing tests**

```python
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
```

- [ ] **Step 2: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_claude_tool_schema_sanitizer.py -v`
Expected: 5 passed (1 from Task 1 + 4 new)

- [ ] **Step 3: Commit**

```bash
git add tests/test_claude_tool_schema_sanitizer.py
git commit -m "Cover nested anyOf/items/exclusive-bounds and multi-type warnings"
```

---

### Task 3: Idempotence and non-mutation guarantees

**Files:**
- Modify: `tests/test_claude_tool_schema_sanitizer.py`

- [ ] **Step 1: Append guarantees tests**

```python
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
```

Add the import at the top of the test file:

```python
import copy
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_claude_tool_schema_sanitizer.py -v`
Expected: 7 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_claude_tool_schema_sanitizer.py
git commit -m "Guarantee sanitizer idempotence and non-mutation of input"
```

---

### Task 4: Wire sanitizer into `translate_tools`

**Files:**
- Modify: `src/feather/providers/claude_translator.py:77-110`
- Modify: `tests/test_claude_translator.py`

- [ ] **Step 1: Write failing test that exercises the wired-up path**

Append to `tests/test_claude_translator.py`:

```python
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
```

Add to the imports at the top of `tests/test_claude_translator.py` (if not already present):

```python
from feather.providers.claude_translator import translate_tools
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_claude_translator.py::test_translate_tools_strips_anthropic_rejected_keywords -v`
Expected: FAIL — `minimum` still present.

- [ ] **Step 3: Wire the sanitizer into `translate_tools`**

Modify `src/feather/providers/claude_translator.py` — replace the existing body of `translate_tools` (the loop at lines 95-110):

```python
    out: list[dict[str, Any]] = []
    for tool in tools:
        if "input_schema" in tool:
            translated = dict(tool)
            translated["input_schema"] = sanitize_anthropic_tool_schema(
                tool["input_schema"]
            )
            out.append(translated)
            continue
        parameters = tool.get("parameters", {"type": "object", "properties": {}})
        translated = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": sanitize_anthropic_tool_schema(parameters),
        }
        if tool.get("strict"):
            translated["strict"] = True
        out.append(translated)
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_claude_translator.py -v`
Expected: all existing tests + 2 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_claude_translator.py src/feather/providers/claude_translator.py
git commit -m "Run Anthropic sanitizer over every tool in translate_tools"
```

---

### Task 5: Sweep — every shipped tool's post-sanitize schema is Anthropic-clean

**Files:**
- Create: `tests/test_tool_schemas_anthropic_compatible.py`

- [ ] **Step 1: Write the sweep test**

```python
"""Regression: every shipped tool's schema is Anthropic-compatible
after the sanitizer pass.

If a future tool author introduces a JSON-Schema keyword Anthropic
rejects, this test fails with the offending tool name + path so the
fix is to either update the sanitizer or change the tool's schema.
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
    samples = [
        AskUserTool(),
        BashTool(),
        GrepTool(),
        ReadFileTool(),
        WriteFileTool(),
        ReadPdfTool(),
        LoadSkillTool(),
        ParallelSearchTool(client=None),  # type: ignore[arg-type]
        RecallMemoryTool(service=None),  # type: ignore[arg-type]
        ManageMemoryTool(service=None),  # type: ignore[arg-type]
        CreateCronTool(store=None, scheduler=None),  # type: ignore[arg-type]
        UpdateCronTool(store=None, scheduler=None),  # type: ignore[arg-type]
        ListCronsTool(store=None),  # type: ignore[arg-type]
        DeleteCronTool(store=None, scheduler=None),  # type: ignore[arg-type]
    ]
    return [(t.name, t.parameters_schema) for t in samples]


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
```

- [ ] **Step 2: Run the sweep**

Run: `uv run pytest tests/test_tool_schemas_anthropic_compatible.py -v`
Expected: all parametrized cases + the end-to-end test pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tool_schemas_anthropic_compatible.py
git commit -m "Sweep every shipped tool's schema through the Anthropic sanitizer"
```

---

### Task 6: Integration — claude_provider request body has no rejected keywords

**Files:**
- Modify: `tests/test_claude_provider.py`

- [ ] **Step 1: Find the existing claude_provider test that posts a request body**

Run: `grep -n "respx\|tools.*custom\|mock_router\|api.anthropic.com" tests/test_claude_provider.py | head`

Identify the most appropriate existing fixture/helper that builds a ClaudeMessagesProvider, stubs the Anthropic endpoint via respx, and inspects the outbound JSON. If none exists, add the smallest possible one.

- [ ] **Step 2: Append an integration test that asserts wire-body cleanliness**

Append to `tests/test_claude_provider.py`:

```python
async def test_claude_provider_strips_rejected_keywords_in_wire_body(
    respx_mock,
) -> None:
    """Regression for 'tools.0.custom: For 'integer' type, property
    'minimum' is not supported'.

    Build a provider, post a tool list that contains the exact shape
    the shipped tools use (``"type": ["integer", "null"]`` plus
    ``minimum``), and assert the JSON body sent to Anthropic has had
    those scrubbed by the time it leaves the translator.
    """

    import json
    import httpx

    from feather.models import ClaudeConfig
    from feather.providers.claude_provider import ClaudeMessagesProvider

    captured: dict[str, Any] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b"event: message_start\n"
                b'data: {"type":"message_start","message":{"id":"m","role":"assistant","content":[],"model":"claude-opus-4-7","stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
                b"event: message_stop\n"
                b'data: {"type":"message_stop"}\n\n'
            ),
        )

    respx_mock.post("https://api.anthropic.com/v1/messages").mock(side_effect=_capture)

    config = ClaudeConfig(model="claude-opus-4-7")
    provider = ClaudeMessagesProvider(config=config, api_key="sk-test")

    tools = [
        {
            "type": "function",
            "name": "grep",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {"type": ["integer", "null"], "minimum": 1},
                },
            },
        }
    ]

    await provider.complete(
        system_prompt="hi",
        input_items=[{"type": "message", "role": "user", "content": "hello"}],
        tools=tools,
        config=None,
    )

    wire_tools = captured["json"]["tools"]
    assert wire_tools[0]["input_schema"]["properties"]["max_results"]["type"] == "integer"
    assert "minimum" not in wire_tools[0]["input_schema"]["properties"]["max_results"]
```

Adjust the `ClaudeMessagesProvider(...)` constructor call to match the actual signature (check `src/feather/providers/claude_provider.py`'s `__init__`); the test above is the intent and the construction details may need a one-line edit.

- [ ] **Step 3: Run the integration test**

Run: `uv run pytest tests/test_claude_provider.py::test_claude_provider_strips_rejected_keywords_in_wire_body -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_claude_provider.py
git commit -m "Integration: assert wire body has no minimum after claude translation"
```

---

### Task 7: Simplify pass

**Files:**
- Touched files from Tasks 1-6

- [ ] **Step 1: Invoke the simplify skill against the changed code**

Run: invoke the `code-simplifier:code-simplifier` agent (via the Agent tool with `subagent_type=code-simplifier:code-simplifier`) and ask it to review:
- `src/feather/providers/claude_translator.py` (added `sanitize_anthropic_tool_schema` + wired into `translate_tools`)
- `tests/test_claude_tool_schema_sanitizer.py`
- `tests/test_tool_schemas_anthropic_compatible.py`
- `tests/test_claude_provider.py` (new test only)

Specifically, ask it to remove:
- Duplicated helper code between the sanitizer file and tests (e.g. if a `_walk` helper duplicates structure used by the sanitizer's recursion).
- Comments that explain WHAT the code does (well-named identifiers should suffice) — keep WHY comments.
- Unused imports introduced in the test files.

- [ ] **Step 2: Re-run full Phase 0 test suite after any simplifier changes**

Run: `uv run pytest tests/test_claude_tool_schema_sanitizer.py tests/test_tool_schemas_anthropic_compatible.py tests/test_claude_translator.py tests/test_claude_provider.py -v`
Expected: all green.

- [ ] **Step 3: Commit any simplifier changes**

```bash
git add -p   # review each hunk
git commit -m "Simplify Anthropic sanitizer per code-simplifier pass"
```

(Skip the commit if the simplifier returned no changes.)

---

### Task 8: Red-team code review

**Files:**
- All Phase 0 changes (review-only).

- [ ] **Step 1: Dispatch the code-reviewer agent**

Invoke `superpowers:code-reviewer` (via the Agent tool with `subagent_type=superpowers:code-reviewer`). Brief:

> Red-team review of Phase 0 (Claude tool-schema sanitizer) against the spec at `docs/superpowers/specs/2026-05-11-config-tui-design.md` (sections 4 and Tests). Hunt for: (1) schemas the sanitizer might miss (nested `oneOf`/`allOf`, definitions, references); (2) idempotence breaks (e.g. recursion that mutates inputs); (3) regressions in the non-Claude providers (translator changes shouldn't affect OpenAI Responses or OpenRouter); (4) performance — sanitizer runs per request, ensure no quadratic walks; (5) the integration test's mock fidelity — does it actually exercise the wire path or short-circuit? Report blocking issues vs nits. Under 400 words.

- [ ] **Step 2: Address any BLOCKING findings**

For each blocking item: write a failing test → fix → re-run all Phase 0 tests → commit with a message starting `Address red-team finding: ...`. NIT findings can be deferred to Phase 3's "registry coverage" sweep.

- [ ] **Step 3: Phase 0 done — push**

```bash
git push -u origin feature/config-tui
```

---

## Phase 0 self-review checklist (before handoff)

- [ ] Sanitizer strips `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` (Task 2)
- [ ] Sanitizer normalises `["integer", "null"]` → `"integer"` (Task 2)
- [ ] Sanitizer is idempotent (Task 3)
- [ ] Sanitizer is non-mutating (Task 3)
- [ ] Sanitizer recurses into `properties`, `items`, `anyOf`, `oneOf`, `allOf` (Tasks 1, 2)
- [ ] `translate_tools` calls the sanitizer on BOTH the flat-Responses path AND the Anthropic-native passthrough path (Task 4)
- [ ] Every shipped tool's schema is verified clean post-sanitize (Task 5)
- [ ] Wire body to Anthropic has no rejected keywords (Task 6)
- [ ] Simplify pass executed (Task 7)
- [ ] Red-team review executed and blockers addressed (Task 8)
