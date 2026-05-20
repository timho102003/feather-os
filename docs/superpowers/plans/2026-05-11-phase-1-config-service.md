# Phase 1 — Headless Config Service + Reload Plumbing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the foundational config layer: a typed schema registry, a comment-preserving writer, a dotted-path resolver, a `ConfigService` orchestration layer, `FeatherRuntime` reload primitives, the supervisor reload envelope, the headless `/config get|set|list|diff|reset` slash subcommands, and the `app.yaml` cleanup (reorder + `memory.operations.*` regrouping + MCP example extraction).

**Architecture:** A hand-authored schema registry (`feather.config_schema`) is the single source of truth for what is editable, where its YAML lives, how to validate it, and which reload class it falls into. A drift tripwire test guards the registry against the dataclasses. `ConfigService` is the only orchestration layer used by both the headless CLI and the future modal — modal save in Phase 2 is just a sequence of `ConfigService.set()` calls plus one `runtime.apply_config_change()`. The supervisor reload envelope uses the existing pipe protocol with a new request/ack pair so worker mode and in-process mode share one TUI-facing API.

**Tech Stack:** Python 3.12+, pytest (auto-mode async), ruamel.yaml for round-trip writes, existing supervisor pipe protocol.

**Worktree:** `/home/dev/feather_v2/.worktrees/config-tui` on branch `feature/config-tui`. The design spec is `docs/superpowers/specs/2026-05-11-config-tui-design.md`. Phase 0 must be merged or complete on this branch before starting Phase 1.

**Dependencies to add:** `ruamel.yaml` (round-trip YAML writer) — added in Task 6.

**Workflow reminder:** Each task is TDD. The phase wraps with a simplify pass (Task 30) and a red-team review (Task 31).

---

## 1A — Config schema foundation

### Task 1: Define `ConfigField`, `ReloadClass`, `FieldType`, `Scope`, `WidgetHint`

**Files:**
- Create: `src/feather/config_schema.py`
- Create: `tests/test_config_schema_types.py`

- [ ] **Step 1: Failing test for the type primitives**

```python
"""Tests for the ConfigField dataclass and supporting enums."""

from __future__ import annotations

import pytest

from feather.config_schema import (
    ConfigField,
    FieldType,
    ReloadClass,
    Scope,
    WidgetHint,
)


def test_reload_class_values() -> None:
    assert {c.value for c in ReloadClass} == {
        "live",
        "next_turn",
        "restart_lead",
        "restart_app",
    }


def test_scope_values() -> None:
    assert {s.value for s in Scope} == {"app", "agent"}


def test_field_type_values() -> None:
    assert {t.value for t in FieldType} == {
        "string",
        "integer",
        "float",
        "boolean",
        "string_list",
        "enum",
    }


def test_widget_hint_values() -> None:
    assert {w.value for w in WidgetHint} == {
        "text",
        "numeric",
        "toggle",
        "dropdown",
        "list_editor",
        "sensitive_readonly",
    }


def test_config_field_validates_enum_consistency() -> None:
    with pytest.raises(ValueError):
        ConfigField(
            path="x.y",
            type=FieldType.ENUM,
            enum=None,
            widget=WidgetHint.DROPDOWN,
            reload=ReloadClass.LIVE,
            scope=Scope.APP,
            description="needs an enum list",
        )


def test_config_field_widget_must_match_type_for_enum() -> None:
    with pytest.raises(ValueError):
        ConfigField(
            path="x.y",
            type=FieldType.ENUM,
            enum=("a", "b"),
            widget=WidgetHint.TEXT,
            reload=ReloadClass.LIVE,
            scope=Scope.APP,
            description="enum must use dropdown",
        )
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_config_schema_types.py -v`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 3: Implement the primitives**

```python
"""Typed schema for editable Feather configuration fields.

The :class:`ConfigField` registry in this module is the single source
of truth for what may be edited from the TUI / ``/config`` slash
subcommands. Each entry binds a dotted path to its YAML type, the
TUI widget hint, validation rules, reload semantics, and a
human-readable description.

A drift tripwire test (``tests/test_config_schema_drift.py``) walks
``AppConfig`` / ``AgentConfig`` recursively and asserts every leaf
path is either in :data:`REGISTRY` or :data:`IGNORED_PATHS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ReloadClass(str, Enum):
    """How invasive applying this field's change is."""

    LIVE = "live"
    NEXT_TURN = "next_turn"
    RESTART_LEAD = "restart_lead"
    RESTART_APP = "restart_app"


class Scope(str, Enum):
    """Which YAML file owns this field."""

    APP = "app"
    AGENT = "agent"


class FieldType(str, Enum):
    """Wire type the field serialises to in YAML."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING_LIST = "string_list"
    ENUM = "enum"


class WidgetHint(str, Enum):
    """Render hint for the modal."""

    TEXT = "text"
    NUMERIC = "numeric"
    TOGGLE = "toggle"
    DROPDOWN = "dropdown"
    LIST_EDITOR = "list_editor"
    SENSITIVE_READONLY = "sensitive_readonly"


Validator = Callable[[Any], None]


@dataclass(slots=True, frozen=True)
class ConfigField:
    """One editable configuration field.

    Attributes:
        path: Dotted path. ``app.*`` for application config,
            ``agents.<name>.*`` for per-agent.
        type: YAML wire type.
        widget: Render hint for the modal.
        reload: How invasive applying this field's change is.
        scope: Which YAML file class owns the field.
        description: One-line user-facing description.
        enum: Allowed values when ``type`` is ``ENUM``.
        validator: Optional callable raising ``ValueError`` on bad value.
        sensitive: True for env-var indirection (read-only in modal).
        default: Documented default; ``None`` means "inherits dataclass default".
    """

    path: str
    type: FieldType
    widget: WidgetHint
    reload: ReloadClass
    scope: Scope
    description: str
    enum: tuple[str, ...] | None = None
    validator: Validator | None = None
    sensitive: bool = False
    default: Any = None

    def __post_init__(self) -> None:
        if self.type is FieldType.ENUM:
            if not self.enum:
                raise ValueError(
                    f"ConfigField {self.path!r}: enum type requires non-empty enum"
                )
            if self.widget is not WidgetHint.DROPDOWN:
                raise ValueError(
                    f"ConfigField {self.path!r}: enum type must use DROPDOWN widget"
                )


REGISTRY: tuple[ConfigField, ...] = ()
IGNORED_PATHS: frozenset[str] = frozenset()
```

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest tests/test_config_schema_types.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/feather/config_schema.py tests/test_config_schema_types.py
git commit -m "Add ConfigField primitives for the editable-config registry"
```

---

### Task 2: Drift tripwire test (fails until registry is filled)

**Files:**
- Create: `tests/test_config_schema_drift.py`

- [ ] **Step 1: Write the tripwire**

```python
"""Drift tripwire: every leaf in AppConfig / AgentConfig must be in
the registry or explicitly ignored.

This test fails the build whenever a new dataclass field is added
without a corresponding registry entry, forcing an explicit decision
(surface it in /config, or add it to IGNORED_PATHS with a reason).
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_origin

from feather.config_schema import IGNORED_PATHS, REGISTRY, Scope
from feather.models import AgentConfig, AppConfig


def _leaf_paths(prefix: str, cls: type) -> list[str]:
    """Walk a dataclass; yield dotted leaf paths."""

    if not is_dataclass(cls):
        return [prefix.rstrip(".")]

    out: list[str] = []
    for f in fields(cls):
        sub = f"{prefix}{f.name}"
        ftype = f.type if isinstance(f.type, type) else None

        # Unwrap Optional[T] / T | None into T
        origin = get_origin(f.type)
        if origin is type(None):
            ftype = None
        elif origin is None:
            ftype = f.type if isinstance(f.type, type) else None
        else:
            args = [a for a in get_args(f.type) if a is not type(None)]
            ftype = args[0] if len(args) == 1 and isinstance(args[0], type) else None

        if ftype is not None and is_dataclass(ftype):
            out.extend(_leaf_paths(f"{sub}.", ftype))
        else:
            out.append(sub)
    return out


def test_app_config_fields_are_in_registry_or_ignored() -> None:
    leaves = {f"app.{p}" for p in _leaf_paths("", AppConfig)}
    addressed = {f.path for f in REGISTRY if f.scope is Scope.APP}
    missing = leaves - addressed - IGNORED_PATHS
    assert not missing, (
        "AppConfig has fields not covered by registry or IGNORED_PATHS: "
        + ", ".join(sorted(missing))
    )


def test_agent_config_fields_are_in_registry_or_ignored() -> None:
    leaves = {f"agents.*.{p}" for p in _leaf_paths("", AgentConfig)}
    addressed = {
        f.path.replace(f.path.split(".")[1], "*", 1)
        for f in REGISTRY
        if f.scope is Scope.AGENT
    }
    missing = leaves - addressed - IGNORED_PATHS
    assert not missing, (
        "AgentConfig has fields not covered: " + ", ".join(sorted(missing))
    )


def test_registry_paths_are_unique() -> None:
    paths = [f.path for f in REGISTRY]
    assert len(paths) == len(set(paths)), "duplicate paths in REGISTRY"
```

- [ ] **Step 2: Run, expect failures listing every leaf**

Run: `uv run pytest tests/test_config_schema_drift.py -v`
Expected: FAIL on both `*_in_registry_or_ignored` tests, with a long missing-paths list. This is the work list for Tasks 3–8.

- [ ] **Step 3: Commit (test only; registry filled in subsequent tasks)**

```bash
git add tests/test_config_schema_drift.py
git commit -m "Add drift tripwire test for the config registry"
```

---

### Task 3: Seed `IGNORED_PATHS` for genuinely non-editable fields

**Files:**
- Modify: `src/feather/config_schema.py`

- [ ] **Step 1: Identify non-editable leaves**

Inspect the tripwire failure list from Task 2. The following are NOT user-editable (set by the runtime, immutable defaults, or covered by other surfaces):

- `app.openai.prompt_cache_key`, `app.openai.prompt_cache_retention`, `app.openai.store` (provider-internal cache plumbing)
- `app.mcp.servers` (servers managed via a different future surface; spec scope says no)
- `app.parallel.api_key_env` etc. — these are env-var indirections; surfaced via `[sensitive]` badge in registry instead — *NOT* ignored, will be in registry
- `agents.*.prompt_modules` (Phase 1 read-only per spec; treat as ignored for now)
- `agents.*.mcp_servers` (Phase 3)
- `agents.*.description`, `agents.*.inline_prompt` (rarely useful for the lead)

- [ ] **Step 2: Update `IGNORED_PATHS`**

```python
IGNORED_PATHS: frozenset[str] = frozenset({
    # Provider-internal cache plumbing (not a user-facing knob)
    "app.openai.prompt_cache_key",
    "app.openai.prompt_cache_retention",
    "app.openai.store",
    # MCP servers managed via a different (future) UI surface
    "app.mcp.servers",
    # Phase 1 keeps prompt_modules read-only — changing it risks
    # loading a non-existent module mid-session.
    "agents.*.prompt_modules",
    # Phase 3 — agent-level MCP overrides
    "agents.*.mcp_servers",
    # Internal: agents.*.description/inline_prompt are spec-defined
    # but not surfaced in Phase 1
    "agents.*.description",
    "agents.*.inline_prompt",
})
```

- [ ] **Step 3: Run tripwire**

Run: `uv run pytest tests/test_config_schema_drift.py -v`
Expected: STILL FAIL — but the missing list is shorter. Remaining failures will be addressed in Tasks 4–8.

- [ ] **Step 4: Commit**

```bash
git add src/feather/config_schema.py
git commit -m "Seed IGNORED_PATHS for non-user-editable config leaves"
```

---

### Task 4: Registry — infrastructure + behavioural defaults section

**Files:**
- Modify: `src/feather/config_schema.py`

- [ ] **Step 1: Add registry entries for `database`, `storage`, `logging`, `compaction`, `skills`, `scheduler`, `self_repair`**

Replace `REGISTRY: tuple[ConfigField, ...] = ()` with the following (keep `IGNORED_PATHS` from Task 3):

```python
def _ratio(v: float) -> None:
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"must be in [0.0, 1.0], got {v}")


def _positive(v: float) -> None:
    if v <= 0:
        raise ValueError(f"must be positive, got {v}")


def _non_negative_int(v: int) -> None:
    if v < 0:
        raise ValueError(f"must be >= 0, got {v}")


REGISTRY: tuple[ConfigField, ...] = (
    # Infrastructure --------------------------------------------------
    ConfigField(
        path="app.database.path",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="SQLite database path (relative to project root or absolute).",
    ),
    ConfigField(
        path="app.storage.temp_directory",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Directory where oversized tool outputs are spilled.",
    ),
    ConfigField(
        path="app.logging.path",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Log file path.",
    ),
    ConfigField(
        path="app.logging.level",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Log level.",
        enum=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    ),
    # Behavioural defaults -------------------------------------------
    ConfigField(
        path="app.compaction.enabled",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Automatically compact context when usage exceeds trigger_ratio.",
    ),
    ConfigField(
        path="app.compaction.trigger_ratio",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Fraction of context window before compaction fires (0.0-1.0).",
        validator=_ratio,
    ),
    ConfigField(
        path="app.compaction.context_window_tokens",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Context-window size used to compute usage_ratio.",
        validator=_positive,
    ),
    ConfigField(
        path="app.compaction.model",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Override the model used for compaction summaries (blank inherits).",
    ),
    ConfigField(
        path="app.compaction.max_output_tokens",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Max output tokens for compaction summaries.",
        validator=_positive,
    ),
    ConfigField(
        path="app.compaction.temperature",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Temperature for compaction summaries.",
    ),
    ConfigField(
        path="app.skills.directory",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Project-relative directory where SKILL.md files live.",
    ),
    ConfigField(
        path="app.scheduler.enabled",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Run the background cron scheduler.",
    ),
    ConfigField(
        path="app.scheduler.poll_interval_seconds",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="How often the scheduler polls for due jobs.",
        validator=_positive,
    ),
    ConfigField(
        path="app.scheduler.failure_retry_seconds",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Delay before retrying a failed scheduled job.",
        validator=_positive,
    ),
    ConfigField(
        path="app.scheduler.max_due_jobs_per_tick",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Max jobs picked up per scheduler tick.",
        validator=_positive,
    ),
    ConfigField(
        path="app.self_repair.enabled",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description=(
            "Run the lead in a worker subprocess for hang detection and self-repair. "
            "Flipping requires a full TUI restart; cannot be applied mid-session."
        ),
    ),
)
```

- [ ] **Step 2: Run tripwire**

Run: `uv run pytest tests/test_config_schema_drift.py -v`
Expected: still failing on remaining sections (providers, memory, agents).

- [ ] **Step 3: Commit**

```bash
git add src/feather/config_schema.py
git commit -m "Registry: cover infrastructure + behavioural defaults"
```

---

### Task 5: Registry — `active_provider` + OpenAI + OpenRouter + Claude + Parallel

**Files:**
- Modify: `src/feather/config_schema.py`

- [ ] **Step 1: Append provider entries to REGISTRY**

Append to the `REGISTRY` tuple (immediately before the closing `)`):

```python
    # Routing --------------------------------------------------------
    ConfigField(
        path="app.active_provider",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="LLM provider every agent routes through unless overridden.",
        enum=("openai", "openrouter", "claude"),
    ),
    # OpenAI ---------------------------------------------------------
    ConfigField(
        path="app.openai.api_key_env",
        type=FieldType.STRING,
        widget=WidgetHint.SENSITIVE_READONLY,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Env-var name holding the OpenAI API key.",
        sensitive=True,
    ),
    ConfigField(
        path="app.openai.model",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Default OpenAI model (e.g. gpt-5-mini).",
    ),
    ConfigField(
        path="app.openai.max_output_tokens",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Upper bound on output tokens per OpenAI turn.",
        validator=_positive,
    ),
    ConfigField(
        path="app.openai.temperature",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Sampling temperature for OpenAI.",
    ),
    ConfigField(
        path="app.openai.parallel_tool_calls",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Allow OpenAI to emit multiple tool calls in one turn.",
    ),
    ConfigField(
        path="app.openai.reasoning.effort",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="OpenAI reasoning effort.",
        enum=("none", "minimal", "low", "medium", "high"),
    ),
    ConfigField(
        path="app.openai.reasoning.summary",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="OpenAI reasoning summary verbosity.",
        enum=("auto", "concise", "detailed"),
    ),
    ConfigField(
        path="app.openai.stream_idle_timeout_seconds",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="How long an idle OpenAI stream may sit before timing out.",
        validator=_positive,
    ),
    # OpenRouter -----------------------------------------------------
    ConfigField(
        path="app.openrouter.api_key_env",
        type=FieldType.STRING,
        widget=WidgetHint.SENSITIVE_READONLY,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Env-var name holding the OpenRouter API key.",
        sensitive=True,
    ),
    ConfigField(
        path="app.openrouter.base_url",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="OpenRouter API base URL.",
    ),
    ConfigField(
        path="app.openrouter.http_referer",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="HTTP Referer header value (optional).",
    ),
    ConfigField(
        path="app.openrouter.app_title",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="X-Title header value (optional).",
    ),
    ConfigField(
        path="app.openrouter.model",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Default OpenRouter model slug.",
    ),
    ConfigField(
        path="app.openrouter.max_output_tokens",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Upper bound on output tokens per OpenRouter turn.",
        validator=_positive,
    ),
    ConfigField(
        path="app.openrouter.temperature",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Sampling temperature for OpenRouter.",
    ),
    ConfigField(
        path="app.openrouter.parallel_tool_calls",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Allow OpenRouter to emit multiple tool calls in one turn.",
    ),
    ConfigField(
        path="app.openrouter.reasoning.effort",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="OpenRouter reasoning effort.",
        enum=("none", "minimal", "low", "medium", "high"),
    ),
    ConfigField(
        path="app.openrouter.reasoning.summary",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="OpenRouter reasoning summary verbosity.",
        enum=("auto", "concise", "detailed"),
    ),
    ConfigField(
        path="app.openrouter.cache_strategy",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Prompt-cache breakpoint placement strategy.",
        enum=("anthropic_breakpoint", "gemini_breakpoint", "none"),
    ),
    ConfigField(
        path="app.openrouter.stream_idle_timeout_seconds",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="How long an idle OpenRouter stream may sit before timing out.",
        validator=_positive,
    ),
    ConfigField(
        path="app.openrouter.request_timeout_seconds",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="HTTP request timeout for OpenRouter calls.",
        validator=_positive,
    ),
    ConfigField(
        path="app.openrouter.max_attempts",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Retry budget for transient OpenRouter failures.",
        validator=_positive,
    ),
    ConfigField(
        path="app.openrouter.supports_multimodal",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Mark this OpenRouter model as accepting image inputs.",
    ),
    ConfigField(
        path="app.openrouter.max_stream_wall_seconds",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Hard wall-clock cap on a single OpenRouter stream.",
        validator=_positive,
    ),
    # Claude ---------------------------------------------------------
    ConfigField(
        path="app.claude.api_key_env",
        type=FieldType.STRING,
        widget=WidgetHint.SENSITIVE_READONLY,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Env-var name holding the Anthropic API key.",
        sensitive=True,
    ),
    ConfigField(
        path="app.claude.base_url",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Anthropic API base URL.",
    ),
    ConfigField(
        path="app.claude.anthropic_version",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Pinned Anthropic API version date.",
    ),
    ConfigField(
        path="app.claude.anthropic_beta",
        type=FieldType.STRING_LIST,
        widget=WidgetHint.LIST_EDITOR,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Beta flags joined into the anthropic-beta header.",
    ),
    ConfigField(
        path="app.claude.model",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Default Anthropic model (e.g. claude-opus-4-7).",
    ),
    ConfigField(
        path="app.claude.max_output_tokens",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Upper bound on output tokens per Claude turn.",
        validator=_positive,
    ),
    ConfigField(
        path="app.claude.temperature",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Sampling temperature for Claude.",
    ),
    ConfigField(
        path="app.claude.parallel_tool_calls",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Allow Claude to emit multiple tool calls in one turn.",
    ),
    ConfigField(
        path="app.claude.thinking.type",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Anthropic extended-thinking mode.",
        enum=("enabled", "adaptive", "disabled"),
    ),
    ConfigField(
        path="app.claude.thinking.budget_tokens",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Thinking-token budget when thinking.type is 'enabled'.",
        validator=_positive,
    ),
    ConfigField(
        path="app.claude.cache_strategy",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Prompt-cache breakpoint placement strategy.",
        enum=("anthropic_breakpoint", "none"),
    ),
    ConfigField(
        path="app.claude.stream_idle_timeout_seconds",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="How long an idle Claude stream may sit before timing out.",
        validator=_positive,
    ),
    ConfigField(
        path="app.claude.request_timeout_seconds",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="HTTP request timeout for Claude calls.",
        validator=_positive,
    ),
    ConfigField(
        path="app.claude.max_attempts",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Retry budget for transient Claude failures.",
        validator=_positive,
    ),
    ConfigField(
        path="app.claude.supports_multimodal",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Mark this Claude model as accepting image inputs.",
    ),
    ConfigField(
        path="app.claude.max_stream_wall_seconds",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Hard wall-clock cap on a single Claude stream.",
        validator=_positive,
    ),
    # Parallel AI ----------------------------------------------------
    ConfigField(
        path="app.parallel.api_key_env",
        type=FieldType.STRING,
        widget=WidgetHint.SENSITIVE_READONLY,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Env-var name holding the Parallel AI API key.",
        sensitive=True,
    ),
    ConfigField(
        path="app.parallel.default_search_mode",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Default Parallel AI search mode.",
        enum=("fast", "balanced", "thorough"),
    ),
    ConfigField(
        path="app.parallel.max_results",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Max search results returned to the model.",
        validator=_positive,
    ),
    ConfigField(
        path="app.parallel.inline_full_content_threshold",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Below this byte threshold, search results inline full content.",
        validator=_non_negative_int,
    ),
)
```

Also ignore the OpenRouter fields not in the registry above (provider_preferences, fallback_models, tracing — Phase 3 work):

Update `IGNORED_PATHS`:

```python
IGNORED_PATHS: frozenset[str] = frozenset({
    # ... entries from Task 3 ...
    "app.openrouter.provider_preferences",
    "app.openrouter.fallback_models",
    "app.openrouter.tracing",
    "app.mcp.enabled",  # toggled via /config when MCP gets surfaced; out of Phase 1
})
```

- [ ] **Step 2: Run tripwire**

Run: `uv run pytest tests/test_config_schema_drift.py -v`
Expected: still failing on memory + agents only.

- [ ] **Step 3: Commit**

```bash
git add src/feather/config_schema.py
git commit -m "Registry: cover active_provider + OpenAI/OpenRouter/Claude/Parallel"
```

---

### Task 6: Registry — `memory` block (including new `memory.operations.*` grouping)

**Files:**
- Modify: `src/feather/config_schema.py`

- [ ] **Step 1: Append memory entries**

The registry must reflect the post-cleanup `memory.operations.*` shape (Task 10 finalises the loader / packaged-default change). The drift tripwire walks `MemoryConfig` which still names them at top level; add to `IGNORED_PATHS` the legacy flat paths and surface the grouped ones.

Append to `REGISTRY` (before closing `)`):

```python
    # Memory: qdrant -------------------------------------------------
    ConfigField(
        path="app.memory.enabled",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Master switch for the long-term memory subsystem.",
    ),
    ConfigField(
        path="app.memory.qdrant.url",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Qdrant server URL (QDRANT_URL env overrides at boot).",
    ),
    ConfigField(
        path="app.memory.qdrant.api_key_env",
        type=FieldType.STRING,
        widget=WidgetHint.SENSITIVE_READONLY,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Env-var name holding the Qdrant API key (optional).",
        sensitive=True,
    ),
    ConfigField(
        path="app.memory.qdrant.collection_name",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Qdrant collection that stores memory points.",
    ),
    ConfigField(
        path="app.memory.qdrant.embedding_dims",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Dimensionality of stored vectors.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.qdrant.hnsw_m",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="HNSW connectivity parameter.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.qdrant.hnsw_ef_construct",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="HNSW build-time search width.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.qdrant.hnsw_ef_search",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="HNSW query-time search width.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.qdrant.hnsw_full_scan_threshold",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Points below this fall back to full-scan search.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.qdrant.indexing_threshold",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Min points before HNSW kicks in.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.qdrant.default_segment_number",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Initial segment count.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.qdrant.on_disk_vectors",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Store vectors on disk (slower, lower RAM).",
    ),
    ConfigField(
        path="app.memory.qdrant.on_disk_payload",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Store payload on disk (slower, lower RAM).",
    ),
    ConfigField(
        path="app.memory.qdrant.prefer_grpc",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Prefer the gRPC transport over HTTP.",
    ),
    ConfigField(
        path="app.memory.qdrant.request_timeout_s",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Qdrant client request timeout.",
        validator=_positive,
    ),
    # Memory: embedding ----------------------------------------------
    ConfigField(
        path="app.memory.embedding.provider",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Embedding provider.",
        enum=("gemini", "openai"),
    ),
    ConfigField(
        path="app.memory.embedding.model",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Embedding model name.",
    ),
    ConfigField(
        path="app.memory.embedding.output_dimensionality",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Output embedding dimensionality.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.embedding.task_type_document",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Gemini task_type for document embeddings.",
    ),
    ConfigField(
        path="app.memory.embedding.task_type_query",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Gemini task_type for query embeddings.",
    ),
    ConfigField(
        path="app.memory.embedding.normalize_reduced_dims",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Renormalise Matryoshka-reduced vectors.",
    ),
    ConfigField(
        path="app.memory.embedding.request_timeout_s",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Embedding API request timeout.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.embedding.max_retries",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Retry budget for embedding API calls.",
        validator=_non_negative_int,
    ),
    ConfigField(
        path="app.memory.embedding.retry_backoff_s",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Embedding retry backoff base.",
        validator=_positive,
    ),
    # Memory: chunking -----------------------------------------------
    ConfigField(
        path="app.memory.chunking.chunk_size_tokens",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Chunk size in tokens for memory ingestion.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.chunking.chunk_overlap_tokens",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Overlap between consecutive chunks.",
        validator=_non_negative_int,
    ),
    ConfigField(
        path="app.memory.chunking.tokenizer",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Tokenizer name (e.g. tiktoken).",
    ),
    ConfigField(
        path="app.memory.chunking.tokenizer_encoding",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Tokenizer encoding (e.g. o200k_base).",
    ),
    # Memory: retrieval ----------------------------------------------
    ConfigField(
        path="app.memory.retrieval.enabled",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Run memory retrieval on each agent turn.",
    ),
    ConfigField(
        path="app.memory.retrieval.top_k_prompt_injection",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Top-K memories injected into the system prompt.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.retrieval.top_k_tool",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Top-K memories returned by recall_memory tool calls.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.retrieval.score_threshold",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Min cosine similarity for a memory to be considered relevant.",
        validator=_ratio,
    ),
    ConfigField(
        path="app.memory.retrieval.classifier_top_k",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Top-K memories fed to the relevance classifier.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.retrieval.classifier_score_threshold",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Classifier confidence threshold.",
        validator=_ratio,
    ),
    ConfigField(
        path="app.memory.retrieval.retrieval_timeout_s",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Hard timeout on the retrieval pipeline.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.retrieval.query_builder_enabled",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Use the LLM-driven query builder before retrieval.",
    ),
    ConfigField(
        path="app.memory.retrieval.query_builder_recent_messages",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="How many recent messages the query builder sees.",
        validator=_positive,
    ),
    # Memory: trigger ------------------------------------------------
    ConfigField(
        path="app.memory.trigger.enabled",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Extract memories from conversation turns.",
    ),
    ConfigField(
        path="app.memory.trigger.trigger_turns",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Run extraction every N turns.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.trigger.skip_compact_messages",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="Skip compaction summaries during extraction.",
    ),
    ConfigField(
        path="app.memory.trigger.background",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.RESTART_LEAD,
        scope=Scope.APP,
        description="Run extraction in a background task (vs inline).",
    ),
    ConfigField(
        path="app.memory.trigger.shutdown_timeout_s",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Grace period for in-flight extractions at shutdown.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.trigger.max_concurrent_extractions_per_session",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Max parallel extraction jobs per session.",
        validator=_positive,
    ),
    # Memory: operations (post-cleanup grouping) ---------------------
    ConfigField(
        path="app.memory.operations.extraction.provider",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Provider for memory extraction (blank inherits active_provider).",
    ),
    ConfigField(
        path="app.memory.operations.extraction.model",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Model for memory extraction.",
    ),
    ConfigField(
        path="app.memory.operations.extraction.max_output_tokens",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Max output tokens for extraction.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.operations.extraction.temperature",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Temperature for extraction.",
    ),
    ConfigField(
        path="app.memory.operations.classification.provider",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Provider for relevance classification.",
    ),
    ConfigField(
        path="app.memory.operations.classification.model",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Model for relevance classification.",
    ),
    ConfigField(
        path="app.memory.operations.classification.max_output_tokens",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Max output tokens for classification.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.operations.classification.temperature",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Temperature for classification.",
    ),
    ConfigField(
        path="app.memory.operations.query_builder.provider",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Provider for query builder.",
    ),
    ConfigField(
        path="app.memory.operations.query_builder.model",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Model for query builder.",
    ),
    ConfigField(
        path="app.memory.operations.query_builder.max_output_tokens",
        type=FieldType.INTEGER,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Max output tokens for query builder.",
        validator=_positive,
    ),
    ConfigField(
        path="app.memory.operations.query_builder.temperature",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Temperature for query builder.",
    ),
)
```

Update `IGNORED_PATHS` to ignore the legacy flat `MemoryConfig` paths (the loader still accepts them, but the writer emits only the grouped shape, so the registry does not need both):

```python
IGNORED_PATHS: frozenset[str] = frozenset({
    # ... prior entries ...
    "app.memory.qdrant.quantization",  # advanced, Phase 3
    # Legacy flat shape — loader still accepts, but writer emits grouped form.
    # Drift tripwire walks AppConfig (which has the flat shape); ignore those
    # paths since their grouped equivalents are in REGISTRY.
    "app.memory.extraction.provider",
    "app.memory.extraction.model",
    "app.memory.extraction.max_output_tokens",
    "app.memory.extraction.temperature",
    "app.memory.classification.provider",
    "app.memory.classification.model",
    "app.memory.classification.max_output_tokens",
    "app.memory.classification.temperature",
    "app.memory.query_builder.provider",
    "app.memory.query_builder.model",
    "app.memory.query_builder.max_output_tokens",
    "app.memory.query_builder.temperature",
})
```

- [ ] **Step 2: Run tripwire**

Run: `uv run pytest tests/test_config_schema_drift.py -v`
Expected: AppConfig test passes; AgentConfig test still failing.

- [ ] **Step 3: Commit**

```bash
git add src/feather/config_schema.py
git commit -m "Registry: cover the memory subsystem (grouped operations shape)"
```

---

### Task 7: Registry — Lead agent (per-agent scoped paths)

**Files:**
- Modify: `src/feather/config_schema.py`

- [ ] **Step 1: Append agent-scoped entries**

Append to `REGISTRY` (before closing `)`):

```python
    # Agent: Lead ----------------------------------------------------
    ConfigField(
        path="agents.Lead.personality",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.AGENT,
        description="One-line personality / voice description.",
    ),
    ConfigField(
        path="agents.Lead.memory_enabled",
        type=FieldType.BOOLEAN,
        widget=WidgetHint.TOGGLE,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.AGENT,
        description="Agent-level memory opt-in (also gated by app.memory.enabled).",
    ),
    ConfigField(
        path="agents.Lead.provider",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.AGENT,
        description="Override the app-level provider for this agent (blank inherits).",
    ),
    ConfigField(
        path="agents.Lead.model",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.AGENT,
        description="Override the provider's default model for this agent (blank inherits).",
    ),
    ConfigField(
        path="agents.Lead.reasoning.effort",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.AGENT,
        description="Per-agent reasoning effort override.",
        enum=("none", "minimal", "low", "medium", "high"),
    ),
    ConfigField(
        path="agents.Lead.reasoning.summary",
        type=FieldType.ENUM,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.AGENT,
        description="Per-agent reasoning summary verbosity.",
        enum=("auto", "concise", "detailed"),
    ),
    ConfigField(
        path="agents.Lead.registered_tools",
        type=FieldType.STRING_LIST,
        widget=WidgetHint.LIST_EDITOR,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.AGENT,
        description="List of tool names this agent may call.",
    ),
)
```

The drift tripwire's agent test normalises any `agents.<name>.*` to `agents.*.*` so the entries above match the wildcard form the test expects.

- [ ] **Step 2: Run full tripwire**

Run: `uv run pytest tests/test_config_schema_drift.py -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add src/feather/config_schema.py
git commit -m "Registry: cover Lead agent per-agent fields"
```

---

### Task 8: Registry self-checks — defaults, validators, exports

**Files:**
- Create: `tests/test_config_schema.py`
- Modify: `src/feather/config_schema.py` (export `lookup`, add helper)

- [ ] **Step 1: Failing test for the lookup helper**

```python
"""Per-entry self-checks for the config registry."""

from __future__ import annotations

import pytest

from feather.config_schema import (
    FieldType,
    REGISTRY,
    WidgetHint,
    lookup,
)


def test_lookup_finds_by_exact_path() -> None:
    field = lookup("app.active_provider")
    assert field is not None
    assert field.enum is not None
    assert "openrouter" in field.enum


def test_lookup_handles_agent_wildcard() -> None:
    field = lookup("agents.Lead.model")
    assert field is not None
    assert field.scope.value == "agent"


def test_lookup_returns_none_for_unknown() -> None:
    assert lookup("does.not.exist") is None


def test_every_entry_has_non_empty_description() -> None:
    for field in REGISTRY:
        assert field.description.strip(), f"{field.path} has empty description"


def test_validator_callables_are_callable() -> None:
    for field in REGISTRY:
        if field.validator is not None:
            assert callable(field.validator)


def test_string_list_widget_must_be_list_editor() -> None:
    for field in REGISTRY:
        if field.type is FieldType.STRING_LIST:
            assert field.widget is WidgetHint.LIST_EDITOR, (
                f"{field.path}: STRING_LIST must use LIST_EDITOR"
            )
```

- [ ] **Step 2: Run, expect ImportError on `lookup`**

Run: `uv run pytest tests/test_config_schema.py -v`
Expected: FAIL — `lookup` not exported.

- [ ] **Step 3: Add `lookup` helper to `config_schema.py`**

Append to `src/feather/config_schema.py`:

```python
def lookup(path: str) -> ConfigField | None:
    """Return the registry entry for ``path``, or ``None``.

    Resolves both literal paths (``app.openai.model``) and per-agent
    paths (``agents.Lead.model`` matches a stored ``agents.Lead.model``
    entry — the agent name is part of the literal path, not a wildcard).
    """

    for field in REGISTRY:
        if field.path == path:
            return field
    return None


__all__ = (
    "ConfigField",
    "FieldType",
    "IGNORED_PATHS",
    "REGISTRY",
    "ReloadClass",
    "Scope",
    "WidgetHint",
    "lookup",
)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config_schema.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_config_schema.py src/feather/config_schema.py
git commit -m "Add lookup() and registry self-checks"
```

---

## 1B — Dotted-path resolver

### Task 9: `config_paths.py` — dotted path → (file, yaml_path, scope)

**Files:**
- Create: `src/feather/config_paths.py`
- Create: `tests/test_config_paths.py`

- [ ] **Step 1: Failing test**

```python
"""Tests for the dotted-path resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.config_paths import (
    ConfigPathResolver,
    PathResolution,
    PathScope,
)
from feather.paths import FeatherPaths


def _paths(tmp_path: Path) -> FeatherPaths:
    return FeatherPaths(project_root=tmp_path / "proj", home=tmp_path / "global")


def test_resolve_app_path_global(tmp_path: Path) -> None:
    resolver = ConfigPathResolver(_paths(tmp_path))
    res = resolver.resolve("app.openai.model", scope=PathScope.GLOBAL)
    assert isinstance(res, PathResolution)
    assert res.file == tmp_path / "global" / "config" / "app.yaml"
    assert res.yaml_path == ["openai", "model"]


def test_resolve_app_path_project(tmp_path: Path) -> None:
    resolver = ConfigPathResolver(_paths(tmp_path))
    res = resolver.resolve("app.openai.model", scope=PathScope.PROJECT)
    assert res.file == tmp_path / "proj" / "config" / "app.yaml"


def test_resolve_agent_path(tmp_path: Path) -> None:
    resolver = ConfigPathResolver(_paths(tmp_path))
    res = resolver.resolve("agents.Lead.model", scope=PathScope.GLOBAL)
    assert res.file == tmp_path / "global" / "config" / "agents" / "Lead.yaml"
    assert res.yaml_path == ["model"]


def test_resolve_rejects_unknown_prefix(tmp_path: Path) -> None:
    resolver = ConfigPathResolver(_paths(tmp_path))
    with pytest.raises(ValueError):
        resolver.resolve("session.foo", scope=PathScope.GLOBAL)


def test_resolve_rejects_short_path(tmp_path: Path) -> None:
    resolver = ConfigPathResolver(_paths(tmp_path))
    with pytest.raises(ValueError):
        resolver.resolve("agents.Lead", scope=PathScope.GLOBAL)
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_config_paths.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
"""Resolve dotted config paths to (file, yaml_path, scope) tuples.

Dotted path conventions:

- ``app.<section>.<...>`` → ``<config_dir>/app.yaml``, with the leading
  ``app.`` stripped from the in-file YAML path.
- ``agents.<name>.<...>`` → ``<config_dir>/agents/<name>.yaml``, with
  ``agents.<name>.`` stripped from the in-file YAML path.

The resolver does not read or write the files — that is the writer's
job. Path inputs are validated for shape only (prefix + length).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from feather.paths import FeatherPaths


class PathScope(str, Enum):
    """Which YAML file the resolver targets."""

    GLOBAL = "global"
    PROJECT = "project"


@dataclass(slots=True, frozen=True)
class PathResolution:
    """Result of resolving a dotted config path."""

    file: Path
    yaml_path: list[str]
    scope: PathScope


class ConfigPathResolver:
    """Map dotted config paths to filesystem + YAML coordinates."""

    def __init__(self, paths: FeatherPaths) -> None:
        self._paths = paths

    def resolve(self, dotted: str, *, scope: PathScope) -> PathResolution:
        """Return the file and in-file YAML path for ``dotted``.

        Args:
            dotted: Path like ``app.openai.model`` or ``agents.Lead.model``.
            scope: Whether to target the global overlay or the project file.

        Returns:
            Resolution with the absolute file path and the residual
            in-file YAML path.

        Raises:
            ValueError: If ``dotted`` is too short or uses an unknown
                top-level prefix.
        """

        parts = dotted.split(".")
        if len(parts) < 3:
            raise ValueError(
                f"config path must have at least 3 segments, got {dotted!r}"
            )

        head, *rest = parts
        if head == "app":
            base = self._app_yaml_dir(scope) / "app.yaml"
            return PathResolution(file=base, yaml_path=rest, scope=scope)

        if head == "agents":
            agent_name, *yaml_path = rest
            base = self._app_yaml_dir(scope) / "agents" / f"{agent_name}.yaml"
            return PathResolution(file=base, yaml_path=yaml_path, scope=scope)

        raise ValueError(f"unknown config path prefix: {head!r}")

    def _app_yaml_dir(self, scope: PathScope) -> Path:
        if scope is PathScope.GLOBAL:
            return Path(self._paths.global_config_dir)
        return Path(self._paths.project_root) / "config"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config_paths.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_config_paths.py src/feather/config_paths.py
git commit -m "Add dotted-path resolver for app and agent scopes"
```

---

## 1C — Config writer

### Task 10: Add `ruamel.yaml` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dep**

Edit `pyproject.toml`. Locate the `dependencies = [...]` list and append:

```
  "ruamel.yaml>=0.18,<0.19",
```

- [ ] **Step 2: Sync**

Run: `uv sync 2>&1 | tail -5`
Expected: includes `+ ruamel-yaml==0.18.x`.

- [ ] **Step 3: Smoke import**

Run: `uv run python -c "from ruamel.yaml import YAML; YAML(typ='rt').dump({'a':1}, __import__('sys').stdout)"`
Expected: prints `a: 1`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add ruamel.yaml dependency for round-trip config writes"
```

---

### Task 11: Writer — strict line-walker for known leaf shapes

**Files:**
- Create: `src/feather/config_writer.py`
- Create: `tests/test_config_writer.py`

- [ ] **Step 1: Failing test**

```python
"""Tests for the comment-preserving config writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.config_writer import write_yaml_value


def test_writer_preserves_inline_comment(tmp_path: Path) -> None:
    src = tmp_path / "app.yaml"
    src.write_text(
        "openai:\n"
        "  model: gpt-5-mini   # default model\n"
        "  temperature: 1.0\n",
        encoding="utf-8",
    )

    write_yaml_value(src, ["openai", "model"], "gpt-5")

    text = src.read_text(encoding="utf-8")
    assert "model: gpt-5   # default model\n" in text
    assert "temperature: 1.0\n" in text


def test_writer_preserves_block_comments_above(tmp_path: Path) -> None:
    src = tmp_path / "app.yaml"
    src.write_text(
        "# top comment\n"
        "openai:\n"
        "  # provider-level note\n"
        "  model: gpt-5-mini\n",
        encoding="utf-8",
    )

    write_yaml_value(src, ["openai", "model"], "gpt-5")

    text = src.read_text(encoding="utf-8")
    assert text.startswith("# top comment\n")
    assert "# provider-level note\n" in text


def test_writer_handles_boolean_lower_case(tmp_path: Path) -> None:
    src = tmp_path / "app.yaml"
    src.write_text("memory:\n  enabled: false\n", encoding="utf-8")

    write_yaml_value(src, ["memory", "enabled"], True)

    assert src.read_text(encoding="utf-8") == "memory:\n  enabled: true\n"


def test_writer_handles_integers(tmp_path: Path) -> None:
    src = tmp_path / "app.yaml"
    src.write_text("memory:\n  retrieval:\n    top_k_tool: 10\n", encoding="utf-8")

    write_yaml_value(src, ["memory", "retrieval", "top_k_tool"], 25)

    assert "top_k_tool: 25\n" in src.read_text(encoding="utf-8")


def test_writer_handles_strings_with_quoting(tmp_path: Path) -> None:
    src = tmp_path / "app.yaml"
    src.write_text("logging:\n  level: INFO\n", encoding="utf-8")

    write_yaml_value(src, ["logging", "level"], "DEBUG")

    assert "level: DEBUG\n" in src.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_config_writer.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement using ruamel.yaml round-trip**

```python
"""Comment-preserving writer for Feather config YAML.

Uses ``ruamel.yaml`` in round-trip mode so existing comments, blank
lines, and key ordering survive a write. Writes are atomic via
tmp-file + rename so an interrupted write never leaves a half-written
``app.yaml`` on disk.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


def _yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def write_yaml_value(file: Path, yaml_path: list[str], value: Any) -> None:
    """Write ``value`` into ``file`` at the dotted ``yaml_path``.

    If ``file`` does not exist, it is created with an empty mapping
    before the write. Intermediate mappings are created as needed
    (so a first write of a nested leaf works on a sparse global
    overlay). Comments and ordering on existing nodes are preserved.

    Args:
        file: Target YAML file.
        yaml_path: Sequence of keys leading to the leaf.
        value: New scalar / list value to write.

    Raises:
        ValueError: If ``yaml_path`` is empty.
    """

    if not yaml_path:
        raise ValueError("yaml_path must be non-empty")

    file.parent.mkdir(parents=True, exist_ok=True)
    yaml = _yaml()
    if file.exists():
        data = yaml.load(file.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    cursor: Any = data
    for key in yaml_path[:-1]:
        existing = cursor.get(key)
        if existing is None:
            cursor[key] = {}
        cursor = cursor[key]
    cursor[yaml_path[-1]] = value

    buffer = io.StringIO()
    yaml.dump(data, buffer)
    new_text = buffer.getvalue()

    tmp = file.with_suffix(file.suffix + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(file)


def delete_yaml_value(file: Path, yaml_path: list[str]) -> bool:
    """Remove the leaf at ``yaml_path`` from ``file``.

    Args:
        file: Target YAML file.
        yaml_path: Sequence of keys leading to the leaf.

    Returns:
        True if a value was removed; False if the file or key did not
        exist.
    """

    if not file.exists():
        return False
    yaml = _yaml()
    data = yaml.load(file.read_text(encoding="utf-8")) or {}

    cursor: Any = data
    for key in yaml_path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            return False
        cursor = cursor[key]
    if not isinstance(cursor, dict) or yaml_path[-1] not in cursor:
        return False
    del cursor[yaml_path[-1]]

    buffer = io.StringIO()
    yaml.dump(data, buffer)
    tmp = file.with_suffix(file.suffix + ".tmp")
    tmp.write_text(buffer.getvalue(), encoding="utf-8")
    tmp.replace(file)
    return True
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config_writer.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_config_writer.py src/feather/config_writer.py
git commit -m "Add ruamel-backed config writer with atomic tmp+rename"
```

---

### Task 12: Writer — create-nested-key and atomicity tests

**Files:**
- Modify: `tests/test_config_writer.py`

- [ ] **Step 1: Append failing tests**

```python
def test_writer_creates_nested_path_in_empty_file(tmp_path: Path) -> None:
    src = tmp_path / "fresh.yaml"

    write_yaml_value(src, ["openai", "reasoning", "effort"], "high")

    text = src.read_text(encoding="utf-8")
    assert "openai:" in text
    assert "effort: high" in text


def test_writer_creates_intermediate_node_in_existing_file(tmp_path: Path) -> None:
    src = tmp_path / "app.yaml"
    src.write_text("openai:\n  model: gpt-5-mini\n", encoding="utf-8")

    write_yaml_value(src, ["openai", "reasoning", "effort"], "high")

    text = src.read_text(encoding="utf-8")
    assert "model: gpt-5-mini" in text
    assert "effort: high" in text


def test_writer_atomic_no_partial_on_failure(tmp_path: Path, monkeypatch) -> None:
    """If the dump raises, the original file is untouched."""

    src = tmp_path / "app.yaml"
    src.write_text("openai:\n  model: original\n", encoding="utf-8")

    from feather import config_writer as cw

    def boom(*args, **kwargs):
        raise RuntimeError("simulated dump failure")

    monkeypatch.setattr(cw.YAML, "dump", boom)

    with pytest.raises(RuntimeError):
        write_yaml_value(src, ["openai", "model"], "gpt-5")

    assert "model: original\n" in src.read_text(encoding="utf-8")


def test_delete_removes_leaf(tmp_path: Path) -> None:
    from feather.config_writer import delete_yaml_value

    src = tmp_path / "app.yaml"
    src.write_text(
        "openai:\n  model: gpt-5-mini\n  temperature: 1.0\n", encoding="utf-8"
    )

    assert delete_yaml_value(src, ["openai", "model"]) is True
    text = src.read_text(encoding="utf-8")
    assert "model" not in text
    assert "temperature: 1.0" in text


def test_delete_missing_leaf_returns_false(tmp_path: Path) -> None:
    from feather.config_writer import delete_yaml_value

    src = tmp_path / "app.yaml"
    src.write_text("openai:\n  model: gpt-5-mini\n", encoding="utf-8")

    assert delete_yaml_value(src, ["openai", "missing"]) is False
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_config_writer.py -v`
Expected: 10 passed (5 from Task 11 + 5 new).

- [ ] **Step 3: Commit**

```bash
git add tests/test_config_writer.py
git commit -m "Cover writer nested-key creation, atomic rollback, and delete"
```

---

## 1D — app.yaml cleanup (packaged default reshape)

### Task 13: Reorder packaged default + group memory.operations.*

**Files:**
- Modify: `src/feather/_resources/config/app.yaml`
- Modify: `src/feather/config.py`
- Modify: `tests/test_config.py` or new `tests/test_config_layered.py`

- [ ] **Step 1: Failing test — loader reads both legacy flat and new grouped memory shapes**

Append to `tests/test_config.py`:

```python
def test_loader_reads_grouped_memory_operations(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "app.yaml").write_text(
        _GROUPED_MEMORY_YAML, encoding="utf-8"
    )

    cfg = load_app_config(tmp_path)

    assert cfg.memory.extraction.provider == "openai"
    assert cfg.memory.extraction.model == "gpt-5.4-nano"
    assert cfg.memory.classification.model == "gpt-5.4-nano"
    assert cfg.memory.query_builder.model == "gpt-5.4-nano"


def test_loader_still_reads_legacy_flat_memory_operations(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "app.yaml").write_text(
        _FLAT_MEMORY_YAML, encoding="utf-8"
    )

    cfg = load_app_config(tmp_path)

    assert cfg.memory.extraction.provider == "openai"
```

Add to `tests/test_config.py` near the top (after imports):

```python
_GROUPED_MEMORY_YAML = """\
database: { path: feather.db }
storage: { temp_directory: tmp }
logging: { path: log, level: INFO }
compaction: { enabled: true, trigger_ratio: 0.8, context_window_tokens: 100, model: null, max_output_tokens: 100, temperature: 0.2 }
skills: { directory: skills }
scheduler: { enabled: true, poll_interval_seconds: 2, failure_retry_seconds: 30, max_due_jobs_per_tick: 10 }
openai:
  api_key_env: OPENAI_API_KEY
  model: gpt-5-mini
  max_output_tokens: 100
  temperature: 1.0
  parallel_tool_calls: true
memory:
  enabled: true
  operations:
    extraction:
      provider: openai
      model: gpt-5.4-nano
      max_output_tokens: 100
      temperature: 0.1
    classification:
      provider: openai
      model: gpt-5.4-nano
    query_builder:
      provider: openai
      model: gpt-5.4-nano
"""

_FLAT_MEMORY_YAML = _GROUPED_MEMORY_YAML.replace(
    "operations:\n    extraction:",
    "extraction:",
).replace("    classification:", "  classification:").replace(
    "    query_builder:", "  query_builder:"
)
```

- [ ] **Step 2: Run — expect failure (loader does not yet read `memory.operations.*`)**

Run: `uv run pytest tests/test_config.py::test_loader_reads_grouped_memory_operations -v`
Expected: FAIL.

- [ ] **Step 3: Update loader to accept both shapes**

In `src/feather/config.py::_parse_memory_config`, after `trigger_raw` is set, replace the three per-operation lines:

```python
    operations_raw = raw.get("operations") or {}
    extraction_raw = operations_raw.get("extraction") or raw.get("extraction") or {}
    classification_raw = operations_raw.get("classification") or raw.get("classification") or {}
    query_builder_raw = operations_raw.get("query_builder") or raw.get("query_builder") or {}
    extraction = _parse_operation_model(
        extraction_raw, default_max=2000, default_temp=0.1
    )
    classification = _parse_operation_model(
        classification_raw, default_max=2000, default_temp=0.1
    )
    query_builder = _parse_operation_model(
        query_builder_raw, default_max=2000, default_temp=0.1
    )
```

Also emit a deprecation log once per process when the flat shape is detected. Add at module top:

```python
_FLAT_MEMORY_OPS_WARNED = False
```

And in `_parse_memory_config`, after the operations resolution:

```python
    global _FLAT_MEMORY_OPS_WARNED
    if not operations_raw and any(
        raw.get(k) for k in ("extraction", "classification", "query_builder")
    ):
        if not _FLAT_MEMORY_OPS_WARNED:
            import logging

            logging.getLogger(__name__).warning(
                "memory.{extraction,classification,query_builder} flat shape is "
                "deprecated; move under memory.operations.{...} (loader still "
                "accepts both for now)."
            )
            _FLAT_MEMORY_OPS_WARNED = True
```

- [ ] **Step 4: Run loader tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: green for both new tests + existing.

- [ ] **Step 5: Reorder the packaged default**

Rewrite `src/feather/_resources/config/app.yaml` so the section order is exactly:

1. `database`
2. `storage`
3. `logging`
4. `compaction`
5. `skills`
6. `scheduler`
7. `self_repair`
8. `active_provider`
9. `openai`
10. `openrouter`
11. `claude`
12. `parallel`
13. `mcp`
14. `memory` (with `memory.operations.{extraction,classification,query_builder}` instead of flat)

Keep every existing comment and value; only the order and the operations grouping change. The MCP comment-examples (current lines 116-149) stay for now — extraction happens in Task 14.

- [ ] **Step 6: Verify loader still produces the same `AppConfig` from the reordered file**

Run: `uv run pytest tests/test_config.py tests/test_config_layered.py -v`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/feather/config.py src/feather/_resources/config/app.yaml tests/test_config.py
git commit -m "Reorder packaged app.yaml and group memory.operations.*"
```

---

### Task 14: Extract MCP example comments into a separate resource

**Files:**
- Modify: `src/feather/_resources/config/app.yaml`
- Create: `src/feather/_resources/config/examples/mcp.example.yaml`

- [ ] **Step 1: Move the example MCP servers to a new file**

Create `src/feather/_resources/config/examples/mcp.example.yaml`:

```yaml
# Example MCP server registrations. Copy any block into your
# app.yaml under ``mcp.servers`` to activate.

# HTTP MCP — OpenAI can receive HTTP MCPs natively;
# OpenRouter uses Feather's proxy after session registration.
docs:
  url: https://developers.openai.com/mcp
  description: OpenAI developer documentation
  providers: [openai, openrouter]
  agents: [Lead]
  allowed_tools: [search_openai_docs, fetch_openai_doc]
  require_approval: never

# stdio MCP — Feather starts the command on demand.
playwright:
  command: npx
  args: ["-y", "@playwright/mcp@latest"]
  description: Browser automation through Playwright
  providers: [openai, openrouter]
  agents: [Lead]
  require_approval: never

# Private HTTP MCP with auth header from env.
# Set PRIVATE_DOCS_AUTH_HEADER='Bearer <token>' before launching feather.
private_docs:
  url: https://example.com/mcp
  description: Private company documentation
  providers: [openai]
  agents: [Lead]
  header_envs:
    Authorization: PRIVATE_DOCS_AUTH_HEADER
  require_approval: never
```

- [ ] **Step 2: Replace the MCP block in `app.yaml`**

Find the `mcp:` block in `src/feather/_resources/config/app.yaml` and replace it with:

```yaml
mcp:
  enabled: false
  servers: {}
  # See _resources/config/examples/mcp.example.yaml for HTTP and stdio templates.
```

- [ ] **Step 3: Verify the loader still parses fine**

Run: `uv run pytest tests/test_config.py tests/test_mcp_config.py -v`
Expected: green.

- [ ] **Step 4: Commit**

```bash
git add src/feather/_resources/config/app.yaml src/feather/_resources/config/examples/mcp.example.yaml
git commit -m "Extract MCP example comments into a separate resource"
```

---

## 1E — ConfigService

### Task 15: `ConfigService.get` — read current value + source

**Files:**
- Create: `src/feather/config_service.py`
- Create: `tests/test_config_service.py`

- [ ] **Step 1: Failing test**

```python
"""Tests for the ConfigService orchestration layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.config import load_app_config
from feather.config_service import ConfigService, ValueSource
from feather.paths import FeatherPaths


def _service(tmp_path: Path) -> ConfigService:
    paths = FeatherPaths(project_root=tmp_path / "proj", home=tmp_path / "global")
    paths.ensure_global_dirs()
    paths.ensure_project_dirs()
    cfg = load_app_config(paths.project_root, paths=paths)
    return ConfigService(paths=paths, app_config=cfg)


def test_get_returns_default_source_when_no_override(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    value = svc.get("app.active_provider")

    assert value.source == ValueSource.DEFAULT
    assert value.current in {"openai", "openrouter", "claude"}
    assert value.field.path == "app.active_provider"


def test_get_returns_global_source_when_override_present(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    (svc.paths.global_config_dir / "app.yaml").write_text(
        "active_provider: claude\n", encoding="utf-8"
    )

    # Re-create after writing so loader picks the new file
    svc2 = _service(tmp_path)
    value = svc2.get("app.active_provider")

    assert value.current == "claude"
    assert value.source == ValueSource.GLOBAL


def test_get_unknown_path_raises(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    with pytest.raises(KeyError):
        svc.get("app.nope.foo")
```

- [ ] **Step 2: Run — expect ImportError**

Run: `uv run pytest tests/test_config_service.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `ConfigService.get`**

```python
"""Orchestration layer for editable Feather configuration.

ConfigService is the single entry point the slash-command CLI and
the future Textual modal both call. It owns:

- field lookup (via :mod:`feather.config_schema`)
- value resolution (current value + source badge)
- write dispatch (via :mod:`feather.config_writer`)
- validation (via the registry's per-field validators + enum lists)

The service is intentionally synchronous; reload orchestration lives
on :class:`feather.runtime.FeatherRuntime`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from feather.config_paths import ConfigPathResolver, PathScope
from feather.config_schema import ConfigField, REGISTRY, lookup
from feather.models import AppConfig
from feather.paths import FeatherPaths


class ValueSource(str, Enum):
    """Where the current value came from."""

    DEFAULT = "default"
    GLOBAL = "global"
    PROJECT = "project"


@dataclass(slots=True, frozen=True)
class ConfigValue:
    """Result of :meth:`ConfigService.get`."""

    field: ConfigField
    current: Any
    source: ValueSource


@dataclass(slots=True, frozen=True)
class WriteResult:
    """Result of :meth:`ConfigService.set` or :meth:`reset`."""

    ok: bool
    path: str
    error: str | None = None


class ConfigService:
    """Read/write editable config through the registry."""

    def __init__(
        self,
        *,
        paths: FeatherPaths,
        app_config: AppConfig,
    ) -> None:
        self._paths = paths
        self._app_config = app_config
        self._resolver = ConfigPathResolver(paths)

    @property
    def paths(self) -> FeatherPaths:
        return self._paths

    def get(self, dotted: str) -> ConfigValue:
        """Resolve ``dotted`` to its current value + source badge.

        Args:
            dotted: A path that exists in :data:`REGISTRY`.

        Raises:
            KeyError: If the path is unknown.
        """

        field_def = lookup(dotted)
        if field_def is None:
            raise KeyError(dotted)
        current, source = self._resolve_value(field_def)
        return ConfigValue(field=field_def, current=current, source=source)

    # ----- internal value lookup ---------------------------------

    def _resolve_value(self, field_def: ConfigField) -> tuple[Any, ValueSource]:
        for scope, source_enum in (
            (PathScope.PROJECT, ValueSource.PROJECT),
            (PathScope.GLOBAL, ValueSource.GLOBAL),
        ):
            res = self._resolver.resolve(field_def.path, scope=scope)
            if res.file.exists():
                data = yaml.safe_load(res.file.read_text(encoding="utf-8")) or {}
                value = self._dig(data, res.yaml_path)
                if value is not None:
                    return value, source_enum
        # Fallback: read the resolved current value from the live
        # AppConfig (which already reflects packaged defaults).
        return self._dig_app_config(field_def.path), ValueSource.DEFAULT

    @staticmethod
    def _dig(data: dict[str, Any], yaml_path: list[str]) -> Any:
        cursor: Any = data
        for key in yaml_path:
            if not isinstance(cursor, dict) or key not in cursor:
                return None
            cursor = cursor[key]
        return cursor

    def _dig_app_config(self, dotted: str) -> Any:
        """Return the AppConfig leaf value for ``dotted``.

        Handles both ``app.*`` paths (walk AppConfig) and
        ``agents.<name>.*`` paths (load the resolved AgentConfig and
        walk it).
        """

        if dotted.startswith("app."):
            cursor: Any = self._app_config
            for part in dotted.split(".")[1:]:
                cursor = getattr(cursor, part)
            return cursor
        if dotted.startswith("agents."):
            parts = dotted.split(".")
            agent_name = parts[1]
            from feather.config import load_agent_config

            agent_cfg = load_agent_config(
                self._paths.project_root, agent_name, paths=self._paths
            )
            cursor = agent_cfg
            for part in parts[2:]:
                cursor = getattr(cursor, part)
            return cursor
        raise KeyError(dotted)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config_service.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_config_service.py src/feather/config_service.py
git commit -m "Add ConfigService.get with source-badge resolution"
```

---

### Task 16: `ConfigService.validate` + `ConfigService.set`

**Files:**
- Modify: `src/feather/config_service.py`
- Modify: `tests/test_config_service.py`

- [ ] **Step 1: Failing tests**

Append to `tests/test_config_service.py`:

```python
def test_validate_accepts_valid_enum(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.validate("app.active_provider", "claude")

    assert result.ok


def test_validate_rejects_unknown_enum(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.validate("app.active_provider", "anthropic")

    assert not result.ok
    assert "openai" in (result.error or "")


def test_validate_rejects_negative_for_positive_validator(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.validate("app.memory.retrieval.top_k_tool", -1)

    assert not result.ok


def test_validate_coerces_strings_to_int(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.validate("app.memory.retrieval.top_k_tool", "12")

    assert result.ok
    assert result.coerced == 12


def test_set_writes_to_global_by_default(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.set("app.active_provider", "claude")

    assert result.ok
    overlay = svc.paths.global_config_dir / "app.yaml"
    assert overlay.exists()
    assert "claude" in overlay.read_text(encoding="utf-8")


def test_set_writes_to_project_when_flagged(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.set("app.active_provider", "claude", scope=PathScope.PROJECT)

    assert result.ok
    proj = tmp_path / "proj" / "config" / "app.yaml"
    assert proj.exists()


def test_set_rejects_invalid_value(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.set("app.active_provider", "anthropic")

    assert not result.ok
    overlay = svc.paths.global_config_dir / "app.yaml"
    assert not overlay.exists() or "anthropic" not in overlay.read_text(encoding="utf-8")
```

Add to imports at top of `tests/test_config_service.py`:

```python
from feather.config_paths import PathScope
```

- [ ] **Step 2: Run — expect failures**

Run: `uv run pytest tests/test_config_service.py -v`
Expected: 7 failures on the new tests.

- [ ] **Step 3: Implement `validate` and `set`**

Append to `src/feather/config_service.py`:

```python
@dataclass(slots=True, frozen=True)
class ValidateResult:
    ok: bool
    coerced: Any = None
    error: str | None = None


def _coerce(value: Any, field_def: ConfigField) -> Any:
    from feather.config_schema import FieldType

    if field_def.type is FieldType.INTEGER:
        return int(value)
    if field_def.type is FieldType.FLOAT:
        return float(value)
    if field_def.type is FieldType.BOOLEAN:
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "yes", "on", "1"):
                return True
            if v in ("false", "no", "off", "0"):
                return False
            raise ValueError(f"not a boolean: {value!r}")
        return bool(value)
    if field_def.type is FieldType.STRING_LIST:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return list(value)
    if field_def.type is FieldType.ENUM:
        if value not in (field_def.enum or ()):
            raise ValueError(
                f"value {value!r} not in allowed: {sorted(field_def.enum or ())}"
            )
        return value
    return str(value)
```

And add these methods to `ConfigService`:

```python
    def validate(self, dotted: str, value: Any) -> ValidateResult:
        """Coerce + validate ``value`` against the registry entry for ``dotted``.

        Returns a :class:`ValidateResult` with ``ok=True`` and the
        coerced value, or ``ok=False`` and a human-readable error.
        """

        field_def = lookup(dotted)
        if field_def is None:
            return ValidateResult(ok=False, error=f"unknown path: {dotted}")

        try:
            coerced = _coerce(value, field_def)
        except (TypeError, ValueError) as exc:
            return ValidateResult(ok=False, error=str(exc))

        if field_def.validator is not None:
            try:
                field_def.validator(coerced)
            except ValueError as exc:
                return ValidateResult(ok=False, error=str(exc))

        return ValidateResult(ok=True, coerced=coerced)

    def set(
        self,
        dotted: str,
        value: Any,
        *,
        scope: PathScope = PathScope.GLOBAL,
    ) -> WriteResult:
        """Validate ``value`` and write it to the resolved YAML file."""

        from feather.config_writer import write_yaml_value

        validate = self.validate(dotted, value)
        if not validate.ok:
            return WriteResult(ok=False, path=dotted, error=validate.error)

        res = self._resolver.resolve(dotted, scope=scope)
        try:
            write_yaml_value(res.file, res.yaml_path, validate.coerced)
        except OSError as exc:
            return WriteResult(ok=False, path=dotted, error=str(exc))
        return WriteResult(ok=True, path=dotted)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config_service.py -v`
Expected: all 10 (3 from Task 15 + 7 new) pass.

- [ ] **Step 5: Commit**

```bash
git add src/feather/config_service.py tests/test_config_service.py
git commit -m "ConfigService: add validate() and scope-aware set()"
```

---

### Task 17: `ConfigService.list`, `diff`, `reset`

**Files:**
- Modify: `src/feather/config_service.py`
- Modify: `tests/test_config_service.py`

- [ ] **Step 1: Failing tests**

Append to `tests/test_config_service.py`:

```python
def test_list_filters_by_section(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    rows = svc.list(section="app.openai")

    paths = [r.field.path for r in rows]
    assert "app.openai.model" in paths
    assert all(p.startswith("app.openai") for p in paths)


def test_list_returns_all_when_section_blank(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    rows = svc.list(section="")

    assert len(rows) >= 50


def test_diff_shows_global_vs_default(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc.set("app.active_provider", "claude")
    svc.set("app.openai.temperature", 0.3)

    diff = svc.diff()

    assert "app.active_provider" in diff
    old, new = diff["app.active_provider"]
    assert new == "claude"
    assert old != new


def test_reset_removes_overlay(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc.set("app.active_provider", "claude")

    result = svc.reset("app.active_provider")

    assert result.ok
    diff = svc.diff()
    assert "app.active_provider" not in diff


def test_reset_no_op_when_no_overlay(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.reset("app.active_provider")

    assert result.ok
```

- [ ] **Step 2: Run — expect failures**

Run: `uv run pytest tests/test_config_service.py -v`
Expected: 5 failures.

- [ ] **Step 3: Implement**

Append to `src/feather/config_service.py`:

```python
@dataclass(slots=True, frozen=True)
class ConfigRow:
    """One row returned by :meth:`ConfigService.list`."""

    field: ConfigField
    current: Any
    source: ValueSource


    # ...inside ConfigService class:

    def list(self, section: str = "") -> list[ConfigRow]:
        """Return registry rows whose path starts with ``section``.

        Empty ``section`` returns every entry.
        """

        out: list[ConfigRow] = []
        for field_def in REGISTRY:
            if section and not field_def.path.startswith(section):
                continue
            current, source = self._resolve_value(field_def)
            out.append(ConfigRow(field=field_def, current=current, source=source))
        return out

    def diff(self) -> dict[str, tuple[Any, Any]]:
        """Return paths whose effective value differs from the packaged default.

        Maps ``path → (default_value, current_value)``.
        """

        out: dict[str, tuple[Any, Any]] = {}
        for field_def in REGISTRY:
            current, source = self._resolve_value(field_def)
            if source is ValueSource.DEFAULT:
                continue
            default = self._dig_app_config(field_def.path)
            # When the overlay restored the default explicitly, skip.
            if default == current:
                continue
            out[field_def.path] = (default, current)
        return out

    def reset(
        self, dotted: str, *, scope: PathScope = PathScope.GLOBAL
    ) -> WriteResult:
        """Remove the overlay value for ``dotted``."""

        field_def = lookup(dotted)
        if field_def is None:
            return WriteResult(ok=False, path=dotted, error=f"unknown path: {dotted}")
        from feather.config_writer import delete_yaml_value

        res = self._resolver.resolve(dotted, scope=scope)
        delete_yaml_value(res.file, res.yaml_path)
        return WriteResult(ok=True, path=dotted)
```

(Move `ConfigRow` definition to before the `ConfigService` class so the type hint resolves cleanly. The implementation block above is the intent — the engineer should keep dataclasses together at module top.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config_service.py -v`
Expected: all 15 pass.

- [ ] **Step 5: Commit**

```bash
git add src/feather/config_service.py tests/test_config_service.py
git commit -m "ConfigService: add list(), diff(), reset()"
```

---

## 1F — Runtime additions

### Task 18: `runtime.reload_config()` — swap `_app_config`

**Files:**
- Modify: `src/feather/runtime.py`
- Create: `tests/test_runtime_reload.py`

- [ ] **Step 1: Failing test**

```python
"""Tests for FeatherRuntime config reload primitives."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.runtime import FeatherRuntime


async def test_reload_config_swaps_app_config(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(project)
    try:
        assert runtime.config.active_provider == "openai"

        (project / "config" / "app.yaml").write_text(
            _MINIMAL_YAML.replace("active_provider: openai", "active_provider: claude"),
            encoding="utf-8",
        )

        await runtime.reload_config()
        assert runtime.config.active_provider == "claude"
    finally:
        await runtime.close()
```

Add `_MINIMAL_YAML` constant at the top of the test file — copy the `_GROUPED_MEMORY_YAML` constant from `tests/test_config.py` but with `active_provider: openai` and `memory.enabled: false` so the runtime starts without a Qdrant requirement.

- [ ] **Step 2: Run — expect AttributeError**

Run: `uv run pytest tests/test_runtime_reload.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `reload_config`**

Add to `src/feather/runtime.py` inside the `FeatherRuntime` class (just after the `config` property, around line 357):

```python
    async def reload_config(self) -> None:
        """Re-read app.yaml + global overlay from disk and swap _app_config.

        This is the LIVE-class reload path. Provider-bound state (HTTP
        clients, models, reasoning config) is NOT reconstructed — call
        :meth:`rebuild_agent` for NEXT_TURN-class changes.
        """

        from feather.paths import FeatherPaths

        # Re-derive paths to use the same resolution the constructor did.
        # The runtime stores ``_root`` already; paths uses the global
        # state dir from ~/.feather.
        paths = FeatherPaths(project_root=self._root)
        new_config = load_app_config(self._root, paths=paths)
        self._app_config = new_config
        logger.info(
            "runtime.config.reloaded active_provider=%s", new_config.active_provider
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_runtime_reload.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_runtime_reload.py src/feather/runtime.py
git commit -m "runtime.reload_config() swaps _app_config from disk"
```

---

### Task 19: `runtime.rebuild_agent()` — fresh provider + agent, preserved session

**Files:**
- Modify: `src/feather/runtime.py`
- Modify: `tests/test_runtime_reload.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_runtime_reload.py`:

```python
async def test_rebuild_agent_uses_new_provider_after_reload(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(project)
    try:
        agent_before = runtime.build_agent("lead")
        provider_before = id(agent_before._provider)

        (project / "config" / "app.yaml").write_text(
            _MINIMAL_YAML.replace("active_provider: openai", "active_provider: claude"),
            encoding="utf-8",
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        await runtime.reload_config()
        runtime.rebuild_agent("lead")

        agent_after = runtime.get_agent("lead")
        assert id(agent_after._provider) != provider_before
    finally:
        await runtime.close()
```

(This test relies on `runtime.get_agent(name)` and the agent storing the provider as `_provider`. If those internals differ, adapt the assertion to whatever the agent factory uses to expose its provider.)

- [ ] **Step 2: Run — expect AttributeError**

Run: `uv run pytest tests/test_runtime_reload.py::test_rebuild_agent_uses_new_provider_after_reload -v`
Expected: FAIL.

- [ ] **Step 3: Implement `rebuild_agent` and `get_agent`**

Inspect `feather/core/agent_factory.py::AgentFactory` to confirm the build signature. Assume `factory.build(name, app_config)` returns a `BaseAgent` and stores its provider on a `_provider` attribute (or similar — check before writing).

Add to `FeatherRuntime`:

```python
    def __init__(self, ... existing ... ) -> None:
        # ... existing body ...
        self._agents: dict[str, BaseAgent] = {}

    def build_agent(self, name: str) -> BaseAgent:
        """Build a fresh agent and remember it for later rebuild calls."""

        agent = self._agent_factory.build(name, self._app_config)
        self._agents[name] = agent
        return agent

    def get_agent(self, name: str) -> BaseAgent:
        if name not in self._agents:
            raise KeyError(f"agent {name!r} not yet built")
        return self._agents[name]

    def rebuild_agent(self, name: str) -> BaseAgent:
        """Reconstruct ``name`` (and its provider) against the current app_config.

        Session cursor (``last_response_id``) is owned by SessionStore
        rather than the agent instance, so the new agent picks up the
        in-flight conversation transparently on the next turn.
        """

        new_agent = self._agent_factory.build(name, self._app_config)
        self._agents[name] = new_agent
        logger.info("runtime.agent.rebuilt name=%s", name)
        return new_agent
```

(Find existing callers of `build_agent` and ensure they still work — the existing code calls `runtime.build_agent("lead")` and stores the result locally; the runtime now ALSO caches it. Non-breaking.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_runtime_reload.py -v`
Expected: 2 pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_runtime_reload.py src/feather/runtime.py
git commit -m "runtime.rebuild_agent() reconstructs agent + provider on demand"
```

---

### Task 20: `runtime.apply_config_change()` — in-process branch

**Files:**
- Modify: `src/feather/runtime.py`
- Modify: `tests/test_runtime_reload.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_runtime_reload.py`:

```python
async def test_apply_config_change_live_reload_only(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(project)
    try:
        (project / "config" / "app.yaml").write_text(
            _MINIMAL_YAML.replace("trigger_ratio: 0.8", "trigger_ratio: 0.5"),
            encoding="utf-8",
        )

        result = await runtime.apply_config_change(
            ["app.compaction.trigger_ratio"]
        )

        assert result.applied == ["app.compaction.trigger_ratio"]
        assert result.needs_restart_lead == []
        assert result.needs_restart_app == []
        assert runtime.config.compaction.trigger_ratio == 0.5
    finally:
        await runtime.close()


async def test_apply_config_change_next_turn_rebuilds_agent(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    runtime = await FeatherRuntime.create(project)
    try:
        agent_before = runtime.build_agent("lead")
        before_id = id(agent_before)

        (project / "config" / "app.yaml").write_text(
            _MINIMAL_YAML.replace("active_provider: openai", "active_provider: claude"),
            encoding="utf-8",
        )

        result = await runtime.apply_config_change(["app.active_provider"])

        assert "app.active_provider" in result.applied
        assert id(runtime.get_agent("lead")) != before_id
    finally:
        await runtime.close()


async def test_apply_config_change_flags_restart_lead(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(project)
    try:
        result = await runtime.apply_config_change(
            ["app.claude.request_timeout_seconds"]
        )

        assert "app.claude.request_timeout_seconds" in result.needs_restart_lead
    finally:
        await runtime.close()


async def test_apply_config_change_flags_restart_app(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(project)
    try:
        result = await runtime.apply_config_change(["app.database.path"])

        assert "app.database.path" in result.needs_restart_app
    finally:
        await runtime.close()
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_runtime_reload.py -v`
Expected: 4 new failures.

- [ ] **Step 3: Implement**

Add at the top of `src/feather/runtime.py` (with other imports):

```python
from feather.config_schema import ReloadClass, lookup as _lookup_field
```

Add a result dataclass and the method:

```python
@dataclass(slots=True, frozen=True)
class ConfigApplyResult:
    """Outcome of :meth:`FeatherRuntime.apply_config_change`."""

    applied: list[str]
    needs_restart_lead: list[str]
    needs_restart_app: list[str]
```

(Place it near the other module-level dataclasses, or in `feather/models.py` if you prefer to keep it in the shared module.)

And inside `FeatherRuntime`:

```python
    async def apply_config_change(
        self, changed_paths: list[str]
    ) -> ConfigApplyResult:
        """Apply the cumulative reload effect of ``changed_paths``.

        Looks up each path's :class:`ReloadClass` from the registry.
        - LIVE-only changes → ``reload_config()`` only.
        - Any NEXT_TURN → ``reload_config()`` + ``rebuild_agent()``.
        - RESTART_LEAD / RESTART_APP entries are surfaced in the
          returned result; the caller (TUI) shows the banner.

        Worker-mode dispatch is added in a later task (Task 21).
        """

        live: list[str] = []
        next_turn: list[str] = []
        restart_lead: list[str] = []
        restart_app: list[str] = []
        for path in changed_paths:
            field_def = _lookup_field(path)
            if field_def is None:
                continue
            bucket = {
                ReloadClass.LIVE: live,
                ReloadClass.NEXT_TURN: next_turn,
                ReloadClass.RESTART_LEAD: restart_lead,
                ReloadClass.RESTART_APP: restart_app,
            }[field_def.reload]
            bucket.append(path)

        applied = list(live)
        if live or next_turn:
            await self.reload_config()
            applied.extend(next_turn)
            if next_turn:
                for agent_name in list(self._agents):
                    self.rebuild_agent(agent_name)

        return ConfigApplyResult(
            applied=applied,
            needs_restart_lead=restart_lead,
            needs_restart_app=restart_app,
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_runtime_reload.py -v`
Expected: 6 pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_runtime_reload.py src/feather/runtime.py
git commit -m "runtime.apply_config_change() — in-process branch by reload class"
```

---

## 1G — Supervisor reload envelope

### Task 21: Define the reload envelope types

**Files:**
- Modify: `src/feather/core/lead_supervisor.py` (or wherever envelope types live — likely in `feather/core/lead_worker_core.py` or a shared protocol module)
- Modify: `tests/test_lead_supervisor.py`

- [ ] **Step 1: Explore — find envelope definitions**

Run: `grep -n "type.*run\|envelope\|enqueue_user_input\|resume_on_inbox" src/feather/core/lead_supervisor.py src/feather/core/lead_worker_core.py | head -40`

Identify the file that declares the request/response envelope dataclasses or TypedDicts. Adjust the file paths below if the structure differs from what's described here.

- [ ] **Step 2: Failing test**

Append to `tests/test_lead_supervisor.py` (create if absent):

```python
from feather.core.lead_supervisor import (
    ConfigReloadEnvelope,
    ConfigReloadAckEnvelope,
)


def test_config_reload_envelope_serialisation() -> None:
    env = ConfigReloadEnvelope(
        correlation_id="cid-1",
        changed_paths=["app.active_provider"],
        reload_class="next_turn",
    )
    encoded = env.to_dict()
    assert encoded == {
        "type": "reload_config",
        "correlation_id": "cid-1",
        "changed_paths": ["app.active_provider"],
        "reload_class": "next_turn",
    }


def test_config_reload_ack_envelope_serialisation() -> None:
    ack = ConfigReloadAckEnvelope(
        correlation_id="cid-1",
        ok=True,
        applied_paths=["app.active_provider"],
        error=None,
    )
    assert ack.to_dict() == {
        "type": "reload_config_ack",
        "correlation_id": "cid-1",
        "ok": True,
        "applied_paths": ["app.active_provider"],
        "error": None,
    }
```

- [ ] **Step 3: Run — expect ImportError**

Run: `uv run pytest tests/test_lead_supervisor.py::test_config_reload_envelope_serialisation -v`
Expected: FAIL.

- [ ] **Step 4: Implement envelope dataclasses**

Locate the existing envelope definitions in `src/feather/core/lead_supervisor.py` (or the shared protocol module) and append:

```python
@dataclass(slots=True, frozen=True)
class ConfigReloadEnvelope:
    """Supervisor → worker: please reload the config + (maybe) rebuild agent."""

    correlation_id: str
    changed_paths: list[str]
    reload_class: str  # "live" | "next_turn"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "reload_config",
            "correlation_id": self.correlation_id,
            "changed_paths": list(self.changed_paths),
            "reload_class": self.reload_class,
        }


@dataclass(slots=True, frozen=True)
class ConfigReloadAckEnvelope:
    """Worker → supervisor: reload attempt outcome."""

    correlation_id: str
    ok: bool
    applied_paths: list[str]
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "reload_config_ack",
            "correlation_id": self.correlation_id,
            "ok": self.ok,
            "applied_paths": list(self.applied_paths),
            "error": self.error,
        }
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_lead_supervisor.py -v`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/feather/core/lead_supervisor.py tests/test_lead_supervisor.py
git commit -m "Add ConfigReload request/ack envelopes"
```

---

### Task 22: `LeadSupervisor.request_config_reload()` — send + await ack

**Files:**
- Modify: `src/feather/core/lead_supervisor.py`
- Modify: `tests/test_lead_supervisor.py`

- [ ] **Step 1: Failing test (round-trip via a fake worker)**

Append to `tests/test_lead_supervisor.py`:

```python
import asyncio
import uuid

import pytest

from feather.core.lead_supervisor import LeadSupervisor


async def test_supervisor_request_config_reload_round_trip(monkeypatch) -> None:
    """The supervisor sends a reload_config envelope and awaits the ack."""

    sent: list[dict] = []

    async def fake_send(envelope: dict) -> None:
        sent.append(envelope)

    async def fake_recv(correlation_id: str) -> dict:
        return {
            "type": "reload_config_ack",
            "correlation_id": correlation_id,
            "ok": True,
            "applied_paths": envelope_paths,
            "error": None,
        }

    envelope_paths = ["app.active_provider"]

    sup = LeadSupervisor.__new__(LeadSupervisor)  # construct without running start()
    monkeypatch.setattr(sup, "_send_envelope", fake_send, raising=False)
    monkeypatch.setattr(sup, "_await_response", fake_recv, raising=False)

    ack = await sup.request_config_reload(
        changed_paths=envelope_paths, reload_class="next_turn"
    )

    assert sent[0]["type"] == "reload_config"
    assert sent[0]["changed_paths"] == envelope_paths
    assert ack.ok is True
    assert ack.applied_paths == envelope_paths
```

(Adapt the monkeypatched attribute names to whatever the actual `LeadSupervisor` uses for outbound send + correlated receive. Inspect the existing `enqueue_user_input` implementation to copy the right helpers.)

- [ ] **Step 2: Run — expect AttributeError**

Run: `uv run pytest tests/test_lead_supervisor.py::test_supervisor_request_config_reload_round_trip -v`
Expected: FAIL.

- [ ] **Step 3: Implement on `LeadSupervisor`**

Add to `LeadSupervisor`:

```python
    async def request_config_reload(
        self,
        *,
        changed_paths: list[str],
        reload_class: str,
        timeout_s: float = 10.0,
    ) -> ConfigReloadAckEnvelope:
        """Ask the worker to reload its app_config + (maybe) rebuild agent.

        Args:
            changed_paths: Paths that changed on disk.
            reload_class: ``"live"`` (no agent rebuild) or
                ``"next_turn"`` (rebuild after reload).
            timeout_s: Cap on how long to wait for the ack.

        Returns:
            The ack envelope (ok or error).
        """

        correlation_id = uuid.uuid4().hex
        envelope = ConfigReloadEnvelope(
            correlation_id=correlation_id,
            changed_paths=list(changed_paths),
            reload_class=reload_class,
        )
        await self._send_envelope(envelope.to_dict())
        raw = await asyncio.wait_for(
            self._await_response(correlation_id), timeout=timeout_s
        )
        return ConfigReloadAckEnvelope(
            correlation_id=raw["correlation_id"],
            ok=bool(raw["ok"]),
            applied_paths=list(raw.get("applied_paths") or []),
            error=raw.get("error"),
        )
```

(Cite the existing `_send_envelope` / `_await_response` helper names — adapt as needed.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_lead_supervisor.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/feather/core/lead_supervisor.py tests/test_lead_supervisor.py
git commit -m "Supervisor: request_config_reload() with correlated ack"
```

---

### Task 23: Worker handler — defer to turn boundary, validate, swap

**Files:**
- Modify: `src/feather/core/lead_worker_core.py`
- Modify: `tests/test_lead_worker.py` (or create)

- [ ] **Step 1: Explore — find the worker's envelope dispatch**

Run: `grep -n "_dispatch\|type.*run\|type.*resume_on_inbox\|_handle_envelope" src/feather/core/lead_worker_core.py | head -20`

Identify how the worker dispatches inbound envelope types. The reload handler hooks into the same dispatch.

- [ ] **Step 2: Failing test**

Append to `tests/test_lead_worker.py`:

```python
import json

import pytest

from feather.core.lead_worker_core import _handle_reload_config


async def test_worker_handles_reload_envelope(tmp_path, monkeypatch):
    """The worker reloads config and rebuilds agents on a NEXT_TURN reload."""

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    # Spin up a runtime + lead agent the same way lead_worker_entry does.
    from feather.runtime import FeatherRuntime

    runtime = await FeatherRuntime.create(project)
    runtime.build_agent("lead")

    (project / "config" / "app.yaml").write_text(
        _MINIMAL_YAML.replace("active_provider: openai", "active_provider: claude"),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    envelope = {
        "type": "reload_config",
        "correlation_id": "cid-1",
        "changed_paths": ["app.active_provider"],
        "reload_class": "next_turn",
    }

    ack = await _handle_reload_config(runtime, envelope)
    assert ack["ok"] is True
    assert ack["applied_paths"] == ["app.active_provider"]
    assert runtime.config.active_provider == "claude"

    await runtime.close()
```

Use the same `_MINIMAL_YAML` constant pattern as Task 18.

- [ ] **Step 3: Run — expect AttributeError**

Run: `uv run pytest tests/test_lead_worker.py::test_worker_handles_reload_envelope -v`
Expected: FAIL.

- [ ] **Step 4: Implement the worker-side handler**

Add to `src/feather/core/lead_worker_core.py`:

```python
async def _handle_reload_config(
    runtime: "FeatherRuntime", envelope: dict[str, Any]
) -> dict[str, Any]:
    """Worker-side reload handler.

    Defers responsibility for waiting until between turns to the
    surrounding event loop (which already serialises run / resume calls
    with this handler via the same dispatch queue). Validates the new
    config by attempting a throwaway agent rebuild before swapping;
    rolls back if validation fails.
    """

    correlation_id = envelope["correlation_id"]
    changed_paths: list[str] = list(envelope.get("changed_paths") or [])
    reload_class = envelope.get("reload_class") or "live"

    prior_config = runtime.config
    try:
        await runtime.reload_config()
        if reload_class == "next_turn":
            for name in list(runtime._agents):
                runtime.rebuild_agent(name)
    except Exception as exc:
        # Roll back — re-attach the prior config.
        runtime._app_config = prior_config
        return {
            "type": "reload_config_ack",
            "correlation_id": correlation_id,
            "ok": False,
            "applied_paths": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "type": "reload_config_ack",
        "correlation_id": correlation_id,
        "ok": True,
        "applied_paths": changed_paths,
        "error": None,
    }
```

Wire the new handler into the worker's envelope dispatch (the loop that already routes `type=="run"` etc.):

```python
        elif envelope.get("type") == "reload_config":
            response = await _handle_reload_config(runtime, envelope)
            await self._send_response(response)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_lead_worker.py -v`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add tests/test_lead_worker.py src/feather/core/lead_worker_core.py
git commit -m "Worker: handle reload_config envelope with validate-then-swap"
```

---

### Task 24: `runtime.apply_config_change()` — worker-mode branch

**Files:**
- Modify: `src/feather/runtime.py`
- Modify: `tests/test_runtime_reload.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_runtime_reload.py`:

```python
async def test_apply_config_change_routes_via_supervisor_when_set(
    tmp_path, monkeypatch
):
    """When a supervisor is attached, runtime fans the reload through it."""

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    runtime = await FeatherRuntime.create(project)

    fake_calls: list[dict] = []

    class FakeSupervisor:
        async def request_config_reload(self, *, changed_paths, reload_class):
            from feather.core.lead_supervisor import ConfigReloadAckEnvelope

            fake_calls.append(
                {"changed_paths": changed_paths, "reload_class": reload_class}
            )
            return ConfigReloadAckEnvelope(
                correlation_id="x",
                ok=True,
                applied_paths=list(changed_paths),
                error=None,
            )

    runtime.attach_supervisor(FakeSupervisor())

    result = await runtime.apply_config_change(["app.active_provider"])

    assert result.applied == ["app.active_provider"]
    assert fake_calls[0]["reload_class"] == "next_turn"

    await runtime.close()
```

- [ ] **Step 2: Run — expect AttributeError**

Run: `uv run pytest tests/test_runtime_reload.py::test_apply_config_change_routes_via_supervisor_when_set -v`
Expected: FAIL.

- [ ] **Step 3: Implement supervisor attachment + worker-mode fanout**

In `FeatherRuntime.__init__` add:

```python
        self._supervisor: Any = None  # set by attach_supervisor when worker mode is on
```

Add methods:

```python
    def attach_supervisor(self, supervisor: Any) -> None:
        """Wire the supervisor reference; called by the TUI when worker mode is active."""

        self._supervisor = supervisor

    def detach_supervisor(self) -> None:
        self._supervisor = None
```

Update `apply_config_change` — replace the existing body with:

```python
    async def apply_config_change(
        self, changed_paths: list[str]
    ) -> ConfigApplyResult:
        live, next_turn, restart_lead, restart_app = self._bucket(changed_paths)

        applied: list[str] = []
        if live or next_turn:
            if self._supervisor is not None:
                # Worker mode: ALSO reload + rebuild locally so the TUI
                # process's view of `runtime.config` stays in sync with
                # the worker's. Then send the envelope so the worker's
                # config gets refreshed too.
                await self.reload_config()
                if next_turn:
                    for name in list(self._agents):
                        self.rebuild_agent(name)
                ack = await self._supervisor.request_config_reload(
                    changed_paths=live + next_turn,
                    reload_class="next_turn" if next_turn else "live",
                )
                if not ack.ok:
                    return ConfigApplyResult(
                        applied=[],
                        needs_restart_lead=restart_lead,
                        needs_restart_app=restart_app,
                    )
                applied = list(ack.applied_paths)
            else:
                await self.reload_config()
                applied = list(live)
                if next_turn:
                    applied.extend(next_turn)
                    for name in list(self._agents):
                        self.rebuild_agent(name)

        return ConfigApplyResult(
            applied=applied,
            needs_restart_lead=restart_lead,
            needs_restart_app=restart_app,
        )

    def _bucket(
        self, changed_paths: list[str]
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        live: list[str] = []
        next_turn: list[str] = []
        restart_lead: list[str] = []
        restart_app: list[str] = []
        for path in changed_paths:
            field_def = _lookup_field(path)
            if field_def is None:
                continue
            bucket = {
                ReloadClass.LIVE: live,
                ReloadClass.NEXT_TURN: next_turn,
                ReloadClass.RESTART_LEAD: restart_lead,
                ReloadClass.RESTART_APP: restart_app,
            }[field_def.reload]
            bucket.append(path)
        return live, next_turn, restart_lead, restart_app
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_runtime_reload.py -v`
Expected: 7 pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_runtime_reload.py src/feather/runtime.py
git commit -m "runtime.apply_config_change: worker-mode fanout via supervisor"
```

---

### Task 25: Worker rejects invalid config and rolls back

**Files:**
- Modify: `tests/test_lead_worker.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_lead_worker.py`:

```python
async def test_worker_rolls_back_on_invalid_config(tmp_path, monkeypatch):
    """If rebuild_agent raises after reload, prior config is restored."""

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config").mkdir()
    (project / "config" / "app.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")

    from feather.runtime import FeatherRuntime

    runtime = await FeatherRuntime.create(project)
    runtime.build_agent("lead")

    # Write an unloadable model name so rebuild raises during provider init.
    (project / "config" / "app.yaml").write_text(
        _MINIMAL_YAML.replace("active_provider: openai", "active_provider: claude"),
        encoding="utf-8",
    )
    # Do NOT set ANTHROPIC_API_KEY → provider construction fails.

    envelope = {
        "type": "reload_config",
        "correlation_id": "cid-roll",
        "changed_paths": ["app.active_provider"],
        "reload_class": "next_turn",
    }

    ack = await _handle_reload_config(runtime, envelope)
    assert ack["ok"] is False
    assert "ANTHROPIC" in (ack["error"] or "")
    # Rollback: app_config reverted to the prior active_provider.
    assert runtime.config.active_provider == "openai"

    await runtime.close()
```

- [ ] **Step 2: Run — expect failure if not yet rolling back**

Run: `uv run pytest tests/test_lead_worker.py::test_worker_rolls_back_on_invalid_config -v`
Expected: PASS if Task 23's rollback works; FAIL if validation isn't covering provider construction. If failing, expand `_handle_reload_config` to perform a dry-run agent rebuild in a try-then-commit pattern.

- [ ] **Step 3: Tighten the handler to do dry-run validate first**

Replace `_handle_reload_config` (from Task 23) with the version below — it dry-runs the rebuild before mutating live agents:

```python
async def _handle_reload_config(
    runtime: "FeatherRuntime", envelope: dict[str, Any]
) -> dict[str, Any]:
    correlation_id = envelope["correlation_id"]
    changed_paths: list[str] = list(envelope.get("changed_paths") or [])
    reload_class = envelope.get("reload_class") or "live"

    prior_config = runtime.config
    try:
        await runtime.reload_config()
        if reload_class == "next_turn":
            # Dry-run: try to build each agent against the NEW config; abort
            # if any fails. Only commit on full success.
            for name in list(runtime._agents):
                runtime._agent_factory.build(name, runtime.config)
            for name in list(runtime._agents):
                runtime.rebuild_agent(name)
    except Exception as exc:
        runtime._app_config = prior_config
        return {
            "type": "reload_config_ack",
            "correlation_id": correlation_id,
            "ok": False,
            "applied_paths": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "type": "reload_config_ack",
        "correlation_id": correlation_id,
        "ok": True,
        "applied_paths": changed_paths,
        "error": None,
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_lead_worker.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_lead_worker.py src/feather/core/lead_worker_core.py
git commit -m "Worker: dry-run agent rebuild before committing reload"
```

---

## 1H — Slash command headless surface

### Task 26: Register `/config` + subcommand dispatcher

**Files:**
- Modify: `src/feather/slash_commands.py`
- Create: `src/feather/config_slash.py`
- Modify: `tests/test_slash_commands.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_slash_commands.py`:

```python
def test_default_registry_includes_config() -> None:
    registry = default_registry()
    cmd = registry.find("config")
    assert cmd is not None
    assert cmd.summary
    assert cmd.usage and "get" in cmd.usage and "set" in cmd.usage
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_slash_commands.py::test_default_registry_includes_config -v`
Expected: FAIL.

- [ ] **Step 3: Register the command**

Edit `src/feather/slash_commands.py::default_registry` — insert into the commands tuple (alphabetised before `copy`):

```python
        SlashCommand(
            name="config",
            summary="Browse and edit Feather application + agent config",
            usage="/config [get|set|list|diff|reset] <path> [value]",
            category="session",
        ),
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_slash_commands.py -v`
Expected: green.

- [ ] **Step 5: Implement the dispatcher skeleton**

Create `src/feather/config_slash.py`:

```python
"""Headless `/config` subcommand dispatcher.

The TUI's slash handler calls :func:`handle_config_command` with the
raw arg string. This module parses the subcommand and dispatches to
the appropriate :class:`feather.config_service.ConfigService` method,
returning a rendered string for the TUI to display.

Interactive (modal) handling is wired separately in
``feather.textual_tui`` (Phase 2). Phase 1 supports headless only.
"""

from __future__ import annotations

from dataclasses import dataclass

from feather.config_paths import PathScope
from feather.config_service import ConfigService


@dataclass(slots=True, frozen=True)
class ConfigCommandResult:
    """Outcome of one `/config <sub>` invocation."""

    ok: bool
    body: str
    requires_apply: list[str] | None = None  # paths to feed apply_config_change


def handle_config_command(
    service: ConfigService, args: str
) -> ConfigCommandResult:
    """Parse and dispatch one `/config <sub> [args]` invocation."""

    tokens = args.strip().split()
    if not tokens:
        # Interactive modal — wired in Phase 2.
        return ConfigCommandResult(
            ok=False, body="Interactive /config modal is not yet wired (Phase 2)."
        )

    sub, *rest = tokens
    if sub == "get":
        return _cmd_get(service, rest)
    if sub == "set":
        return _cmd_set(service, rest)
    if sub == "list":
        return _cmd_list(service, rest)
    if sub == "diff":
        return _cmd_diff(service, rest)
    if sub == "reset":
        return _cmd_reset(service, rest)
    return ConfigCommandResult(
        ok=False, body=f"unknown subcommand: {sub} (expected get|set|list|diff|reset)"
    )


def _cmd_get(service: ConfigService, rest: list[str]) -> ConfigCommandResult:
    if len(rest) != 1:
        return ConfigCommandResult(ok=False, body="usage: /config get <path>")
    path = rest[0]
    try:
        value = service.get(path)
    except KeyError:
        return ConfigCommandResult(ok=False, body=f"unknown path: {path}")
    body = f"{path} = {value.current!r}  [{value.source.value}]"
    return ConfigCommandResult(ok=True, body=body)


def _parse_scope(rest: list[str]) -> tuple[PathScope, list[str]]:
    scope = PathScope.GLOBAL
    remaining: list[str] = []
    for token in rest:
        if token == "--global":
            scope = PathScope.GLOBAL
        elif token == "--project":
            scope = PathScope.PROJECT
        else:
            remaining.append(token)
    return scope, remaining


def _cmd_set(service: ConfigService, rest: list[str]) -> ConfigCommandResult:
    scope, positional = _parse_scope(rest)
    if len(positional) < 2:
        return ConfigCommandResult(
            ok=False, body="usage: /config set <path> <value> [--project|--global]"
        )
    path, *value_parts = positional
    value = " ".join(value_parts)
    write = service.set(path, value, scope=scope)
    if not write.ok:
        return ConfigCommandResult(ok=False, body=f"{path}: {write.error}")
    return ConfigCommandResult(
        ok=True,
        body=f"{path} = {value} (saved to {scope.value})",
        requires_apply=[path],
    )


def _cmd_list(service: ConfigService, rest: list[str]) -> ConfigCommandResult:
    section = rest[0] if rest else ""
    rows = service.list(section=section)
    if not rows:
        return ConfigCommandResult(
            ok=True, body=f"no fields under {section!r}"
        )
    lines = [
        f"{row.field.path}  =  {row.current!r}  [{row.source.value}]"
        for row in rows
    ]
    return ConfigCommandResult(ok=True, body="\n".join(lines))


def _cmd_diff(service: ConfigService, rest: list[str]) -> ConfigCommandResult:
    diff = service.diff()
    if not diff:
        return ConfigCommandResult(ok=True, body="no overrides")
    lines = [f"{path}: {old!r} → {new!r}" for path, (old, new) in sorted(diff.items())]
    return ConfigCommandResult(ok=True, body="\n".join(lines))


def _cmd_reset(service: ConfigService, rest: list[str]) -> ConfigCommandResult:
    scope, positional = _parse_scope(rest)
    if len(positional) != 1:
        return ConfigCommandResult(
            ok=False, body="usage: /config reset <path> [--project|--global]"
        )
    path = positional[0]
    write = service.reset(path, scope=scope)
    if not write.ok:
        return ConfigCommandResult(ok=False, body=f"{path}: {write.error}")
    return ConfigCommandResult(
        ok=True,
        body=f"{path}: reset (scope={scope.value})",
        requires_apply=[path],
    )
```

- [ ] **Step 6: Commit**

```bash
git add src/feather/slash_commands.py src/feather/config_slash.py tests/test_slash_commands.py
git commit -m "Register /config + headless subcommand dispatcher skeleton"
```

---

### Task 27: Cover every `/config` subcommand with tests

**Files:**
- Create: `tests/test_config_slash.py`

- [ ] **Step 1: Write the dispatcher tests**

```python
"""Tests for the /config slash dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.config import load_app_config
from feather.config_paths import PathScope
from feather.config_service import ConfigService
from feather.config_slash import handle_config_command
from feather.paths import FeatherPaths


def _service(tmp_path: Path) -> ConfigService:
    paths = FeatherPaths(project_root=tmp_path / "proj", home=tmp_path / "global")
    paths.ensure_global_dirs()
    paths.ensure_project_dirs()
    cfg = load_app_config(paths.project_root, paths=paths)
    return ConfigService(paths=paths, app_config=cfg)


def test_get_returns_current_value(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "get app.active_provider")

    assert result.ok
    assert "app.active_provider" in result.body
    assert "[default]" in result.body


def test_get_unknown_path(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "get app.does.not.exist")

    assert not result.ok
    assert "unknown" in result.body.lower()


def test_set_writes_to_global(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "set app.active_provider claude")

    assert result.ok
    assert result.requires_apply == ["app.active_provider"]
    overlay = svc.paths.global_config_dir / "app.yaml"
    assert "claude" in overlay.read_text(encoding="utf-8")


def test_set_with_project_flag(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "set app.active_provider claude --project")

    assert result.ok
    proj = tmp_path / "proj" / "config" / "app.yaml"
    assert proj.exists()


def test_set_rejects_invalid(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "set app.active_provider anthropic")

    assert not result.ok


def test_list_section(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "list app.openai")

    assert result.ok
    assert "app.openai.model" in result.body


def test_diff_after_set(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    handle_config_command(svc, "set app.active_provider claude")

    result = handle_config_command(svc, "diff")

    assert result.ok
    assert "app.active_provider" in result.body


def test_reset_after_set(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    handle_config_command(svc, "set app.active_provider claude")

    result = handle_config_command(svc, "reset app.active_provider")

    assert result.ok
    diff = handle_config_command(svc, "diff").body
    assert "app.active_provider" not in diff


def test_unknown_subcommand(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "xyz")

    assert not result.ok
    assert "unknown" in result.body.lower()


def test_bare_config_returns_modal_pending(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "")

    assert not result.ok
    assert "Phase 2" in result.body
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_config_slash.py -v`
Expected: 10 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_config_slash.py
git commit -m "Cover /config get|set|list|diff|reset subcommand dispatch"
```

---

### Task 28: Wire `/config` into the textual TUI

**Files:**
- Modify: `src/feather/textual_tui.py`

- [ ] **Step 1: Smoke-test that the slash registry test now passes**

The existing `tests/test_slash_commands.py::test_default_registry_includes_config` (Task 26) is the binding-coverage guard. The TUI handler is exercised by the modal integration tests in Phase 2 (`tests/test_textual_config_screen.py::test_save_calls_apply_config_change`). No additional `tests/test_textual_tui.py` test is required for this task — instead, verify the existing TUI tests do not regress:

Run: `uv run pytest tests/test_textual_tui.py tests/test_slash_commands.py -v`
Expected: every existing test green; `test_default_registry_includes_config` passes.

- [ ] **Step 2: Implement the handler**

In `src/feather/textual_tui.py`, find `_register_default_handlers` (around line 1124) and add:

```python
            "config": self._cmd_config,
```

In the `accepts_args` opt-in set (around line 1178), add `"config"`.

Add the handler method:

```python
    def _cmd_config(self, args: str) -> None:
        """Dispatch the `/config <sub> [args]` slash command."""

        from feather.config_service import ConfigService
        from feather.config_slash import handle_config_command

        assert self._runtime is not None
        service = ConfigService(
            paths=self._paths,  # set by __init__ when paths-aware mode is on
            app_config=self._runtime.config,
        )
        result = handle_config_command(service, args)
        self._write_marker(
            "Config",
            result.body,
            style="cyan" if result.ok else "red",
        )

        if result.ok and result.requires_apply:
            async def _apply() -> None:
                outcome = await self._runtime.apply_config_change(
                    list(result.requires_apply or [])
                )
                msg_parts: list[str] = []
                if outcome.applied:
                    msg_parts.append(
                        f"Applied: {', '.join(outcome.applied)}"
                    )
                if outcome.needs_restart_lead:
                    msg_parts.append(
                        "Needs /restart-lead: "
                        + ", ".join(outcome.needs_restart_lead)
                    )
                if outcome.needs_restart_app:
                    msg_parts.append(
                        "Needs full restart: "
                        + ", ".join(outcome.needs_restart_app)
                    )
                self._write_marker(
                    "Config apply",
                    "\n".join(msg_parts) or "no changes applied",
                    style="cyan",
                )

            self.run_worker(_apply(), exclusive=False)
```

(`self._paths` is added if not already present — the TUI accepts a `paths` kwarg in newer code; if absent, instantiate a fresh `FeatherPaths(project_root=self._root)`.)

- [ ] **Step 3: Smoke-run the TUI test suite**

Run: `uv run pytest tests/test_textual_tui.py -v -k config`
Expected: green.

- [ ] **Step 4: Commit**

```bash
git add src/feather/textual_tui.py
git commit -m "Wire /config into the textual TUI slash dispatcher"
```

---

## 1I — Phase wrap

### Task 29: Re-run the full test suite

**Files:**
- (verification only)

- [ ] **Step 1: Run everything**

Run: `uv run pytest -x -q 2>&1 | tail -25`
Expected: 1190 prior tests + ~120 new tests all green.

- [ ] **Step 2: If anything red, fix in place and commit per-fix**

Each fix gets its own commit with `Fix: <what>` prefix. Do not bundle.

---

### Task 30: Simplify pass

**Files:**
- All Phase 1 changes.

- [ ] **Step 1: Invoke the simplifier on the new modules**

Dispatch `code-simplifier:code-simplifier` via the Agent tool. Brief:

> Simplify the Phase 1 additions for the config service feature:
> - `src/feather/config_schema.py`
> - `src/feather/config_paths.py`
> - `src/feather/config_writer.py`
> - `src/feather/config_service.py`
> - `src/feather/config_slash.py`
> - `src/feather/runtime.py` (new methods only — do not touch pre-existing code)
> - `src/feather/core/lead_supervisor.py` (envelope + request_config_reload only)
> - `src/feather/core/lead_worker_core.py` (_handle_reload_config only)
>
> Apply: remove duplicated coerce/parse logic, kill comments that explain WHAT rather than WHY, drop unused imports, collapse trivially-wrapping helpers. Do NOT change public API surface (registry path strings, ConfigService method names, slash subcommand names). Do NOT touch unrelated files.

- [ ] **Step 2: Re-run full suite after any simplifier changes**

Run: `uv run pytest -x -q 2>&1 | tail -10`
Expected: green.

- [ ] **Step 3: Commit per change-cluster**

```bash
git add -p
git commit -m "Simplify Phase 1 modules per code-simplifier pass"
```

(Skip if the simplifier returned no changes.)

---

### Task 31: Red-team code review

**Files:**
- All Phase 1 changes.

- [ ] **Step 1: Dispatch the reviewer**

Invoke `superpowers:code-reviewer` via the Agent tool. Brief:

> Red-team review of Phase 1 (config schema + writer + service + reload plumbing) against `docs/superpowers/specs/2026-05-11-config-tui-design.md`. Hunt:
>
> 1. **Concurrency:** The supervisor's `_send_envelope` / `_await_response` are shared with `enqueue_user_input` and `run` envelopes — is the reload envelope correctly serialised against an in-flight `run`? Does the worker's dispatch loop drain reload envelopes between turns or could a reload interleave with a streaming response?
> 2. **Rollback completeness:** When `_handle_reload_config` rolls back `_app_config`, does it also need to re-build any agents that may have been rebuilt PARTIALLY before the dry-run noticed the failure?
> 3. **YAML safety:** `ruamel.yaml`'s round-trip mode loads arbitrary YAML — could a maliciously-crafted overlay file run code, escalate file paths, or cause OOM? (Out of scope for a single-user TUI, but flag if any of the user-input → YAML write paths could let `/config set` smuggle YAML anchors or aliases.)
> 4. **Registry drift:** The tripwire test covers `AppConfig` + `AgentConfig` leaves. What about `MemoryConfig`'s sub-dataclasses? The recursive walker should cover them — double-check by running the test in dry-run and confirming `app.memory.qdrant.embedding_dims` etc. appear in the resolved leaf list.
> 5. **Project-vs-global semantics:** When `--project` writes a value and then `--global` writes a different value for the same path, what does `get` return? Spec says project beats global. Verify a test covers this.
> 6. **`self_repair.enabled` carve-out:** The spec requires `--force` for set on this field. Did we implement it? (Check `_cmd_set` and `ConfigService.set`.) **If missing, this is a blocker — add it.**
>
> Report blocking vs nit. Under 500 words.

- [ ] **Step 2: Address every BLOCKING finding**

For each blocker: failing test → fix → re-run all Phase 1 tests → commit with `Address red-team finding: <summary>`.

- [ ] **Step 3: Push**

```bash
git push origin feature/config-tui
```

---

## Phase 1 self-review checklist

- [ ] `ConfigField`/`ReloadClass`/`Scope`/`FieldType`/`WidgetHint` types defined (Task 1)
- [ ] Drift tripwire test covers `AppConfig` and `AgentConfig` (Tasks 2, 7)
- [ ] `IGNORED_PATHS` covers genuinely non-editable fields (Task 3)
- [ ] Registry covers all app.* and agents.* leaves (Tasks 4–7)
- [ ] Registry self-checks (Task 8)
- [ ] Path resolver handles app + agent scopes (Task 9)
- [ ] ruamel.yaml writer preserves comments, atomic, creates nested keys (Tasks 10–12)
- [ ] app.yaml reorder + memory.operations.* (Task 13)
- [ ] MCP examples extracted (Task 14)
- [ ] ConfigService.get with source badge (Task 15)
- [ ] ConfigService.validate + set (Task 16)
- [ ] ConfigService.list + diff + reset (Task 17)
- [ ] runtime.reload_config + rebuild_agent + apply_config_change in-process (Tasks 18–20)
- [ ] Supervisor reload envelope (Tasks 21–22)
- [ ] Worker reload handler with validate-then-swap (Tasks 23, 25)
- [ ] runtime.apply_config_change worker-mode fanout (Task 24)
- [ ] /config registered + dispatcher (Tasks 26–28)
- [ ] Full suite green (Task 29)
- [ ] Simplify pass (Task 30)
- [ ] Red-team review with blockers addressed (Task 31)
- [ ] `self_repair.enabled --force` requirement implemented (per Task 31 blocker — add to ConfigService.set + _cmd_set if not already covered)
