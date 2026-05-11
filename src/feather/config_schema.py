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

from dataclasses import dataclass
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
