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
        enum: Allowed values when ``type`` is ``ENUM`` — picker is strict
            (validator rejects anything outside this set).
        validator: Optional callable raising ``ValueError`` on bad value.
        sensitive: True for env-var indirection (read-only in modal).
        default: Documented default; ``None`` means "inherits dataclass default".
        hint: Short human-readable range/help (e.g. ``"0.0-1.0"``) shown in
            the modal's input placeholder. ``None`` falls back to a generic
            label.
        choices: Suggested values for a non-strict picker. Unlike ``enum``,
            free-form input is still accepted by validation; the picker only
            displays these as quick-pick options. Used for fields like model
            names where the catalog isn't exhaustive.
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
    hint: str | None = None
    choices: tuple[str, ...] | None = None
    #: When True the DROPDOWN choices are resolved by the modal at
    #: picker-open time (e.g. from the YAML model catalog or from a
    #: sibling provider field), so the registry doesn't ship them.
    #: Required to opt out of the "DROPDOWN must have choices" check.
    dynamic_choices: bool = False

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
        if (
            self.widget is WidgetHint.DROPDOWN
            and not (self.enum or self.choices)
            and not self.dynamic_choices
        ):
            # DROPDOWN must have something to display — either strict enum,
            # suggested choices, or an opt-in declaration that the modal
            # resolves them dynamically. Otherwise the picker has nothing
            # to show.
            raise ValueError(
                f"ConfigField {self.path!r}: DROPDOWN widget requires enum, "
                f"choices, or dynamic_choices=True"
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


# Mapping from validator identity to a human-readable range string.
# Used by ``hint_for()`` to provide a placeholder hint in the modal
# without having to backfill every numeric registry entry.
_VALIDATOR_HINTS: dict[Validator, str] = {
    _ratio: "0.0-1.0",
    _positive: "> 0",
    _non_negative_int: ">= 0",
}


def hint_for(field: "ConfigField") -> str | None:
    """Return the user-facing hint string for ``field``.

    Resolution order:
      1. Explicit ``field.hint`` if set.
      2. Validator-derived hint from :data:`_VALIDATOR_HINTS`.
      3. ``None`` (no hint).
    """

    if field.hint:
        return field.hint
    if field.validator is not None and field.validator in _VALIDATOR_HINTS:
        return _VALIDATOR_HINTS[field.validator]
    return None


# ---------------------------------------------------------------------------
# Model catalog (non-strict suggestions)
# ---------------------------------------------------------------------------
#
# The dropdown choices for ``app.<provider>.model``, ``agents.<name>.model``,
# and ``app.memory.operations.<op>.model`` are all sourced at picker-open
# time from the YAML catalog (``feather.config.model_catalog``) — see
# ``ConfigScreen._picker_choices_for``. That keeps every model picker
# pointing at one source of truth instead of forking a hard-coded
# ``MODEL_CATALOG`` here that had to be kept in sync by hand (it wasn't,
# and the app-level pickers ended up showing fewer slugs than the per-agent
# ones).


#: Sentinel value the modal's picker writes when the user chooses
#: "(inherit)" — translates to a reset of the overlay key, so the lower
#: layer's value (e.g. app.active_provider) takes effect. Defined here so
#: schema + modal + tests use the same string.
INHERIT_SENTINEL = "(inherit)"

#: Choices for any field that overrides a provider with an inherit option:
#: agents.<name>.provider AND app.memory.operations.<op>.provider. Picker
#: order matches the conceptual hierarchy (inherit first, then concrete
#: providers in their canonical app-level ordering).
_PROVIDER_OVERRIDE_CHOICES: tuple[str, ...] = (
    INHERIT_SENTINEL,
    "openai",
    "openrouter",
    "claude",
)


def _url(v: str) -> None:
    """Validator: value must look like an HTTP(S) URL.

    Doesn't go beyond the prefix check — full URL parsing belongs to the
    HTTP client; this catches the obvious "user pasted ``api.example.com``"
    mistake at edit time.
    """

    if not isinstance(v, str) or not v.strip():
        raise ValueError("URL must be a non-empty string")
    if not v.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")


#: Suggested Gemini ``task_type`` values for the embedding API.
GEMINI_TASK_TYPES: tuple[str, ...] = (
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
)

#: Suggested tiktoken encoding names — newest first.
TIKTOKEN_ENCODINGS: tuple[str, ...] = (
    "o200k_base",
    "cl100k_base",
    "p50k_base",
    "r50k_base",
)

#: Suggested embedding model slugs — kept short; user can still type any
#: model via /config set since these are choices (non-strict).
EMBEDDING_MODEL_CATALOG: tuple[str, ...] = (
    "gemini-embedding-2-preview",
    "text-embedding-3-large",
    "text-embedding-3-small",
)


def _agent_fields(name: str) -> tuple[ConfigField, ...]:
    """Return the 9 standard per-agent :class:`ConfigField` entries for ``name``.

    Args:
        name: Agent name as it appears in the YAML ``name:`` field (e.g.
            ``"Lead"``, ``"Explore"``).  The exact capitalisation is used
            in all generated paths so it must match the agent YAML.

    Returns:
        Tuple of nine :class:`ConfigField` instances covering personality,
        memory_enabled, provider, model, temperature, max_output_tokens,
        reasoning.effort, reasoning.summary, and registered_tools.
    """

    return (
        ConfigField(
            path=f"agents.{name}.personality",
            type=FieldType.STRING,
            widget=WidgetHint.TEXT,
            reload=ReloadClass.NEXT_TURN,
            scope=Scope.AGENT,
            description="One-line personality / voice description.",
        ),
        ConfigField(
            path=f"agents.{name}.memory_enabled",
            type=FieldType.BOOLEAN,
            widget=WidgetHint.TOGGLE,
            reload=ReloadClass.NEXT_TURN,
            scope=Scope.AGENT,
            description="Agent-level memory opt-in (also gated by app.memory.enabled).",
        ),
        ConfigField(
            path=f"agents.{name}.provider",
            type=FieldType.STRING,
            widget=WidgetHint.DROPDOWN,
            reload=ReloadClass.NEXT_TURN,
            scope=Scope.AGENT,
            description="Override the app-level provider for this agent (pick (inherit) to use app default).",
            choices=_PROVIDER_OVERRIDE_CHOICES,
        ),
        ConfigField(
            path=f"agents.{name}.model",
            type=FieldType.STRING,
            widget=WidgetHint.DROPDOWN,
            reload=ReloadClass.NEXT_TURN,
            scope=Scope.AGENT,
            description="Override the provider's default model for this agent (pick (inherit) to use provider default).",
            # Choices are computed dynamically by the modal at picker-open
            # time — they depend on the agent's resolved provider, which
            # the schema cannot know.
            dynamic_choices=True,
        ),
        ConfigField(
            path=f"agents.{name}.temperature",
            type=FieldType.FLOAT,
            widget=WidgetHint.NUMERIC,
            reload=ReloadClass.NEXT_TURN,
            scope=Scope.AGENT,
            description=(
                "Per-agent sampling temperature override (blank inherits the "
                "provider's app-level temperature)."
            ),
        ),
        ConfigField(
            path=f"agents.{name}.max_output_tokens",
            type=FieldType.INTEGER,
            widget=WidgetHint.NUMERIC,
            reload=ReloadClass.NEXT_TURN,
            scope=Scope.AGENT,
            description=(
                "Per-agent max output tokens override (blank inherits the "
                "provider's app-level max_output_tokens)."
            ),
            validator=_positive,
        ),
        ConfigField(
            path=f"agents.{name}.reasoning.effort",
            type=FieldType.ENUM,
            widget=WidgetHint.DROPDOWN,
            reload=ReloadClass.NEXT_TURN,
            scope=Scope.AGENT,
            description="Per-agent reasoning effort override.",
            enum=("none", "minimal", "low", "medium", "high"),
        ),
        ConfigField(
            path=f"agents.{name}.reasoning.summary",
            type=FieldType.ENUM,
            widget=WidgetHint.DROPDOWN,
            reload=ReloadClass.NEXT_TURN,
            scope=Scope.AGENT,
            description="Per-agent reasoning summary verbosity.",
            enum=("auto", "concise", "detailed"),
        ),
        ConfigField(
            path=f"agents.{name}.registered_tools",
            type=FieldType.STRING_LIST,
            widget=WidgetHint.LIST_EDITOR,
            reload=ReloadClass.NEXT_TURN,
            scope=Scope.AGENT,
            description="List of tool names this agent may call.",
        ),
    )


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
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Default OpenAI model (e.g. gpt-5-mini).",
        # Choices resolved from the YAML catalog at picker-open time.
        dynamic_choices=True,
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
        validator=_url,
        hint="https://...",
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
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Default OpenRouter model slug.",
        # Choices resolved from the YAML catalog at picker-open time.
        dynamic_choices=True,
    ),
    ConfigField(
        path="app.openrouter.fallback_models",
        type=FieldType.STRING_LIST,
        widget=WidgetHint.LIST_EDITOR,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description=(
            "OpenRouter ``models`` array — routes to the first available slug "
            "when the primary model 4xx/5xx's. Leave empty unless you "
            "explicitly want silent substitution; otherwise OpenRouter may "
            "swap your selection out without telling you."
        ),
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
        validator=_url,
        hint="https://...",
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
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.NEXT_TURN,
        scope=Scope.APP,
        description="Default Anthropic model (e.g. claude-opus-4-7).",
        # Choices resolved from the YAML catalog at picker-open time.
        dynamic_choices=True,
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
        validator=_url,
        hint="http(s)://host:port",
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
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Embedding model name.",
        choices=EMBEDDING_MODEL_CATALOG,
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
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Gemini task_type for document embeddings.",
        choices=GEMINI_TASK_TYPES,
    ),
    ConfigField(
        path="app.memory.embedding.task_type_query",
        type=FieldType.STRING,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Gemini task_type for query embeddings.",
        choices=GEMINI_TASK_TYPES,
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
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Tokenizer encoding (e.g. o200k_base).",
        choices=TIKTOKEN_ENCODINGS,
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
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Provider for memory extraction (pick (inherit) to use active_provider).",
        choices=_PROVIDER_OVERRIDE_CHOICES,
    ),
    ConfigField(
        path="app.memory.operations.extraction.model",
        type=FieldType.STRING,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Model for memory extraction (pick (inherit) to use provider default).",
        # Dynamic choices computed by the modal: catalog of the resolved
        # op-provider, prefixed with (inherit). Same mechanism as agent.model.
        dynamic_choices=True,
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
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Provider for relevance classification (pick (inherit) to use active_provider).",
        choices=_PROVIDER_OVERRIDE_CHOICES,
    ),
    ConfigField(
        path="app.memory.operations.classification.model",
        type=FieldType.STRING,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Model for relevance classification (pick (inherit) to use provider default).",
        dynamic_choices=True,
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
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Provider for query builder (pick (inherit) to use active_provider).",
        choices=_PROVIDER_OVERRIDE_CHOICES,
    ),
    ConfigField(
        path="app.memory.operations.query_builder.model",
        type=FieldType.STRING,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.RESTART_APP,
        scope=Scope.APP,
        description="Model for query builder (pick (inherit) to use provider default).",
        dynamic_choices=True,
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
    # Agent: Lead, Explore, Research, Validate -----------------------
    *_agent_fields("Lead"),
    *_agent_fields("Explore"),
    *_agent_fields("Research"),
    *_agent_fields("Validate"),
)
IGNORED_PATHS: frozenset[str] = frozenset({
    # Provider-internal cache plumbing (not a user-facing knob)
    "app.openai.prompt_cache_key",
    "app.openai.prompt_cache_retention",
    "app.openai.store",
    # MCP servers managed via a different (future) UI surface
    "app.mcp.servers",
    # Default lead agent name — selected by which lead YAMLs exist + the
    # multi-lead UI, not a free-text /config knob in this phase.
    "app.default_lead",
    # Phase 1 keeps prompt_modules read-only — changing it risks
    # loading a non-existent module mid-session.
    "agents.*.prompt_modules",
    # Phase 3 — agent-level MCP overrides
    "agents.*.mcp_servers",
    # Internal: agents.*.description/inline_prompt are spec-defined
    # but not surfaced in Phase 1
    "agents.*.description",
    "agents.*.inline_prompt",
    # OpenRouter advanced fields — Phase 3 work
    "app.openrouter.provider_preferences",
    "app.openrouter.tracing",
    # MCP enabled toggled via /config when MCP gets surfaced; out of Phase 1
    "app.mcp.enabled",
    # advanced, Phase 3
    "app.memory.qdrant.quantization",
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
    # Composite container fields — models.py uses `from __future__ import
    # annotations`, so the drift walker sees only top-level field names (not
    # their nested leaves). These paths represent the container dataclasses
    # whose individual sub-fields are each individually registered above.
    "app.database",
    "app.storage",
    "app.logging",
    "app.compaction",
    "app.skills",
    "app.scheduler",
    "app.self_repair",
    "app.openai",
    "app.openrouter",
    "app.claude",
    "app.parallel",
    "app.memory",
    "app.mcp",
    # Agent composite field — ReasoningConfig sub-fields are individually
    # registered as agents.Lead.reasoning.effort / .summary above.
    "agents.*.reasoning",
    # Agent identity fields — name and role are structural, not user-editable.
    "agents.*.name",
    "agents.*.role",
    # Lead-identity + display + capability fields. Edited by hand in the agent
    # YAML (and, in a later phase, the in-TUI lead manager); not surfaced as
    # individual /config knobs in this phase.
    "agents.*.soul",
    "agents.*.color",
    "agents.*.emoji",
    "agents.*.capabilities",
})


# Path -> field index for O(1) lookup. Built from ``reversed(REGISTRY)`` so
# that on the (unexpected) duplicate path the earliest REGISTRY entry wins,
# preserving the previous linear-scan "first match" semantics.
_REGISTRY_BY_PATH: dict[str, ConfigField] = {f.path: f for f in reversed(REGISTRY)}


def lookup(path: str) -> ConfigField | None:
    """Return the registry entry for ``path``, or ``None``.

    Resolves both literal paths (``app.openai.model``) and per-agent
    paths (``agents.Lead.model`` matches a stored ``agents.Lead.model``
    entry — the agent name is part of the literal path, not a wildcard).
    """

    return _REGISTRY_BY_PATH.get(path)


__all__ = (
    "ConfigField",
    "EMBEDDING_MODEL_CATALOG",
    "FieldType",
    "GEMINI_TASK_TYPES",
    "IGNORED_PATHS",
    "INHERIT_SENTINEL",
    "REGISTRY",
    "ReloadClass",
    "Scope",
    "TIKTOKEN_ENCODINGS",
    "WidgetHint",
    "hint_for",
    "lookup",
)
