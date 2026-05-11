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
