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
