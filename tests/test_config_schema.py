"""Per-entry self-checks for the config registry."""

from __future__ import annotations

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
