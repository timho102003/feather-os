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


# ---------------------------------------------------------------------------
# Phase 2 enrichment: hint + choices + MODEL_CATALOG
# ---------------------------------------------------------------------------


def test_config_field_supports_hint_attribute() -> None:
    """ConfigField must expose an optional ``hint`` for the modal placeholder."""

    from feather.config_schema import ConfigField, FieldType, WidgetHint, ReloadClass, Scope

    f = ConfigField(
        path="x.y",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="d",
        hint="0.0-1.0",
    )
    assert f.hint == "0.0-1.0"


def test_config_field_supports_choices_attribute() -> None:
    """ConfigField must expose an optional ``choices`` (non-strict suggestions)."""

    from feather.config_schema import ConfigField, FieldType, WidgetHint, ReloadClass, Scope

    f = ConfigField(
        path="x.y",
        type=FieldType.STRING,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="d",
        choices=("a", "b"),
    )
    assert f.choices == ("a", "b")


def test_ratio_validator_has_range_hint() -> None:
    """Fields validated by _ratio should advertise the 0.0-1.0 range via hint_for."""

    from feather.config_schema import hint_for

    field = lookup("app.compaction.trigger_ratio")
    assert field is not None
    h = hint_for(field)
    assert h is not None and "0" in h and "1" in h, (
        f"trigger_ratio should have a 0.0-1.0 hint, got {h!r}"
    )


def test_positive_validator_has_positivity_hint() -> None:
    """Fields validated by _positive should advertise > 0 via hint_for."""

    from feather.config_schema import hint_for

    field = lookup("app.compaction.context_window_tokens")
    assert field is not None
    h = hint_for(field)
    assert h is not None and ">" in h, (
        f"_positive-validated fields should hint at > 0; got {h!r}"
    )


def test_explicit_hint_beats_validator_derived() -> None:
    """When ConfigField.hint is set, hint_for returns it instead of the derived value."""

    from feather.config_schema import (
        ConfigField,
        FieldType,
        ReloadClass,
        Scope,
        WidgetHint,
        _ratio,
        hint_for,
    )

    f = ConfigField(
        path="x.y",
        type=FieldType.FLOAT,
        widget=WidgetHint.NUMERIC,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="d",
        validator=_ratio,
        hint="custom override",
    )
    assert hint_for(f) == "custom override"


def test_hint_for_returns_none_when_no_signal() -> None:
    """No validator + no explicit hint → hint_for returns None."""

    from feather.config_schema import (
        ConfigField,
        FieldType,
        ReloadClass,
        Scope,
        WidgetHint,
        hint_for,
    )

    f = ConfigField(
        path="x.y",
        type=FieldType.STRING,
        widget=WidgetHint.TEXT,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="d",
    )
    assert hint_for(f) is None


def test_model_catalog_exists_and_covers_providers() -> None:
    """MODEL_CATALOG ships suggestions for each first-party provider."""

    from feather.config_schema import MODEL_CATALOG

    assert "openai" in MODEL_CATALOG
    assert "claude" in MODEL_CATALOG
    assert "openrouter" in MODEL_CATALOG
    assert all(isinstance(m, str) and m for m in MODEL_CATALOG["openai"])
    # Sanity: catalog entries should be non-empty.
    assert MODEL_CATALOG["openai"]
    assert MODEL_CATALOG["claude"]
    assert MODEL_CATALOG["openrouter"]


def test_provider_model_fields_use_dropdown_with_choices() -> None:
    """Per-provider model fields should be DROPDOWN populated from the catalog."""

    from feather.config_schema import MODEL_CATALOG

    for path, key in (
        ("app.openai.model", "openai"),
        ("app.claude.model", "claude"),
        ("app.openrouter.model", "openrouter"),
    ):
        field = lookup(path)
        assert field is not None, f"{path} missing"
        assert field.widget is WidgetHint.DROPDOWN, (
            f"{path} should be DROPDOWN, got {field.widget}"
        )
        assert field.choices is not None
        assert set(MODEL_CATALOG[key]).issubset(field.choices)


def test_choices_field_is_not_enum_validated() -> None:
    """choices != enum: a non-enum DROPDOWN must not raise __post_init__ on free input."""

    field = lookup("app.openai.model")
    assert field is not None
    assert field.type is FieldType.STRING, "model fields are strings, not enums"
    # enum is None — validation does not constrain to choices.
    assert field.enum is None


def test_agent_registry_exposes_temperature_and_max_output_tokens() -> None:
    """Each agent tab carries per-agent temperature and max_output_tokens
    overrides so users can tune one agent without touching app.* defaults."""

    for name in ("Lead", "Explore", "Research", "Validate"):
        for leaf in ("temperature", "max_output_tokens"):
            path = f"agents.{name}.{leaf}"
            field = lookup(path)
            assert field is not None, f"missing {path} in registry"
            assert field.scope.value == "agent", f"{path} should be Scope.AGENT"


def test_dropdown_without_enum_or_choices_is_invalid() -> None:
    """A DROPDOWN field must offer either enum (strict) or choices (suggestions)."""

    import pytest

    from feather.config_schema import ConfigField, FieldType, WidgetHint, ReloadClass, Scope

    with pytest.raises(ValueError, match="enum or choices"):
        ConfigField(
            path="x.y",
            type=FieldType.STRING,
            widget=WidgetHint.DROPDOWN,
            reload=ReloadClass.LIVE,
            scope=Scope.APP,
            description="d",
        )
