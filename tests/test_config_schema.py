"""Per-entry self-checks for the config registry."""

from __future__ import annotations

from feather.config.schema import (
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


def test_lookup_index_matches_first_registry_match() -> None:
    """The O(1) path index returns the same field a linear scan would.

    Pins the lookup optimization: ``lookup`` must resolve to the *first*
    REGISTRY entry for a path (first-occurrence-wins), identical to the
    previous linear scan, and unknown paths still return ``None``.
    """

    from feather.config.schema import _REGISTRY_BY_PATH

    for field in REGISTRY:
        first = next(f for f in REGISTRY if f.path == field.path)
        assert lookup(field.path) is first
        assert _REGISTRY_BY_PATH[field.path] is first
    assert lookup("definitely.not.a.real.path") is None


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

    from feather.config.schema import ConfigField, FieldType, WidgetHint, ReloadClass, Scope

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

    from feather.config.schema import ConfigField, FieldType, WidgetHint, ReloadClass, Scope

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

    from feather.config.schema import hint_for

    field = lookup("app.compaction.trigger_ratio")
    assert field is not None
    h = hint_for(field)
    assert h is not None and "0" in h and "1" in h, (
        f"trigger_ratio should have a 0.0-1.0 hint, got {h!r}"
    )


def test_positive_validator_has_positivity_hint() -> None:
    """Fields validated by _positive should advertise > 0 via hint_for."""

    from feather.config.schema import hint_for

    field = lookup("app.compaction.context_window_tokens")
    assert field is not None
    h = hint_for(field)
    assert h is not None and ">" in h, (
        f"_positive-validated fields should hint at > 0; got {h!r}"
    )


def test_explicit_hint_beats_validator_derived() -> None:
    """When ConfigField.hint is set, hint_for returns it instead of the derived value."""

    from feather.config.schema import (
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

    from feather.config.schema import (
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


def test_model_catalog_covers_first_party_providers() -> None:
    """The YAML model catalog exposes at least one slug for each provider.

    Note: the catalog keys Anthropic models under ``anthropic`` (matching
    the SDK vendor name), while the app-level config field uses
    ``app.claude.*``; the TUI maps between them via
    ``_provider_to_catalog_key``.
    """

    from feather.config.model_catalog import load_catalog

    catalog = load_catalog()
    for provider in ("openai", "anthropic", "openrouter"):
        slugs = catalog.slugs_for(provider)
        assert slugs, f"catalog has no slugs for {provider!r}"
        assert all(isinstance(s, str) and s for s in slugs)


def test_provider_model_fields_use_dropdown() -> None:
    """``app.<provider>.model`` is DROPDOWN; choices come from the catalog at
    picker-open time, so the field itself ships without a static list."""

    for path in (
        "app.openai.model",
        "app.claude.model",
        "app.openrouter.model",
    ):
        field = lookup(path)
        assert field is not None, f"{path} missing"
        assert field.widget is WidgetHint.DROPDOWN, (
            f"{path} should be DROPDOWN, got {field.widget}"
        )
        # Choices are intentionally empty in the registry — they are
        # resolved dynamically by ConfigScreen._picker_choices_for so the
        # picker and the per-agent picker share one source of truth.
        assert not field.choices


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
    """A DROPDOWN field must offer either enum (strict), choices
    (suggestions), or ``dynamic_choices=True`` (resolved by the modal)."""

    import pytest

    from feather.config.schema import ConfigField, FieldType, WidgetHint, ReloadClass, Scope

    with pytest.raises(ValueError, match="enum, choices, or dynamic_choices"):
        ConfigField(
            path="x.y",
            type=FieldType.STRING,
            widget=WidgetHint.DROPDOWN,
            reload=ReloadClass.LIVE,
            scope=Scope.APP,
            description="d",
        )

    # dynamic_choices=True opts out of the check.
    ConfigField(
        path="x.y",
        type=FieldType.STRING,
        widget=WidgetHint.DROPDOWN,
        reload=ReloadClass.LIVE,
        scope=Scope.APP,
        description="d",
        dynamic_choices=True,
    )
