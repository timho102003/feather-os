"""Tests for :mod:`feather.models_catalog`."""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# ModelCapability dataclass
# ---------------------------------------------------------------------------


def test_model_capability_supports_minimal_fields() -> None:
    """ModelCapability accepts only the required fields with sensible defaults."""

    from feather.models_catalog import ModelCapability

    cap = ModelCapability(
        slug="test-model",
        family="test",
        context_window=128000,
        default_max_output=4096,
    )
    assert cap.slug == "test-model"
    assert cap.supports_temperature is False
    assert cap.temperature_range is None
    assert cap.supports_reasoning is False
    assert cap.reasoning_efforts == ()
    assert cap.supports_thinking is False
    assert cap.thinking_types == ()
    assert cap.supports_parallel_tool_calls is False
    assert cap.supports_multimodal is False
    assert cap.supports_cache_control is False
    assert cap.deprecated is False


def test_model_capability_temperature_range_consistency() -> None:
    """temperature_range must be None when supports_temperature=False."""

    from feather.models_catalog import ModelCapability

    with pytest.raises(ValueError, match="temperature_range"):
        ModelCapability(
            slug="x",
            family="x",
            context_window=1,
            default_max_output=1,
            supports_temperature=False,
            temperature_range=(0.0, 1.0),
        )


def test_model_capability_reasoning_efforts_only_when_supported() -> None:
    """reasoning_efforts must be empty when supports_reasoning=False."""

    from feather.models_catalog import ModelCapability

    with pytest.raises(ValueError, match="reasoning_efforts"):
        ModelCapability(
            slug="x",
            family="x",
            context_window=1,
            default_max_output=1,
            supports_reasoning=False,
            reasoning_efforts=("low",),
        )


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


def test_load_catalog_returns_packaged_models() -> None:
    """Loading the packaged catalog yields openai/anthropic/openrouter entries."""

    from feather.models_catalog import load_catalog

    catalog = load_catalog()
    assert "openai" in catalog.providers
    assert "anthropic" in catalog.providers
    assert "openrouter" in catalog.providers


def test_load_catalog_contains_shipped_app_yaml_defaults() -> None:
    """Every model slug referenced from the shipped app.yaml must be in catalog.

    Catches the obvious regression where a user picks the shipped default
    and the catalog has no opinion (the picker would lock them out of
    fields the model actually supports).
    """

    from feather.models_catalog import load_catalog

    catalog = load_catalog()
    assert catalog.capability("openai", "gpt-5-mini") is not None, (
        "app.yaml default openai.model not in catalog"
    )
    assert catalog.capability("anthropic", "claude-opus-4-7") is not None
    assert catalog.capability("openrouter", "qwen/qwen3.6-plus") is not None


def test_inherits_resolves_eagerly() -> None:
    """A child entry with `inherits:` is flattened so all fields are filled."""

    from feather.models_catalog import load_catalog

    catalog = load_catalog()
    # openrouter:anthropic/claude-opus-4-7 inherits from anthropic:claude-opus-4-7
    child = catalog.capability("openrouter", "anthropic/claude-opus-4-7")
    parent = catalog.capability("anthropic", "claude-opus-4-7")
    assert child is not None and parent is not None
    assert child.context_window == parent.context_window
    assert child.supports_thinking is parent.supports_thinking
    assert child.temperature_range == parent.temperature_range


def test_inherits_child_overrides_win() -> None:
    """A child entry overrides individual fields from its parent."""

    from feather.models_catalog import load_catalog

    catalog = load_catalog()
    child = catalog.capability("openrouter", "anthropic/claude-opus-4-7-fast")
    parent = catalog.capability("anthropic", "claude-opus-4-7")
    assert child is not None and parent is not None
    # Same context window (inherited), but child has an explicit notes
    # override that doesn't match the parent.
    assert child.context_window == parent.context_window
    assert child.notes != parent.notes


def test_inherits_unknown_parent_raises_at_load() -> None:
    """Pointing inherits: at a nonexistent slug fails loudly at load time."""

    from feather.models_catalog import _resolve_inherits

    raw = {
        "openrouter": {
            "anthropic/orphan": {"inherits": "anthropic:does-not-exist"},
        }
    }
    with pytest.raises(ValueError, match="inherits"):
        _resolve_inherits(raw)


def test_inherits_cycle_detected_at_load() -> None:
    """A → B → A inherit cycle is detected and raised."""

    from feather.models_catalog import _resolve_inherits

    raw = {
        "openai": {
            "a": {"inherits": "openai:b"},
            "b": {"inherits": "openai:a"},
        }
    }
    with pytest.raises(ValueError, match="cycle"):
        _resolve_inherits(raw)


def test_global_overlay_deep_merges_per_slug(tmp_path: Path) -> None:
    """`~/.feather/models/catalog.yaml` patches individual fields without
    requiring a full re-declaration of the entry."""

    from feather.models_catalog import load_catalog
    from feather.paths import FeatherPaths

    paths = FeatherPaths(project_root=tmp_path / "proj", home=tmp_path / "global")
    paths.ensure_global_dirs()
    # Override one field on an existing entry.
    (paths.global_root / "models").mkdir(parents=True, exist_ok=True)
    (paths.global_root / "models" / "catalog.yaml").write_text(
        "openai:\n"
        "  gpt-5-mini:\n"
        "    suggested_reasoning_effort: high\n"
        "    notes: \"User-tuned override.\"\n",
        encoding="utf-8",
    )

    catalog = load_catalog(paths=paths)
    cap = catalog.capability("openai", "gpt-5-mini")
    assert cap is not None
    # Patched fields
    assert cap.suggested_reasoning_effort == "high"
    assert cap.notes == "User-tuned override."
    # Untouched fields inherited from packaged
    assert cap.context_window == 400000
    assert cap.supports_reasoning is True


def test_global_overlay_can_register_a_brand_new_slug(tmp_path: Path) -> None:
    """User overlay can introduce a slug the packaged catalog doesn't know."""

    from feather.models_catalog import load_catalog
    from feather.paths import FeatherPaths

    paths = FeatherPaths(project_root=tmp_path / "proj", home=tmp_path / "global")
    paths.ensure_global_dirs()
    (paths.global_root / "models").mkdir(parents=True, exist_ok=True)
    (paths.global_root / "models" / "catalog.yaml").write_text(
        "openai:\n"
        "  gpt-7-experimental:\n"
        "    family: gpt-7\n"
        "    context_window: 200000\n"
        "    default_max_output: 8000\n"
        "    supports_temperature: false\n"
        "    supports_reasoning: true\n"
        "    reasoning_efforts: [low, medium, high]\n",
        encoding="utf-8",
    )

    catalog = load_catalog(paths=paths)
    cap = catalog.capability("openai", "gpt-7-experimental")
    assert cap is not None
    assert cap.family == "gpt-7"
    assert cap.supports_reasoning is True


# ---------------------------------------------------------------------------
# can_edit() — capability-driven field gating
# ---------------------------------------------------------------------------


def test_can_edit_temperature_reflects_support() -> None:
    from feather.models_catalog import load_catalog

    catalog = load_catalog()
    gpt5 = catalog.capability("openai", "gpt-5-mini")
    gpt4o = catalog.capability("openai", "gpt-4o")
    assert gpt5 is not None and gpt4o is not None

    # Same field path; opposite verdict based on capability.
    assert catalog.can_edit("app.openai.temperature", gpt5) is False
    assert catalog.can_edit("app.openai.temperature", gpt4o) is True


def test_can_edit_reasoning_effort_reflects_support() -> None:
    from feather.models_catalog import load_catalog

    catalog = load_catalog()
    gpt5 = catalog.capability("openai", "gpt-5-mini")
    gpt4o = catalog.capability("openai", "gpt-4o")
    assert gpt5 is not None and gpt4o is not None

    assert catalog.can_edit("app.openai.reasoning.effort", gpt5) is True
    assert catalog.can_edit("app.openai.reasoning.effort", gpt4o) is False


def test_can_edit_thinking_only_on_supporting_models() -> None:
    from feather.models_catalog import load_catalog

    catalog = load_catalog()
    sonnet = catalog.capability("anthropic", "claude-sonnet-4-6")
    opus = catalog.capability("anthropic", "claude-opus-4-7")
    haiku = catalog.capability("anthropic", "claude-haiku-4-5")
    assert sonnet is not None and opus is not None and haiku is not None

    # thinking.type editable when supports_thinking
    assert catalog.can_edit("app.claude.thinking.type", sonnet) is True
    assert catalog.can_edit("app.claude.thinking.type", opus) is True
    # budget_tokens only when thinking_types includes 'enabled' (opus is adaptive-only)
    assert catalog.can_edit("app.claude.thinking.budget_tokens", sonnet) is True
    assert catalog.can_edit("app.claude.thinking.budget_tokens", opus) is False
    assert catalog.can_edit("app.claude.thinking.budget_tokens", haiku) is True


def test_can_edit_parallel_tool_calls() -> None:
    from feather.models_catalog import load_catalog

    catalog = load_catalog()
    gpt4o = catalog.capability("openai", "gpt-4o")
    o3 = catalog.capability("openai", "o3")
    assert gpt4o is not None and o3 is not None

    assert catalog.can_edit("app.openai.parallel_tool_calls", gpt4o) is True
    assert catalog.can_edit("app.openai.parallel_tool_calls", o3) is False


def test_can_edit_returns_true_for_unknown_field_paths() -> None:
    """Fields the catalog has no opinion on (e.g. paths, log level) must
    remain editable — the catalog is a permission layer, not an allow-list."""

    from feather.models_catalog import load_catalog

    catalog = load_catalog()
    gpt5 = catalog.capability("openai", "gpt-5-mini")
    assert gpt5 is not None
    assert catalog.can_edit("app.logging.level", gpt5) is True
    assert catalog.can_edit("app.database.path", gpt5) is True


# ---------------------------------------------------------------------------
# slugs_for() — populating the picker
# ---------------------------------------------------------------------------


def test_slugs_for_returns_per_provider_list() -> None:
    from feather.models_catalog import load_catalog

    catalog = load_catalog()
    openai_slugs = catalog.slugs_for("openai")
    anthropic_slugs = catalog.slugs_for("anthropic")
    openrouter_slugs = catalog.slugs_for("openrouter")

    assert "gpt-5-mini" in openai_slugs
    assert "claude-opus-4-7" in anthropic_slugs
    assert "qwen/qwen3.6-plus" in openrouter_slugs

    # Deprecated entries are excluded by default.
    assert "claude-opus-4-0" not in anthropic_slugs
    assert "claude-sonnet-4-0" not in anthropic_slugs


def test_slugs_for_unknown_provider_returns_empty() -> None:
    from feather.models_catalog import load_catalog

    assert load_catalog().slugs_for("nonexistent") == ()


# ---------------------------------------------------------------------------
# Coverage: every slug returned by slugs_for() resolves to capability metadata
# ---------------------------------------------------------------------------


def test_every_slug_returned_by_slugs_for_has_capability_metadata() -> None:
    """``slugs_for(provider)`` populates every model picker in the TUI, so
    each slug it emits must also resolve through ``capability()`` — otherwise
    the picker could surface a slug that the gating layer cannot reason about
    (no temperature range, no thinking support, etc.).
    """

    from feather.models_catalog import load_catalog

    catalog = load_catalog()
    missing: list[str] = []
    for provider in ("openai", "anthropic", "openrouter"):
        for slug in catalog.slugs_for(provider):
            if catalog.capability(provider, slug) is None:
                missing.append(f"{provider}:{slug}")
    assert not missing, f"slugs returned by slugs_for() but missing capability: {missing}"
