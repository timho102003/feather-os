"""Model capability catalog.

Loads ``_resources/models/catalog.yaml`` (packaged with the wheel) and
optionally merges a user overlay at ``~/.feather/models/catalog.yaml`` to
produce a lookup table of :class:`ModelCapability` records, one per
(provider, slug) pair.

The catalog drives the ``/config`` modal: when a user picks a model,
fields the model's API doesn't accept render as ``[N/A]`` in the form
and refuse to open an editor. The single source of truth lives in YAML
so adding a new model is a one-file PR — no code change required.

``inherits:`` shortcut::

    openrouter:
      anthropic/claude-opus-4-7:
        inherits: anthropic:claude-opus-4-7   # eager-resolved at load
        notes: "Served via OpenRouter."        # optional per-field override

The loader resolves ``inherits:`` eagerly so every downstream call sees a
fully-flat record. Cycles and dangling references are caught at load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from feather.resources import packaged_root

if TYPE_CHECKING:
    from feather.paths import FeatherPaths


_PACKAGED_CATALOG_PATH = packaged_root() / "models" / "catalog.yaml"


@dataclass(frozen=True, slots=True)
class ModelCapability:
    """Per-model API capability record.

    Drives :class:`ModelCatalog.can_edit` and the picker's range/effort
    hints. Every attribute reflects what the upstream provider's API
    actually accepts — flags set to ``False`` mean the API will reject (or
    silently ignore) the corresponding parameter for this model.

    Attributes:
        slug: Provider-scoped model identifier (e.g. ``gpt-5-mini`` for
            OpenAI, ``anthropic/claude-opus-4-7`` for OpenRouter).
        family: Coarse generation label (``gpt-5``, ``claude-4``, ``o``)
            used for UI grouping.
        context_window: Maximum input tokens accepted in a single request.
        default_max_output: Recommended max output tokens (not a hard cap;
            user can configure higher subject to provider limits).
        supports_temperature: ``True`` when the API accepts a temperature
            parameter. ``False`` means the parameter is rejected or
            ignored (e.g. OpenAI reasoning models).
        temperature_range: Inclusive ``(low, high)`` range when
            ``supports_temperature`` is ``True``; ``None`` otherwise.
        supports_reasoning: ``True`` when the API accepts
            ``reasoning.effort`` / ``reasoning.summary``. OpenAI-specific.
        reasoning_efforts: Allowed effort values when supported.
        supports_thinking: ``True`` when the API accepts a ``thinking``
            block. Anthropic-specific.
        thinking_types: Allowed ``thinking.type`` values when supported.
        supports_parallel_tool_calls: ``True`` when the model accepts
            multiple tool calls in a single turn.
        supports_multimodal: ``True`` when the model accepts image input.
        supports_cache_control: ``True`` when the model honors
            ``cache_control`` blocks (Claude / OpenRouter).
        suggested_temperature: Picker default when supports_temperature.
        suggested_reasoning_effort: Picker default when supports_reasoning.
        deprecated: ``True`` for models past their sunset date.
        deprecation_date: ISO date string when the model is retired.
        notes: Free-form one-liner shown in the modal's description.
    """

    slug: str
    family: str
    context_window: int
    default_max_output: int
    supports_temperature: bool = False
    temperature_range: tuple[float, float] | None = None
    supports_reasoning: bool = False
    reasoning_efforts: tuple[str, ...] = ()
    supports_thinking: bool = False
    thinking_types: tuple[str, ...] = ()
    supports_parallel_tool_calls: bool = False
    supports_multimodal: bool = False
    supports_cache_control: bool = False
    suggested_temperature: float | None = None
    suggested_reasoning_effort: str | None = None
    deprecated: bool = False
    deprecation_date: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.supports_temperature and self.temperature_range is not None:
            raise ValueError(
                f"{self.slug}: temperature_range set but supports_temperature=False"
            )
        if not self.supports_reasoning and self.reasoning_efforts:
            raise ValueError(
                f"{self.slug}: reasoning_efforts set but supports_reasoning=False"
            )
        if not self.supports_thinking and self.thinking_types:
            raise ValueError(
                f"{self.slug}: thinking_types set but supports_thinking=False"
            )


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """Loaded catalog organised by provider name.

    Attributes:
        providers: Mapping from provider name (``openai``, ``anthropic``,
            ``openrouter``) to per-slug capability records.
    """

    providers: dict[str, dict[str, ModelCapability]] = field(default_factory=dict)

    def capability(self, provider: str, slug: str) -> ModelCapability | None:
        """Return the capability record for ``(provider, slug)`` or ``None``."""

        return self.providers.get(provider, {}).get(slug)

    def slugs_for(self, provider: str) -> tuple[str, ...]:
        """Return non-deprecated model slugs for ``provider`` in insertion order."""

        entries = self.providers.get(provider, {})
        return tuple(slug for slug, cap in entries.items() if not cap.deprecated)

    def can_edit(self, field_path: str, model: ModelCapability) -> bool:
        """Return whether ``field_path`` is editable given the model's caps.

        Args:
            field_path: Dotted registry path (e.g. ``app.openai.temperature``,
                ``agents.Lead.reasoning.effort``).
            model: The model whose capabilities gate the field.

        Returns:
            ``False`` only for fields the model's API would reject; ``True``
            for everything else (including fields the catalog has no
            opinion about, like paths or log levels).
        """

        # The check operates on the trailing segment(s) so it applies
        # uniformly to app.<provider>.<knob> and agents.<name>.<knob>.
        for suffix, allowed in _FIELD_GATES.items():
            if field_path.endswith(suffix):
                return allowed(model)
        return True


# Field-suffix → predicate map. Each predicate returns True when the model
# accepts the field; False means render as N/A and refuse to edit.
# Suffixes are checked in declaration order; the first match wins.
_FIELD_GATES: dict[str, Any] = {
    ".thinking.budget_tokens": lambda m: m.supports_thinking
    and "enabled" in m.thinking_types,
    ".thinking.type": lambda m: m.supports_thinking,
    ".reasoning.effort": lambda m: m.supports_reasoning,
    ".reasoning.summary": lambda m: m.supports_reasoning,
    ".temperature": lambda m: m.supports_temperature,
    ".parallel_tool_calls": lambda m: m.supports_parallel_tool_calls,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_catalog(*, paths: "FeatherPaths | None" = None) -> ModelCatalog:
    """Load the packaged catalog with an optional user-overlay merge.

    Args:
        paths: When provided, merges ``paths.global_root/models/catalog.yaml``
            on top of the packaged catalog per (provider, slug) pair using
            a deep-merge — so a user overlay can patch a single field without
            re-declaring the whole entry.

    Returns:
        Fully-resolved :class:`ModelCatalog` with all ``inherits:`` flattened.
    """

    raw = _read_yaml(_PACKAGED_CATALOG_PATH)
    if paths is not None:
        overlay_path = paths.global_root / "models" / "catalog.yaml"
        if overlay_path.exists():
            overlay = _read_yaml(overlay_path)
            raw = _deep_merge(raw, overlay)
    resolved = _resolve_inherits(raw)
    providers: dict[str, dict[str, ModelCapability]] = {}
    for provider, slugs in resolved.items():
        providers[provider] = {}
        for slug, attrs in slugs.items():
            providers[provider][slug] = _capability_from_attrs(slug, attrs)
    return ModelCatalog(providers=providers)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: catalog must be a mapping at top level")
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` over ``base`` without mutating either.

    Mirrors :func:`feather.config._deep_merge` semantics: overlay scalars
    win, dicts merge key-by-key, anything else is replaced wholesale.
    """

    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_inherits(raw: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Flatten ``inherits:`` references so every entry is self-contained.

    Raises:
        ValueError: If an ``inherits:`` reference targets an unknown
            ``provider:slug`` pair, or if the inherits graph contains a cycle.
    """

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for provider, slugs in raw.items():
        if not isinstance(slugs, dict):
            continue
        out[provider] = {}
        for slug, attrs in slugs.items():
            if not isinstance(attrs, dict):
                continue
            out[provider][slug] = dict(attrs)

    # Resolve each entry's inherits chain. Eager flatten with cycle detection.
    for provider, slugs in out.items():
        for slug in list(slugs.keys()):
            _resolve_one(out, provider, slug, visiting=set())
    return out


def _resolve_one(
    catalog: dict[str, dict[str, dict[str, Any]]],
    provider: str,
    slug: str,
    *,
    visiting: set[str],
) -> dict[str, Any]:
    """Resolve one entry's inheritance, flattening into a self-contained dict."""

    key = f"{provider}:{slug}"
    if key in visiting:
        raise ValueError(f"inherits cycle detected involving {key}")
    entry = catalog[provider][slug]
    parent_ref = entry.get("inherits")
    if parent_ref is None:
        return entry
    if not isinstance(parent_ref, str) or ":" not in parent_ref:
        raise ValueError(
            f"{key}: inherits must be 'provider:slug', got {parent_ref!r}"
        )
    parent_provider, parent_slug = parent_ref.split(":", 1)
    if (
        parent_provider not in catalog
        or parent_slug not in catalog[parent_provider]
    ):
        raise ValueError(
            f"{key}: inherits target {parent_ref!r} not in catalog"
        )
    visiting = visiting | {key}
    parent_resolved = _resolve_one(
        catalog, parent_provider, parent_slug, visiting=visiting
    )
    # Child fields override parent; drop the inherits marker.
    merged = {**parent_resolved, **entry}
    merged.pop("inherits", None)
    catalog[provider][slug] = merged
    return merged


def _capability_from_attrs(slug: str, attrs: dict[str, Any]) -> ModelCapability:
    """Build a :class:`ModelCapability` from a flattened YAML attrs dict."""

    temp_range_raw = attrs.get("temperature_range")
    temperature_range: tuple[float, float] | None = None
    if temp_range_raw is not None:
        temperature_range = (float(temp_range_raw[0]), float(temp_range_raw[1]))
    return ModelCapability(
        slug=slug,
        family=str(attrs.get("family", "unknown")),
        context_window=int(attrs.get("context_window", 0)),
        default_max_output=int(attrs.get("default_max_output", 0)),
        supports_temperature=bool(attrs.get("supports_temperature", False)),
        temperature_range=temperature_range,
        supports_reasoning=bool(attrs.get("supports_reasoning", False)),
        reasoning_efforts=tuple(attrs.get("reasoning_efforts") or ()),
        supports_thinking=bool(attrs.get("supports_thinking", False)),
        thinking_types=tuple(attrs.get("thinking_types") or ()),
        supports_parallel_tool_calls=bool(
            attrs.get("supports_parallel_tool_calls", False)
        ),
        supports_multimodal=bool(attrs.get("supports_multimodal", False)),
        supports_cache_control=bool(attrs.get("supports_cache_control", False)),
        suggested_temperature=(
            float(attrs["suggested_temperature"])
            if "suggested_temperature" in attrs
            else None
        ),
        suggested_reasoning_effort=attrs.get("suggested_reasoning_effort"),
        deprecated=bool(attrs.get("deprecated", False)),
        deprecation_date=attrs.get("deprecation_date"),
        notes=str(attrs.get("notes") or ""),
    )


__all__ = (
    "ModelCapability",
    "ModelCatalog",
    "load_catalog",
)
