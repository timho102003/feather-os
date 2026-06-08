"""The Soul value object + load_soul parser."""

from __future__ import annotations

import pytest

from feather.core.leads.soul import Soul, SoulError, load_soul

_VALID = {
    "title": "The Systems Thinker",
    "personality": "Calm, systems-first, decisive.",
    "prose": "You think in wholes before parts and reason in trade-offs.",
    "color": "#5B8DEF",
    "emoji": "🏛️",
    "tags": ["analytical", "big-picture"],
}


def test_load_soul_happy_path() -> None:
    soul = load_soul("systems-thinker", _VALID)
    assert isinstance(soul, Soul)
    assert soul.id == "systems-thinker"
    assert soul.title == "The Systems Thinker"
    assert soul.tags == ("analytical", "big-picture")


def test_load_soul_normalizes_tags() -> None:
    raw = {**_VALID, "tags": ["  data  ", "", "  ", "ml"]}
    soul = load_soul("vega", raw)
    assert soul.tags == ("data", "ml")  # stripped + blanks dropped


def test_load_soul_missing_tags_is_empty() -> None:
    raw = {k: v for k, v in _VALID.items() if k != "tags"}
    soul = load_soul("x", raw)
    assert soul.tags == ()


@pytest.mark.parametrize("missing", ["title", "personality", "prose", "color", "emoji"])
def test_load_soul_missing_required_field_raises(missing: str) -> None:
    raw = {k: v for k, v in _VALID.items() if k != missing}
    with pytest.raises(SoulError) as exc:
        load_soul("broken", raw)
    assert missing in str(exc.value)


def test_load_soul_blank_required_field_raises() -> None:
    raw = {**_VALID, "personality": "   "}
    with pytest.raises(SoulError):
        load_soul("broken", raw)


def test_load_soul_non_mapping_raises() -> None:
    with pytest.raises(SoulError):
        load_soul("broken", "not a mapping")  # type: ignore[arg-type]
