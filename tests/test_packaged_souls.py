"""Content guard for the bundled 20-soul library.

These assertions pin the *shape* of the packaged souls, not their exact prose,
so they survive re-authoring while still catching a malformed or missing soul.
"""

from __future__ import annotations

import re

import yaml

from feather.core.leads.soul import load_soul
from feather.resources import iter_packaged_soul_names, packaged_soul_yaml_text

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _all_souls():
    return [
        load_soul(soul_id, yaml.safe_load(packaged_soul_yaml_text(soul_id)) or {})
        for soul_id in iter_packaged_soul_names()
    ]


def test_exactly_twenty_packaged_souls() -> None:
    assert len(list(iter_packaged_soul_names())) == 20


def test_every_packaged_soul_loads() -> None:
    souls = _all_souls()
    assert len(souls) == 20


def test_ids_and_titles_unique() -> None:
    souls = _all_souls()
    assert len({s.id for s in souls}) == 20
    assert len({s.title for s in souls}) == 20


def test_required_fields_well_formed() -> None:
    for soul in _all_souls():
        assert soul.title.strip()
        assert "\n" not in soul.personality and 0 < len(soul.personality) <= 120
        assert _HEX.match(soul.color), f"{soul.id} color={soul.color!r}"
        assert soul.emoji.strip() and not soul.emoji.isascii()
        assert soul.tags  # at least one tag


def test_prose_is_second_person_temperament() -> None:
    for soul in _all_souls():
        words = soul.prose.split()
        assert 50 <= len(words) <= 200, f"{soul.id} prose has {len(words)} words"
        # A reusable temperament is written in second person.
        assert soul.prose.lower().startswith("you")


_LEAK_RE = re.compile(r"\bnamed [A-Z]|\bYou are [A-Z]")


def test_prose_has_no_leaked_identity() -> None:
    """Souls must be identity-free so they're reusable across agents."""
    # Cheap regression guard against the old name/backstory style. A proper name
    # after "named "/"You are " is a leak; "the named example" is fine.
    for soul in _all_souls():
        body = soul.prose.lower()
        assert "grew up" not in body and "hometown" not in body, f"{soul.id} has backstory"
        assert not _LEAK_RE.search(soul.prose), f"{soul.id} prose leaks a proper name"
