"""The ``Soul`` value object — one selectable lead personality preset.

A soul is a *preset*, not an agent: rich persona prose plus display metadata a
user picks when creating a lead (see :mod:`feather.core.leads.scaffold`). The
preset's fields are baked into the new lead's YAML, so a soul never becomes a
``role: lead`` itself and adds nothing to startup. Loaded from layered YAML by
:class:`feather.core.leads.soul_library.SoulLibrary`.

Follows the house pattern for config-from-YAML (``AgentConfig`` + a manual
loader): a frozen ``@dataclass`` plus :func:`load_soul`, which raises
``ValueError`` on malformed input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ("Soul", "load_soul", "SoulError")

_REQUIRED_FIELDS = ("title", "personality", "prose", "color", "emoji")


class SoulError(ValueError):
    """Raised when a soul YAML is missing a required field or is malformed."""


@dataclass(slots=True, frozen=True)
class Soul:
    """One reusable working-temperament preset, assignable to any agent.

    A soul is a *disposition*, not a person — it carries no name, profession,
    or backstory, so the same soul can be applied to many different leads
    without collision. ``title`` is the archetype label shown in the picker
    (e.g. "The Skeptic"); ``personality`` maps to the lead's one-line
    ``AgentConfig.personality`` and ``prose`` to ``AgentConfig.soul`` (the
    ``<agent_soul>`` prompt block). ``color``/``emoji`` are temperament display
    hints. ``id`` is the filename stem and the selection key. Applying a soul
    never sets the lead's name — the lead keeps the name the user gives it.
    """

    id: str
    title: str
    personality: str
    prose: str
    color: str
    emoji: str
    tags: tuple[str, ...] = ()


def load_soul(soul_id: str, raw: Mapping[str, Any]) -> Soul:
    """Build a :class:`Soul` from a parsed YAML mapping.

    Args:
        soul_id: The selection key (filename stem); becomes ``Soul.id``.
        raw: The parsed YAML mapping for one soul file.

    Raises:
        SoulError: If ``raw`` is not a mapping or any required field is
            missing or blank after stripping.
    """

    if not isinstance(raw, Mapping):
        raise SoulError(f"soul {soul_id!r} must be a mapping, got {type(raw).__name__}")
    values: dict[str, str] = {}
    for key in _REQUIRED_FIELDS:
        value = str(raw.get(key) or "").strip()
        if not value:
            raise SoulError(f"soul {soul_id!r} missing/blank field: {key}")
        values[key] = value
    raw_tags = raw.get("tags") or ()
    tags = tuple(
        stripped
        for tag in raw_tags
        if (stripped := str(tag).strip())
    )
    return Soul(
        id=soul_id,
        title=values["title"],
        personality=values["personality"],
        prose=values["prose"],
        color=values["color"],
        emoji=values["emoji"],
        tags=tags,
    )
