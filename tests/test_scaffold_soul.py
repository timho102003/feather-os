"""scaffold_lead_yaml with a Soul preset bakes the preset into the lead YAML."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from feather.config import load_agent_config
from feather.core.leads.scaffold import scaffold_lead_yaml
from feather.core.leads.soul import Soul

_PRESET = Soul(
    id="systems-thinker",
    title="The Systems Thinker",
    personality="Calm, systems-first, decisive.",
    prose="You think in wholes before parts and reason in trade-offs, never absolutes.",
    color="#5B8DEF",
    emoji="🏛️",
    tags=("analytical", "big-picture"),
)


def test_preset_applies_temperament_keeps_user_name(tmp_path: Path) -> None:
    scaffold_lead_yaml(tmp_path, "backend", soul_preset=_PRESET)
    cfg = load_agent_config(tmp_path, "backend")
    assert cfg.role == "lead"
    assert cfg.name == "Backend"                     # lead name is the USER's, not the soul
    assert cfg.personality == "Calm, systems-first, decisive."
    assert "trade-offs" in cfg.soul                  # prose → soul block
    assert cfg.color == "#5B8DEF"                    # color quoted, not a comment
    assert cfg.emoji == "🏛️"


def test_same_soul_assignable_to_two_leads(tmp_path: Path) -> None:
    scaffold_lead_yaml(tmp_path, "backend", soul_preset=_PRESET)
    scaffold_lead_yaml(tmp_path, "frontend", soul_preset=_PRESET)
    back = load_agent_config(tmp_path, "backend")
    front = load_agent_config(tmp_path, "frontend")
    assert back.name == "Backend" and front.name == "Frontend"   # distinct identities
    assert back.soul == front.soul                                # shared temperament


def test_no_preset_path_unchanged(tmp_path: Path) -> None:
    scaffold_lead_yaml(tmp_path, "tim", "You are Tim, a pragmatic operator.")
    cfg = load_agent_config(tmp_path, "tim")
    assert cfg.name == "Tim"                          # name.capitalize()
    assert "pragmatic operator" in cfg.soul
    assert cfg.color is None and cfg.emoji is None     # no identity block


def test_preset_is_idempotent(tmp_path: Path) -> None:
    scaffold_lead_yaml(tmp_path, "backend", soul_preset=_PRESET)
    other = replace(_PRESET, personality="Different.", color="#000000")
    scaffold_lead_yaml(tmp_path, "backend", soul_preset=other)  # must not overwrite
    cfg = load_agent_config(tmp_path, "backend")
    assert cfg.personality == "Calm, systems-first, decisive."
