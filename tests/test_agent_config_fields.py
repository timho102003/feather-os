"""load_agent_config round-trips the lead-identity + capability fields."""

from __future__ import annotations

from pathlib import Path

from feather.config import load_agent_config
from feather.core.agent.capabilities import CapabilityProfile


def _write_agent(root: Path, name: str, body: str) -> None:
    agents = root / "config" / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / f"{name}.yaml").write_text(body, encoding="utf-8")


def test_soul_and_display_fields_round_trip(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "tim",
        """name: Tim
role: lead
personality: Decisive and warm.
soul: |
  You are Tim, a pragmatic operator who keeps the team moving.
color: "#22d3ee"
emoji: "🧭"
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
registered_tools: []
""",
    )
    cfg = load_agent_config(tmp_path, "tim")
    assert cfg.name == "Tim"
    assert "pragmatic operator" in cfg.soul
    assert cfg.color == "#22d3ee"
    assert cfg.emoji == "🧭"
    assert cfg.capabilities == {}


def test_capability_overrides_round_trip_and_apply(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "explore-plus-custom",
        """name: ExplorePlus
role: explore
personality: Curious.
capabilities:
  can_spawn: true
  memory_enabled: true
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
registered_tools: []
""",
    )
    cfg = load_agent_config(tmp_path, "explore-plus-custom")
    assert cfg.capabilities == {"can_spawn": True, "memory_enabled": True}
    profile = CapabilityProfile.from_config(cfg)
    assert profile.can_spawn is True  # override beats the explore role default
    assert profile.memory_enabled is True
    assert profile.is_lead is False


def test_missing_optional_fields_default_empty(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "plain",
        """name: Plain
role: explore
personality: Minimal.
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
registered_tools: []
""",
    )
    cfg = load_agent_config(tmp_path, "plain")
    assert cfg.soul == ""
    assert cfg.color is None
    assert cfg.emoji is None
    assert cfg.capabilities == {}
