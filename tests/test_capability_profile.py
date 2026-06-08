"""CapabilityProfile derives an agent's allowed features from its config.

The pivot: "a sub-agent is the lead with features disabled." The profile is
the single source of truth for that distinction.
"""

from __future__ import annotations

from feather.core.agent.capabilities import CapabilityProfile
from feather.models import AgentConfig


def _cfg(role: str, **kw) -> AgentConfig:
    base = dict(
        name=role.title(),
        role=role,
        personality="",
        prompt_modules=[],
        registered_tools=[],
    )
    base.update(kw)
    return AgentConfig(**base)


def test_lead_profile_has_everything():
    p = CapabilityProfile.from_config(_cfg("lead", memory_enabled=True))
    assert p.is_lead and not p.dispatchable
    assert p.can_spawn and p.can_message_user and p.can_schedule
    assert p.memory_enabled


def test_subagent_profile_is_reduced():
    for role in ("explore", "research", "validate", "custom"):
        p = CapabilityProfile.from_config(_cfg(role))
        assert not p.is_lead and p.dispatchable
        assert not p.can_spawn and not p.can_message_user and not p.can_schedule


def test_unknown_role_defaults_to_subagent():
    p = CapabilityProfile.from_config(_cfg("weird-role"))
    assert not p.is_lead and p.dispatchable and not p.can_spawn


def test_memory_flag_follows_config():
    assert not CapabilityProfile.from_config(_cfg("explore", memory_enabled=False)).memory_enabled
    assert CapabilityProfile.from_config(_cfg("explore", memory_enabled=True)).memory_enabled


# --- YAML-level capability overrides (Task 3 adds the AgentConfig field) ---


def test_yaml_capability_override_enables_spawn_on_subagent():
    cfg = _cfg("explore", capabilities={"can_spawn": True})
    p = CapabilityProfile.from_config(cfg)
    assert p.can_spawn and not p.is_lead  # gate is real, not role-hardcoded


def test_yaml_capability_override_disables_lead_memory():
    cfg = _cfg("lead", memory_enabled=True, capabilities={"memory_enabled": False})
    assert not CapabilityProfile.from_config(cfg).memory_enabled


def test_frozen_value_object():
    import dataclasses
    import pytest

    p = CapabilityProfile.from_config(_cfg("lead"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.is_lead = False  # type: ignore[misc]
