"""Explicit capability model: a sub-agent is the lead with features disabled.

Historically the lead/sub-agent distinction lived in scattered places — a
``_LEAD_ONLY_TOOLS`` frozenset filtered by ``role == "lead"``, empty marker
subclasses, a ``_NON_DISPATCHABLE_ROLES`` set, and ad-hoc role checks. This
module collapses all of that into one declarative value object derived from
the agent's config, so every gate consults the same source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feather.models import AgentConfig

__all__ = ("CapabilityProfile",)

_LEAD_ROLE = "lead"


@dataclass(slots=True, frozen=True)
class CapabilityProfile:
    """What an agent may do.

    The lead has every capability on; a sub-agent is the same agent with some
    turned off. Derived from the agent's role (defaults) and any explicit
    ``capabilities:`` overrides declared in its YAML.

    Attributes:
        is_lead: Top-level identity — supervised, switchable in the TUI, and
            in direct conversation with the user.
        dispatchable: May be launched as a sub-agent via ``spawn_agent``.
        can_spawn: Gets the orchestration tools (``spawn_agent``,
            ``terminate_agent``, ``task_create``/``task_stop``/``task_resume``).
        can_message_user: Gets the user-facing tools (``ask_user``,
            ``manage_memory``, ``user_info``).
        can_schedule: Gets the cron tools.
        memory_enabled: Long-term memory read augmentation + write-back.
    """

    is_lead: bool
    dispatchable: bool
    can_spawn: bool
    can_message_user: bool
    can_schedule: bool
    memory_enabled: bool

    @classmethod
    def from_config(cls, config: "AgentConfig") -> "CapabilityProfile":
        """Derive the profile from role defaults, layered with YAML overrides.

        Role defaults: a ``lead`` gets everything; any other role is a
        reduced sub-agent. An optional ``capabilities`` mapping on the config
        (e.g. ``{"can_spawn": true}``) overrides individual fields, so a
        custom YAML can grant or revoke one capability without inventing a new
        role. Unknown override keys are ignored.
        """

        is_lead = config.role == _LEAD_ROLE
        defaults: dict[str, bool] = {
            "is_lead": is_lead,
            "dispatchable": not is_lead,
            "can_spawn": is_lead,
            "can_message_user": is_lead,
            "can_schedule": is_lead,
            "memory_enabled": bool(config.memory_enabled),
        }
        overrides = getattr(config, "capabilities", None) or {}
        for key, value in overrides.items():
            if key in defaults:
                defaults[key] = bool(value)
        return cls(**defaults)
