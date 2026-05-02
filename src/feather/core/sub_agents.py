"""Specialist sub-agent classes dispatched by the lead via spawn_agent."""

from __future__ import annotations

from feather.core.base_agent import BaseAgent


class ExploreAgent(BaseAgent):
    """Thin codebase-exploration sub-agent over the reusable base loop."""


class ResearchAgent(BaseAgent):
    """Thin web-research sub-agent over the reusable base loop."""


class ValidateAgent(BaseAgent):
    """Thin verification sub-agent over the reusable base loop."""


class CustomAgent(BaseAgent):
    """User-defined sub-agent fully configured via its YAML + inline_prompt.

    Custom agents do not carry any bespoke class-level behavior. They exist as
    a named class so operators can introspect and log the agent type without
    conflating user-authored agents with the internal ``DefaultAgent``
    fallback used for unknown roles.
    """
