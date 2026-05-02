"""Lead-agent orchestration loop."""

from __future__ import annotations

from feather.core.base_agent import BaseAgent


class LeadAgent(BaseAgent):
    """Thin lead-agent wrapper over the reusable base agent loop."""
