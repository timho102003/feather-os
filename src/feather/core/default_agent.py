"""Default agent wrapper used when no specialized subclass is registered."""

from __future__ import annotations

from feather.core.base_agent import BaseAgent


class DefaultAgent(BaseAgent):
    """Thin default wrapper over the reusable base agent loop."""
