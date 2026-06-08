"""Core agent runtime package.

Public surface re-exported for ergonomic imports. The implementation is
organized into focused sub-packages:

- ``core.agent``     — the agent loop, factory, catalog, prompt builder, compaction
- ``core.leads``     — multi-lead orchestration (manager, supervisor, worker core)
- ``core.subagents`` — sub-agent lifecycle (registry, reaper)
- ``core.session``   — per-session run coordination + user-input queue
- ``core.ipc``       — worker stdin/stdout codecs
- ``core.scheduling``— cron scheduler
- ``core.prompts``   — prompt modules (referenced by agent YAML ``prompt_modules``)
"""

from __future__ import annotations

from feather.core.agent.base import BaseAgent
from feather.core.agent.catalog import AgentCatalog
from feather.core.agent.factory import AgentFactory
from feather.core.agent.prompt_builder import PromptBuilder

__all__ = ("BaseAgent", "AgentFactory", "AgentCatalog", "PromptBuilder")
