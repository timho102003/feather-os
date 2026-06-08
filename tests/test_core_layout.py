"""Guards on load-bearing module paths after the core/ reorg.

These paths are referenced as strings outside the Python import graph
(agent YAML ``prompt_modules``; ``python -m`` subprocess targets), so a
future reorg that silently breaks them would not be caught by ordinary
import errors. Pin them here.
"""

from __future__ import annotations

import importlib


def test_prompt_modules_still_importable():
    # Agent YAMLs reference these as dotted ``prompt_modules`` strings; the
    # reorg must NOT move ``feather.core.prompts``.
    for target in (
        "feather.core.prompts.base_agent_prompt",
        "feather.core.prompts.lead_agent_prompt",
        "feather.core.prompts.agent_messaging_protocol",
        "feather.core.prompts.explore_agent_prompt",
        "feather.core.prompts.research_agent_prompt",
        "feather.core.prompts.validate_agent_prompt",
    ):
        assert importlib.import_module(target) is not None


def test_subprocess_entrypoints_importable():
    # ``python -m`` targets used by the lead/sub-agent process model.
    assert importlib.import_module("feather.subagent_entry") is not None
    assert importlib.import_module("feather.lead_worker_entry") is not None


def test_core_public_reexports():
    from feather.core import AgentCatalog, AgentFactory, BaseAgent, PromptBuilder

    assert all((AgentCatalog, AgentFactory, BaseAgent, PromptBuilder))


def test_new_subpackages_importable():
    for target in (
        "feather.core.agent.base",
        "feather.core.agent.factory",
        "feather.core.agent.catalog",
        "feather.core.agent.compaction",
        "feather.core.agent.prompt_builder",
        "feather.core.leads.supervisor",
        "feather.core.leads.worker_core",
        "feather.core.subagents.registry",
        "feather.core.subagents.reaper",
        "feather.core.session.coordinator",
        "feather.core.session.input_queue",
        "feather.core.ipc.event_codec",
        "feather.core.ipc.command_codec",
        "feather.core.scheduling.cron_scheduler",
    ):
        assert importlib.import_module(target) is not None
