"""Tests for PromptBuilder.memory_block injection."""

from __future__ import annotations

from pathlib import Path

from feather.core.agent.prompt_builder import PromptBuilder
from feather.models import AgentConfig
from feather.skills.catalog import SkillCatalog
from feather.tools.ask_user_tool import AskUserTool
from feather.tools.registry import ToolRegistry


def _builder(tmp_path: Path) -> PromptBuilder:
    skill_catalog = SkillCatalog(tmp_path / ".feather" / "skills")
    tool_registry = ToolRegistry([AskUserTool()])
    return PromptBuilder(skill_catalog, tool_registry)


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="Lead",
        role="lead",
        personality="Decisive",
        prompt_modules=[
            "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
        ],
        registered_tools=["ask_user"],
    )


def test_memory_block_param_is_optional_and_default_omits_block(tmp_path: Path) -> None:
    """build() without memory_block produces no 'Relevant memory' section."""
    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    builder = _builder(tmp_path)
    prompt = builder.build(_agent_config(), loaded_skill_names=[])
    assert "Relevant memory" not in prompt


def test_memory_block_appears_in_dynamic_suffix_when_provided(tmp_path: Path) -> None:
    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    builder = _builder(tmp_path)
    block = "## Relevant memory from past conversations\n1. [0.9] the user prefers Python"
    sections = builder.build_sections(
        _agent_config(), loaded_skill_names=[], memory_block=block
    )
    assert block in sections.dynamic_suffix
    assert block not in sections.cached_prefix


def test_memory_block_does_not_appear_when_empty_or_whitespace(tmp_path: Path) -> None:
    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    builder = _builder(tmp_path)
    sections_empty = builder.build_sections(
        _agent_config(), loaded_skill_names=[], memory_block=""
    )
    sections_ws = builder.build_sections(
        _agent_config(), loaded_skill_names=[], memory_block="   \n   "
    )
    sections_none = builder.build_sections(
        _agent_config(), loaded_skill_names=[], memory_block=None
    )
    # None of them include a memory section header
    for s in (sections_empty, sections_ws, sections_none):
        assert "Relevant memory" not in s.render()


def test_memory_block_does_not_pollute_cached_prefix(tmp_path: Path) -> None:
    """Memory changes per turn — must NOT live in the prompt-cache prefix."""
    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    builder = _builder(tmp_path)
    s_a = builder.build_sections(
        _agent_config(), loaded_skill_names=[], memory_block="MEM_A"
    )
    s_b = builder.build_sections(
        _agent_config(), loaded_skill_names=[], memory_block="MEM_B"
    )
    assert s_a.cached_prefix == s_b.cached_prefix
    assert "MEM_A" in s_a.dynamic_suffix
    assert "MEM_B" in s_b.dynamic_suffix
