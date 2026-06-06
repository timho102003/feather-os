"""Tests for prompt assembly."""

from pathlib import Path

from feather.core.agent.prompt_builder import PromptBuilder
from feather.models import AgentConfig, MCPServerConfig
from feather.skills.catalog import SkillCatalog
from feather.tools.ask_user_tool import AskUserTool
from feather.tools.registry import ToolRegistry


def test_prompt_builder_includes_tools_skill_catalog_and_loaded_skill(tmp_path: Path) -> None:
    """Prompt assembly should include tool prompts, skill metadata, and loaded skill content."""

    skill_dir = tmp_path / ".feather" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo-skill
description: Demo skill
---

# Demo Skill

Loaded skill body.
""",
        encoding="utf-8",
    )

    skill_catalog = SkillCatalog(tmp_path / ".feather" / "skills")
    tool_registry = ToolRegistry([AskUserTool()])
    builder = PromptBuilder(skill_catalog, tool_registry)
    agent_config = AgentConfig(
        name="Lead",
        role="lead",
        personality="Decisive",
        prompt_modules=[
            "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
            "feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT",
        ],
        registered_tools=["ask_user"],
    )

    sections = builder.build_sections(agent_config, loaded_skill_names=["demo-skill"])
    prompt = sections.render()

    assert prompt.startswith('<feather_system_prompt version="3">')
    assert "<static_cached_prefix>" in prompt
    assert "<dynamic_prompt_extensions>" in prompt
    assert "<base_prompt>" in prompt
    assert "<agent_prompt>" in prompt
    assert "<lead_agent_identity>" in prompt
    assert "You are Feather's lead agent." in prompt
    assert "You are a Feather agent." in prompt
    assert "use `read_file` when available to inspect it" in prompt
    assert "Use `list_mcp_servers`" in prompt
    assert "register only the MCP servers needed for the current task" in prompt
    assert "`ask_user`" in prompt
    assert "demo-skill: Demo skill" in prompt
    assert '<skill name="demo-skill">' in prompt
    assert "Loaded skill body." in prompt


def test_prompt_builder_keeps_cacheable_prefix_stable_when_loaded_skills_change(tmp_path: Path) -> None:
    """Loaded skill bodies should live in the dynamic suffix, not the cacheable prefix."""

    skill_dir = tmp_path / ".feather" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo-skill
description: Demo skill
---

# Demo Skill

Loaded skill body.
""",
        encoding="utf-8",
    )

    skill_catalog = SkillCatalog(tmp_path / ".feather" / "skills")
    tool_registry = ToolRegistry([AskUserTool()])
    builder = PromptBuilder(skill_catalog, tool_registry)
    agent_config = AgentConfig(
        name="Lead",
        role="lead",
        personality="Decisive",
        prompt_modules=[
            "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
            "feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT",
        ],
        registered_tools=["ask_user"],
    )

    without_skill = builder.build_sections(agent_config, loaded_skill_names=[])
    with_skill = builder.build_sections(agent_config, loaded_skill_names=["demo-skill"])

    assert without_skill.cached_prefix == with_skill.cached_prefix
    assert "Loaded skill body." not in with_skill.cached_prefix
    assert "Loaded skill body." in with_skill.dynamic_suffix


def test_cache_prefix_is_leading_substring_of_render(tmp_path: Path) -> None:
    """The split point the providers slice on must be an exact leading substring.

    The breakpoint translators do ``instructions[len(cache_prefix):]``; if
    ``cache_prefix`` were not a true prefix of ``render()`` the slice would
    corrupt the system prompt (and silently fall back to one cached block).
    """

    builder = _basic_builder(tmp_path)
    sections = builder.build_sections(
        _agent("lead"),
        loaded_skill_names=[],
        memory_block="## Relevant memory\n- recalled fact",
        user_profile_block="name: Tim",
    )

    assert sections.cache_prefix  # non-empty
    rendered = sections.render()
    assert rendered.startswith(sections.cache_prefix)
    assert sections.cache_prefix != rendered  # there IS a dynamic remainder to exclude


def test_cached_prefix_byte_identical_across_memory_changes(tmp_path: Path) -> None:
    """Per-turn memory must never perturb the cached prefix (byte-for-byte)."""

    builder = _basic_builder(tmp_path)
    cfg = _agent("lead")
    base = builder.build_sections(cfg, loaded_skill_names=[]).cache_prefix
    variants = [
        builder.build_sections(
            cfg, loaded_skill_names=[], memory_block="## Relevant memory\n- fact A"
        ).cache_prefix,
        builder.build_sections(
            cfg, loaded_skill_names=[], memory_block="completely different recall"
        ).cache_prefix,
    ]
    for prefix in variants:
        assert prefix.encode("utf-8") == base.encode("utf-8")


def test_cached_prefix_is_deterministic(tmp_path: Path) -> None:
    """Same inputs → identical prefix bytes every build.

    Guards against a future regression that smuggles a timestamp / uuid /
    unordered-dict render into the cached prefix, which would silently drop the
    runtime cache-hit rate to zero.
    """

    builder = _basic_builder(tmp_path)
    cfg = _agent("lead")
    prefixes = {
        builder.build_sections(cfg, loaded_skill_names=[]).cache_prefix for _ in range(20)
    }
    assert len(prefixes) == 1


def test_static_first_dynamic_last(tmp_path: Path) -> None:
    """Static content lands before the cache boundary; dynamic content after it."""

    builder = _basic_builder(tmp_path)
    sections = builder.build_sections(
        _agent("lead"),
        loaded_skill_names=[],
        memory_block="recalled fact xyz",
        user_profile_block="name: Tim",
    )
    rendered = sections.render()
    boundary = rendered.index("</static_cached_prefix>")
    # Dynamic content lives strictly after the static boundary.
    assert rendered.index("recalled fact xyz") > boundary
    assert rendered.index("<long_term_memory>") > boundary
    # The user profile is intentionally cached, so it sits before the boundary.
    assert rendered.index("name: Tim") < boundary


def test_prompt_builder_includes_mcp_catalog_metadata_without_connection_details(
    tmp_path: Path,
) -> None:
    """Agents should see assigned MCP metadata without eager remote tool schemas."""

    skill_catalog = SkillCatalog(tmp_path / ".feather" / "skills")
    tool_registry = ToolRegistry([AskUserTool()])
    builder = PromptBuilder(skill_catalog, tool_registry)
    agent_config = AgentConfig(
        name="Lead",
        role="lead",
        personality="Decisive",
        prompt_modules=[
            "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
            "feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT",
        ],
        registered_tools=["ask_user"],
        mcp_servers=(
            MCPServerConfig(
                label="docs",
                server_url="https://example.internal/mcp",
                server_description="Private documentation search",
                allowed_tools=("search", "fetch"),
                headers={"Authorization": "Bearer secret"},
            ),
            MCPServerConfig(
                label="playwright",
                transport="stdio",
                command="npx",
                args=("-y", "@playwright/mcp@latest"),
                server_description="Browser automation",
            ),
        ),
    )

    prompt = builder.build(agent_config, loaded_skill_names=[])

    assert "<available_mcp_servers>" in prompt
    assert "docs: Private documentation search" in prompt
    assert "allowed_tools: search, fetch" in prompt
    assert "playwright: Browser automation" in prompt
    assert "https://example.internal/mcp" not in prompt
    assert "Bearer secret" not in prompt
    assert "@playwright/mcp" not in prompt


def test_prompt_builder_includes_user_profile_block_in_cached_prefix(tmp_path: Path) -> None:
    """When provided, the user profile renders inside the cached prefix."""

    skill_catalog = SkillCatalog(tmp_path / ".feather" / "skills")
    tool_registry = ToolRegistry([AskUserTool()])
    builder = PromptBuilder(skill_catalog, tool_registry)
    agent_config = AgentConfig(
        name="Lead",
        role="lead",
        personality="Decisive",
        prompt_modules=[
            "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
            "feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT",
        ],
        registered_tools=["ask_user"],
    )

    sections = builder.build_sections(
        agent_config,
        loaded_skill_names=[],
        user_profile_block="---\nname: Tim\n---\n\n## About\nHi.",
    )

    assert "<user_profile>" in sections.cached_prefix
    assert "name: Tim" in sections.cached_prefix
    assert "name: Tim" not in sections.dynamic_suffix


def test_prompt_builder_renders_placeholder_when_profile_absent(tmp_path: Path) -> None:
    """When no profile is supplied, a clear placeholder is rendered."""

    skill_catalog = SkillCatalog(tmp_path / ".feather" / "skills")
    tool_registry = ToolRegistry([AskUserTool()])
    builder = PromptBuilder(skill_catalog, tool_registry)
    agent_config = AgentConfig(
        name="Lead",
        role="lead",
        personality="Decisive",
        prompt_modules=[
            "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
            "feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT",
        ],
        registered_tools=["ask_user"],
    )

    prompt = builder.build(agent_config, loaded_skill_names=[])
    assert "<user_profile>" in prompt
    assert "No user profile available yet" in prompt


def test_prompt_builder_supports_additional_static_prompt_modules(tmp_path: Path) -> None:
    """Additional prompt modules should render after base and agent prompt sections."""

    skill_catalog = SkillCatalog(tmp_path / ".feather" / "skills")
    tool_registry = ToolRegistry([AskUserTool()])
    builder = PromptBuilder(skill_catalog, tool_registry)
    agent_config = AgentConfig(
        name="Lead",
        role="lead",
        personality="Decisive",
        prompt_modules=[
            "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
            "feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT",
            "feather.core.prompts.default_agent_prompt:DEFAULT_AGENT_PROMPT",
        ],
        registered_tools=["ask_user"],
    )

    prompt = builder.build(agent_config, loaded_skill_names=[])

    assert "<additional_static_prompts>" in prompt
    assert '<prompt_module index="3">' in prompt


def _basic_builder(tmp_path: Path, agent_catalog=None) -> PromptBuilder:
    skill_catalog = SkillCatalog(tmp_path / ".feather" / "skills")
    tool_registry = ToolRegistry([AskUserTool()])
    return PromptBuilder(skill_catalog, tool_registry, agent_catalog=agent_catalog)


def _agent(role: str, **kw) -> AgentConfig:
    base = dict(
        name=role.title(),
        role=role,
        personality="Decisive",
        prompt_modules=["feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT"],
        registered_tools=["ask_user"],
    )
    base.update(kw)
    return AgentConfig(**base)


def test_prompt_builder_injects_soul_when_present(tmp_path: Path) -> None:
    builder = _basic_builder(tmp_path)
    cfg = _agent("lead", soul="You are Tim, a pragmatic operator.")
    prompt = builder.build_sections(cfg, loaded_skill_names=[]).render()
    assert "<agent_soul>" in prompt
    assert "pragmatic operator" in prompt


def test_prompt_builder_omits_soul_when_absent(tmp_path: Path) -> None:
    builder = _basic_builder(tmp_path)
    prompt = builder.build_sections(_agent("lead"), loaded_skill_names=[]).render()
    assert "<agent_soul>" not in prompt


def test_dispatch_catalog_gated_by_spawn_capability(tmp_path: Path) -> None:
    from feather.core.agent.catalog import AgentCatalogEntry

    class _FakeCatalog:
        def list_entries(self):
            return [
                AgentCatalogEntry(name="lead", role="lead", description="", personality=""),
                AgentCatalogEntry(
                    name="explore", role="explore", description="Find code", personality=""
                ),
            ]

    builder = _basic_builder(tmp_path, agent_catalog=_FakeCatalog())

    # A lead (can_spawn) sees the dispatchable catalog (and not itself).
    lead_prompt = builder.build_sections(_agent("lead"), loaded_skill_names=[]).render()
    assert "<dispatchable_agents>" in lead_prompt
    assert "`explore`" in lead_prompt

    # A sub-agent (no can_spawn) never gets the catalog.
    sub_prompt = builder.build_sections(_agent("explore"), loaded_skill_names=[]).render()
    assert "<dispatchable_agents>" not in sub_prompt


def test_dispatch_catalog_follows_capability_override(tmp_path: Path) -> None:
    """A sub-agent YAML that grants can_spawn DOES get the catalog — the gate
    is the capability, not the role string."""
    from feather.core.agent.catalog import AgentCatalogEntry

    class _FakeCatalog:
        def list_entries(self):
            return [
                AgentCatalogEntry(name="explore", role="explore", description="x", personality="")
            ]

    builder = _basic_builder(tmp_path, agent_catalog=_FakeCatalog())
    cfg = _agent("explore", capabilities={"can_spawn": True})
    prompt = builder.build_sections(cfg, loaded_skill_names=[]).render()
    assert "<dispatchable_agents>" in prompt
