"""Prompt assembly for Feather agents."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from feather.core.agent.capabilities import CapabilityProfile
from feather.core.agent.catalog import AgentCatalog, render_catalog_block
from feather.models import AgentConfig, MCPServerConfig
from feather.skills.catalog import SkillCatalog
from feather.tools.registry import ToolRegistry


_MAX_MCP_ALLOWED_TOOLS_IN_PROMPT = 12


@dataclass(slots=True, frozen=True)
class PromptSections:
    """Cache-aware prompt sections for one agent prompt build."""

    cached_prefix: str
    dynamic_suffix: str

    def render(self) -> str:
        """Render the full prompt from its cache-aware sections."""

        return "\n\n".join(part for part in [self.cached_prefix.strip(), self.dynamic_suffix.strip()] if part)

    @property
    def cache_prefix(self) -> str:
        """The exact stable text :meth:`render` emits at the very front.

        Providers that place an explicit cache breakpoint (Anthropic /
        OpenRouter) split the rendered system prompt here so the breakpoint
        anchors the *static* prefix only — the per-turn dynamic suffix stays
        outside the cached region. Guaranteed to be a leading substring of
        :meth:`render` so the split is a pure prefix slice.
        """

        return self.cached_prefix.strip()


class PromptBuilder:
    """Build the system instructions for an agent."""

    def __init__(
        self,
        skill_catalog: SkillCatalog,
        tool_registry: ToolRegistry,
        agent_catalog: AgentCatalog | None = None,
    ) -> None:
        self._skill_catalog = skill_catalog
        self._tool_registry = tool_registry
        self._agent_catalog = agent_catalog

    def build(
        self,
        agent_config: AgentConfig,
        loaded_skill_names: list[str],
        *,
        memory_block: str | None = None,
        user_profile_block: str | None = None,
    ) -> str:
        """Construct a full system prompt for one agent.

        Args:
            agent_config: Current agent configuration.
            loaded_skill_names: Skills that should be injected in full.
            memory_block: Optional pre-rendered long-term-memory section
                produced by the read path. Empty / whitespace-only blocks
                are silently dropped so an absent block never produces a
                bare ``## Relevant memory`` header.
            user_profile_block: Optional verbatim contents of the
                ``.feather/user.md`` profile. Rendered inside the cached
                prefix so prompt caching stays effective for sessions
                where the profile is stable.

        Returns:
            Complete system instructions.
        """

        return self.build_sections(
            agent_config,
            loaded_skill_names,
            memory_block=memory_block,
            user_profile_block=user_profile_block,
        ).render()

    def build_sections(
        self,
        agent_config: AgentConfig,
        loaded_skill_names: list[str],
        *,
        memory_block: str | None = None,
        user_profile_block: str | None = None,
    ) -> PromptSections:
        """Build prompt sections with a stable prefix and dynamic suffix.

        The stable prefix is kept first so prompt caching can reuse an exact
        shared prefix across requests. Session-specific dynamic content, such as
        loaded skill bodies and the per-turn memory block, is appended later
        so changes to those sections never invalidate the cache.
        """

        prompt_module_texts = self._load_prompt_symbols(agent_config.prompt_modules)
        if agent_config.inline_prompt:
            prompt_module_texts.append(agent_config.inline_prompt)
        tool_prompt = "\n".join(self._tool_registry.prompts_for(agent_config.registered_tools))
        catalog_prompt = "\n".join(
            f"- {meta.name}: {meta.description}" for meta in self._skill_catalog.list_metadata()
        )
        mcp_catalog_prompt = self._render_mcp_catalog(agent_config.mcp_servers)
        loaded_sections: list[str] = []
        for skill_name in loaded_skill_names:
            loaded = self._skill_catalog.load_skill(skill_name)
            loaded_sections.append(
                "\n".join(
                    [
                        f'<skill name="{loaded.metadata.name}">',
                        loaded.content,
                        "</skill>",
                    ]
                )
            )

        static_prompt_sections = self._render_static_prompt_sections(prompt_module_texts)
        profile_text = (user_profile_block or "").strip() or "- No user profile available yet."
        # The dispatchable-agent catalog is fixed for the agent's whole session
        # (it depends only on the agent config + catalog, never per-turn state),
        # so it lives in the cached prefix — caching it is both correct and a
        # win. Gated on can_spawn, so non-spawn sub-agents emit nothing.
        dispatch_block = self._render_dispatchable_agents(agent_config)
        cached_prefix = "\n\n".join(
            [
                '<feather_system_prompt version="3">',
                "<static_cached_prefix>",
                static_prompt_sections,
                "<agent_profile>",
                f"<agent_name>{agent_config.name}</agent_name>",
                f"<agent_role>{agent_config.role}</agent_role>",
                f"<agent_personality>{agent_config.personality}</agent_personality>",
                *(
                    [f"<agent_soul>\n{agent_config.soul.strip()}\n</agent_soul>"]
                    if agent_config.soul
                    else []
                ),
                "</agent_profile>",
                "<user_profile>",
                profile_text,
                "</user_profile>",
                "<available_tools>",
                tool_prompt or "- No tools registered.",
                "</available_tools>",
                "<available_skills>",
                catalog_prompt or "- No skills available.",
                "</available_skills>",
                "<available_mcp_servers>",
                mcp_catalog_prompt,
                "</available_mcp_servers>",
                *(
                    ["<dispatchable_agents>", dispatch_block, "</dispatchable_agents>"]
                    if dispatch_block
                    else []
                ),
                "</static_cached_prefix>",
            ]
        )

        dynamic_parts: list[str] = [
            "<dynamic_prompt_extensions>",
            "<loaded_skills>",
            "\n\n".join(loaded_sections) if loaded_sections else "- No skills loaded in this session.",
            "</loaded_skills>",
        ]
        if memory_block and memory_block.strip():
            dynamic_parts.extend(
                [
                    "<long_term_memory>",
                    memory_block.strip(),
                    "</long_term_memory>",
                ]
            )
        dynamic_parts.extend(
            [
                "</dynamic_prompt_extensions>",
                "</feather_system_prompt>",
            ]
        )
        dynamic_suffix = "\n\n".join(dynamic_parts)
        return PromptSections(cached_prefix=cached_prefix, dynamic_suffix=dynamic_suffix)

    def _render_dispatchable_agents(self, agent_config: AgentConfig) -> str:
        """Render the catalog block for agents that can spawn sub-agents.

        Gated on the ``can_spawn`` capability (the lead by default), so the
        catalog appears only for agents that actually have the ``spawn_agent``
        tool — not hard-coded to ``role == "lead"``.
        """

        if self._agent_catalog is None:
            return ""
        if not CapabilityProfile.from_config(agent_config).can_spawn:
            return ""
        return render_catalog_block(self._agent_catalog.list_entries())

    def _render_mcp_catalog(self, servers: tuple[MCPServerConfig, ...]) -> str:
        """Render configured MCP metadata without connection or tool schemas."""

        if not servers:
            return "- No MCP servers available."
        return "\n".join(self._render_mcp_entry(server) for server in servers)

    def _render_mcp_entry(self, server: MCPServerConfig) -> str:
        """Render one safe MCP catalog row for prompt selection."""

        description = server.server_description or "No description provided."
        details = [f"transport: {server.transport}"]
        if server.allowed_tools:
            allowed = list(server.allowed_tools[:_MAX_MCP_ALLOWED_TOOLS_IN_PROMPT])
            if len(server.allowed_tools) > _MAX_MCP_ALLOWED_TOOLS_IN_PROMPT:
                allowed.append("...")
            details.append(f"allowed_tools: {', '.join(allowed)}")
        return f"- {server.label}: {description} ({'; '.join(details)})"

    def _load_prompt_symbols(self, targets: list[str]) -> list[str]:
        """Import the configured ordered prompt symbols.

        Args:
            targets: Ordered import targets in `module:symbol` form.

        Returns:
            Ordered prompt texts.
        """

        texts: list[str] = []
        for target in targets:
            module_name, symbol_name = target.split(":", maxsplit=1)
            module = importlib.import_module(module_name)
            texts.append(str(getattr(module, symbol_name)))
        return texts

    def _render_static_prompt_sections(self, prompt_module_texts: list[str]) -> str:
        """Render ordered prompt modules into explicit static sections."""

        parts: list[str] = []
        if prompt_module_texts:
            parts.extend(
                [
                    "<base_prompt>",
                    prompt_module_texts[0],
                    "</base_prompt>",
                ]
            )
        if len(prompt_module_texts) >= 2:
            parts.extend(
                [
                    "<agent_prompt>",
                    prompt_module_texts[1],
                    "</agent_prompt>",
                ]
            )
        if len(prompt_module_texts) > 2:
            extra_sections = []
            for index, prompt_text in enumerate(prompt_module_texts[2:], start=3):
                extra_sections.extend(
                    [
                        f'<prompt_module index="{index}">',
                        prompt_text,
                        "</prompt_module>",
                    ]
                )
            parts.extend(
                [
                    "<additional_static_prompts>",
                    "\n\n".join(extra_sections),
                    "</additional_static_prompts>",
                ]
            )
        return "\n\n".join(parts)
