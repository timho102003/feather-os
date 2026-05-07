"""Shared runtime factory for building configured Feather agents."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

from feather.config import load_agent_config
from feather.core.agent_catalog import AgentCatalog
from feather.core.base_agent import BaseAgent
from feather.core.compaction import ContextCompactor
from feather.core.default_agent import DefaultAgent
from feather.core.input_queue import UserInputQueue
from feather.core.install_mode import detect_install_mode
from feather.core.lead_agent import LeadAgent
from feather.core.prompt_builder import PromptBuilder
from feather.core.session_run_coordinator import SessionRunCoordinator
from feather.core.subagent_registry import SubagentRegistry
from feather.storage.agent_message_store import AgentMessageStore
from feather.core.sub_agents import CustomAgent, ExploreAgent, ResearchAgent, ValidateAgent
from feather.memory.context import current_session_id
from feather.memory.reader import NoOpMemoryReader
from feather.memory.runtime import MemoryStack
from feather.memory.trigger import NoOpMemoryTrigger
from feather.mcp_client import (
    MCPClientManager,
    MCPProxyTool,
    mcp_servers_for,
    should_proxy_mcp_server,
)
from feather.models import AgentConfig, AppConfig, MCPServerConfig
from feather.providers.base import BaseLLMProvider
from feather.providers.claude_provider import ClaudeMessagesProvider
from feather.providers.openai_provider import OpenAIResponsesProvider
from feather.providers.openrouter_provider import OpenRouterChatProvider
from feather.profile import UserProfileStore
from feather.providers.parallel_client import ParallelClient
from feather.skills.catalog import SkillCatalog
from feather.storage.attachment_store import AttachmentStore
from feather.storage.cron_store import CronJobStore
from feather.storage.session_store import SessionStore
from feather.storage.task_store import TaskStore
from feather.storage.tool_output_store import ToolOutputStore
from feather.tools.ask_user_tool import AskUserTool
from feather.tools.base import BaseTool
from feather.tools.bash_tool import BashTool
from feather.tools.cron_tools import CreateCronTool, DeleteCronTool, ListCronsTool, UpdateCronTool
from feather.tools.grep_tool import GrepTool
from feather.tools.manage_memory_tool import ManageMemoryTool
from feather.tools.mcp_tools import ListMCPServersTool, RegisterMCPServerTool
from feather.tools.parallel_extract_tool import ParallelExtractTool
from feather.tools.parallel_search_tool import ParallelSearchTool
from feather.tools.pdf_tool import ReadPdfTool
from feather.tools.read_file_tool import ReadFileTool
from feather.tools.recall_memory_tool import RecallMemoryTool
from feather.tools.request_restart_tool import RequestRestartTool
from feather.tools.submit_github_report_tool import SubmitGithubReportTool
from feather.tools.write_file_tool import WriteFileTool
from feather.tools.registry import ToolRegistry
from feather.tools.send_message_tool import SendMessageTool
from feather.tools.skill_tool import LoadSkillTool
from feather.tools.spawn_agent_tool import SpawnAgentTool
from feather.tools.task_tools import (
    RequestInputTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskResumeTool,
    TaskStopTool,
    TaskUpdateTool,
)
from feather.tools.terminate_agent_tool import TerminateAgentTool

logger = logging.getLogger(__name__)

ToolBuilder = Callable[[], BaseTool]

_LEAD_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "spawn_agent",
        "terminate_agent",
        "create_cron",
        "update_cron",
        "delete_cron",
        "list_crons",
        "ask_user",
        # Only the lead is in direct conversation with the user, so only
        # the lead can act on a "remember/forget" request. Sub-agents
        # writing user memory behind the user's back is a footgun.
        "manage_memory",
        # Same rationale as manage_memory: the lead is the only agent
        # talking to the user, so only the lead can mutate the persistent
        # user profile.
        "user_info",
        "task_create",
        "task_stop",
        "task_resume",
    }
)


class AgentFactory:
    """Build configured agents with shared runtime services."""

    def __init__(
        self,
        *,
        root: Path,
        app_config: AppConfig,
        provider: BaseLLMProvider,
        session_store: SessionStore,
        cron_store: CronJobStore | None = None,
        tool_output_store: ToolOutputStore,
        skill_catalog: SkillCatalog,
        parallel_client: ParallelClient | None = None,
        memory_stack: MemoryStack | None = None,
        run_coordinator: SessionRunCoordinator | None = None,
        input_queue: UserInputQueue | None = None,
        agent_message_store: AgentMessageStore | None = None,
        task_store: TaskStore | None = None,
        subagent_registry: SubagentRegistry | None = None,
        attachment_store: AttachmentStore | None = None,
        agent_classes: Mapping[str, type[BaseAgent]] | None = None,
        tool_builders: Mapping[str, ToolBuilder] | None = None,
        profile_store: UserProfileStore | None = None,
        paths: object = None,
    ) -> None:
        self._root = root
        self._paths = paths
        self._app_config = app_config
        self._provider = provider
        self._session_store = session_store
        self._cron_store = cron_store
        self._tool_output_store = tool_output_store
        self._skill_catalog = skill_catalog
        self._parallel_client = parallel_client
        self._memory_stack = memory_stack
        self._run_coordinator = run_coordinator or SessionRunCoordinator()
        self._input_queue = input_queue
        self._agent_message_store = agent_message_store
        self._task_store = task_store
        self._subagent_registry = subagent_registry
        self._attachment_store = attachment_store
        self._agent_classes = {
            "lead": LeadAgent,
            "explore": ExploreAgent,
            "research": ResearchAgent,
            "validate": ValidateAgent,
            "custom": CustomAgent,
            **(dict(agent_classes) if agent_classes is not None else {}),
        }
        self._current_agent_name: str = ""
        self._profile_store = profile_store
        self._tool_builders = dict(tool_builders) if tool_builders is not None else self._default_tool_builders()
        self._agent_catalog = AgentCatalog(root, paths=self._paths)
        self._mcp_manager = MCPClientManager()
        # Cache of providers keyed by provider name, so a per-agent override
        # reuses one httpx client across rebuilds instead of spawning a new
        # one per build() call.
        self._providers_by_name: dict[str, BaseLLMProvider] = {
            self._default_provider_name(): provider,
        }

    async def aclose(self) -> None:
        """Close any per-provider resources the factory built on demand.

        The shared app-level ``self._provider`` is owned by the runtime
        (``FeatherRuntime`` builds it, the runtime controls its lifetime).
        Any *additional* providers spun up for per-agent overrides were
        built by the factory and are closed here so httpx clients don't
        leak when the runtime shuts down.
        """

        default_name = self._default_provider_name()
        await self._mcp_manager.aclose()
        for name, provider in list(self._providers_by_name.items()):
            if name == default_name:
                continue
            closer = getattr(provider, "aclose", None)
            if closer is None:
                continue
            try:
                await closer()
            except Exception:  # noqa: BLE001
                logger.exception("agent_factory.provider_close_error provider=%s", name)

    def _build_spawn_agent_builder(self) -> ToolBuilder:
        """Return a builder for SpawnAgentTool; bound to the current agent's name."""

        def builder() -> BaseTool:
            parent_name = self._current_agent_name or "lead"
            registry = self._subagent_registry
            if registry is None:
                raise RuntimeError(
                    "spawn_agent requested but SubagentRegistry is not available"
                )
            return SpawnAgentTool(
                root=self._root,
                agent_catalog=self._agent_catalog,
                registry=registry,
                parent_agent_name=parent_name,
                task_store=self._task_store,
            )

        return builder

    def _build_terminate_agent_builder(self) -> ToolBuilder:
        """Return a builder for TerminateAgentTool; bound to the current agent's name."""

        def builder() -> BaseTool:
            parent_name = self._current_agent_name or "lead"
            registry = self._subagent_registry
            if registry is None:
                raise RuntimeError(
                    "terminate_agent requested but SubagentRegistry is not available"
                )
            message_store = self._agent_message_store
            if message_store is None:
                raise RuntimeError(
                    "terminate_agent requested but AgentMessageStore is not available"
                )
            return TerminateAgentTool(
                registry=registry,
                agent_message_store=message_store,
                parent_agent_name=parent_name,
            )

        return builder

    def _build_send_message_builder(
        self, message_store: AgentMessageStore
    ) -> ToolBuilder:
        """Return a builder that binds send_message to the current agent name.

        The agent name is only known when :meth:`build` selects one; the
        builder is therefore a proxy that reads ``self._current_agent_name``
        — set for the duration of a single ``build`` call.
        """

        def builder() -> BaseTool:
            sender = self._current_agent_name
            if not sender:
                raise RuntimeError(
                    "send_message builder invoked outside AgentFactory.build"
                )
            return SendMessageTool(
                message_store,
                from_agent_name=sender,
                session_store=self._session_store,
                subagent_registry=self._subagent_registry,
            )

        return builder

    def _default_provider_name(self) -> str:
        """Return the name of the app-level default provider."""

        return (self._app_config.active_provider or "openai").strip().lower()

    def _resolve_provider(self, agent_config: AgentConfig) -> BaseLLMProvider:
        """Resolve which provider to use for ``agent_config``.

        Returns the shared app-level provider unless the agent YAML
        specifies a different ``provider:`` value. Built providers are
        cached by name so repeated builds of the same agent, or of
        different agents sharing one override, reuse one httpx client.
        """

        agent_provider = (agent_config.provider or "").strip().lower()
        if not agent_provider or agent_provider == self._default_provider_name():
            return self._provider
        cached = self._providers_by_name.get(agent_provider)
        if cached is not None:
            return cached
        if agent_provider == "openai":
            built: BaseLLMProvider = OpenAIResponsesProvider(self._app_config.openai)
        elif agent_provider == "openrouter":
            if self._app_config.openrouter is None:
                raise ValueError(
                    f"Agent `{agent_config.name}` requested provider=openrouter "
                    "but no `openrouter:` block in app.yaml"
                )
            built = OpenRouterChatProvider(self._app_config.openrouter)
        elif agent_provider == "claude":
            if self._app_config.claude is None:
                raise ValueError(
                    f"Agent `{agent_config.name}` requested provider=claude "
                    "but no `claude:` block in app.yaml"
                )
            built = ClaudeMessagesProvider(self._app_config.claude)
        else:
            raise ValueError(
                f"Agent `{agent_config.name}` requested unknown provider "
                f"`{agent_provider}` "
                "(expected 'openai', 'openrouter', or 'claude')"
            )
        self._providers_by_name[agent_provider] = built
        return built

    def _resolve_model_name(self, agent_config: AgentConfig) -> str:
        """Resolve the conversation model name for this agent.

        Used by :class:`BaseAgent` for memory-subsystem "inherit model"
        defaults. Order of precedence: explicit agent override → chosen
        provider's default model → app-level openai default (legacy).
        """

        if agent_config.model:
            return agent_config.model
        provider_name = (agent_config.provider or self._default_provider_name()).strip().lower()
        if provider_name == "openrouter" and self._app_config.openrouter is not None:
            return self._app_config.openrouter.model
        if provider_name == "claude" and self._app_config.claude is not None:
            return self._app_config.claude.model
        return self._app_config.openai.model

    def _resolve_provider_name(self, agent_config: AgentConfig) -> str:
        """Return the provider key that will back ``agent_config``."""

        return (agent_config.provider or self._default_provider_name()).strip().lower()

    def build(self, agent_name: str) -> BaseAgent:
        """Build one agent instance from its config file."""

        raw_config = load_agent_config(self._root, agent_name)
        self._current_agent_name = raw_config.name
        provider_name = self._resolve_provider_name(raw_config)
        mcp_servers = self._supported_mcp_servers(
            mcp_servers_for(
                self._app_config.mcp,
                provider_name=provider_name,
                agent_name=raw_config.name,
            ),
            provider_name=provider_name,
            agent_name=raw_config.name,
        )
        visible_tools, all_tools = self._build_tools(
            raw_config.registered_tools,
            agent_config=raw_config,
            provider_name=provider_name,
            mcp_servers=mcp_servers,
        )
        # Keep the agent's tool list in sync with what actually got registered so
        # downstream consumers (PromptBuilder, the OpenAI tools array) don't
        # reference names that were dropped for policy reasons.
        agent_config = dataclasses.replace(
            raw_config,
            registered_tools=[tool.name for tool in visible_tools],
            mcp_servers=mcp_servers,
        )
        tool_registry = ToolRegistry(all_tools)
        prompt_builder = PromptBuilder(
            self._skill_catalog,
            tool_registry,
            agent_catalog=self._agent_catalog,
        )
        provider = self._resolve_provider(agent_config)
        compactor = ContextCompactor(
            config=self._app_config.compaction,
            provider=provider,
            session_store=self._session_store,
        )
        # Memory wiring: pick live reader/trigger only when the agent opts in
        # AND the runtime stack itself is live.
        memory_reader = NoOpMemoryReader()
        memory_trigger = NoOpMemoryTrigger()
        if (
            self._memory_stack is not None
            and self._memory_stack.enabled
            and agent_config.memory_enabled
        ):
            memory_reader = self._memory_stack.reader
            memory_trigger = self._memory_stack.trigger
        agent_cls = self._resolve_agent_class(agent_config)
        logger.info(
            "building agent name=%s role=%s class=%s tools=%s memory=%s",
            agent_config.name,
            agent_config.role,
            agent_cls.__name__,
            agent_config.registered_tools,
            agent_config.memory_enabled and (self._memory_stack is not None and self._memory_stack.enabled),
        )
        return agent_cls(
            agent_config=agent_config,
            prompt_builder=prompt_builder,
            provider=provider,
            session_store=self._session_store,
            tool_output_store=self._tool_output_store,
            tool_registry=tool_registry,
            compactor=compactor,
            memory_reader=memory_reader,
            memory_trigger=memory_trigger,
            model_name=self._resolve_model_name(agent_config),
            memory_recent_messages=self._app_config.memory.retrieval.query_builder_recent_messages,
            run_coordinator=self._run_coordinator,
            input_queue=self._input_queue,
            agent_message_store=self._agent_message_store,
            task_store=self._task_store,
            provider_name=provider_name,
            mcp_servers=mcp_servers,
            mcp_client_manager=self._mcp_manager,
            profile_store=self._profile_store,
            attachment_store=self._attachment_store,
            supports_multimodal_attachments=self._supports_multimodal_attachments(
                provider_name
            ),
        )

    def _resolve_agent_class(self, agent_config: AgentConfig) -> type[BaseAgent]:
        """Resolve the runtime class for one configured agent."""

        return self._agent_classes.get(agent_config.role, DefaultAgent)

    def _supports_multimodal_attachments(self, provider_name: str) -> bool:
        """Return whether the selected provider/model accepts image/PDF blocks."""

        if provider_name == "openrouter":
            if self._app_config.openrouter is None:
                return False
            return self._app_config.openrouter.supports_multimodal
        if provider_name == "claude":
            if self._app_config.claude is None:
                return False
            return self._app_config.claude.supports_multimodal
        return True

    def _supported_mcp_servers(
        self,
        servers: tuple[MCPServerConfig, ...],
        *,
        provider_name: str,
        agent_name: str,
    ) -> tuple[MCPServerConfig, ...]:
        """Filter MCP servers to integrations Feather can safely activate."""

        supported: list[MCPServerConfig] = []
        for server in servers:
            if server.require_approval not in (None, "never"):
                logger.warning(
                    "skipping MCP server `%s` for agent=%s provider=%s: "
                    "require_approval is not supported yet",
                    server.label,
                    agent_name,
                    provider_name,
                )
                continue
            supported.append(server)
        return tuple(supported)

    def _build_tools(
        self,
        tool_names: list[str],
        *,
        agent_config: AgentConfig,
        provider_name: str,
        mcp_servers: tuple[MCPServerConfig, ...] = (),
    ) -> tuple[list[BaseTool], list[BaseTool]]:
        """Instantiate the configured tools for one agent.

        ``recall_memory`` is silently dropped (with a warning) when the
        runtime memory stack isn't live, so a YAML that lists it doesn't
        crash a memory-disabled deployment.
        """

        tools: list[BaseTool] = []
        memory_live = (
            self._memory_stack is not None
            and self._memory_stack.enabled
            and agent_config.memory_enabled
        )
        is_lead = agent_config.role == "lead"
        for tool_name in tool_names:
            if tool_name in {"recall_memory", "manage_memory"} and not memory_live:
                logger.warning(
                    "skipping memory tool `%s`: memory subsystem not live "
                    "for agent=%s",
                    tool_name,
                    agent_config.name,
                )
                continue
            if tool_name in _LEAD_ONLY_TOOLS and not is_lead:
                logger.warning(
                    "skipping lead-only tool `%s` for non-lead agent=%s role=%s",
                    tool_name,
                    agent_config.name,
                    agent_config.role,
                )
                continue
            builder = self._tool_builders.get(tool_name)
            if builder is None:
                known = ", ".join(sorted(self._tool_builders))
                raise ValueError(f"Agent requested unknown tool `{tool_name}`. Known tools: {known}")
            tools.append(builder())
        if mcp_servers:
            tools.append(
                ListMCPServersTool(
                    mcp_servers=mcp_servers,
                    provider_name=provider_name,
                    session_store=self._session_store,
                )
            )
            tools.append(
                RegisterMCPServerTool(
                    mcp_servers=mcp_servers,
                    provider_name=provider_name,
                    session_store=self._session_store,
                    manager=self._mcp_manager,
                )
            )
        all_tools = list(tools)
        existing = {tool.name for tool in all_tools}
        for server in mcp_servers:
            if not should_proxy_mcp_server(server, provider_name):
                continue
            if server.require_approval not in (None, "never"):
                logger.warning(
                    "skipping MCP server `%s` proxy for agent=%s: "
                    "require_approval is not supported by the proxy tool",
                    server.label,
                    agent_config.name,
                )
                continue
            proxy_tool = MCPProxyTool(server, manager=self._mcp_manager)
            if proxy_tool.name in existing:
                raise ValueError(
                    f"MCP server `{server.label}` maps to duplicate tool "
                    f"`{proxy_tool.name}`."
                )
            all_tools.append(proxy_tool)
            existing.add(proxy_tool.name)
        return tools, all_tools

    def _default_tool_builders(self) -> dict[str, ToolBuilder]:
        """Return the built-in tool builders available to agents."""

        builders: dict[str, ToolBuilder] = {
            "read_file": lambda: ReadFileTool(self._root, paths=self._paths),
            "read_pdf": lambda: ReadPdfTool(self._root),
            "write_file": lambda: WriteFileTool(self._root, paths=self._paths),
            "grep": lambda: GrepTool(self._root),
            "bash": lambda: BashTool(self._root),
            "ask_user": AskUserTool,
            "load_skill": lambda: LoadSkillTool(self._skill_catalog),
            "spawn_agent": self._build_spawn_agent_builder(),
            "terminate_agent": self._build_terminate_agent_builder(),
        }
        # request_restart: self-repair primitive. Probes install-mode once at
        # build time so each tool instance carries an accurate upgrade-
        # durability warning in its response without re-detecting on every call.
        install_info = detect_install_mode()
        session_store = self._session_store
        builders["request_restart"] = lambda: RequestRestartTool(
            session_store, install_info
        )
        # submit_github_report: file an issue upstream via the gh CLI.
        # No state to inject; the tool detects gh availability lazily on
        # each call so the factory builds even on machines without gh.
        builders["submit_github_report"] = SubmitGithubReportTool
        # send_message: available to every role (lead, sub-agents, custom).
        # The sender identity is baked into the tool instance at build time
        # so the tool execution call doesn't need to rediscover it.
        if self._agent_message_store is not None:
            message_store = self._agent_message_store
            # `builders[...]` takes a zero-arg callable; we capture the
            # current agent name via a default-argument closure so multiple
            # factory.build() calls don't share the wrong identity.
            builders["send_message"] = self._build_send_message_builder(
                message_store
            )
        if self._task_store is not None:
            task_store = self._task_store
            builders["task_create"] = lambda: TaskCreateTool(task_store)
            builders["task_list"] = lambda: TaskListTool(task_store)
            builders["task_get"] = lambda: TaskGetTool(task_store)
            builders["task_update"] = lambda: TaskUpdateTool(task_store)
            builders["task_output"] = lambda: TaskOutputTool(task_store, root=self._root)
            if self._agent_message_store is not None:
                message_store = self._agent_message_store
                builders["request_input"] = lambda: RequestInputTool(
                    task_store, message_store
                )
            if self._subagent_registry is not None:
                registry = self._subagent_registry
                if self._agent_message_store is not None:
                    message_store = self._agent_message_store
                    builders["task_stop"] = lambda: TaskStopTool(
                        task_store, registry, message_store
                    )
                    builders["task_resume"] = lambda: TaskResumeTool(
                        root=self._root,
                        agent_catalog=self._agent_catalog,
                        registry=registry,
                        task_store=task_store,
                        message_store=message_store,
                    )
        if self._cron_store is not None:
            cron_store = self._cron_store
            builders["create_cron"] = lambda: CreateCronTool(cron_store)
            builders["update_cron"] = lambda: UpdateCronTool(cron_store)
            builders["delete_cron"] = lambda: DeleteCronTool(cron_store)
            builders["list_crons"] = lambda: ListCronsTool(cron_store)
        if self._parallel_client is not None and self._app_config.parallel is not None:
            parallel_client = self._parallel_client
            parallel_config = self._app_config.parallel
            builders["web_search"] = lambda: ParallelSearchTool(
                parallel_client, parallel_config
            )
            builders["web_fetch"] = lambda: ParallelExtractTool(
                parallel_client, parallel_config, self._tool_output_store
            )
        if self._profile_store is not None:
            from feather.tools.user_info_tool import UserInfoTool

            store = self._profile_store
            builders["user_info"] = lambda: UserInfoTool(store)
        if self._memory_stack is not None and self._memory_stack.enabled:
            stack = self._memory_stack
            retrieval_cfg = self._app_config.memory.retrieval
            builders["recall_memory"] = lambda: RecallMemoryTool(
                reader=stack.reader,
                cfg=retrieval_cfg,
                session_id_resolver=current_session_id.get,
            )
            # `manage_memory` requires the live write-path service. The
            # `enabled` gate above guarantees `stack.service is not None`.
            service = stack.service
            if service is not None:
                builders["manage_memory"] = lambda: ManageMemoryTool(
                    service=service,
                    session_id_resolver=current_session_id.get,
                )
        return builders
