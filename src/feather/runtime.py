"""Shared Feather runtime bootstrap and agent construction."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from feather.core.lead_supervisor import LeadSupervisor

import httpx

from feather.config import load_app_config
from feather.config_schema import ReloadClass
from feather.config_schema import lookup as _lookup_field
from feather.core.agent_factory import AgentFactory
from feather.core.base_agent import BaseAgent
from feather.core.cron_scheduler import CronScheduler
from feather.core.input_queue import UserInputQueue
from feather.core.session_run_coordinator import SessionRunCoordinator
from feather.core.subagent_registry import SubagentRegistry
from feather.core.subagent_reaper import SubagentReaper
from feather.env import load_dotenv
from feather.logging_utils import configure_logging
from feather.memory.runtime import MemoryStack, build_memory_stack
from feather.messaging.adapters.line import LineAdapter
from feather.messaging.adapters.telegram import TelegramAdapter
from feather.messaging.adapters.whatsapp import WhatsAppAdapter
from feather.messaging.models import Platform
from feather.messaging.router import MessagingRouter
from feather.messaging.service import MessagingService
from feather.messaging.store import MessagingStore
from feather.messaging.webhook_server import WebhookServer
from feather.models import AppConfig, EventHandler, TaskRunStatus, TaskStatus
from feather.profile import UserProfileStore
from feather.providers.base import BaseLLMProvider
from feather.providers.claude_provider import ClaudeMessagesProvider
from feather.providers.openai_provider import OpenAIResponsesProvider
from feather.providers.openrouter_provider import OpenRouterChatProvider
from feather.providers.parallel_client import ParallelClient
from feather.resources import packaged_skills_root
from feather.skills.catalog import SkillCatalog
from feather.storage.agent_message_store import AgentMessageStore
from feather.storage.attachment_store import AttachmentStore
from feather.storage.cron_store import CronJobStore
from feather.storage.session_store import SessionStore
from feather.storage.task_store import TaskStore
from feather.storage.tool_output_store import ToolOutputStore

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ConfigApplyResult:
    """Outcome of :meth:`FeatherRuntime.apply_config_change`.

    Attributes:
        applied: Paths that were applied (LIVE or NEXT_TURN class).
        needs_restart_lead: Paths that require a lead-agent restart.
        needs_restart_app: Paths that require a full application restart.
    """

    applied: list[str]
    needs_restart_lead: list[str]
    needs_restart_app: list[str]


_TERMINAL_TASK_STATUSES = {
    TaskStatus.COMPLETED_WITH_REPORT,
    TaskStatus.COMPLETED_WITH_ARTIFACTS,
    TaskStatus.COMPLETED_WITHOUT_ARTIFACTS,
    TaskStatus.FAILED,
    TaskStatus.STOPPED,
}


class FeatherRuntime:
    """Application runtime that owns shared services and builds agents."""

    def __init__(
        self,
        *,
        root: Path,
        session_store: SessionStore,
        cron_store: CronJobStore,
        agent_factory: AgentFactory,
        skill_catalog: SkillCatalog,
        cron_scheduler: CronScheduler,
        memory_stack: MemoryStack,
        input_queue: UserInputQueue,
        agent_message_store: AgentMessageStore,
        task_store: TaskStore,
        attachment_store: AttachmentStore,
        subagent_registry: SubagentRegistry,
        subagent_reaper: SubagentReaper,
        messaging_service: MessagingService,
        default_provider: BaseLLMProvider,
        app_config: AppConfig,
        shutdown_timeout_s: float = 30.0,
    ) -> None:
        self._root = root
        self._session_store = session_store
        self._cron_store = cron_store
        self._agent_factory = agent_factory
        self._skill_catalog = skill_catalog
        self._cron_scheduler = cron_scheduler
        self._memory_stack = memory_stack
        self._input_queue = input_queue
        self._agent_message_store = agent_message_store
        self._task_store = task_store
        self._attachment_store = attachment_store
        self._subagent_registry = subagent_registry
        self._subagent_reaper = subagent_reaper
        self._messaging_service = messaging_service
        self._default_provider = default_provider
        self._app_config = app_config
        self._shutdown_timeout_s = shutdown_timeout_s
        self._session_event_handlers: dict[str, EventHandler] = {}
        self._agents: dict[str, BaseAgent] = {}
        self._supervisor: "LeadSupervisor | None" = None
        # Callbacks fired after :meth:`rebuild_agent` swaps the cached
        # instance. Used by CLI / TUI drivers to refresh their captured
        # agent reference so the next turn sees the new provider/model.
        self._agent_rebuilt_listeners: list[Callable[[str, BaseAgent], None]] = []

    @classmethod
    async def create(
        cls,
        root: Path,
        *,
        provider_factory: Callable[[Any], BaseLLMProvider] | None = None,
        agent_classes: Mapping[str, type[BaseAgent]] | None = None,
        paths: object = None,
    ) -> FeatherRuntime:
        """Create and initialize the shared Feather runtime.

        Args:
            root: Working directory. In project mode this is the
                directory that contains ``.feather/`` (per CLI
                walk-up); in global-only mode it's a placeholder for
                back-compat with code that still threads a ``root``
                argument and is otherwise unused for path resolution.
            provider_factory: Optional override for provider construction.
            agent_classes: Optional extra agent classes to register.
            paths: Optional :class:`feather.paths.FeatherPaths`. When
                provided, every store, log, and tmp dir is rooted at
                either the project (``paths.project_root``) or the
                global state dir (``paths.global_state_dir``) per
                ``paths.is_project_mode``. When ``None``, behaviour is
                unchanged for back-compat with existing tests.
        """

        # Load secrets in dependency order. The global ~/.feather/.env
        # provides the user-scoped baseline (API keys are personal, not
        # repo-specific); a project-local ./.env can still override
        # individual entries when one is present.
        from feather.paths import FeatherPaths

        if paths is None:
            _paths = FeatherPaths(project_root=root)
        else:
            _paths = paths  # type: ignore[assignment]
        if _paths.env_file.exists():
            load_dotenv(_paths.env_file)
        load_dotenv(root / ".env", override=True)
        app_config = load_app_config(root, paths=_paths)
        configure_logging(root, app_config.logging)

        # In global-only mode the runtime stores sessions etc. under
        # ~/.feather/state/ instead of creating an unwanted .feather/
        # tree wherever the user happened to be.
        if paths is not None and not _paths.is_project_mode:
            _paths.ensure_global_dirs()
            db_path = _paths.global_sessions_db
            tmp_root = _paths.global_state_dir
            tmp_subdir = "tmp"
            (tmp_root / tmp_subdir).mkdir(parents=True, exist_ok=True)
        else:
            db_path = (root / app_config.database.path).resolve()
            tmp_root = root
            tmp_subdir = app_config.storage.temp_directory
        session_store = SessionStore(db_path)
        await session_store.initialize()
        cron_store = CronJobStore(db_path)
        await cron_store.initialize()
        agent_message_store = AgentMessageStore(db_path)
        await agent_message_store.initialize()
        task_store = TaskStore(db_path)
        await task_store.initialize()

        if provider_factory is not None:
            provider = provider_factory(app_config)
        else:
            provider = _build_default_provider(app_config)
        parallel_client = _try_build_parallel_client(app_config)
        memory_stack = build_memory_stack(
            cfg=app_config.memory,
            default_provider=provider,
            app_config=app_config,
            session_store=session_store,
        )
        if memory_stack.enabled and memory_stack.service is not None:
            await memory_stack.service.initialize()
        run_coordinator = SessionRunCoordinator()
        input_queue = UserInputQueue()
        subagent_registry = SubagentRegistry()
        subagent_reaper = SubagentReaper(
            registry=subagent_registry,
            agent_message_store=agent_message_store,
            task_store=task_store,
        )
        # Profile prefers a project-local user.md when present (override),
        # otherwise falls back to the global one written by the wizard.
        if (root / ".feather" / "user.md").exists():
            profile_path = (root / ".feather" / "user.md").resolve()
        elif _paths.global_user_md.exists():
            profile_path = _paths.global_user_md
        else:
            profile_path = (root / ".feather" / "user.md").resolve()
        profile_store = UserProfileStore(profile_path)
        attachment_store = AttachmentStore(
            root=root,
            session_store=session_store,
            memory_service=memory_stack.service if memory_stack.enabled else None,
        )
        skills_sources: list = [packaged_skills_root()]
        if _paths.global_skills_dir.is_dir():
            skills_sources.append(_paths.global_skills_dir)
        skills_sources.append((root / app_config.skills.directory).resolve())
        skill_catalog = SkillCatalog(skills_sources)
        agent_factory = AgentFactory(
            root=root,
            app_config=app_config,
            provider=provider,
            session_store=session_store,
            cron_store=cron_store,
            tool_output_store=ToolOutputStore(tmp_root, tmp_subdir),
            skill_catalog=skill_catalog,
            parallel_client=parallel_client,
            memory_stack=memory_stack,
            run_coordinator=run_coordinator,
            input_queue=input_queue,
            agent_message_store=agent_message_store,
            task_store=task_store,
            attachment_store=attachment_store,
            subagent_registry=subagent_registry,
            agent_classes=agent_classes,
            profile_store=profile_store,
            paths=_paths,
        )
        messaging_store = MessagingStore(db_path)
        await messaging_store.initialize()
        messaging_http_client = httpx.AsyncClient()
        webhook_server = WebhookServer()
        # Lazily build the lead agent the first time a platform message
        # arrives. Building eagerly here would force every test fixture
        # (subagent, isolated explore-only fixtures, etc.) to stage a
        # lead.yaml even when the messaging service is never used.
        lead_agent_holder: list[BaseAgent] = []

        def _lead_agent() -> BaseAgent:
            if not lead_agent_holder:
                lead_agent_holder.append(agent_factory.build("lead"))
            return lead_agent_holder[0]

        async def _create_lead_session() -> str:
            return await _lead_agent().create_session()

        async def _run_lead_agent(session_id: str, text: str):
            return await _lead_agent().run(session_id, text, None)

        async def _is_session_busy(session_id: str) -> bool:
            return run_coordinator.is_busy(session_id)

        messaging_router = MessagingRouter(
            store=messaging_store,
            create_session=_create_lead_session,
            run_agent=_run_lead_agent,
            is_session_busy=_is_session_busy,
            input_queue=input_queue,
        )
        adapter_factories = {
            Platform.TELEGRAM: lambda svc, cfg: TelegramAdapter(svc, cfg),
            Platform.LINE: lambda svc, cfg: LineAdapter(svc, cfg),
            Platform.WHATSAPP: lambda svc, cfg: WhatsAppAdapter(svc, cfg),
        }
        messaging_service = MessagingService(
            store=messaging_store,
            router=messaging_router,
            webhook_server=webhook_server,
            http_client=messaging_http_client,
            adapter_factories=adapter_factories,
            own_http_client=True,
        )
        runtime = cls(
            root=root,
            session_store=session_store,
            cron_store=cron_store,
            agent_factory=agent_factory,
            skill_catalog=skill_catalog,
            cron_scheduler=CronScheduler(
                config=app_config.scheduler,
                cron_store=cron_store,
                # Mailbox-routed: scheduler delivers prompts via
                # agent_messages so the lead's existing inbox path
                # (resume_on_inbox) processes them, instead of the
                # scheduler building its own in-process BaseAgent.
                # Works in both default and self-repair modes.
                message_store=agent_message_store,
            ),
            memory_stack=memory_stack,
            input_queue=input_queue,
            agent_message_store=agent_message_store,
            task_store=task_store,
            attachment_store=attachment_store,
            subagent_registry=subagent_registry,
            subagent_reaper=subagent_reaper,
            messaging_service=messaging_service,
            default_provider=provider,
            app_config=app_config,
            shutdown_timeout_s=app_config.memory.trigger.shutdown_timeout_s,
        )
        runtime._cron_scheduler.set_event_handler_resolver(runtime._resolve_session_event_handler)
        return runtime

    def build_agent(self, agent_name: str) -> BaseAgent:
        """Build a fresh agent and cache it for later :meth:`rebuild_agent` calls.

        Args:
            agent_name: Logical agent name (e.g. ``"lead"``).

        Returns:
            The newly built agent instance.
        """

        agent = self._agent_factory.build(agent_name)
        self._agents[agent_name] = agent
        return agent

    def get_agent(self, name: str) -> BaseAgent:
        """Return the cached agent built via :meth:`build_agent`.

        Args:
            name: Logical agent name.

        Raises:
            KeyError: If no agent with ``name`` has been built yet.
        """

        if name not in self._agents:
            raise KeyError(f"agent {name!r} not yet built")
        return self._agents[name]

    def rebuild_agent(self, name: str) -> BaseAgent:
        """Reconstruct ``name`` (and its provider) against the current app_config.

        Session cursor (``last_response_id``) is owned by
        :class:`~feather.storage.session_store.SessionStore` rather than the
        agent instance, so the new agent picks up the in-flight conversation
        transparently on the next turn.

        The factory's internal ``_app_config`` AND ``_provider`` are updated
        to the runtime's current config so provider-bound state (model,
        reasoning, HTTP clients) reflects the freshly reloaded values.

        Args:
            name: Logical agent name.

        Returns:
            The newly built agent instance, already stored in the cache.
        """

        # Sync the factory's config view with whatever reload_config() loaded.
        self._agent_factory._app_config = self._app_config
        # Rebuild the factory's default provider so the new agent gets a fresh
        # provider client (correct model, reasoning config, HTTP client, etc.)
        # that matches the new active_provider setting.
        new_default_provider = _build_default_provider(self._app_config)
        active = (self._app_config.active_provider or "openai").strip().lower()
        self._agent_factory._provider = new_default_provider
        self._agent_factory._providers_by_name[active] = new_default_provider
        new_agent = self._agent_factory.build(name)
        self._agents[name] = new_agent
        logger.info("runtime.agent.rebuilt name=%s provider=%s", name, active)
        # Notify subscribers (CLI / TUI drivers) so they can refresh any
        # captured agent reference. Without this, callers that took a
        # snapshot of ``runtime._agents[name]`` keep talking to the old
        # provider after /config saves a NEXT_TURN-class field — the
        # exact symptom that caused the OpenRouter model swap to look
        # saved but never actually take effect.
        for listener in list(self._agent_rebuilt_listeners):
            try:
                listener(name, new_agent)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "runtime.agent_rebuilt_listener.error name=%s", name
                )
        return new_agent

    def register_agent_rebuilt_listener(
        self, listener: Callable[[str, BaseAgent], None]
    ) -> Callable[[], None]:
        """Subscribe to ``rebuild_agent`` events.

        ``listener`` is called as ``listener(name, new_agent)`` after the
        cache swap, on the same task that triggered the rebuild. Returns
        an unsubscribe callable that removes the listener (idempotent).

        Args:
            listener: Function to invoke after each rebuild.

        Returns:
            A zero-arg callable; invoking it removes the listener.
        """

        self._agent_rebuilt_listeners.append(listener)

        def _unsubscribe() -> None:
            try:
                self._agent_rebuilt_listeners.remove(listener)
            except ValueError:
                pass

        return _unsubscribe

    def attach_supervisor(self, supervisor: "LeadSupervisor") -> None:
        """Register a :class:`~feather.core.lead_supervisor.LeadSupervisor`.

        When a supervisor is attached, :meth:`apply_config_change` fans out
        the reload to the lead worker subprocess in addition to applying it
        in-process (so the TUI process's config view stays consistent).

        Args:
            supervisor: The running supervisor that owns the lead worker.
        """

        self._supervisor = supervisor
        logger.info("runtime.supervisor.attached")

    def detach_supervisor(self) -> None:
        """Unregister the attached supervisor.

        Safe to call even when no supervisor is attached.
        """

        self._supervisor = None
        logger.info("runtime.supervisor.detached")

    @property
    def input_queue(self) -> UserInputQueue:
        """Return the shared per-session user-input queue."""

        return self._input_queue

    @property
    def agent_message_store(self) -> AgentMessageStore:
        """Return the shared agent-to-agent message store."""

        return self._agent_message_store

    @property
    def session_store(self) -> SessionStore:
        """Return the shared session store.

        Used by the supervisor-side ``RestartWatcher`` to poll the
        ``restart_requested_at`` flag set by the worker's
        ``request_restart`` tool.
        """

        return self._session_store

    @property
    def subagent_registry(self) -> SubagentRegistry:
        """Return the live sub-agent subprocess registry."""

        return self._subagent_registry

    @property
    def task_store(self) -> TaskStore:
        """Return the durable task store."""

        return self._task_store

    @property
    def skill_catalog(self) -> SkillCatalog:
        """Return the shared skill catalog."""

        return self._skill_catalog

    @property
    def messaging_service(self) -> MessagingService:
        """Return the shared messaging-integration service."""

        return self._messaging_service

    @property
    def config(self) -> AppConfig:
        """Return the loaded application configuration."""

        return self._app_config

    async def reload_config(self) -> None:
        """Re-read app.yaml + global overlay from disk and swap ``_app_config``.

        This is the LIVE-class reload path. Provider-bound state (HTTP clients,
        models, reasoning config) is NOT reconstructed — call
        :meth:`rebuild_agent` for NEXT_TURN-class changes.
        """

        from feather.paths import FeatherPaths

        # Re-derive paths using the same resolution the constructor used.
        # The runtime stores ``_root`` already; FeatherPaths resolves the
        # global state dir from ``~/.feather`` automatically.
        paths = FeatherPaths(project_root=self._root)
        new_config = load_app_config(self._root, paths=paths)
        self._app_config = new_config
        logger.info(
            "runtime.config.reloaded active_provider=%s", new_config.active_provider
        )

    async def apply_config_change(
        self, changed_paths: list[str]
    ) -> ConfigApplyResult:
        """Apply the cumulative reload effect of ``changed_paths``.

        Looks up each path's :class:`~feather.config_schema.ReloadClass` from
        the registry and fans out accordingly:

        - ``LIVE``-only changes → :meth:`reload_config` only.
        - Any ``NEXT_TURN`` → :meth:`reload_config` + :meth:`rebuild_agent`
          for every cached agent.
        - ``RESTART_LEAD`` / ``RESTART_APP`` paths are surfaced in the
          returned :class:`ConfigApplyResult`; the caller (TUI) shows the
          appropriate banner.

        Args:
            changed_paths: Dotted config paths that were written to disk
                (e.g. ``["app.active_provider", "app.openai.model"]``).

        Returns:
            A :class:`ConfigApplyResult` describing what was applied and what
            requires a manual restart.
        """

        live: list[str] = []
        next_turn: list[str] = []
        restart_lead: list[str] = []
        restart_app: list[str] = []

        for path in changed_paths:
            field_def = _lookup_field(path)
            if field_def is None:
                continue
            bucket = {
                ReloadClass.LIVE: live,
                ReloadClass.NEXT_TURN: next_turn,
                ReloadClass.RESTART_LEAD: restart_lead,
                ReloadClass.RESTART_APP: restart_app,
            }[field_def.reload]
            bucket.append(path)

        applied = list(live)
        if live or next_turn:
            await self.reload_config()
            applied.extend(next_turn)
            if next_turn:
                for agent_name in list(self._agents):
                    self.rebuild_agent(agent_name)

        # Worker-mode fanout: when a supervisor is attached, propagate the same
        # reload to the lead worker subprocess.  The in-process reload above
        # keeps the TUI process's config view consistent; the worker handles
        # its own validate-then-swap internally.
        if self._supervisor is not None and (live or next_turn):
            reload_class = (
                ReloadClass.NEXT_TURN.value if next_turn else ReloadClass.LIVE.value
            )
            worker_paths = list(live) + list(next_turn)
            try:
                ack = await self._supervisor.request_config_reload(
                    worker_paths, reload_class
                )
                if not ack.ok:
                    logger.warning(
                        "runtime.apply_config_change worker reload failed: %s",
                        ack.error,
                    )
                    # Return empty applied list so the caller knows the
                    # worker-side apply did not succeed.
                    return ConfigApplyResult(
                        applied=[],
                        needs_restart_lead=restart_lead,
                        needs_restart_app=restart_app,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "runtime.apply_config_change supervisor fanout error: %s", exc
                )
                return ConfigApplyResult(
                    applied=[],
                    needs_restart_lead=restart_lead,
                    needs_restart_app=restart_app,
                )

        return ConfigApplyResult(
            applied=applied,
            needs_restart_lead=restart_lead,
            needs_restart_app=restart_app,
        )

    async def start_background_services(
        self, *, lead_in_subprocess: bool = False
    ) -> None:
        """Start shared background services such as the cron scheduler.

        Args:
            lead_in_subprocess: When True, the lead agent is running in a
                separate worker process (see
                :class:`feather.core.lead_supervisor.LeadSupervisor`).
                The **messaging router** is NOT started in that mode —
                its inbound queue is the in-process
                :class:`UserInputQueue` which the worker can't see. The
                **cron scheduler** runs in both modes: it now routes
                through the ``agent_messages`` mailbox (see
                :mod:`feather.core.cron_scheduler`), which is
                process-shared via SQLite, so the worker's existing
                ``resume_on_inbox`` path processes the cron-triggered
                turns naturally with no race on session state. The
                sub-agent reaper always runs because sub-agents are
                spawned by tools, not by background services.
        """

        await self._cron_scheduler.start()
        if lead_in_subprocess:
            logger.info(
                "runtime.start_background_services skipping messaging "
                "(lead_in_subprocess=True): the messaging router enqueues "
                "into the in-process UserInputQueue which the worker "
                "can't see. Set self_repair.enabled=false to use Telegram "
                "/ LINE / WhatsApp."
            )
        else:
            await self._messaging_service.start()
        await self._subagent_reaper.start()

    async def run_pending_cron_jobs(self) -> int:
        """Run one scheduler tick immediately."""

        return await self._cron_scheduler.run_pending()

    def set_session_event_handler(self, session_id: str, handler: EventHandler | None) -> None:
        """Bind or clear a runtime event handler for one session."""

        if handler is None:
            self._session_event_handlers.pop(session_id, None)
            return
        self._session_event_handlers[session_id] = handler

    async def close(self) -> None:
        """Close shared runtime resources (alias for :meth:`shutdown`)."""

        await self.shutdown()

    async def shutdown(self) -> None:
        """Stop background services, drain memory tasks, then close shared stores."""

        await self._cron_scheduler.stop()
        await self._subagent_reaper.stop()
        try:
            await self._messaging_service.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("runtime.shutdown.messaging_close_error")
        await self._terminate_live_subagents()
        self._session_event_handlers.clear()
        try:
            await self._agent_factory.aclose()
        except Exception:  # noqa: BLE001
            logger.exception("runtime.shutdown.agent_factory_close_error")
        try:
            await self._attachment_store.drain_indexing(
                timeout_s=self._shutdown_timeout_s
            )
        except Exception:  # noqa: BLE001
            logger.exception("runtime.shutdown.attachment_index_drain_error")
        # The agent factory skips the app-level default provider on purpose
        # (runtime owns it). Close it here so raw httpx clients don't leak
        # open sockets when active_provider=openrouter.
        default_provider_closer = getattr(self._default_provider, "aclose", None)
        if default_provider_closer is not None:
            try:
                await default_provider_closer()
            except Exception:  # noqa: BLE001
                logger.exception("runtime.shutdown.default_provider_close_error")
        try:
            await self._memory_stack.trigger.drain(self._shutdown_timeout_s)
        except Exception:  # noqa: BLE001
            logger.exception("memory.shutdown.drain_error")
        try:
            if self._memory_stack.service is not None:
                store = getattr(self._memory_stack.service, "_store", None)
                client = getattr(store, "_client", None)
                if client is not None and hasattr(client, "close"):
                    await client.close()
        except Exception:  # noqa: BLE001
            logger.exception("memory.shutdown.qdrant_close_error")
        try:
            await self._memory_stack.aclose()
        except Exception:  # noqa: BLE001
            logger.exception("memory.shutdown.alternate_providers_close_error")
        await self._cron_store.close()
        await self._task_store.close()
        await self._agent_message_store.close()
        await self._session_store.close()

    def _resolve_session_event_handler(self, session_id: str) -> EventHandler | None:
        return self._session_event_handlers.get(session_id)

    async def _terminate_live_subagents(self) -> None:
        """Best-effort: terminate any sub-agent subprocesses still running.

        Called during shutdown AFTER the reaper has stopped. Any child that
        is still live at this point will not have its final report
        delivered — the parent is exiting, so there's nobody to drain the
        inbox anyway. The goal is just to avoid orphaned processes.
        """

        import asyncio as _asyncio

        live = await self._subagent_registry.snapshot()
        for entry in live:
            proc = entry.process
            killed = False
            if proc.returncode is None:
                try:
                    proc.terminate()
                    killed = True
                except ProcessLookupError:
                    pass
                else:
                    try:
                        await _asyncio.wait_for(proc.wait(), timeout=2.0)
                    except _asyncio.TimeoutError:
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                        else:
                            try:
                                await _asyncio.wait_for(proc.wait(), timeout=2.0)
                            except _asyncio.TimeoutError:
                                logger.warning(
                                    "sub-agent %s did not exit after SIGKILL",
                                    entry.session_id,
                                )
            await self._finalize_live_subagent_on_shutdown(entry, killed=killed)
        for entry in live:
            for drainer in entry.drainers:
                if not drainer.done():
                    drainer.cancel()
                try:
                    await drainer
                except (_asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            await self._subagent_registry.remove(entry.session_id)

    async def _finalize_live_subagent_on_shutdown(self, entry: Any, *, killed: bool) -> None:
        """Persist durable task/run state for a child left live at shutdown."""

        task_id = getattr(entry, "task_id", None)
        task_run_id = getattr(entry, "task_run_id", None)
        proc = entry.process
        exit_code = getattr(proc, "returncode", None)
        if killed:
            run_status = TaskRunStatus.KILLED
            task_status = TaskStatus.STOPPED
            message = "runtime shutdown terminated live sub-agent"
        else:
            run_status = TaskRunStatus.EXITED if exit_code == 0 else TaskRunStatus.CRASHED
            task_status = TaskStatus.FAILED
            message = "runtime shutdown before reaper delivered final report"

        try:
            if task_run_id is not None:
                envelope = getattr(entry, "envelope", None) or {}
                envelope_status = envelope.get("status") if isinstance(envelope, dict) else None
                await self._task_store.finish_run(
                    task_run_id,
                    status=run_status,
                    exit_code=exit_code,
                    envelope_status=(
                        str(envelope_status) if envelope_status is not None else None
                    ),
                    error=message,
                )
            if task_id is not None:
                task = await self._task_store.get_task(task_id)
                if task.status not in _TERMINAL_TASK_STATUSES:
                    await self._task_store.update_task(
                        task_id,
                        status=task_status,
                        error=message,
                    )
        except Exception:  # noqa: BLE001
            logger.exception(
                "runtime.shutdown.task_finalize_error task_id=%s session_id=%s",
                task_id,
                getattr(entry, "session_id", None),
            )


def _build_default_provider(app_config: Any) -> BaseLLMProvider:
    """Pick the provider implementation for the session-wide active provider.

    ``app_config.active_provider`` defaults to ``"openai"`` and preserves
    existing sessions exactly. Setting it to ``"openrouter"`` or
    ``"claude"`` flips every agent to that provider's path; missing the
    matching config block raises so operator misconfiguration fails
    loudly rather than silently falling back.
    """

    active = (app_config.active_provider or "openai").strip().lower()
    if active == "openrouter":
        if app_config.openrouter is None:
            raise ValueError(
                "active_provider=openrouter but no `openrouter:` block in app.yaml"
            )
        return OpenRouterChatProvider(app_config.openrouter)
    if active == "claude":
        if app_config.claude is None:
            raise ValueError(
                "active_provider=claude but no `claude:` block in app.yaml"
            )
        return ClaudeMessagesProvider(app_config.claude)
    if active == "openai":
        return OpenAIResponsesProvider(app_config.openai)
    raise ValueError(
        f"unsupported active_provider={active!r} "
        "(expected 'openai', 'openrouter', or 'claude')"
    )


def _try_build_parallel_client(app_config: Any) -> ParallelClient | None:
    """Instantiate the Parallel AI client when config and API key are present."""

    parallel_config = getattr(app_config, "parallel", None)
    if parallel_config is None:
        return None
    try:
        return ParallelClient(parallel_config)
    except ValueError as exc:
        logger.warning("parallel web tools disabled: %s", exc)
        return None
