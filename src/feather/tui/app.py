"""Textual-powered terminal UI for Feather sessions."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.geometry import Region
from textual.screen import ModalScreen
from textual.widgets import OptionList, RichLog, Static, TextArea
from textual.widgets.option_list import Option

from feather.integrations.attachments.parse import parse_attachment_drops, render_attachment_message
from feather.core.agent.catalog import AgentCatalog
from feather.core.leads.scaffold import scaffold_lead_yaml
from feather.core.leads.supervisor import LeadSupervisor
from feather.core.log_triage_bot import LogTriageBot
from feather.core.restart_watcher import RestartWatcher
from feather.models import AgentOutcome, RuntimeEvent, TaskRecord, TaskStatus
from feather.runtime import FeatherRuntime
from feather.tui.slash_commands import (
    SlashCommand,
    SlashCommandRegistry,
    default_registry,
    parse_slash_input,
)
from feather.tui import (
    _TASK_TOOL_NAMES,
    _INBOX_WAKE,
    _event_title,
    _failed_tool_title,
    _format_tool_error,
    _format_tool_payload,
    _format_tool_result,
    _indent_lines,
    _status_style,
    _system_event_style,
    _tool_finished_title,
    _tool_started_title,
    preview_inline,
)
from feather.tui.lead_worker import (
    _HANG_WATCHER_POLL_SECONDS,
    _LEAD_WORKER_ENV,
    _env_says_use_lead_worker,
    _should_use_lead_worker,
    decide_hang_alert,
)
from feather.tui.widgets import (
    ComposerTextArea,
    SlashCommandDropdown,
    render_dropdown_text,
    render_help_text,
)
from feather.tui.drill import SubagentDrillScreen
from feather.tui.render import (
    _LEAD_STATUS_GLYPH,
    _agent_message_sender,
    _normalize_pasted_attachment_token,
    build_header_text,
    build_lead_strip,
    build_transcript_text,
    build_work_text,
    format_agent_message_event,
    format_task_row,
    format_transcript_block,
    is_exit_command,
    normalize_pasted_attachment_text,
    region_contains_point,
    summarize_agent_message_update,
    summarize_user_input_for_display,
)

logger = logging.getLogger(__name__)


_SlashHandler = Callable[[str], None]




#: Braille-dot spinner frames. Cycling these at ~10 fps renders as a smooth
#: rotating dot pattern in any monospace terminal. Used inline next to the
#: "<Agent> streaming" label so the user sees the turn is still alive even
#: when reasoning models pause between text deltas.
_SPINNER_FRAMES: tuple[str, ...] = (
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
)

#: Tick interval for advancing :data:`_SPINNER_FRAMES`. 100 ms is the upper
#: bound of "smooth" for terminal animation; faster trips repaint cost on
#: long conversations without adding perceived motion.
_SPINNER_INTERVAL_SECONDS: float = 0.1


@dataclass(slots=True, frozen=True)
class _ConversationBlock:
    """One styled conversation block rendered by the Textual transcript."""

    title: str
    body: str
    label_style: str
    body_style: str


@dataclass
class PerLeadState:
    """All mutable state for one lead in the multi-lead cockpit.

    Every lead — active or backgrounded — owns one of these. The active lead's
    state is what the widgets render; a backgrounded lead keeps accumulating
    here (its driver runs concurrently) and is shown the moment you switch to
    it. Execution primitives (handle, queues, events, tasks) are always
    per-lead so leads run truly independently.
    """

    name: str
    display_name: str
    handle: Any
    session_id: str
    agent: Any = None
    supervisor: Any = None
    color: str | None = None
    emoji: str | None = None
    # --- display state (rendered when this lead is active) ---
    conversation_blocks: list[_ConversationBlock] = field(default_factory=list)
    transcript_blocks: list[str] = field(default_factory=list)
    assistant_parts: list[str] = field(default_factory=list)
    status: str = "idle"
    latest_usage_ratio: float | None = None
    active_tool: str | None = None
    queue_depth: int = 0
    queued_messages: tuple[str, ...] = ()
    running_agents: tuple[str, ...] = ()
    task_rows: tuple[str, ...] = ()
    task_updates: list[str] = field(default_factory=list)
    # --- execution state (always per-lead) ---
    pending_answer: asyncio.Queue[str] = field(
        default_factory=lambda: asyncio.Queue(maxsize=1)
    )
    new_run_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    busy_event: asyncio.Event = field(default_factory=asyncio.Event)
    awaiting_event: asyncio.Event = field(default_factory=asyncio.Event)
    run_task: asyncio.Task[Any] | None = None
    driver_task: asyncio.Task[None] | None = None


def _lead_prop(field_name: str) -> property:
    """Build a property delegating ``self._<scalar>`` to the active lead's state.

    Lets the existing display code and tests keep using ``self._status`` etc.
    while the real storage moved onto the per-lead :class:`PerLeadState`.
    """

    def getter(self: "FeatherTextualApp") -> Any:
        return getattr(self._active(), field_name)

    def setter(self: "FeatherTextualApp", value: Any) -> None:
        setattr(self._active(), field_name, value)

    return property(getter, setter)


class FeatherTextualApp(App[None]):
    """Full-screen Textual application for operating a Feather session."""

    CSS = """
    Screen {
        layout: vertical;
        background: #090b10;
        color: white;
    }

    #header {
        height: 3;
        border: solid #666666;
        padding: 0 1;
    }

    #conversation {
        height: 1fr;
        border: solid #666666;
        padding: 0 1;
        background: #090b10;
    }

    #work {
        height: 9;
        border: solid #666666;
        padding: 0 1;
        overflow-y: auto;
    }

    #work_content {
        height: auto;
    }

    #slash_dropdown {
        height: auto;
        max-height: 10;
        border: solid #4f7fff;
        background: #11151c;
        color: white;
        padding: 0 1;
        display: none;
    }

    #slash_dropdown.-open {
        display: block;
    }

    #composer {
        height: 6;
        border: solid #666666;
        background: #090b10;
        color: white;
    }

    TextArea .text-area--cursor {
        background: white;
        color: black;
    }

    TextArea .text-area--selection {
        background: #2f5f9f;
    }
    """

    BINDINGS = [
        Binding("enter", "submit", "Send", priority=True, show=False),
        Binding("escape", "interrupt", "Interrupt", priority=True),
        Binding("pageup", "conversation_page_up", "Older", priority=True),
        Binding("pagedown", "conversation_page_down", "Newer", priority=True),
        Binding("home", "conversation_home", "Top", priority=True),
        Binding("end", "conversation_end", "Latest", priority=True),
        Binding("shift+pageup", "work_page_up", "Work older", priority=True, show=False),
        Binding("shift+pagedown", "work_page_down", "Work newer", priority=True, show=False),
        Binding("shift+home", "work_home", "Work top", priority=True, show=False),
        Binding("shift+end", "work_end", "Work latest", priority=True, show=False),
        Binding("ctrl+c", "copy_selection_or_transcript", "Copy", priority=True, show=False),
        Binding("ctrl+y", "copy_transcript", "Copy transcript", priority=True, show=False),
        # Switch which lead is on screen. ctrl+←/→ always works; bare ←/→ also
        # switch, but only when the composer is empty (handled in
        # ComposerTextArea._on_key) so text editing is never disrupted.
        Binding("ctrl+left", "lead_prev", "◀ Lead", priority=True),
        Binding("ctrl+right", "lead_next", "Lead ▶", priority=True),
        Binding("ctrl+o", "open_subagents", "Sub-agents", priority=True),
    ]

    # Actions disabled while a ModalScreen (e.g. ConfigScreen) is on top —
    # otherwise the App's priority Enter/Esc bindings preempt the modal's
    # own bindings and the user can't edit fields or close the modal.
    _ACTIONS_MUTED_UNDER_MODAL = frozenset(
        {
            "submit",
            "interrupt",
            "conversation_page_up",
            "conversation_page_down",
            "conversation_home",
            "conversation_end",
            "work_page_up",
            "work_page_down",
            "work_home",
            "work_end",
            "lead_prev",
            "lead_next",
            "open_subagents",
        }
    )

    def check_action(
        self, action: str, parameters: tuple[object, ...]
    ) -> bool | None:
        """Disable App-level priority bindings while a ModalScreen is active.

        Textual fires App-level priority bindings before the focused screen's
        bindings, so without this override Enter/Esc inside a modal would hit
        the composer's submit/interrupt actions instead of the modal's own
        Enter (edit) / Esc (close) bindings.
        """

        if action in self._ACTIONS_MUTED_UNDER_MODAL and isinstance(
            self.screen, ModalScreen
        ):
            return False
        return super().check_action(action, parameters)

    def __init__(
        self,
        *,
        root: Path,
        session_id: str | None = None,
        slash_registry: SlashCommandRegistry | None = None,
    ) -> None:
        super().__init__()
        self._root = root
        self._requested_session_id = session_id
        self._runtime: FeatherRuntime | None = None
        self._log_triage_bot: LogTriageBot | None = None
        self._restart_watcher: RestartWatcher | None = None
        self._hang_watcher_task: asyncio.Task[None] | None = None
        self._active_lead_name: str | None = None
        # Per-lead state lives in PerLeadState; the active lead's is what the
        # widgets render. The scalar ``self._<field>`` names below are
        # properties (see ``_lead_prop``) that delegate to the active lead, so
        # display code and tests keep working; a ``_bootstrap_state`` backs
        # them before on_mount wires the first real lead.
        self._leads: dict[str, PerLeadState] = {}
        self._bootstrap_state = PerLeadState(
            name="", display_name="Lead", handle=None, session_id=""
        )
        # Animated streaming indicator. Braille dots cycling at 10 fps
        # render as a smooth spinner in any monospaced terminal. The tick
        # only triggers a conversation re-render while ``_assistant_parts``
        # is non-empty, so idle sessions pay nothing.
        self._spinner_frame: int = 0
        self._stop_event = asyncio.Event()
        self._watcher_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self.slash_registry: SlashCommandRegistry = (
            slash_registry if slash_registry is not None else default_registry()
        )
        # Map every registered name + alias to its handler so dispatch can
        # resolve aliases without re-walking the registry.
        self._slash_handlers: dict[str, _SlashHandler] = {}
        self._register_default_handlers()
        # Reference set by ``compose`` once the widget is mounted; the
        # composer's key handler peeks at it via ``getattr``.
        self._slash_dropdown_widget: SlashCommandDropdown | None = None
        # Set by ``ComposerTextArea._on_key`` when Tab autocompletes — the
        # programmatic ``composer.text = ...`` write triggers a Changed
        # event that arrives after this handler returns, which would
        # otherwise re-open the dropdown for the just-completed command
        # (review fix C1).
        self._slash_suppress_next_change: bool = False

    def _active(self) -> PerLeadState:
        """Return the active lead's state (a bootstrap placeholder pre-mount)."""

        name = self._active_lead_name
        if name is not None and name in self._leads:
            return self._leads[name]
        return self._bootstrap_state

    def _lead_names(self) -> list[str]:
        """Active lead names in stable display order."""

        return sorted(self._leads)

    def action_open_subagents(self) -> None:
        """Open the read-only sub-agent transcript drill-down for the active lead."""

        if self._runtime is None or not self._active_session_id:
            return
        display = self._agent.config.name if self._agent is not None else (
            self._active_lead_name or "Lead"
        )
        self.push_screen(
            SubagentDrillScreen(
                runtime=self._runtime,
                parent_session_id=self._active_session_id,
                lead_display_name=display,
            )
        )

    def action_lead_prev(self) -> None:
        """Switch to the previous lead (ctrl+← / bare ← when composer empty)."""

        self._switch_lead(-1)

    def action_lead_next(self) -> None:
        """Switch to the next lead (ctrl+→ / bare → when composer empty)."""

        self._switch_lead(1)

    def _switch_lead(self, delta: int) -> None:
        """Rotate the active lead by ``delta`` over the sorted lead names."""

        names = self._lead_names()
        if len(names) <= 1 or self._active_lead_name is None:
            return
        try:
            idx = names.index(self._active_lead_name)
        except ValueError:
            return
        self._active_lead_name = names[(idx + delta) % len(names)]
        self._render_active()

    def _render_active(self) -> None:
        """Repaint header + conversation + work from the active lead's state."""

        self._render_conversation()
        self._update_header()
        self._update_work()

    # Scalar accessors delegating to the active lead's PerLeadState. Display
    # code and tests read/write these as before; the active lead is the target.
    _agent = _lead_prop("agent")
    _supervisor = _lead_prop("supervisor")
    _active_session_id = _lead_prop("session_id")
    _assistant_parts = _lead_prop("assistant_parts")
    _latest_usage_ratio = _lead_prop("latest_usage_ratio")
    _queue_depth = _lead_prop("queue_depth")
    _queued_messages = _lead_prop("queued_messages")
    _running_agents = _lead_prop("running_agents")
    _task_rows = _lead_prop("task_rows")
    _task_updates = _lead_prop("task_updates")
    _conversation_blocks = _lead_prop("conversation_blocks")
    _transcript_blocks = _lead_prop("transcript_blocks")
    _active_tool = _lead_prop("active_tool")
    _status = _lead_prop("status")
    _pending_answer = _lead_prop("pending_answer")
    _new_run_queue = _lead_prop("new_run_queue")
    _busy_event = _lead_prop("busy_event")
    _awaiting_event = _lead_prop("awaiting_event")
    _run_task = _lead_prop("run_task")
    _driver_task = _lead_prop("driver_task")

    def compose(self) -> ComposeResult:
        """Compose the app's primary sections."""

        yield Static(id="header")
        yield RichLog(id="conversation", wrap=True, markup=False, highlight=False)
        yield VerticalScroll(Static(id="work_content"), id="work")
        dropdown = SlashCommandDropdown(self.slash_registry, id="slash_dropdown")
        self._slash_dropdown_widget = dropdown
        yield dropdown
        yield ComposerTextArea(
            "",
            id="composer",
            soft_wrap=True,
            show_line_numbers=False,
            placeholder="Type a message. Press / for commands. Enter submits.",
        )

    def _make_lead_event_handler(self, lead_name: str):
        """Return an event handler bound to one lead's state."""

        def handler(event: RuntimeEvent) -> None:
            self._dispatch_lead_event(lead_name, event)

        return handler

    def _dispatch_lead_event(self, lead_name: str, event: RuntimeEvent) -> None:
        """Route a runtime event to the owning lead's state.

        The widgets are only touched when the event belongs to the active
        lead; a backgrounded lead's state still accumulates so switching to it
        shows the full picture.
        """

        state = self._leads.get(lead_name)
        if state is None:
            return
        self._apply_event(state, event)

    async def _bootstrap_lead(
        self, lead_name: str, *, requested_session_id: str | None = None
    ) -> PerLeadState:
        """Build one lead's agent + durable session + state, ready to drive.

        Reused for the default lead at startup, for any additional discovered
        lead, and for leads created at runtime via ``/lead new``.
        """

        assert self._runtime is not None
        agent = self._runtime.build_agent(lead_name)
        state = PerLeadState(
            name=lead_name,
            display_name=agent.config.name,
            handle=None,
            session_id="",
            agent=agent,
            color=agent.config.color,
            emoji=agent.config.emoji,
        )
        self._leads[lead_name] = state
        store = self._runtime.lead_session_store
        if requested_session_id:
            session_id = await agent.ensure_session_with_id(requested_session_id)
        else:
            existing = await store.get(lead_name)
            if existing is not None:
                session_id = await agent.ensure_session_with_id(existing)
            else:
                session_id = await agent.create_session()
        await store.upsert(lead_name, session_id)
        state.session_id = session_id
        self._runtime.set_session_event_handler(
            session_id, self._make_lead_event_handler(lead_name)
        )
        return state

    async def on_mount(self) -> None:
        """Initialize Feather runtime services once Textual is mounted."""

        self._runtime = await FeatherRuntime.create(self._root)
        lead_name = self._runtime.default_lead_name
        # Re-bind the active agent whenever the runtime rebuilds a lead
        # (e.g. after /config saves a NEXT_TURN-class field).
        self._runtime.register_agent_rebuilt_listener(self._on_agent_rebuilt)
        # Bootstrap the default lead (resumes its durable session). The
        # ``--session-id`` override applies to the default lead.
        await self._bootstrap_lead(
            lead_name, requested_session_id=self._requested_session_id
        )
        self._active_lead_name = lead_name
        worker_mode = _should_use_lead_worker(
            yaml_enabled=self._runtime.config.self_repair.enabled,
        )
        if worker_mode:
            await self._start_lead_worker_supervisor()
        await self._runtime.start_background_services(
            lead_in_subprocess=worker_mode
        )
        await self._refresh_monitor()
        self._update_header()
        self._update_work()
        self._write_conversation(
            "Feather",
            "Session ready.",
            label_style="bold grey70",
            body_style="grey70",
        )
        self.query_one("#composer", TextArea).focus()
        # Bring up every other discovered lead (in-process), so leads run
        # concurrently and switching is instant. The default lead keeps the
        # worker-mode self-repair treatment above; additional leads run
        # in-process in this phase.
        for entry in self._runtime.agent_catalog.list_leads():
            if entry.name != lead_name and entry.name not in self._leads:
                try:
                    await self._bootstrap_lead(entry.name)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "textual_tui.bootstrap_lead_failed", extra={"lead": entry.name}
                    )
        # Start one concurrent driver per active lead.
        for state in self._leads.values():
            state.driver_task = asyncio.create_task(self._run_lead_driver(state))
        self._watcher_task = asyncio.create_task(self._inbox_watcher())
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        # Drive the streaming-spinner animation. The tick is cheap when
        # nothing is streaming (early-out below); during a streaming
        # window it re-renders the conversation once per frame so the
        # spinner glyph advances visually even when no new text deltas
        # are arriving (e.g., the model is mid-reasoning).
        self.set_interval(_SPINNER_INTERVAL_SECONDS, self._tick_spinner)

    async def on_unmount(self) -> None:
        """Stop runtime services and background tasks."""

        self._stop_event.set()
        # Cancel + await every background task in one pass. Per-lead drivers +
        # run tasks (one set per active lead) plus the app-global watcher /
        # monitor / hang-watcher. ``_hang_watcher_task`` shares the same shape
        # as the rest — a raw asyncio.Task with no ``stop()`` method.
        background_tasks: list[asyncio.Task[Any] | None] = [
            self._watcher_task,
            self._monitor_task,
            self._hang_watcher_task,
        ]
        for state in self._leads.values():
            background_tasks.append(state.driver_task)
            background_tasks.append(state.run_task)
        for task in background_tasks:
            if task is not None and not task.done():
                task.cancel()
        for task in background_tasks:
            if task is None:
                continue
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._hang_watcher_task = None
        # Aux subsystems with a stop/shutdown coroutine. Each is independently
        # guarded so one failure logs and yields rather than aborting the rest
        # of the teardown and leaking the remaining subsystems.
        if self._restart_watcher is not None:
            try:
                await self._restart_watcher.stop()
            except Exception:  # noqa: BLE001
                logger.exception("textual_tui.restart_watcher_stop_failed")
            self._restart_watcher = None
        if self._log_triage_bot is not None:
            try:
                await self._log_triage_bot.stop()
            except Exception:  # noqa: BLE001
                logger.exception("textual_tui.log_triage_bot_stop_failed")
            self._log_triage_bot = None
        # Shut down every lead's supervisor (worker mode) and unregister its
        # event handler.
        for state in self._leads.values():
            if state.supervisor is not None:
                if self._runtime is not None and self._is_active(state):
                    self._runtime.detach_supervisor()
                try:
                    await state.supervisor.shutdown()
                except Exception:  # noqa: BLE001
                    logger.exception("textual_tui.lead_supervisor_shutdown_failed")
                state.supervisor = None
            if self._runtime is not None and state.session_id:
                self._runtime.set_session_event_handler(state.session_id, None)
        if self._runtime is not None:
            await self._runtime.close()

    async def _hang_watcher(self) -> None:
        """Background poll: surface a banner when the worker heartbeat goes stale.

        Two-state machine — only fires on transitions, so a sustained
        hang shows one banner, not one per tick. Recovery emits a follow-up
        message so the user knows the worker is healthy again without
        having to re-test it themselves.
        """

        if self._supervisor is None:
            return
        prev_stale = False
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=_HANG_WATCHER_POLL_SECONDS,
                )
                return
            except asyncio.TimeoutError:
                pass
            try:
                current_stale = await self._supervisor.is_stale()
            except Exception:  # noqa: BLE001
                logger.exception("textual_tui.hang_watcher_check_failed")
                continue
            transition = decide_hang_alert(prev_stale, current_stale)
            prev_stale = current_stale
            if transition == "alert":
                self._write_marker(
                    "Lead unresponsive",
                    "the lead worker has stopped sending heartbeats. "
                    "Try /restart-lead to recover. Conversation history "
                    "is preserved across restarts.",
                    style="red",
                )
            elif transition == "recover":
                self._write_marker(
                    "Lead recovered",
                    "heartbeats resumed; the worker is responsive again.",
                    style="green",
                )

    async def _start_lead_worker_supervisor(self) -> None:
        """Spawn the lead-worker subprocess and wire it as the lead handle.

        Opt-in via ``FEATHER_USE_LEAD_WORKER=1``. The in-process ``self._agent``
        is retained so display-only properties (``config.name``) and idempotent
        reads (``has_pending_inbox`` over the shared SQLite mailbox) keep
        working — but every ``run`` / ``resume_on_inbox`` / mid-turn input call
        is routed to the supervisor below.

        Known limitations in this slice (will be addressed in subsequent
        steps of the lead-worker roadmap):

        * Messaging integrations (Telegram, LINE, WhatsApp) enqueue into
          the in-process ``runtime.input_queue``; that queue lives in the
          TUI process, so its contents are not visible to the worker. Use
          worker mode only from the TUI for now.
        * The cron scheduler still drives the in-process agent reference,
          so scheduled prompts bypass the worker. Disable scheduled jobs
          when running in worker mode if you need them processed by the
          worker's run cycle.
        """

        assert self._runtime is not None
        assert self._active_session_id is not None
        self._supervisor = self._runtime.build_lead_supervisor(
            self._active_lead_name or self._runtime.default_lead_name
        )
        self._runtime.attach_supervisor(self._supervisor)
        await self._supervisor.start(self._active_session_id)
        # Crash-recovery: a previous TUI session may have been SIGKILLed
        # mid-restart, leaving `restart_requested_at` set on disk. If we
        # let the watcher see that flag on its first tick it would fire
        # an immediate, surprising restart against a worker that's only
        # been alive for ~1.5 s. Clear any pre-existing flag now so only
        # NEW request_restart calls fire restarts.
        await self._runtime.session_store.clear_restart_request(
            self._active_session_id
        )
        # Surface ERROR-level log entries from the worker (and from the
        # supervisor process itself) into the lead's mailbox. The lead's
        # existing inbox watcher picks them up via resume_on_inbox.
        log_path = Path(self._runtime.config.logging.path)
        if not log_path.is_absolute():
            log_path = (self._root / log_path).resolve()
        self._log_triage_bot = LogTriageBot(
            log_path=log_path,
            message_store=self._runtime.agent_message_store,
            lead_session_id=self._active_session_id,
        )
        await self._log_triage_bot.start()
        # Self-repair: poll the session row for request_restart flags and
        # respawn the worker on the same session id when the lead asks.
        self._restart_watcher = RestartWatcher(
            session_store=self._runtime.session_store,
            message_store=self._runtime.agent_message_store,
            lead_session_id=self._active_session_id,
            restart_fn=self._supervisor.restart,
            cancel_in_flight_run=self._cancel_in_flight_run,
        )
        await self._restart_watcher.start()
        # Watch the heartbeat for hangs and surface a banner to the user.
        self._hang_watcher_task = asyncio.create_task(
            self._hang_watcher(), name="textual_tui.hang_watcher"
        )
        logger.info(
            "textual_tui.lead_worker_started session_id=%s",
            self._active_session_id,
        )

    def _on_agent_rebuilt(self, name: str, new_agent: Any) -> None:
        """Rebind the rebuilt lead's in-process agent after a config reload.

        Wired in :meth:`on_mount` via ``register_agent_rebuilt_listener``.
        Targets the specific lead by name (not just the active one), so a
        background lead picks up its new provider/model too. When that lead is
        worker-supervised, the in-process agent is not on the request route —
        the supervisor handles its own swap via ``request_config_reload`` — so
        we skip the rebind to avoid masking supervisor-side reload failures.
        """

        state = self._leads.get(name)
        if state is None or state.supervisor is not None:
            return
        state.agent = new_agent

    async def _cancel_in_flight_run(self) -> bool:
        """Cancel the agent driver's current run task, if any.

        Used by the restart watcher so the LeadSupervisor.shutdown
        invariant ("no concurrent run/shutdown") holds when the
        supervisor restarts the worker. Returns True iff a task was
        actually cancelled.
        """

        run_task = self._run_task
        if run_task is None or run_task.done():
            return False
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        self._run_task = None
        return True

    async def action_submit(self) -> None:
        """Submit the focused composer.

        When the slash-command dropdown is open with a highlighted
        command, the composer text is replaced with the canonical
        ``/<name>`` form before submission so aliases and abbreviations
        always dispatch to the same handler.
        """

        if self.focused is not self.query_one("#composer", TextArea):
            return
        dropdown = self._slash_dropdown_widget
        if dropdown is not None and dropdown.is_open and dropdown.selected_command is not None:
            composer = self.query_one("#composer", TextArea)
            cmd = dropdown.selected_command
            # Preserve any args the user has already typed past the first
            # whitespace — matching the dropdown rule that args close it,
            # this branch should rarely fire, but defend anyway so we do
            # not throw away typed text.
            parsed = parse_slash_input(composer.text)
            args = parsed.args if parsed is not None else ""
            composer.text = cmd.display + (f" {args}" if args else "")
            dropdown.close()
        await self._submit_composer()

    async def action_interrupt(self) -> None:
        """Interrupt the active run, or close the slash dropdown first.

        Escape has two roles depending on context: when the dropdown is
        open it dismisses the dropdown; otherwise it cancels the active
        run (or notes there is nothing to cancel).
        """

        dropdown = self._slash_dropdown_widget
        if dropdown is not None and dropdown.is_open:
            dropdown.close()
            return
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
            return
        self._write_marker("Interrupt", "No active run to cancel.", style="yellow")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Update the slash-command dropdown when the composer changes."""

        if event.text_area.id != "composer":
            return
        dropdown = self._slash_dropdown_widget
        if dropdown is None:
            return
        if self._slash_suppress_next_change:
            self._slash_suppress_next_change = False
            return
        dropdown.update_for_text(event.text_area.text)

    def action_conversation_page_up(self) -> None:
        """Scroll conversation older by one page."""

        self.query_one("#conversation", RichLog).scroll_page_up(animate=False)

    def action_conversation_page_down(self) -> None:
        """Scroll conversation newer by one page."""

        self.query_one("#conversation", RichLog).scroll_page_down(animate=False)

    def action_conversation_home(self) -> None:
        """Jump to the oldest conversation content."""

        self.query_one("#conversation", RichLog).scroll_home(animate=False)

    def action_conversation_end(self) -> None:
        """Jump to the latest conversation content."""

        self.query_one("#conversation", RichLog).scroll_end(animate=False)

    def action_work_page_up(self) -> None:
        """Scroll the task monitor older by one page."""

        self.query_one("#work", VerticalScroll).scroll_page_up(animate=False)

    def action_work_page_down(self) -> None:
        """Scroll the task monitor newer by one page."""

        self.query_one("#work", VerticalScroll).scroll_page_down(animate=False)

    def action_work_home(self) -> None:
        """Jump to the top of the task monitor."""

        self.query_one("#work", VerticalScroll).scroll_home(animate=False)

    def action_work_end(self) -> None:
        """Jump to the bottom of the task monitor."""

        self.query_one("#work", VerticalScroll).scroll_end(animate=False)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Route wheel-up to the panel under the pointer."""

        self._route_mouse_scroll(event, direction=-1)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        """Route wheel-down to the panel under the pointer."""

        self._route_mouse_scroll(event, direction=1)

    def _route_mouse_scroll(
        self,
        event: events.MouseScrollDown | events.MouseScrollUp,
        *,
        direction: int,
    ) -> None:
        """Scroll conversation or task monitor based on pointer position."""

        target = self._mouse_scroll_target(event)
        if direction < 0:
            target.scroll_up(animate=False)
        else:
            target.scroll_down(animate=False)
        event.prevent_default()
        event.stop()
        self.query_one("#composer", TextArea).focus()

    def _mouse_scroll_target(
        self,
        event: events.MouseScrollDown | events.MouseScrollUp,
    ) -> RichLog | VerticalScroll:
        """Return the scrollable panel under the pointer.

        Header/composer wheel movement falls back to the conversation; the
        composer should never consume wheel movement as text-cursor movement.
        """

        work = self.query_one("#work", VerticalScroll)
        x = event.screen_x if event.screen_x is not None else event.x
        y = event.screen_y if event.screen_y is not None else event.y
        if region_contains_point(work.region, x, y):
            return work
        return self.query_one("#conversation", RichLog)

    def action_copy_selection_or_transcript(self) -> None:
        """Copy selected text, falling back to the full transcript."""

        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            self._write_marker("Copied", "selected text copied to clipboard.")
            return
        self.action_copy_transcript()

    def action_copy_transcript(self) -> None:
        """Copy the conversation transcript to the terminal clipboard."""

        transcript_blocks = list(self._transcript_blocks)
        if self._assistant_parts:
            agent_name = self._agent.config.name if self._agent is not None else "Lead"
            transcript_blocks.append(
                format_transcript_block(
                    f"{agent_name} streaming",
                    "".join(self._assistant_parts),
                )
            )
        transcript = build_transcript_text(tuple(transcript_blocks))
        if not transcript:
            self._write_marker("Copied", "no conversation text yet.", style="yellow")
            return
        self.copy_to_clipboard(transcript)
        self._write_marker("Copied", "conversation transcript copied to clipboard.")

    async def _submit_composer(self) -> None:
        composer = self.query_one("#composer", TextArea)
        raw_text = composer.text
        text = raw_text.strip()
        if not text:
            composer.clear()
            return
        composer.clear()
        dropdown = self._slash_dropdown_widget
        if dropdown is not None:
            dropdown.close()

        if self._dispatch_slash_input(raw_text):
            return

        display_text = summarize_user_input_for_display(text, self._root)
        if self._awaiting_event.is_set():
            self._write_conversation("You", display_text, label_style="bold cyan")
            try:
                self._pending_answer.put_nowait(text)
                self._awaiting_event.clear()
            except asyncio.QueueFull:
                await self._enqueue_user_text(text)
            return

        if self._busy_event.is_set():
            await self._enqueue_user_text(text)
            return

        self._write_conversation("You", display_text, label_style="bold cyan")
        self._status = "running"
        self._update_header()
        await self._new_run_queue.put(text)

    async def _enqueue_user_text(self, text: str) -> None:
        assert self._runtime is not None
        assert self._active_session_id is not None
        # In worker mode the lead's own input queue lives inside the
        # worker process. Forwarding via the supervisor is the only path
        # that actually reaches the agent loop; the in-process queue
        # would never drain (no in-process agent runs) and would grow
        # unbounded — also producing a misleading queue-depth display.
        if self._supervisor is not None:
            try:
                await self._supervisor.enqueue_user_input(
                    self._active_session_id, text
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "textual_tui.lead_worker_enqueue_failed session_id=%s",
                    self._active_session_id,
                )
                return
            await self._refresh_monitor()
            self._update_work()
            return
        ok = await self._runtime.input_queue.enqueue(self._active_session_id, text)
        if ok:
            await self._refresh_monitor()
            self._update_work()

    def _register_default_handlers(self) -> None:
        """Bind every registered slash command to its callable handler.

        Called from ``__init__`` so unit tests can interact with the
        dispatch table without mounting the Textual app.

        Raises:
            RuntimeError: If a registered command has no handler bound.
                Catches "added a command but forgot to wire the handler"
                bugs at app construction instead of letting them surface
                only when a user types the command (review nit 1).
        """

        # ``self.slash_registry`` is populated by ``__init__`` before this
        # method is called; keep handlers narrow so each one stays
        # individually testable.
        binding: dict[str, _SlashHandler] = {
            "help": self._cmd_help,
            "exit": self._cmd_exit,
            "onboard": self._cmd_onboard,
            "qdrant": self._cmd_qdrant,
            "clear": self._cmd_clear,
            "config": self._cmd_config,
            "copy": self._cmd_copy,
            "queue": self._cmd_queue,
            "agents": self._cmd_agents,
            "lead": self._cmd_lead,
            "tasks": self._cmd_tasks,
            "session": self._cmd_session,
            "skills": self._cmd_skills,
            "integrations": self._cmd_integrations,
            "telegram": self._cmd_telegram,
            "line": self._cmd_line,
            "whatsapp": self._cmd_whatsapp,
            "restart-lead": self._cmd_restart_lead,
        }
        handlers: dict[str, _SlashHandler] = {}
        unbound: list[str] = []
        for cmd in self.slash_registry.all():
            handler = binding.get(cmd.name)
            if handler is None:
                unbound.append(cmd.name)
                continue
            handlers[cmd.name.lower()] = handler
            for alias in cmd.aliases:
                handlers[alias.lower()] = handler
        if unbound:
            raise RuntimeError(
                "slash commands are registered but have no handler bound: "
                + ", ".join(unbound)
            )
        self._slash_handlers = handlers
        # Names of handlers that intentionally consume the ``args`` payload.
        # Anything else triggers a "discarded extra args" warning so pasted
        # multi-line bodies after a slash command do not silently vanish
        # (review fix M1).
        self._slash_handlers_accepts_args: set[str] = {
            "config",
            "telegram",
            "line",
            "whatsapp",
            "qdrant",
            "lead",
        }

    def _dispatch_slash_input(self, text: str) -> bool:
        """Route ``text`` through the slash-command dispatcher.

        Args:
            text: Raw composer text.

        Returns:
            True when the text was a slash command (recognised or not);
            False when the text is plain user input that should follow
            the normal awaiting/queue/run path.
        """

        parsed = parse_slash_input(text)
        if parsed is None:
            return False

        token = parsed.name_token.strip()
        if not token:
            self._write_marker(
                "Slash command",
                "Type a command name after '/' or press Esc to cancel.",
                style="yellow",
            )
            return True

        handler = self._slash_handlers.get(token.lower())
        if handler is None:
            cmd = self.slash_registry.find(token)
            if cmd is None:
                self._write_marker(
                    "Unknown command",
                    f"/{token} is not a known command. Type /help for the list.",
                    style="yellow",
                )
                return True
            handler = self._slash_handlers.get(cmd.name.lower())
            if handler is None:
                self._write_marker(
                    "Slash command",
                    f"/{cmd.name} is registered but has no handler bound.",
                    style="red",
                )
                return True

        # Resolve the canonical command name so the args/awaiting checks
        # below behave consistently regardless of which alias the user
        # typed.
        canonical = self.slash_registry.find(token)
        canonical_name = canonical.name.lower() if canonical else token.lower()
        args_text = parsed.args
        accepts_args = canonical_name in self._slash_handlers_accepts_args
        try:
            handler(args_text.strip())
        except Exception as exc:  # noqa: BLE001
            logger.exception("textual_tui.slash_command_failed token=%s", token)
            self._write_marker(
                "Slash command failed",
                f"/{token}: {type(exc).__name__}: {exc}",
                style="red",
            )
            return True

        if not accepts_args and args_text.strip():
            # Review fix M1: warn so pasted multi-line bodies do not vanish
            # silently. The discarded text is included verbatim so the
            # user can copy it back into a fresh message.
            self._write_marker(
                "Slash command",
                f"/{token} ignored extra text: {args_text.strip()}",
                style="yellow",
            )

        if self._awaiting_event.is_set():
            # Review fix N7: surface a reminder so users do not lose track
            # of the still-paused agent after running a local slash command.
            self._write_marker(
                "Awaiting your answer",
                "agent is still paused on its last question",
                style="yellow",
            )
        return True

    def _cmd_help(self, args: str) -> None:
        """Render the full slash-command help block in the conversation."""

        del args
        body = render_help_text(self.slash_registry)
        self._write_conversation(
            "Slash commands",
            body.plain,
            label_style="bold cyan",
            body_style="white",
        )

    def _cmd_exit(self, args: str) -> None:
        """Leave the TUI."""

        del args
        self.exit()

    def _cmd_onboard(self, args: str) -> None:
        """Reset the onboarding markers and exit so the wizard runs.

        The wizard is interactive (sync ``input``/``getpass``) and would
        fight Textual's alternate-screen mode if launched in-process.
        Clearing the completion markers here means the very next
        ``feather tui`` (or ``feather onboard --force``) launches the
        wizard from a clean state.
        """

        del args
        cleared: list[str] = []
        try:
            for name in ("onboarded.json", "user.md"):
                marker = self._root / ".feather" / name
                if marker.exists():
                    marker.unlink()
                    cleared.append(str(marker))
        except OSError as exc:
            self._write_marker(
                "Onboard",
                f"failed to clear markers: {type(exc).__name__}: {exc}",
                style="red",
            )
            return
        body_lines = [
            "Onboarding markers cleared.",
        ]
        if cleared:
            body_lines.append("removed:")
            body_lines.extend(f"  {p}" for p in cleared)
        else:
            body_lines.append("(no completion markers were present)")
        body_lines.append(
            "Restart the TUI (`uv run feather tui`) to launch the wizard, "
            "or run `feather onboard --force` for the standalone flow."
        )
        self._write_conversation(
            "Onboard",
            "\n".join(body_lines),
            label_style="bold cyan",
            body_style="white",
        )
        # Exit so the user can immediately re-run and hit the wizard.
        self.exit()

    def _cmd_qdrant(self, args: str) -> None:
        """Manage the local Qdrant Docker container.

        Subcommands: ``status`` (default), ``start``, ``stop``,
        ``remove``, ``help``. All Docker calls run in a worker thread
        via :func:`asyncio.to_thread` so the TUI event loop stays
        responsive while the Docker daemon does its work.
        """

        parts = args.split(maxsplit=1)
        sub = (parts[0] if parts else "status").lower()
        if sub in {"help", "?", "-h", "--help"}:
            self._write_conversation(
                "/qdrant help",
                (
                    "/qdrant status   show whether the Qdrant container is running\n"
                    "/qdrant start    start (or create) the local container\n"
                    "/qdrant stop     stop the running container — data preserved\n"
                    "/qdrant remove   stop + remove the container — data preserved\n"
                    "                  (run `docker volume rm feather-qdrant-data` "
                    "to wipe vectors)"
                ),
                label_style="bold cyan",
                body_style="white",
            )
            return
        if sub == "status":
            self._spawn_async_command(self._qdrant_status_async())
            return
        if sub == "start":
            self._spawn_async_command(self._qdrant_start_async())
            return
        if sub == "stop":
            self._spawn_async_command(self._qdrant_stop_async())
            return
        if sub == "remove":
            self._spawn_async_command(self._qdrant_remove_async())
            return
        self._write_marker(
            "/qdrant",
            f"unknown subcommand '{sub}'. Try '/qdrant help'.",
            style="yellow",
        )

    async def _qdrant_status_async(self) -> None:
        """Probe Qdrant's ``/readyz`` and the local docker container.

        ``/readyz`` works regardless of how Qdrant is hosted (compose
        sibling, bare-metal docker, cloud) so it's the primary signal.
        Docker container state is a useful secondary detail when we're
        bare-metal and the daemon is reachable; in compose it's not.
        """

        url, source = _resolve_qdrant_url()
        reachable = await _probe_qdrant_readyz(url)
        body_lines = [
            f"endpoint: {url} ({source})",
            f"reachable: {'yes' if reachable else 'no'}",
        ]
        if not _is_compose_managed_qdrant():
            from feather.onboarding import (
                DockerNotAvailable,
                docker_available,
                qdrant_container_state,
            )

            if await asyncio.to_thread(docker_available):
                try:
                    status = await asyncio.to_thread(qdrant_container_state)
                except DockerNotAvailable:
                    pass
                else:
                    body_lines.append(
                        f"container 'feather-qdrant': {status.state}"
                    )
        self._write_conversation(
            "/qdrant status",
            "\n".join(body_lines),
            label_style="bold cyan",
            body_style="white",
        )

    async def _qdrant_start_async(self) -> None:
        if _is_compose_managed_qdrant():
            self._write_marker(
                "/qdrant",
                "Qdrant is managed by docker compose — run "
                "'docker compose start qdrant' on the host.",
                style="yellow",
            )
            return
        from feather.onboarding import (
            DockerNotAvailable,
            QdrantStartFailed,
            ensure_local_qdrant_container,
        )

        say_buffer: list[str] = []
        try:
            url = await asyncio.to_thread(
                ensure_local_qdrant_container,
                say=say_buffer.append,
            )
        except DockerNotAvailable as exc:
            self._write_marker("/qdrant", str(exc), style="red")
            return
        except QdrantStartFailed as exc:
            self._write_marker(
                "/qdrant", f"start failed: {exc}", style="red"
            )
            return
        self._write_conversation(
            "/qdrant start",
            "\n".join(say_buffer + [f"URL: {url}"]),
            label_style="bold green",
            body_style="white",
        )

    async def _qdrant_stop_async(self) -> None:
        if _is_compose_managed_qdrant():
            self._write_marker(
                "/qdrant",
                "Qdrant is managed by docker compose — run "
                "'docker compose stop qdrant' on the host.",
                style="yellow",
            )
            return
        from feather.onboarding import (
            DockerNotAvailable,
            QdrantStartFailed,
            stop_local_qdrant_container,
        )

        say_buffer: list[str] = []
        try:
            state = await asyncio.to_thread(
                stop_local_qdrant_container,
                say=say_buffer.append,
            )
        except DockerNotAvailable as exc:
            self._write_marker("/qdrant", str(exc), style="red")
            return
        except QdrantStartFailed as exc:
            self._write_marker(
                "/qdrant", f"stop failed: {exc}", style="red"
            )
            return
        self._write_conversation(
            "/qdrant stop",
            "\n".join(say_buffer + [f"state: {state}"]),
            label_style="bold cyan",
            body_style="white",
        )

    async def _qdrant_remove_async(self) -> None:
        if _is_compose_managed_qdrant():
            self._write_marker(
                "/qdrant",
                "Qdrant is managed by docker compose — run "
                "'docker compose down qdrant' on the host. "
                "(Add `-v` to wipe the volume and lose all vectors.)",
                style="yellow",
            )
            return
        from feather.onboarding import (
            DockerNotAvailable,
            QdrantStartFailed,
            remove_local_qdrant_container,
        )

        say_buffer: list[str] = []
        try:
            state = await asyncio.to_thread(
                remove_local_qdrant_container,
                say=say_buffer.append,
            )
        except DockerNotAvailable as exc:
            self._write_marker("/qdrant", str(exc), style="red")
            return
        except QdrantStartFailed as exc:
            self._write_marker(
                "/qdrant", f"remove failed: {exc}", style="red"
            )
            return
        self._write_conversation(
            "/qdrant remove",
            "\n".join(say_buffer + [f"state: {state}"]),
            label_style="bold cyan",
            body_style="white",
        )

    def _cmd_restart_lead(self, args: str) -> None:
        """Respawn the lead worker subprocess (worker mode only).

        Manual recovery hook for the hang banner and for "I just patched
        the lead's code, please reload it." The supervisor's restart()
        does the SIGTERM→SIGKILL dance and respawns on the same
        ``--session-id``, so conversation history is preserved.
        """

        del args
        if self._supervisor is None:
            self._write_marker(
                "/restart-lead",
                (
                    "lead worker mode is off — the lead is running in this "
                    "process, so a worker restart is meaningless. "
                    "Set FEATHER_USE_LEAD_WORKER=1 and relaunch feather."
                ),
                style="yellow",
            )
            return
        self._spawn_async_command(self._restart_lead_async())

    async def _restart_lead_async(self) -> None:
        assert self._supervisor is not None
        # Cancel any in-flight turn so the supervisor.restart() invariant
        # ("no run() racing the shutdown") holds. The agent driver records
        # "Interrupted" and loops back to await new input.
        await self._cancel_in_flight_run()
        self._write_marker(
            "/restart-lead", "restarting lead worker…", style="cyan"
        )
        try:
            await self._supervisor.restart()
        except Exception as exc:  # noqa: BLE001
            logger.exception("textual_tui.restart_lead_failed")
            # Clear the busy/awaiting state on the failure path too —
            # otherwise a slash-driven restart that hits a spawn error
            # leaves the TUI stuck "running" with no recovery path
            # short of quitting.
            self._busy_event.clear()
            self._awaiting_event.clear()
            self._status = "idle"
            self._update_header()
            self._write_marker(
                "/restart-lead",
                f"restart failed: {type(exc).__name__}: {exc}. "
                "Type /exit and relaunch feather if the worker stays down.",
                style="red",
            )
            return
        self._busy_event.clear()
        self._awaiting_event.clear()
        self._status = "idle"
        self._update_header()
        self._write_marker(
            "/restart-lead",
            "lead worker restarted. Conversation history is preserved; "
            "type your next message to continue.",
            style="green",
        )

    def _cmd_clear(self, args: str) -> None:
        """Clear the on-screen transcript without touching session state."""

        del args
        self._conversation_blocks = []
        self._transcript_blocks = []
        self._assistant_parts = []
        self._render_conversation()
        self._write_marker("Cleared", "transcript cleared (session history kept)")

    def _cmd_config(self, args: str) -> None:
        """Dispatch the `/config <sub> [args]` slash command.

        When invoked with no arguments (bare ``/config``), pushes the
        interactive :class:`~feather.tui.config_screen.ConfigScreen`
        modal. With subcommands, falls through to the headless dispatcher.

        Args:
            args: Raw argument string after the ``/config`` token, e.g.
                ``"get app.active_provider"`` or ``"set app.active_provider claude"``.
        """

        from feather.config.service import ConfigService
        from feather.config.slash import handle_config_command
        from feather.paths import FeatherPaths as _Paths

        assert self._runtime is not None
        # Get paths defensively — fallback to a fresh FeatherPaths if the TUI
        # doesn't track them explicitly.
        paths = getattr(self, "_paths", None) or _Paths(project_root=self._root)

        service = ConfigService(
            paths=paths,
            app_config=self._runtime.config,
        )

        # Bare /config → open the interactive modal (Phase 2).
        if not args.strip():
            from feather.tui.config_screen import ConfigScreen

            self.push_screen(ConfigScreen(service=service, runtime=self._runtime))
            return

        result = handle_config_command(service, args)
        self._write_marker(
            "Config",
            result.body,
            style="cyan" if result.ok else "red",
        )

        if result.ok and result.requires_apply:
            runtime = self._runtime

            async def _apply() -> None:
                try:
                    outcome = await runtime.apply_config_change(
                        list(result.requires_apply or [])
                    )
                except Exception as exc:  # noqa: BLE001 - surface apply errors to user
                    self._write_marker(
                        "Config apply error",
                        f"apply error: {type(exc).__name__}: {exc}",
                        style="red",
                    )
                    return
                msg_parts: list[str] = []
                if outcome.applied:
                    msg_parts.append(f"Applied: {', '.join(outcome.applied)}")
                if outcome.needs_restart_lead:
                    msg_parts.append(
                        "Needs /restart-lead: " + ", ".join(outcome.needs_restart_lead)
                    )
                if outcome.needs_restart_app:
                    msg_parts.append(
                        "Needs full restart: " + ", ".join(outcome.needs_restart_app)
                    )
                self._write_marker(
                    "Config apply",
                    "\n".join(msg_parts) or "no changes applied",
                    style="cyan",
                )

            self._spawn_async_command(_apply())

    def _cmd_copy(self, args: str) -> None:
        """Copy the transcript to the terminal clipboard."""

        del args
        self.action_copy_transcript()

    def _cmd_queue(self, args: str) -> None:
        """Render the current input queue snapshot."""

        del args
        if not self._queued_messages:
            self._write_marker(
                "Queue",
                f"depth {self._queue_depth} (no queued user inputs)",
                style="cyan",
            )
            return
        rendered = "\n".join(
            f"{i}. {message}" for i, message in enumerate(self._queued_messages, 1)
        )
        self._write_conversation(
            "Queue",
            f"depth {self._queue_depth}\n{rendered}",
            label_style="bold cyan",
            body_style="white",
        )

    def _cmd_agents(self, args: str) -> None:
        """Show currently live sub-agents (snapshot from the registry)."""

        del args
        if not self._running_agents:
            self._write_marker("Agents", "no live sub-agents", style="cyan")
            return
        body = "\n".join(self._running_agents)
        self._write_conversation(
            "Live sub-agents",
            body,
            label_style="bold cyan",
            body_style="white",
        )

    def _cmd_lead(self, args: str) -> None:
        """Manage leads: ``/lead list | souls | switch <name> | new <name> [--soul <id>] [soul]``."""

        parts = args.split(maxsplit=1) if args.strip() else []
        sub = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""
        if sub in ("list", "ls"):
            self._lead_list()
        elif sub in ("souls", "soul"):
            self._lead_souls()
        elif sub == "switch":
            self._lead_switch(rest)
        elif sub == "new":
            self._lead_new(rest)
        else:
            self._write_marker(
                "Lead",
                "usage: /lead list | souls | switch <name> | new <name> [--soul <id>] [soul]",
                style="yellow",
            )

    def _lead_souls(self) -> None:
        """List the selectable personality presets from the soul library."""

        if self._runtime is None:
            self._write_marker("Souls", "soul library unavailable", style="yellow")
            return
        lines = [
            f"{soul.emoji} {soul.title} [{soul.id}] — {soul.personality}"
            for soul in self._runtime.soul_library.list()
        ]
        self._write_conversation(
            "Souls",
            "\n".join(lines) or "(none — packaged souls missing)",
            label_style="bold cyan",
            body_style="white",
        )

    def _lead_list(self) -> None:
        lines: list[str] = []
        for name in self._lead_names():
            state = self._leads[name]
            mark = "  ← active" if name == self._active_lead_name else ""
            glyph = state.emoji or "•"
            lines.append(f"{glyph} {state.display_name} [{name}] · {state.status}{mark}")
        if self._runtime is not None:
            active = set(self._leads)
            for entry in self._runtime.agent_catalog.list_leads():
                if entry.name not in active:
                    lines.append(f"• {entry.name} (not started — /lead switch to start)")
        self._write_conversation(
            "Leads",
            "\n".join(lines) or "(none)",
            label_style="bold cyan",
            body_style="white",
        )

    def _lead_switch(self, name: str) -> None:
        if not name:
            self._write_marker("Lead", "usage: /lead switch <name>", style="yellow")
            return
        name = name.lower()
        if name not in self._leads:
            self._write_marker("Lead", f"lead {name!r} is not active", style="yellow")
            return
        self._active_lead_name = name
        self._render_active()

    def _lead_new(self, rest: str) -> None:
        if not rest:
            self._write_marker(
                "Lead", "usage: /lead new <name> [--soul <id>] [soul]", style="yellow"
            )
            return
        name, soul_id, soul = self._parse_lead_new(rest)
        if not AgentCatalog.is_valid_name(name):
            self._write_marker(
                "Lead", f"invalid lead name: {name!r} (use letters, digits, _ , -)", style="yellow"
            )
            return
        name = name.lower()
        if name in self._leads:
            self._write_marker("Lead", f"lead {name!r} is already active", style="yellow")
            return
        if soul_id is not None and (
            self._runtime is None or self._runtime.soul_library.get(soul_id) is None
        ):
            self._write_marker(
                "Lead",
                f"unknown soul: {soul_id!r} — see /lead souls",
                style="yellow",
            )
            return
        asyncio.create_task(self._lead_new_async(name, soul, soul_id))

    @staticmethod
    def _parse_lead_new(rest: str) -> tuple[str, str | None, str]:
        """Parse ``<name> [--soul <id>] [free text]`` → (name, soul_id, free_text)."""

        tokens = rest.split()
        name = tokens[0] if tokens else ""
        soul_id: str | None = None
        text_parts: list[str] = []
        i = 1
        while i < len(tokens):
            if tokens[i] in ("--soul", "-s") and i + 1 < len(tokens):
                soul_id = tokens[i + 1]
                i += 2
                continue
            text_parts.append(tokens[i])
            i += 1
        return name, soul_id, " ".join(text_parts)

    async def _lead_new_async(self, name: str, soul: str, soul_id: str | None = None) -> None:
        try:
            self._scaffold_lead_yaml(name, soul, soul_id)
            state = await self._bootstrap_lead(name)
            state.driver_task = asyncio.create_task(self._run_lead_driver(state))
            self._active_lead_name = name
            self._render_active()
            self._write_marker(
                "Lead", f"created and switched to {state.display_name}", style="green"
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("textual_tui.lead_new_failed")
            self._leads.pop(name, None)
            self._write_marker("Lead", f"failed to create lead: {exc}", style="red")

    def _scaffold_lead_yaml(self, name: str, soul: str, soul_id: str | None = None) -> None:
        """Write a project lead YAML, applying a soul preset when ``soul_id`` is set."""

        preset = None
        if soul_id and self._runtime is not None:
            preset = self._runtime.soul_library.get(soul_id)
        scaffold_lead_yaml(self._root, name, soul, soul_preset=preset)

    def _cmd_tasks(self, args: str) -> None:
        """Show tracked tasks."""

        del args
        if not self._task_rows:
            self._write_marker("Tasks", "no tracked tasks", style="cyan")
            return
        body = "\n".join(self._task_rows)
        self._write_conversation(
            "Tasks",
            body,
            label_style="bold cyan",
            body_style="white",
        )

    def _cmd_session(self, args: str) -> None:
        """Show session metadata (id, agent, context usage)."""

        del args
        session_id = self._active_session_id or "(starting)"
        agent_name = self._agent.config.name if self._agent is not None else "Lead"
        ctx = (
            "ctx --"
            if self._latest_usage_ratio is None
            else f"ctx {round(self._latest_usage_ratio * 100)}%"
        )
        body = f"agent {agent_name}\nsession {session_id}\n{ctx}\nstatus {self._status}"
        self._write_conversation(
            "Session",
            body,
            label_style="bold cyan",
            body_style="white",
        )

    def _cmd_skills(self, args: str) -> None:
        """List skills available to the lead agent (without loading them)."""

        del args
        runtime = self._runtime
        if runtime is None:
            self._write_marker(
                "Skills",
                "runtime not ready yet",
                style="yellow",
            )
            return
        try:
            entries = runtime.skill_catalog.list_metadata()
        except Exception as exc:  # noqa: BLE001
            self._write_marker(
                "Skills",
                f"failed to load catalog: {type(exc).__name__}: {exc}",
                style="red",
            )
            return
        if not entries:
            self._write_marker("Skills", "no skills installed", style="cyan")
            return
        rows = [f"{meta.name}  {meta.description}".rstrip() for meta in entries]
        self._write_conversation(
            "Skills",
            "\n".join(rows),
            label_style="bold cyan",
            body_style="white",
        )

    # ---- Messaging integration slash commands --------------------------

    def _cmd_integrations(self, args: str) -> None:
        """Show one-line status for every messaging platform."""

        del args
        if self._runtime is None:
            self._write_marker(
                "Integrations",
                "runtime not ready yet",
                style="yellow",
            )
            return
        self._spawn_async_command(self._integrations_async())

    async def _integrations_async(self) -> None:
        from feather.messaging.models import Platform

        runtime = self._runtime
        if runtime is None:
            return
        statuses = await runtime.messaging_service.status()
        rows = []
        for platform in (Platform.TELEGRAM, Platform.LINE, Platform.WHATSAPP):
            status = statuses.get(platform)
            if status is None:
                continue
            row = (
                f"{platform.value:<9} {status.state.value:<13} "
                f"chats={status.connected_chat_count:<3} {status.detail}"
            )
            if status.last_error:
                row += f"  ! {status.last_error}"
            rows.append(row)
        self._write_conversation(
            "Integrations",
            "\n".join(rows) if rows else "no integrations registered",
            label_style="bold cyan",
            body_style="white",
        )

    def _cmd_telegram(self, args: str) -> None:
        """Manage the Telegram bot integration."""

        from feather.messaging.models import Platform

        self._dispatch_messaging_command(
            platform=Platform.TELEGRAM,
            args=args,
            connect_help=(
                "/telegram connect <bot_token>"
                "\n  bot_token comes from @BotFather (e.g. 1234:ABC...)."
                "\n  Long polling — no public URL needed."
            ),
            connect_parser=_parse_telegram_args,
        )

    def _cmd_line(self, args: str) -> None:
        """Manage the LINE Messaging API integration."""

        from feather.messaging.models import Platform

        self._dispatch_messaging_command(
            platform=Platform.LINE,
            args=args,
            connect_help=(
                "/line connect <channel_secret> <channel_token>"
                "\n  Find both in the LINE Developers console (Messaging API channel)."
                "\n  Webhook URL: <local-base>/line/webhook — expose via ngrok / cloudflared."
            ),
            connect_parser=_parse_line_args,
        )

    def _cmd_whatsapp(self, args: str) -> None:
        """Manage the WhatsApp Cloud API integration."""

        from feather.messaging.models import Platform

        self._dispatch_messaging_command(
            platform=Platform.WHATSAPP,
            args=args,
            connect_help=(
                "/whatsapp connect <phone_number_id> <access_token> "
                "<verify_token> <app_secret>"
                "\n  phone_number_id and access_token are from Meta for Developers."
                "\n  verify_token is anything you choose (paste the same value into Meta's webhook config)."
                "\n  app_secret is the App Secret shown next to the App ID in your Meta app dashboard."
                "\n  Webhook URL: <local-base>/whatsapp/webhook — expose via ngrok / cloudflared."
            ),
            connect_parser=_parse_whatsapp_args,
        )

    def _dispatch_messaging_command(
        self,
        *,
        platform: Any,
        args: str,
        connect_help: str,
        connect_parser: Callable[[str], dict[str, Any]],
    ) -> None:
        """Shared subcommand parser for /telegram /line /whatsapp."""

        if self._runtime is None:
            self._write_marker(
                platform.value.title(),
                "runtime not ready yet",
                style="yellow",
            )
            return
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "status"
        rest = parts[1] if len(parts) > 1 else ""

        if sub == "" or sub == "status":
            self._spawn_async_command(self._messaging_status_async(platform))
            return
        if sub in {"help", "?", "-h", "--help"}:
            self._write_conversation(
                f"/{platform.value} help",
                connect_help,
                label_style="bold cyan",
                body_style="white",
            )
            return
        if sub == "disconnect":
            self._spawn_async_command(self._messaging_disconnect_async(platform))
            return
        if sub == "connect":
            try:
                config = connect_parser(rest)
            except ValueError as exc:
                self._write_marker(
                    f"/{platform.value} connect",
                    str(exc),
                    style="red",
                )
                return
            self._spawn_async_command(
                self._messaging_connect_async(platform, config)
            )
            return
        self._write_marker(
            f"/{platform.value}",
            f"unknown subcommand '{sub}'. Try '/{platform.value} help'.",
            style="yellow",
        )

    async def _messaging_status_async(self, platform: Any) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        statuses = await runtime.messaging_service.status()
        status = statuses.get(platform)
        if status is None:
            self._write_marker(
                f"/{platform.value}",
                "platform unknown",
                style="yellow",
            )
            return
        body = (
            f"state: {status.state.value}\n"
            f"chats mapped: {status.connected_chat_count}\n"
            f"detail: {status.detail or '-'}\n"
        )
        if status.last_error:
            body += f"last error: {status.last_error}\n"
        self._write_conversation(
            f"/{platform.value} status",
            body.strip(),
            label_style="bold cyan",
            body_style="white",
        )

    async def _messaging_connect_async(
        self, platform: Any, config: dict[str, Any]
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        try:
            status = await runtime.messaging_service.connect(platform, config)
        except ValueError as exc:
            # Validation errors from adapter __init__ — show the exact message.
            self._write_marker(
                f"/{platform.value} connect",
                str(exc),
                style="red",
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._write_marker(
                f"/{platform.value} connect",
                f"{type(exc).__name__}: {exc}",
                style="red",
            )
            return
        self._write_conversation(
            f"/{platform.value} connected",
            (
                f"state: {status.state.value}\n"
                f"detail: {status.detail or '-'}"
            ),
            label_style="bold green",
            body_style="white",
        )

    async def _messaging_disconnect_async(self, platform: Any) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        try:
            await runtime.messaging_service.disconnect(platform)
        except Exception as exc:  # noqa: BLE001
            self._write_marker(
                f"/{platform.value} disconnect",
                f"{type(exc).__name__}: {exc}",
                style="red",
            )
            return
        self._write_marker(
            f"/{platform.value}",
            "disconnected and credentials removed",
            style="cyan",
        )

    def _spawn_async_command(self, coro: Any) -> None:
        """Schedule an async command from a sync slash handler.

        Errors are caught and surfaced via ``_write_marker``.
        """

        async def _runner() -> None:
            try:
                await coro
            except Exception as exc:  # noqa: BLE001
                logger.exception("textual_tui.async_command_failed")
                self._write_marker(
                    "Slash command failed",
                    f"{type(exc).__name__}: {exc}",
                    style="red",
                )

        try:
            asyncio.create_task(_runner())
        except RuntimeError:
            # No running loop (unit tests); just drop it.
            coro.close()

    def _is_active(self, state: PerLeadState) -> bool:
        """Whether ``state`` is the lead currently shown in the widgets.

        Identity against ``_active()`` so the pre-mount bootstrap placeholder
        (which ``_active()`` returns before a real lead is wired) also counts
        as active and renders.
        """

        return state is self._active()

    def _handle_for(self, state: PerLeadState) -> Any:
        """The run surface for ``state`` (supervisor when worker-mode, else agent)."""

        return state.supervisor if state.supervisor is not None else state.agent

    def _handle_runtime_event(self, event: RuntimeEvent) -> None:
        """Apply an event to the active lead (back-compat entry point)."""

        self._apply_event(self._active(), event)

    def _apply_event(self, state: PerLeadState, event: RuntimeEvent) -> None:
        """Apply one runtime event to ``state``; refresh widgets only if active."""

        active = self._is_active(state)
        if event.kind == "assistant_text_delta":
            state.status = "running"
            state.assistant_parts.append(event.text or "")
            if active:
                self._update_header()
                self._render_conversation()
            return
        if event.kind == "usage_updated":
            ratio = (event.payload or {}).get("usage_ratio")
            if isinstance(ratio, (int, float)):
                state.latest_usage_ratio = max(0.0, min(1.0, float(ratio)))
                if active:
                    self._update_header()
            return

        self._finish_assistant_turn(state)
        if event.kind == "tool_started":
            state.status = "running"
            state.active_tool = event.tool_name
            if active:
                self._update_header()
            if event.tool_name != "ask_user":
                self._write_tool_started(event, state=state)
        elif event.kind == "tool_finished":
            state.active_tool = None
            if active:
                self._update_header()
            if event.tool_name == "ask_user":
                return
            failed_title = _failed_tool_title(event.tool_name, event.text)
            if failed_title:
                self._write_tool_finished(event, failed_title=failed_title, state=state)
            else:
                self._write_tool_finished(event, failed_title=None, state=state)
                if event.tool_name == "spawn_agent":
                    self._record_task_update(
                        f"started {_format_tool_result(event.tool_name, event.text)}",
                        state=state,
                    )
        elif event.kind == "awaiting_user":
            state.status = "awaiting user"
            self._write_conversation(
                f"{state.agent.config.name} asks",
                event.text or "",
                label_style="bold green",
                state=state,
            )
            if active:
                self._update_header()
        elif event.kind == "user_message_injected":
            self._write_conversation(
                "Queued input",
                event.text or "",
                label_style="bold grey70",
                body_style="grey70",
                state=state,
            )
        elif event.kind == "agent_message_received":
            self._record_task_update(summarize_agent_message_update(event), state=state)
            self._write_conversation(
                "Sub-agent completed",
                format_agent_message_event(event),
                label_style="bold magenta",
                state=state,
            )
        elif event.kind.startswith("compaction_"):
            self._write_marker(
                _event_title(event.kind),
                event.text or "",
                style=_system_event_style(event.kind),
                state=state,
            )
        elif event.kind.startswith("scheduled_task_"):
            self._write_marker(
                _event_title(event.kind),
                event.text or "",
                style=_system_event_style(event.kind),
                state=state,
            )

    def _finish_assistant_turn(self, state: PerLeadState | None = None) -> None:
        state = state if state is not None else self._active()
        if not state.assistant_parts:
            return
        body = "".join(state.assistant_parts)
        state.assistant_parts.clear()
        self._write_conversation(
            state.agent.config.name,
            body,
            label_style="bold green",
            state=state,
        )

    async def _run_lead_driver(self, state: PerLeadState) -> None:
        """Per-lead run loop. Runs concurrently for every active lead; refreshes
        widgets only while ``state`` is the lead on screen."""

        on_event = self._make_lead_event_handler(state.name)
        while not self._stop_event.is_set():
            get_task = asyncio.create_task(state.new_run_queue.get())
            stop_task = asyncio.create_task(self._stop_event.wait())
            await asyncio.wait(
                {get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if self._stop_event.is_set():
                get_task.cancel()
                return
            stop_task.cancel()
            user_message = get_task.result()

            state.busy_event.set()
            state.status = "running"
            if self._is_active(state):
                self._update_header()
            try:
                handle = self._handle_for(state)
                if user_message == _INBOX_WAKE:
                    state.run_task = asyncio.create_task(
                        handle.resume_on_inbox(state.session_id, on_event)
                    )
                else:
                    state.run_task = asyncio.create_task(
                        handle.run(state.session_id, user_message, on_event)
                    )
                try:
                    result = await state.run_task
                except asyncio.CancelledError:
                    if self._stop_event.is_set():
                        return
                    self._record_interrupted(state=state)
                    continue
                finally:
                    state.run_task = None
                if result is None:
                    state.status = "idle"
                    if self._is_active(state):
                        self._update_header()
                    continue

                self._finish_assistant_turn(state)
                await self._refresh_monitor(state)
                if result.status == AgentOutcome.COMPLETED:
                    state.status = "idle"
                if self._is_active(state):
                    self._update_header()
                    self._update_work()

                while (
                    result.status == AgentOutcome.AWAITING_USER
                    and result.question is not None
                    and not self._stop_event.is_set()
                ):
                    state.awaiting_event.set()
                    answer_task = asyncio.create_task(state.pending_answer.get())
                    stop_task = asyncio.create_task(self._stop_event.wait())
                    await asyncio.wait(
                        {answer_task, stop_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if self._stop_event.is_set():
                        answer_task.cancel()
                        state.awaiting_event.clear()
                        return
                    stop_task.cancel()
                    answer = answer_task.result()
                    state.run_task = asyncio.create_task(
                        handle.run(state.session_id, answer, on_event)
                    )
                    try:
                        result = await state.run_task
                    except asyncio.CancelledError:
                        if self._stop_event.is_set():
                            return
                        self._record_interrupted(state=state)
                        break
                    finally:
                        state.run_task = None
                    self._finish_assistant_turn(state)
                    await self._refresh_monitor(state)
                    if result.status == AgentOutcome.COMPLETED:
                        state.status = "idle"
                    if self._is_active(state):
                        self._update_header()
                        self._update_work()
            except Exception as exc:  # noqa: BLE001
                logger.exception("textual_tui.agent_driver_crashed")
                self._write_marker(
                    "Agent error", f"{type(exc).__name__}: {exc}", style="red", state=state
                )
                state.status = "idle"
                if self._is_active(state):
                    self._update_header()
            finally:
                state.busy_event.clear()
                state.awaiting_event.clear()

    async def _inbox_watcher(self) -> None:
        """Wake any idle lead whose SQLite inbox has pending peer messages."""

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=0.5)
                return
            except asyncio.TimeoutError:
                pass
            for state in list(self._leads.values()):
                if state.busy_event.is_set() or state.awaiting_event.is_set():
                    continue
                if state.agent is None or not state.session_id:
                    continue
                try:
                    has_pending = await state.agent.has_pending_inbox(state.session_id)
                except Exception:  # noqa: BLE001
                    continue
                if has_pending and state.new_run_queue.empty():
                    await state.new_run_queue.put(_INBOX_WAKE)

    async def _monitor_loop(self) -> None:
        """Refresh durable task and queue state for every lead, even when idle."""

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                return
            except asyncio.TimeoutError:
                pass
            for state in list(self._leads.values()):
                try:
                    await self._refresh_monitor(state)
                except Exception:  # noqa: BLE001
                    logger.exception("textual_tui.monitor_refresh_failed")

    async def _refresh_monitor(self, state: PerLeadState | None = None) -> None:
        """Refresh one lead's queue/sub-agent/task rows; repaint work if active."""

        assert self._runtime is not None
        state = state if state is not None else self._active()
        session_id = state.session_id
        if not session_id:
            return
        depth = await self._runtime.input_queue.depth(session_id)
        pending = await self._runtime.input_queue.peek(session_id)
        state.queue_depth = depth
        state.queued_messages = tuple(
            summarize_user_input_for_display(message, self._root)
            for message in pending
        )
        live = await self._runtime.subagent_registry.snapshot()
        live = [entry for entry in live if entry.parent_session_id == session_id]
        live_sessions = frozenset(entry.session_id for entry in live)
        state.running_agents = tuple(
            f"{entry.agent_name} {entry.session_id[:8]}: "
            f"{preview_inline(entry.task_text, limit=80)}"
            for entry in live
        )
        tasks = await self._runtime.task_store.list_tasks(
            lead_session_id=session_id,
            limit=50,
        )
        state.task_rows = tuple(
            format_task_row(task, live_sessions=live_sessions) for task in tasks
        )
        if self._is_active(state):
            self._update_header()
            self._update_work()

    def _record_interrupted(self, state: PerLeadState | None = None) -> None:
        state = state if state is not None else self._active()
        self._finish_assistant_turn(state)
        self._write_marker(
            "Interrupted",
            "Esc pressed; active run cancelled.",
            style="yellow",
            state=state,
        )
        state.status = "idle"
        if self._is_active(state):
            self._update_header()

    def _record_task_update(self, update: str, *, state: PerLeadState | None = None) -> None:
        state = state if state is not None else self._active()
        update = update.strip()
        if not update:
            return
        state.task_updates.append(update)
        del state.task_updates[:-5]
        if self._is_active(state):
            self._update_work()

    def _write_tool_started(
        self, event: RuntimeEvent, *, state: PerLeadState | None = None
    ) -> None:
        state = state if state is not None else self._active()
        title = _tool_started_title(event.tool_name)
        detail = _format_tool_payload(event.tool_name, event.payload or {})
        body = title if not detail else f"{title}\n{detail}"
        self._write_conversation(
            state.agent.config.name,
            body,
            label_style="bold green",
            body_style="grey70",
            state=state,
        )

    def _write_tool_finished(
        self,
        event: RuntimeEvent,
        *,
        failed_title: str | None,
        state: PerLeadState | None = None,
    ) -> None:
        state = state if state is not None else self._active()
        if failed_title:
            body = f"{failed_title} - Error"
            detail = _format_tool_error(event.tool_name, event.text)
            if detail:
                body = f"{body}\n{detail}"
            self._write_conversation(
                state.agent.config.name,
                body,
                label_style="bold green",
                body_style="red",
                state=state,
            )
            return

        title = _tool_finished_title(event.tool_name)
        detail = _format_tool_result(event.tool_name, event.text)
        if event.tool_name in _TASK_TOOL_NAMES:
            body = title
            body_style = "green"
        else:
            body = title
            body_style = "grey70"
        if detail:
            body = f"{body}\n{detail}"
        self._write_conversation(
            state.agent.config.name,
            body,
            label_style="bold green",
            body_style=body_style,
            state=state,
        )

    def _write_marker(
        self,
        title: str,
        text: str = "",
        *,
        style: str = "grey70",
        state: PerLeadState | None = None,
    ) -> None:
        marker = title if not text else f"{title} · {text}"
        self._write_conversation(
            "Feather",
            marker,
            label_style=f"bold {style}",
            body_style=style,
            state=state,
        )

    def _write_conversation(
        self,
        title: str,
        body: str,
        *,
        label_style: str,
        body_style: str = "bold white",
        state: PerLeadState | None = None,
    ) -> None:
        state = state if state is not None else self._active()
        state.conversation_blocks.append(
            _ConversationBlock(
                title=title,
                body=body,
                label_style=label_style,
                body_style=body_style,
            )
        )
        state.transcript_blocks.append(format_transcript_block(title, body))
        if self._is_active(state):
            self._render_conversation()

    def _tick_spinner(self) -> None:
        """Advance the streaming spinner frame and re-render if active.

        Cheap when nothing is streaming — the early-out skips the
        ``_render_conversation`` repaint, so an idle session pays only
        the cost of one attribute read per 100 ms tick. During an active
        streaming window the conversation repaints at the spinner cadence
        so the glyph cycles even when text deltas pause (e.g., the model
        is mid-reasoning between visible-output chunks).
        """

        if not self._assistant_parts:
            return
        self._spinner_frame = (self._spinner_frame + 1) % len(_SPINNER_FRAMES)
        self._render_conversation()

    def _render_conversation(self) -> None:
        log = self.query_one("#conversation", RichLog)
        log.clear()
        for block in self._conversation_blocks:
            self._write_conversation_block(log, block)
        if self._assistant_parts:
            agent_name = self._agent.config.name if self._agent is not None else "Lead"
            spinner = _SPINNER_FRAMES[self._spinner_frame % len(_SPINNER_FRAMES)]
            self._write_conversation_block(
                log,
                _ConversationBlock(
                    title=f"{spinner} {agent_name} streaming",
                    body="".join(self._assistant_parts),
                    label_style="bold green",
                    body_style="bold white",
                ),
            )

    def _write_conversation_block(
        self,
        log: RichLog,
        block: _ConversationBlock,
    ) -> None:
        log.write(Text(block.title, style=block.label_style), scroll_end=True)
        if block.body:
            log.write(
                Text(_indent_lines(block.body), style=block.body_style),
                scroll_end=True,
            )
        log.write("", scroll_end=True)

    def _update_header(self) -> None:
        header = self.query_one("#header", Static)
        session_id = self._active_session_id or "(starting)"
        agent_name = self._agent.config.name if self._agent is not None else "Lead"
        strip = self._build_lead_strip()
        header.styles.height = 4 if strip is not None else 3
        header.update(
            build_header_text(
                agent_name=agent_name,
                status=self._status,
                context_ratio=self._latest_usage_ratio,
                queue_depth=self._queue_depth,
                active_tool=self._active_tool,
                session_id=session_id,
                lead_strip=strip,
            )
        )

    def _build_lead_strip(self) -> Text | None:
        """Lead strip when more than one lead is active, else ``None``."""

        if len(self._leads) <= 1 or self._active_lead_name is None:
            return None
        leads = tuple(
            (s.display_name, s.emoji, s.status)
            for _, s in sorted(self._leads.items())
        )
        active = self._leads[self._active_lead_name].display_name
        return build_lead_strip(leads, active)

    def _update_work(self) -> None:
        work = self.query_one("#work_content", Static)
        work.update(
            build_work_text(
                queue_depth=self._queue_depth,
                queued_messages=self._queued_messages,
                running_agents=self._running_agents,
                task_rows=self._task_rows,
                task_updates=tuple(self._task_updates),
            )
        )




async def run_textual_tui(root: Path, session_id: str | None) -> None:
    """Run the Textual TUI app and force-exit when it returns.

    Why the explicit ``os._exit``: when the user hits Esc during a turn,
    the active asyncio Task awaiting :func:`asyncio.to_thread` (e.g.,
    inside ``parallel_client.search``) is cancelled, but the underlying
    worker thread keeps running the sync HTTP call until it completes.
    Cancellation cannot be propagated into a running sync function from
    outside the thread.

    Python 3.12's :func:`asyncio.run` then enters ``Runner.close`` which
    calls ``loop.shutdown_default_executor(timeout=THREAD_JOIN_TIMEOUT)``.
    That constant is **300 seconds** — so a user-initiated ``/exit``
    after Esc'ing a slow tool call can stall for up to five minutes
    waiting for an orphaned HTTP request that nobody is reading anymore.
    The user-observed symptom is exactly this: ``KeyboardInterrupt``
    inside ``Runner.close`` followed by a second ``KeyboardInterrupt``
    inside ``threading._shutdown`` joining the same orphan thread.

    By the time ``await app.run_async`` returns here, ``on_unmount`` has
    already completed — every DB store, HTTP client, subagent process,
    and background task we own has been closed/cancelled. The orphan
    default-executor threads carry no work we still care about.
    ``os._exit`` skips the asyncio cleanup AND the atexit thread joins,
    terminating the process immediately.

    Stdout/stderr/logging are flushed first so no diagnostic output is
    lost. Map exit codes to standard shell conventions: 0 on clean exit,
    130 on Ctrl+C, 1 on any other fatal error.
    """

    app = FeatherTextualApp(root=root, session_id=session_id)
    exit_code = 0
    try:
        await app.run_async(mouse=_mouse_enabled())
    except KeyboardInterrupt:
        exit_code = 130
    except BaseException:  # noqa: BLE001
        logger.exception("textual_tui.fatal_error")
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        logging.shutdown()
        os._exit(exit_code)


def _mouse_enabled() -> bool:
    """Return whether Textual should request terminal mouse reporting."""

    value = os.environ.get("FEATHER_TUI_MOUSE")
    if value is None or not value.strip():
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _parse_telegram_args(args: str) -> dict[str, Any]:
    """Parse ``/telegram connect <bot_token>`` and return the config blob."""

    parts = args.split()
    if len(parts) != 1:
        raise ValueError(
            "expected exactly one argument: bot_token. "
            "Try '/telegram help'."
        )
    return {"bot_token": parts[0]}


def _parse_line_args(args: str) -> dict[str, Any]:
    """Parse ``/line connect <secret> <token>``."""

    parts = args.split()
    if len(parts) != 2:
        raise ValueError(
            "expected two arguments: channel_secret channel_token. "
            "Try '/line help'."
        )
    return {"channel_secret": parts[0], "channel_token": parts[1]}


def _parse_whatsapp_args(args: str) -> dict[str, Any]:
    """Parse ``/whatsapp connect <phone_id> <token> <verify> <app_secret>``."""

    parts = args.split()
    if len(parts) != 4:
        raise ValueError(
            "expected four arguments: "
            "phone_number_id access_token verify_token app_secret. "
            "Try '/whatsapp help'."
        )
    return {
        "phone_number_id": parts[0],
        "access_token": parts[1],
        "verify_token": parts[2],
        "app_secret": parts[3],
    }


# ---- Qdrant context helpers (used by /qdrant subcommands) ----------------

_QDRANT_DEFAULT_URL = "http://localhost:6333"
_QDRANT_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1", ""})


def _resolve_qdrant_url() -> tuple[str, str]:
    """Return ``(url, source)`` for the current Qdrant endpoint.

    ``source`` is one of ``"env"`` (QDRANT_URL was set), ``"default"``
    (no env, fallback to localhost). The TUI uses this so ``/qdrant
    status`` can show the user *why* it is probing a particular URL.
    """

    env = (os.environ.get("QDRANT_URL", "") or "").strip()
    if env:
        return env, "env"
    return _QDRANT_DEFAULT_URL, "default"


def _is_compose_managed_qdrant() -> bool:
    """Detect when Qdrant is owned by an external orchestrator.

    Heuristics (any one is sufficient):

    - ``/.dockerenv`` exists — we are inside a Docker container, so
      managing Docker from here makes no sense without a mounted socket.
    - ``QDRANT_URL`` points at a hostname other than localhost — the
      user explicitly told us to talk to a remote Qdrant; lifecycle
      commands shouldn't touch a local container.
    """

    if Path("/.dockerenv").exists():
        return True
    url, source = _resolve_qdrant_url()
    if source != "env":
        return False
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return host not in _QDRANT_LOCAL_HOSTNAMES


async def _probe_qdrant_readyz(url: str, *, timeout_s: float = 3.0) -> bool:
    """Return True when ``<url>/readyz`` responds 2xx within ``timeout_s``."""

    base = url.rstrip("/")
    target = f"{base}/readyz"
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(target)
        return 200 <= response.status_code < 300
    except Exception:  # noqa: BLE001
        return False
