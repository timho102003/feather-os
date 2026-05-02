"""Textual-powered terminal UI for Feather sessions."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.geometry import Region
from textual.widgets import RichLog, Static, TextArea

from feather.attachments import parse_attachment_drops, render_attachment_message
from feather.models import AgentOutcome, RuntimeEvent, TaskRecord, TaskStatus
from feather.runtime import FeatherRuntime
from feather.slash_commands import (
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

logger = logging.getLogger(__name__)


_SlashHandler = Callable[[str], None]


@dataclass(slots=True, frozen=True)
class _ConversationBlock:
    """One styled conversation block rendered by the Textual transcript."""

    title: str
    body: str
    label_style: str
    body_style: str


def render_dropdown_text(
    matches: tuple[SlashCommand, ...],
    *,
    selected_index: int,
    first_visible: int = 0,
    max_visible: int | None = None,
) -> Text:
    """Render the slash-command dropdown body as a Rich :class:`Text`.

    Args:
        matches: Full ranked match set.
        selected_index: Absolute index of the currently highlighted command.
        first_visible: Absolute index of the topmost rendered row. Use 0
            to disable viewport scrolling.
        max_visible: Maximum number of command rows to render at once. A
            value of ``None`` renders every match.

    Returns:
        Styled text suitable for a :class:`Static` widget. When the
        viewport hides entries above or below the visible window, a
        ``↑ N more above`` / ``↓ N more below`` hint is inserted so the
        user knows there is more to scroll to.
    """

    if not matches:
        empty = Text()
        empty.append("No commands match", style="italic dim")
        empty.append("\n  ", style="dim")
        empty.append("Press Esc to dismiss or keep typing.", style="dim")
        return empty

    total = len(matches)
    if max_visible is None or max_visible >= total:
        window_start = 0
        window_end = total
    else:
        window_start = max(0, min(first_visible, total - max_visible))
        window_end = window_start + max_visible

    body = Text()
    hidden_above = window_start
    hidden_below = total - window_end
    if hidden_above > 0:
        body.append(f"↑ {hidden_above} more above\n", style="dim")

    visible_matches = matches[window_start:window_end]
    for offset, cmd in enumerate(visible_matches):
        absolute_index = window_start + offset
        is_selected = absolute_index == selected_index
        marker = "› " if is_selected else "  "
        marker_style = "bold cyan" if is_selected else "dim"
        name_style = "bold cyan" if is_selected else "white"
        summary_style = "white" if is_selected else "grey70"
        body.append(marker, style=marker_style)
        body.append(cmd.display, style=name_style)
        body.append("  ", style="dim")
        body.append(cmd.summary, style=summary_style)
        if cmd.aliases:
            body.append("  ", style="dim")
            body.append(
                "(" + ", ".join(f"/{a}" for a in cmd.aliases) + ")",
                style="dim",
            )
        if offset < len(visible_matches) - 1 or hidden_below > 0:
            body.append("\n")

    if hidden_below > 0:
        body.append(f"↓ {hidden_below} more below", style="dim")
    return body


def render_help_text(registry: SlashCommandRegistry) -> Text:
    """Render the body shown when the user invokes ``/help``.

    Commands are grouped by category — each category header appears
    exactly once, even when the registry interleaves categories.
    """

    grouped: dict[str, list[SlashCommand]] = {}
    category_order: list[str] = []
    for cmd in registry.all():
        if cmd.category not in grouped:
            grouped[cmd.category] = []
            category_order.append(cmd.category)
        grouped[cmd.category].append(cmd)

    body = Text()
    body.append("Slash commands\n", style="bold white")
    for index, category in enumerate(category_order):
        if index > 0:
            body.append("\n")
        body.append(f"[{category}]\n", style="bold grey50")
        for cmd in grouped[category]:
            body.append(f"  {cmd.display}", style="bold cyan")
            if cmd.aliases:
                body.append(
                    "  (" + ", ".join(f"/{a}" for a in cmd.aliases) + ")",
                    style="dim",
                )
            body.append(f"  {cmd.summary}\n", style="white")
    return body


class SlashCommandDropdown(Static):
    """Floating panel that lists matching slash commands above the composer.

    The widget is purely a view; the owning app drives state via
    :meth:`update_for_text` (called on each composer change) and reads
    :attr:`selected_command` / :meth:`completion_text` to apply user
    selections back to the composer.

    The visible area is capped by :attr:`max_visible_rows`. When the
    match list is taller than the window, the highlight is kept in view
    by sliding ``first_visible_index`` so the user never appears to
    "lose" the cursor at the bottom of the list.
    """

    DEFAULT_CSS = """
    SlashCommandDropdown {
        height: auto;
        max-height: 10;
        border: solid #4f7fff;
        background: #11151c;
        color: white;
        padding: 0 1;
        display: none;
    }

    SlashCommandDropdown.-open {
        display: block;
    }
    """

    # ``max-height: 10`` minus the 2-row border leaves 8 content rows.
    # Reserve up to 2 rows for ``↑/↓ N more`` hints so 6 commands always
    # render fully even at the largest hidden-window size.
    _DEFAULT_MAX_VISIBLE = 6

    def __init__(
        self,
        registry: SlashCommandRegistry,
        *,
        id: str | None = None,
        max_visible: int | None = None,
    ) -> None:
        """Initialise the dropdown.

        Args:
            registry: Source of slash commands to filter and display.
            id: Optional Textual widget id.
            max_visible: Maximum number of command rows shown at once.
                Defaults to :attr:`_DEFAULT_MAX_VISIBLE`. Tests may
                override to exercise viewport behaviour with smaller
                windows.
        """

        super().__init__(id=id)
        self._registry = registry
        self._matches: tuple[SlashCommand, ...] = ()
        self._selected: int = 0
        self._first_visible: int = 0
        self._max_visible: int = (
            max_visible if max_visible is not None else self._DEFAULT_MAX_VISIBLE
        )
        self._open: bool = False

    @property
    def max_visible_rows(self) -> int:
        """Return the cap on simultaneously rendered command rows."""

        return self._max_visible

    @property
    def first_visible_index(self) -> int:
        """Return the absolute index of the topmost rendered command."""

        return self._first_visible

    @property
    def is_open(self) -> bool:
        """Return True when the dropdown is currently visible."""

        return self._open

    @property
    def matches(self) -> tuple[SlashCommand, ...]:
        """Return the currently displayed matches."""

        return self._matches

    @property
    def selected_command(self) -> SlashCommand | None:
        """Return the currently highlighted command, if any."""

        if not self._open or not self._matches:
            return None
        if not 0 <= self._selected < len(self._matches):
            return None
        return self._matches[self._selected]

    def completion_text(self) -> str | None:
        """Return the text the composer should be replaced with on Tab.

        A trailing space is included so the user can type arguments
        without having to add a separator.
        """

        cmd = self.selected_command
        if cmd is None:
            return None
        return f"{cmd.display} "

    def update_for_text(self, text: str) -> None:
        """Recompute matches based on the current composer text.

        Args:
            text: Raw composer text. Anything not starting with ``/``
                (after stripping leading whitespace) closes the dropdown.
                Anything past the first whitespace inside the command
                also closes the dropdown — the user has moved on to
                typing arguments.

        Selection is preserved across updates that produce the same set
        of matches (in the same order), so refining a partial match by
        retyping does not reset the highlight to the top.
        """

        parsed = parse_slash_input(text)
        if parsed is None:
            self.close()
            return
        if parsed.has_args:
            self.close()
            return
        matches = self._registry.match(parsed.name_token)
        previous_names = tuple(c.name for c in self._matches)
        new_names = tuple(c.name for c in matches)
        if new_names == previous_names and self._open and matches:
            # Same matches, in the same order — keep the user's highlight
            # and viewport position so refining a partial match by
            # retyping does not jump.
            self._matches = matches
        else:
            self._matches = matches
            self._selected = 0
            self._first_visible = 0
        self._open = True
        self._clamp_viewport()
        self._apply_open_state()
        self._redraw()

    def move_selection(self, delta: int) -> None:
        """Cycle the highlighted entry by ``delta`` positions.

        Wraps around at both ends, and slides the viewport so the new
        selection is always rendered. No-op when the dropdown is closed
        or empty.
        """

        if not self._matches:
            return
        count = len(self._matches)
        self._selected = (self._selected + delta) % count
        self._clamp_viewport()
        self._redraw()

    def _clamp_viewport(self) -> None:
        """Slide ``_first_visible`` so the selection is always rendered."""

        count = len(self._matches)
        if count == 0:
            self._first_visible = 0
            return
        window = self._max_visible
        if window <= 0 or window >= count:
            self._first_visible = 0
            return
        if self._selected < self._first_visible:
            self._first_visible = self._selected
        elif self._selected >= self._first_visible + window:
            self._first_visible = self._selected - window + 1
        # Cap so we never scroll past the end.
        max_first = max(0, count - window)
        self._first_visible = max(0, min(self._first_visible, max_first))

    def close(self) -> None:
        """Hide the dropdown and clear its match list."""

        if not self._open and not self._matches:
            return
        self._open = False
        self._matches = ()
        self._selected = 0
        self._first_visible = 0
        self._apply_open_state()
        try:
            self.update("")
        except Exception:  # noqa: BLE001
            # Widget may not be mounted yet during unit tests; ignore.
            pass

    def _apply_open_state(self) -> None:
        """Reflect ``self._open`` onto the widget's CSS class state."""

        try:
            if self._open:
                self.add_class("-open")
            else:
                self.remove_class("-open")
        except Exception:  # noqa: BLE001
            # In tests the widget may be instantiated outside an App.
            pass

    def _redraw(self) -> None:
        try:
            self.update(
                render_dropdown_text(
                    self._matches,
                    selected_index=self._selected,
                    first_visible=self._first_visible,
                    max_visible=self._max_visible,
                )
            )
        except Exception:  # noqa: BLE001
            pass


class ComposerTextArea(TextArea):
    """Text input that keeps mouse wheel movement out of the edit cursor."""

    async def _on_paste(self, event: events.Paste) -> None:
        """Normalize terminal file drops before inserting pasted text."""

        root = getattr(self.app, "_root", None)
        if isinstance(root, Path):
            event.text = normalize_pasted_attachment_text(event.text, root)
        await super()._on_paste(event)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Route wheel-up over the composer to the conversation panel."""

        event.prevent_default()
        event.stop()
        self.app.query_one("#conversation", RichLog).scroll_up(animate=False)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        """Route wheel-down over the composer to the conversation panel."""

        event.prevent_default()
        event.stop()
        self.app.query_one("#conversation", RichLog).scroll_down(animate=False)

    async def _on_key(self, event: events.Key) -> None:
        """Intercept arrow/tab keys when the slash dropdown is open."""

        dropdown = getattr(self.app, "_slash_dropdown_widget", None)
        if dropdown is not None and dropdown.is_open:
            if event.key == "up":
                dropdown.move_selection(-1)
                event.prevent_default()
                event.stop()
                return
            if event.key == "down":
                dropdown.move_selection(1)
                event.prevent_default()
                event.stop()
                return
            if event.key == "tab":
                completion = dropdown.completion_text()
                if completion is not None:
                    # Programmatic ``self.text =`` posts a TextArea.Changed
                    # message that is delivered after this handler returns,
                    # which would re-open the dropdown a frame later
                    # (review fix C1). Tag the app to suppress the next
                    # change-driven reopen.
                    setattr(self.app, "_slash_suppress_next_change", True)
                    self.text = completion
                    self.move_cursor((0, len(completion)))
                dropdown.close()
                event.prevent_default()
                event.stop()
                return
        await super()._on_key(event)


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
    ]

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
        self._agent: Any = None
        self._active_session_id: str | None = None
        self._assistant_parts: list[str] = []
        self._latest_usage_ratio: float | None = None
        self._queue_depth = 0
        self._queued_messages: tuple[str, ...] = ()
        self._running_agents: tuple[str, ...] = ()
        self._task_rows: tuple[str, ...] = ()
        self._task_updates: list[str] = []
        self._conversation_blocks: list[_ConversationBlock] = []
        self._transcript_blocks: list[str] = []
        self._active_tool: str | None = None
        self._status = "idle"
        self._stop_event = asyncio.Event()
        self._pending_answer: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self._new_run_queue: asyncio.Queue[str] = asyncio.Queue()
        self._busy_event = asyncio.Event()
        self._awaiting_event = asyncio.Event()
        self._run_task: asyncio.Task[Any] | None = None
        self._driver_task: asyncio.Task[None] | None = None
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

    async def on_mount(self) -> None:
        """Initialize Feather runtime services once Textual is mounted."""

        self._runtime = await FeatherRuntime.create(self._root)
        self._agent = self._runtime.build_agent("lead")
        self._active_session_id = (
            self._requested_session_id or await self._agent.create_session()
        )
        self._runtime.set_session_event_handler(
            self._active_session_id,
            self._handle_runtime_event,
        )
        await self._runtime.start_background_services()
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
        self._driver_task = asyncio.create_task(self._agent_driver())
        self._watcher_task = asyncio.create_task(self._inbox_watcher())
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def on_unmount(self) -> None:
        """Stop runtime services and background tasks."""

        self._stop_event.set()
        for task in (
            self._run_task,
            self._driver_task,
            self._watcher_task,
            self._monitor_task,
        ):
            if task is not None and not task.done():
                task.cancel()
        for task in (
            self._run_task,
            self._driver_task,
            self._watcher_task,
            self._monitor_task,
        ):
            if task is None:
                continue
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._runtime is not None:
            if self._active_session_id is not None:
                self._runtime.set_session_event_handler(self._active_session_id, None)
            await self._runtime.close()

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
            "copy": self._cmd_copy,
            "queue": self._cmd_queue,
            "agents": self._cmd_agents,
            "tasks": self._cmd_tasks,
            "session": self._cmd_session,
            "skills": self._cmd_skills,
            "integrations": self._cmd_integrations,
            "telegram": self._cmd_telegram,
            "line": self._cmd_line,
            "whatsapp": self._cmd_whatsapp,
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
            "telegram",
            "line",
            "whatsapp",
            "qdrant",
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

    def _cmd_clear(self, args: str) -> None:
        """Clear the on-screen transcript without touching session state."""

        del args
        self._conversation_blocks = []
        self._transcript_blocks = []
        self._assistant_parts = []
        self._render_conversation()
        self._write_marker("Cleared", "transcript cleared (session history kept)")

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

    def _handle_runtime_event(self, event: RuntimeEvent) -> None:
        if event.kind == "assistant_text_delta":
            self._status = "running"
            self._assistant_parts.append(event.text or "")
            self._update_header()
            self._render_conversation()
            return
        if event.kind == "usage_updated":
            ratio = (event.payload or {}).get("usage_ratio")
            if isinstance(ratio, (int, float)):
                self._latest_usage_ratio = max(0.0, min(1.0, float(ratio)))
                self._update_header()
            return

        self._finish_assistant_turn()
        if event.kind == "tool_started":
            self._status = "running"
            self._active_tool = event.tool_name
            self._update_header()
            if event.tool_name != "ask_user":
                self._write_tool_started(event)
        elif event.kind == "tool_finished":
            self._active_tool = None
            self._update_header()
            if event.tool_name == "ask_user":
                return
            failed_title = _failed_tool_title(event.tool_name, event.text)
            if failed_title:
                self._write_tool_finished(event, failed_title=failed_title)
            else:
                self._write_tool_finished(event, failed_title=None)
                if event.tool_name == "spawn_agent":
                    self._record_task_update(
                        f"started {_format_tool_result(event.tool_name, event.text)}"
                    )
        elif event.kind == "awaiting_user":
            self._status = "awaiting user"
            self._write_conversation(
                f"{self._agent.config.name} asks",
                event.text or "",
                label_style="bold green",
            )
            self._update_header()
        elif event.kind == "user_message_injected":
            self._write_conversation(
                "Queued input",
                event.text or "",
                label_style="bold grey70",
                body_style="grey70",
            )
        elif event.kind == "agent_message_received":
            self._record_task_update(summarize_agent_message_update(event))
            self._write_conversation(
                "Sub-agent completed",
                format_agent_message_event(event),
                label_style="bold magenta",
            )
        elif event.kind.startswith("compaction_"):
            self._write_marker(
                _event_title(event.kind),
                event.text or "",
                style=_system_event_style(event.kind),
            )
        elif event.kind.startswith("scheduled_task_"):
            self._write_marker(
                _event_title(event.kind),
                event.text or "",
                style=_system_event_style(event.kind),
            )

    def _finish_assistant_turn(self) -> None:
        if not self._assistant_parts:
            return
        body = "".join(self._assistant_parts)
        self._assistant_parts.clear()
        self._write_conversation(
            self._agent.config.name,
            body,
            label_style="bold green",
        )

    async def _agent_driver(self) -> None:
        assert self._active_session_id is not None
        assert self._agent is not None
        while not self._stop_event.is_set():
            get_task = asyncio.create_task(self._new_run_queue.get())
            stop_task = asyncio.create_task(self._stop_event.wait())
            await asyncio.wait(
                {get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if self._stop_event.is_set():
                get_task.cancel()
                return
            stop_task.cancel()
            user_message = get_task.result()

            self._busy_event.set()
            self._status = "running"
            self._update_header()
            try:
                if user_message == _INBOX_WAKE:
                    self._run_task = asyncio.create_task(
                        self._agent.resume_on_inbox(
                            self._active_session_id,
                            self._handle_runtime_event,
                        )
                    )
                else:
                    self._run_task = asyncio.create_task(
                        self._agent.run(
                            self._active_session_id,
                            user_message,
                            self._handle_runtime_event,
                        )
                    )
                try:
                    result = await self._run_task
                except asyncio.CancelledError:
                    if self._stop_event.is_set():
                        return
                    self._record_interrupted()
                    continue
                finally:
                    self._run_task = None
                if result is None:
                    self._status = "idle"
                    self._update_header()
                    continue

                self._finish_assistant_turn()
                await self._refresh_monitor()
                if result.status == AgentOutcome.COMPLETED:
                    self._status = "idle"
                self._update_header()
                self._update_work()

                while (
                    result.status == AgentOutcome.AWAITING_USER
                    and result.question is not None
                    and not self._stop_event.is_set()
                ):
                    self._awaiting_event.set()
                    answer_task = asyncio.create_task(self._pending_answer.get())
                    stop_task = asyncio.create_task(self._stop_event.wait())
                    await asyncio.wait(
                        {answer_task, stop_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if self._stop_event.is_set():
                        answer_task.cancel()
                        self._awaiting_event.clear()
                        return
                    stop_task.cancel()
                    answer = answer_task.result()
                    self._run_task = asyncio.create_task(
                        self._agent.run(
                            self._active_session_id,
                            answer,
                            self._handle_runtime_event,
                        )
                    )
                    try:
                        result = await self._run_task
                    except asyncio.CancelledError:
                        if self._stop_event.is_set():
                            return
                        self._record_interrupted()
                        break
                    finally:
                        self._run_task = None
                    self._finish_assistant_turn()
                    await self._refresh_monitor()
                    if result.status == AgentOutcome.COMPLETED:
                        self._status = "idle"
                    self._update_header()
                    self._update_work()
            except Exception as exc:  # noqa: BLE001
                logger.exception("textual_tui.agent_driver_crashed")
                self._write_marker("Agent error", f"{type(exc).__name__}: {exc}", style="red")
                self._status = "idle"
                self._update_header()
            finally:
                self._busy_event.clear()
                self._awaiting_event.clear()

    async def _inbox_watcher(self) -> None:
        assert self._active_session_id is not None
        assert self._agent is not None
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=0.5)
                return
            except asyncio.TimeoutError:
                pass
            if self._busy_event.is_set() or self._awaiting_event.is_set():
                continue
            try:
                has_pending = await self._agent.has_pending_inbox(self._active_session_id)
            except Exception:  # noqa: BLE001
                continue
            if has_pending and self._new_run_queue.empty():
                await self._new_run_queue.put(_INBOX_WAKE)

    async def _monitor_loop(self) -> None:
        """Refresh durable task and queue state even when no events arrive."""

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                return
            except asyncio.TimeoutError:
                pass
            try:
                await self._refresh_monitor()
            except Exception:  # noqa: BLE001
                logger.exception("textual_tui.monitor_refresh_failed")

    async def _refresh_monitor(self) -> None:
        assert self._runtime is not None
        assert self._active_session_id is not None
        depth = await self._runtime.input_queue.depth(self._active_session_id)
        pending = await self._runtime.input_queue.peek(self._active_session_id)
        self._queue_depth = depth
        self._queued_messages = tuple(
            summarize_user_input_for_display(message, self._root)
            for message in pending
        )
        live = await self._runtime.subagent_registry.snapshot()
        live = [
            entry for entry in live if entry.parent_session_id == self._active_session_id
        ]
        live_sessions = frozenset(entry.session_id for entry in live)
        self._running_agents = tuple(
            f"{entry.agent_name} {entry.session_id[:8]}: "
            f"{preview_inline(entry.task_text, limit=80)}"
            for entry in live
        )
        tasks = await self._runtime.task_store.list_tasks(
            lead_session_id=self._active_session_id,
            limit=50,
        )
        self._task_rows = tuple(
            format_task_row(task, live_sessions=live_sessions) for task in tasks
        )
        self._update_header()
        self._update_work()

    def _record_interrupted(self) -> None:
        self._finish_assistant_turn()
        self._write_marker(
            "Interrupted",
            "Esc pressed; active run cancelled.",
            style="yellow",
        )
        self._status = "idle"
        self._update_header()

    def _record_task_update(self, update: str) -> None:
        update = update.strip()
        if not update:
            return
        self._task_updates.append(update)
        self._task_updates = self._task_updates[-5:]
        self._update_work()

    def _write_tool_started(self, event: RuntimeEvent) -> None:
        title = _tool_started_title(event.tool_name)
        detail = _format_tool_payload(event.tool_name, event.payload or {})
        body = title if not detail else f"{title}\n{detail}"
        self._write_conversation(
            self._agent.config.name,
            body,
            label_style="bold green",
            body_style="grey70",
        )

    def _write_tool_finished(
        self,
        event: RuntimeEvent,
        *,
        failed_title: str | None,
    ) -> None:
        if failed_title:
            body = f"{failed_title} - Error"
            detail = _format_tool_error(event.tool_name, event.text)
            if detail:
                body = f"{body}\n{detail}"
            self._write_conversation(
                self._agent.config.name,
                body,
                label_style="bold green",
                body_style="red",
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
            self._agent.config.name,
            body,
            label_style="bold green",
            body_style=body_style,
        )

    def _write_marker(self, title: str, text: str = "", *, style: str = "grey70") -> None:
        marker = title if not text else f"{title} · {text}"
        self._write_conversation(
            "Feather",
            marker,
            label_style=f"bold {style}",
            body_style=style,
        )

    def _write_conversation(
        self,
        title: str,
        body: str,
        *,
        label_style: str,
        body_style: str = "bold white",
    ) -> None:
        self._conversation_blocks.append(
            _ConversationBlock(
                title=title,
                body=body,
                label_style=label_style,
                body_style=body_style,
            )
        )
        self._transcript_blocks.append(format_transcript_block(title, body))
        self._render_conversation()

    def _render_conversation(self) -> None:
        log = self.query_one("#conversation", RichLog)
        log.clear()
        for block in self._conversation_blocks:
            self._write_conversation_block(log, block)
        if self._assistant_parts:
            agent_name = self._agent.config.name if self._agent is not None else "Lead"
            self._write_conversation_block(
                log,
                _ConversationBlock(
                    title=f"{agent_name} streaming",
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
        ctx = "ctx --"
        if self._latest_usage_ratio is not None:
            ctx = f"ctx {round(self._latest_usage_ratio * 100)}%"
        active = f"active {self._active_tool}" if self._active_tool else "no active tool"
        session_id = self._active_session_id or "(starting)"
        agent_name = self._agent.config.name if self._agent is not None else "Lead"
        header.update(
            build_header_text(
                agent_name=agent_name,
                status=self._status,
                context_ratio=self._latest_usage_ratio,
                queue_depth=self._queue_depth,
                active_tool=self._active_tool,
                session_id=session_id,
            )
        )

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


def build_header_text(
    *,
    agent_name: str,
    status: str,
    context_ratio: float | None,
    queue_depth: int,
    active_tool: str | None,
    session_id: str,
) -> Text:
    """Build the Textual header renderable."""

    ctx = "ctx --" if context_ratio is None else f"ctx {round(context_ratio * 100)}%"
    active = f"active {active_tool}" if active_tool else "no active tool"
    return Text.assemble(
        ("Feather", "bold white"),
        "  ",
        (agent_name, "white"),
        "  ",
        (status, _status_style(status)),
        "  ",
        (ctx, "white"),
        "  ",
        (f"queued {queue_depth}", "magenta" if queue_depth else "dim"),
        "  ",
        (active, "grey70" if active_tool else "dim"),
        "\n",
        (f"lead session {session_id}", "dim"),
    )


def build_work_text(
    *,
    queue_depth: int,
    queued_messages: tuple[str, ...],
    running_agents: tuple[str, ...],
    task_rows: tuple[str, ...] = (),
    task_updates: tuple[str, ...] = (),
) -> Text:
    """Build the queued/future-work renderable."""

    rows = Text()
    has_rows = False
    if running_agents:
        has_rows = True
        rows.append("Live sub-agents\n", style="bold white")
        for agent in running_agents:
            rows.append(_indent_lines(agent), style="white")
    if queue_depth:
        if has_rows:
            rows.append("\n")
        has_rows = True
        rows.append("Queued queries\n", style="bold white")
        for index, message in enumerate(queued_messages, 1):
            rows.append(_indent_lines(f"{index}. {message}"), style="white")
    if task_rows:
        if has_rows:
            rows.append("\n")
        has_rows = True
        rows.append("Tasks\n", style="bold white")
        for task in task_rows:
            rows.append(_indent_lines(task), style="white")
    if task_updates:
        if has_rows:
            rows.append("\n")
        has_rows = True
        rows.append("Recent task updates\n", style="bold white")
        for update in task_updates[-3:]:
            rows.append(_indent_lines(update), style="white")
    if not has_rows:
        rows.append(
            "No queued queries, tracked tasks, live sub-agents, or recent task updates.",
            style="dim",
        )
    return rows


def format_task_row(
    task: TaskRecord,
    *,
    live_sessions: frozenset[str] = frozenset(),
) -> str:
    """Render one task row for the Textual monitor."""

    label = {
        "queued": "QUEUE",
        "running": "RUN",
        "blocked_needs_input": "BLOCK",
        "completed_with_report": "DONE",
        "completed_with_artifacts": "DONE",
        "completed_without_artifacts": "DONE",
        "failed": "FAIL",
        "stopped": "STOP",
    }.get(task.status.value, task.status.value.upper())
    if (
        task.status == TaskStatus.RUNNING
        and task.responsible_session_id in live_sessions
    ):
        label = "LIVE"
    agent = task.responsible_agent_name or "-"
    session = (
        task.responsible_session_id[:8]
        if task.responsible_session_id is not None
        else "-"
    )
    detail = task.blocked_question or task.error or task.title
    return (
        f"{label:<5} {agent:<10} {session:<8} "
        f"{preview_inline(detail, limit=92)}"
    )


def format_transcript_block(title: str, body: str) -> str:
    """Render one conversation block as plain text for copying."""

    body = body.strip()
    if not body:
        return title.strip()
    return f"{title.strip()}\n{_indent_lines(body).rstrip()}"


def build_transcript_text(blocks: tuple[str, ...]) -> str:
    """Build the plain-text transcript copied from the TUI."""

    return "\n\n".join(block.strip() for block in blocks if block.strip())


def summarize_user_input_for_display(text: str, root: Path) -> str:
    """Render dropped local paths as compact attachment placeholders."""

    draft = parse_attachment_drops(text, root=root)
    if not draft.attachments:
        return text.strip()
    effective_text = draft.text or "Please review the attached file(s)."
    return render_attachment_message(effective_text, draft.attachments)


def normalize_pasted_attachment_text(text: str, root: Path) -> str:
    """Convert pasted external file paths into explicit file URIs.

    Terminal drag/drop usually arrives as a bracketed paste containing one or
    more absolute file paths. Core agent parsing still ignores plain
    out-of-workspace absolute paths; this TUI-only conversion marks a paste/drop
    as explicit without changing normal typed path mentions.
    """

    stripped = text.strip()
    if not stripped:
        return text
    try:
        tokens = shlex.split(stripped, posix=True)
    except ValueError:
        return text
    if not tokens:
        return text

    converted: list[str] = []
    changed = False
    for token in tokens:
        converted_token = _normalize_pasted_attachment_token(token, root=root)
        if converted_token is None:
            return text
        converted.append(converted_token)
        changed = changed or converted_token != token
    return " ".join(converted) if changed else text


def _normalize_pasted_attachment_token(token: str, *, root: Path) -> str | None:
    if token.startswith("file://"):
        return token
    candidate = Path(token).expanduser()
    if not candidate.is_absolute():
        return None
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.is_file():
        return None
    root = root.resolve()
    if resolved == root or root in resolved.parents:
        return token
    return resolved.as_uri()


def is_exit_command(text: str) -> bool:
    """Return whether composer text requests leaving the TUI."""

    return text.strip() == "/exit"


def region_contains_point(region: Region, x: float | int, y: float | int) -> bool:
    """Return whether a Textual region contains screen coordinates."""

    return region.contains_point((int(x), int(y)))


def format_agent_message_event(event: RuntimeEvent) -> str:
    """Render an agent-message event with full delivered bodies when available."""

    payload = event.payload or {}
    bodies = payload.get("bodies")
    if not isinstance(bodies, list) or not bodies:
        return event.text or ""

    sender = _agent_message_sender(payload)
    count = len(bodies)
    plural = "message" if count == 1 else "messages"
    header = f"{sender} delivered {count} {plural}."
    rendered: list[str] = [header]
    for index, body in enumerate(bodies, 1):
        content = str(body).strip() or "(empty body)"
        if count == 1:
            rendered.append(content)
        else:
            rendered.append(f"Message {index}\n{content}")
    return "\n\n".join(rendered)


def summarize_agent_message_update(event: RuntimeEvent) -> str:
    """Build a one-line completion summary for the work panel."""

    payload = event.payload or {}
    sender = _agent_message_sender(payload)
    count = payload.get("count")
    total_chars = payload.get("total_chars")
    if isinstance(count, int) and isinstance(total_chars, int):
        plural = "message" if count == 1 else "messages"
        return f"completed {sender} ({count} {plural}, {total_chars} chars)"
    return f"completed {preview_inline(event.text or 'sub-agent report', limit=80)}"


def _agent_message_sender(payload: dict[str, Any]) -> str:
    name = payload.get("from_agent_name")
    session = payload.get("from_session_id")
    if isinstance(name, str) and isinstance(session, str) and session:
        return f"{name} {session[:8]}"
    if isinstance(name, str):
        return name
    return "sub-agent"


async def run_textual_tui(root: Path, session_id: str | None) -> None:
    """Run the Textual TUI app."""

    app = FeatherTextualApp(root=root, session_id=session_id)
    await app.run_async(mouse=_mouse_enabled())


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
