"""Friendly Rich-based terminal UI for Feather sessions."""

from __future__ import annotations

import argparse
import asyncio
import os
import logging
import re
import sys
import termios
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from feather.models import AgentOutcome, RuntimeEvent
from feather.runtime import FeatherRuntime


_DETAIL_LIMIT = 360
_INLINE_LIMIT = 180
_PASTE_CHAR_THRESHOLD = 1200
_PASTE_LINE_THRESHOLD = 20
_INBOX_WAKE = "\x00__inbox_wake__\x00"
_INBOX_POLL_INTERVAL_SECONDS = 0.5
_SPINNER_FRAMES = ("|", "/", "-", "\\")
_MAX_RENDERED_CONVERSATION_LINES = 28
_MAX_RENDERED_WORK_LINES = 8
_CONVERSATION_SCROLL_STEP = 6
_TASK_TOOL_NAMES = {
    "task_create",
    "task_list",
    "task_get",
    "task_update",
    "task_output",
    "task_stop",
    "task_resume",
}
_ESCAPE_ACTIONS = {
    "\x1b[A": "input_up",
    "\x1b[B": "input_down",
    "\x1b[C": "input_right",
    "\x1b[D": "input_left",
    "\x1b[5~": "page_up",
    "\x1b[6~": "page_down",
    "\x1b[H": "home",
    "\x1b[F": "end",
    "\x1b[1~": "home",
    "\x1b[4~": "end",
    "\x1b[7~": "home",
    "\x1b[8~": "end",
}


@dataclass(slots=True, frozen=True)
class TuiItem:
    """One conversation row in the terminal UI."""

    title: str
    text: str = ""
    style: str = "white"
    detail_style: str = "dim"


@dataclass(slots=True)
class TuiState:
    """Current display state for one Feather TUI session."""

    session_id: str
    agent_name: str
    status: str = "idle"
    context_ratio: float | None = None
    queue_depth: int = 0
    queued_messages: tuple[str, ...] = ()
    active_tool: str | None = None
    input_text: str = ""
    input_cursor: int = 0
    conversation_scroll_offset: int = 0
    running_agents: tuple[str, ...] = ()
    transcript: list[TuiItem] = field(default_factory=list)


class TuiEventPrinter:
    """Collect runtime events and render a friendly terminal dashboard."""

    def __init__(
        self,
        console: Console,
        *,
        session_id: str,
        agent_name: str,
        transcript_limit: int | None = None,
        auto_refresh: bool = False,
    ) -> None:
        self._console = console
        self._transcript_limit = transcript_limit
        self._auto_refresh = auto_refresh
        self._live: Live | None = None
        self._assistant_parts: list[str] = []
        self._spinner_index = 0
        self.state = TuiState(session_id=session_id, agent_name=agent_name)

    def __call__(self, event: RuntimeEvent) -> None:
        """Handle one runtime event.

        Args:
            event: Event emitted by the Feather runtime.
        """

        if event.kind == "assistant_text_delta":
            self.state.status = "running"
            self._assistant_parts.append(event.text or "")
            self._maybe_auto_refresh()
            return

        if event.kind == "usage_updated":
            ratio = (event.payload or {}).get("usage_ratio")
            if isinstance(ratio, (int, float)):
                self.state.context_ratio = max(0.0, min(1.0, float(ratio)))
            self._maybe_auto_refresh()
            return

        self.finish_turn()
        if event.kind == "tool_started":
            self.state.status = "running"
            self.state.active_tool = event.tool_name
            if event.tool_name == "ask_user":
                self._maybe_auto_refresh()
                return
            self._append_activity(
                TuiItem(
                    title=_tool_started_title(event.tool_name),
                    text=_format_tool_payload(event.tool_name, event.payload or {}),
                    style="grey70",
                    detail_style="grey50",
                )
            )
        elif event.kind == "tool_finished":
            self.state.active_tool = None
            if event.tool_name == "ask_user":
                self._maybe_auto_refresh()
                return
            failed_title = _failed_tool_title(event.tool_name, event.text)
            self._append_activity(
                TuiItem(
                    title=failed_title or _tool_finished_title(event.tool_name),
                    text=(
                        _format_tool_error(event.tool_name, event.text)
                        if failed_title
                        else _format_tool_result(event.tool_name, event.text)
                    ),
                    style="red" if failed_title else "grey70",
                    detail_style="grey50",
                )
            )
        elif event.kind == "awaiting_user":
            self.state.status = "awaiting user"
            self._append_transcript(
                TuiItem(
                    title=f"{self.state.agent_name} asks",
                    text=event.text or "",
                    style="white",
                )
            )
        elif event.kind == "user_message_injected":
            self._append_transcript(
                TuiItem(
                    title="Queued input",
                    text=event.text or "",
                    style="white",
                    detail_style="dim",
                )
            )
        elif event.kind == "agent_message_received":
            self._append_transcript(
                TuiItem(
                    title="Sub-agent",
                    text=event.text or "",
                    style="grey70",
                    detail_style="grey50",
                )
            )
        elif event.kind.startswith("compaction_"):
            self._append_activity(
                TuiItem(
                    title=_event_title(event.kind),
                    text=preview_detail(event.text),
                    style=_system_event_style(event.kind),
                )
            )
        elif event.kind.startswith("scheduled_task_"):
            self._append_activity(
                TuiItem(
                    title=_event_title(event.kind),
                    text=preview_detail(event.text),
                    style=_system_event_style(event.kind),
                )
            )
        self._maybe_auto_refresh()

    def mark_running(self) -> None:
        """Mark the lead session as actively working."""

        self.state.status = "running"

    def mark_idle(self) -> None:
        """Mark the lead session as idle unless it is waiting on the user."""

        if self.state.status != "awaiting user":
            self.state.status = "idle"

    def record_user_message(self, text: str) -> None:
        """Add a user-entered message to the visible transcript.

        Args:
            text: Raw user text. Large pasted content is summarized for display.
        """

        if self.state.status == "awaiting user" and self.state.transcript:
            previous = self.state.transcript[-1]
            if previous.title == f"{self.state.agent_name} asks":
                self.state.transcript[-1] = TuiItem(
                    title=previous.title,
                    text="Asked for clarification.",
                    style="white",
                    detail_style="dim",
                )
        self._append_transcript(
            TuiItem(title="You", text=text, style="white")
        )

    def set_queue_snapshot(self, depth: int, messages: tuple[str, ...]) -> None:
        """Update the sticky queue strip shown above the input prompt.

        Args:
            depth: Current queued message count.
            messages: Pending queued messages.
        """

        self.state.queue_depth = max(0, depth)
        self.state.queued_messages = messages

    def set_input_text(self, text: str, cursor: int | None = None) -> None:
        """Update the raw input currently visible in the footer."""

        self.state.input_text = text
        if cursor is None:
            self.state.input_cursor = len(text)
        else:
            self.state.input_cursor = max(0, min(len(text), cursor))

    def set_running_agents(self, agents: tuple[str, ...]) -> None:
        """Update the running sub-agent labels shown in the footer."""

        self.state.running_agents = agents

    def scroll_conversation(self, delta_items: int) -> None:
        """Move the visible conversation window by item count."""

        max_offset = max(0, len(self._conversation_items()) - 1)
        self.state.conversation_scroll_offset = max(
            0,
            min(max_offset, self.state.conversation_scroll_offset + delta_items),
        )

    def scroll_conversation_home(self) -> None:
        """Jump to the oldest available conversation items."""

        self.state.conversation_scroll_offset = max(0, len(self._conversation_items()) - 1)

    def scroll_conversation_end(self) -> None:
        """Jump back to the latest conversation items."""

        self.state.conversation_scroll_offset = 0

    def record_activity(self, title: str, text: str = "", *, style: str = "white") -> None:
        """Append a user-visible status marker to the conversation."""

        self._append_activity(TuiItem(title=title, text=text, style=style))

    def bind_live(self, live: Live | None) -> None:
        """Bind a Rich Live display for in-place refreshes."""

        self._live = live

    def finish_turn(self) -> None:
        """Flush streamed assistant deltas into the transcript."""

        if self._assistant_parts:
            self._append_transcript(
                TuiItem(
                    title=self.state.agent_name,
                    text="".join(self._assistant_parts),
                    style="cyan",
                )
            )
            self._assistant_parts.clear()
        if self.state.status != "awaiting user":
            self.state.status = "running" if self.state.active_tool else "idle"

    def refresh(self) -> None:
        """Redraw the dashboard."""

        if self.state.status == "running":
            self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
        renderable = self.render()
        if self._live is not None:
            self._live.update(renderable, refresh=True)
            return
        self._console.clear()
        self._console.print(renderable)

    def print_prompt(self) -> None:
        """Draw the active input prompt after a dashboard refresh."""
        return

    def render(self) -> RenderableType:
        """Build the current dashboard renderable.

        Returns:
            A Rich renderable representing the dashboard.
        """

        return Group(
            self._render_header(),
            self._render_transcript(),
            self._render_work(),
            self._render_footer(),
        )

    def _append_transcript(self, item: TuiItem) -> None:
        self.state.transcript.append(item)
        self.scroll_conversation_end()
        if self._transcript_limit is not None and len(self.state.transcript) > self._transcript_limit:
            del self.state.transcript[: len(self.state.transcript) - self._transcript_limit]

    def _append_activity(self, item: TuiItem) -> None:
        """Render a concise tool/status marker inline in the conversation."""

        text = item.title if not item.text else f"{item.title} · {item.text}"
        self._append_transcript(
            TuiItem(
                title="Feather",
                text=text,
                style=item.style,
                detail_style=item.detail_style,
            )
        )

    def _maybe_auto_refresh(self) -> None:
        if not self._auto_refresh:
            return
        self.refresh()
        self.print_prompt()

    def _render_header(self) -> Panel:
        ctx = "ctx --"
        if self.state.context_ratio is not None:
            ctx = f"ctx {round(self.state.context_ratio * 100)}%"
        active = f"active {self.state.active_tool}" if self.state.active_tool else "no active tool"
        status_label = self.state.status
        if self.state.status == "running":
            status_label = f"{_SPINNER_FRAMES[self._spinner_index]} running"
        text = Text.assemble(
            ("Feather", "bold"),
            "  ",
            (self.state.agent_name, "white"),
            "  ",
            (status_label, _status_style(self.state.status)),
            "  ",
            (ctx, "white"),
            "  ",
            (f"queued {self.state.queue_depth}", "magenta" if self.state.queue_depth else "dim"),
            "  ",
            (active, "grey70" if self.state.active_tool else "dim"),
            "\n",
            (f"session {self.state.session_id}", "dim"),
        )
        return Panel(text, border_style="grey50", padding=(0, 1))

    def _render_transcript(self) -> Panel:
        rows = Text()
        items = self._conversation_items()
        if not items:
            rows.append("No conversation yet.\n", style="dim")
        visible_items, hidden_older, hidden_newer = _visible_conversation_items(
            items,
            max_lines=_MAX_RENDERED_CONVERSATION_LINES,
            scroll_offset=self.state.conversation_scroll_offset,
        )
        if hidden_older:
            rows.append(
                f"... {hidden_older} older items above (PageUp/Home)\n\n",
                style="dim",
            )
        for item in visible_items:
            rows.append(f"{item.title}\n", style=_conversation_label_style(item.title))
            if item.text:
                rows.append(_indent_lines(item.text), style=_conversation_body_style(item.title))
                rows.append("\n")
        if hidden_newer:
            rows.append(
                f"... {hidden_newer} newer items below (PageDown/End)\n",
                style="dim",
            )
        return Panel(rows, title="Conversation", border_style="grey50", padding=(0, 1))

    def _conversation_items(self) -> list[TuiItem]:
        items = list(self.state.transcript)
        if self._assistant_parts:
            items.append(
                TuiItem(
                    title=f"{self.state.agent_name} streaming",
                    text="".join(self._assistant_parts),
                    style="cyan",
                )
            )
        return items

    def _render_work(self) -> Panel:
        rows = Text()
        has_rows = False
        if self.state.queue_depth:
            has_rows = True
            rows.append("Queued queries\n", style="bold white")
            for index, message in enumerate(self.state.queued_messages, 1):
                rows.append(
                    _indent_lines(preview_detail(f"{index}. {message}", lines=2)),
                    style="white",
                )
        if self.state.running_agents:
            if has_rows:
                rows.append("\n")
            has_rows = True
            rows.append("Future tasks\n", style="bold white")
            for agent in self.state.running_agents[:_MAX_RENDERED_WORK_LINES]:
                rows.append(_indent_lines(preview_inline(agent, limit=140)), style="white")
            hidden = len(self.state.running_agents) - _MAX_RENDERED_WORK_LINES
            if hidden > 0:
                rows.append(_indent_lines(f"... {hidden} more running tasks"), style="dim")
        if not has_rows:
            rows.append("No queued queries or running future tasks.\n", style="dim")
        return Panel(rows, title="Queued / Future Work", border_style="grey50", padding=(0, 1))

    def _render_footer(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        label = "answer> " if self.state.status == "awaiting user" else "you> "
        row = Text.assemble((label, "bold white"))
        if self.state.input_text:
            row.append(_render_input_with_cursor(self.state.input_text, self.state.input_cursor))
        else:
            row.append("|", style="bold white")
            row.append("  type /exit to leave", style="dim")
        table.add_row(row)
        return Panel(table, title="Input", border_style="grey50", padding=(0, 1))


class TuiInputBuffer:
    """Small raw-terminal input buffer for the live TUI footer."""

    def __init__(self) -> None:
        self.text = ""
        self.cursor = 0
        self._submitted: list[str] = []
        self._actions: list[str] = []
        self._paste_mode = False

    def feed(self, chunk: str) -> None:
        """Apply terminal input to the buffer.

        Args:
            chunk: Decoded terminal input.
        """

        index = 0
        while index < len(chunk):
            if chunk.startswith("\x1b[200~", index):
                self._paste_mode = True
                index += len("\x1b[200~")
                continue
            if chunk.startswith("\x1b[201~", index):
                self._paste_mode = False
                index += len("\x1b[201~")
                continue
            matched_sequence = self._match_escape_action(chunk, index)
            if matched_sequence is not None:
                sequence, action = matched_sequence
                self._handle_action(action)
                index += len(sequence)
                continue

            char = chunk[index]
            index += 1

            if char == "\x1b":
                self._actions.append("interrupt")
                continue
            if char == "\x03":
                raise KeyboardInterrupt
            if char == "\x04":
                raise EOFError
            if char in {"\r", "\n"} and not self._paste_mode:
                self._submitted.append(self.text)
                self.text = ""
                self.cursor = 0
                continue
            if char in {"\x7f", "\b"} and not self._paste_mode:
                if self.cursor > 0:
                    self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
                    self.cursor -= 1
                continue
            self.text = self.text[: self.cursor] + char + self.text[self.cursor :]
            self.cursor += 1

    def pop_line(self) -> str | None:
        """Return the next submitted line, if any."""

        if not self._submitted:
            return None
        return self._submitted.pop(0)

    def pop_actions(self) -> tuple[str, ...]:
        """Return pending non-text input actions."""

        actions = tuple(self._actions)
        self._actions.clear()
        return actions

    def _match_escape_action(self, chunk: str, index: int) -> tuple[str, str] | None:
        for sequence, action in _ESCAPE_ACTIONS.items():
            if chunk.startswith(sequence, index):
                return sequence, action
        return None

    def _handle_action(self, action: str) -> None:
        if action == "input_left":
            self.cursor = max(0, self.cursor - 1)
        elif action == "input_right":
            self.cursor = min(len(self.text), self.cursor + 1)
        elif action == "input_up":
            self._move_vertical(-1)
        elif action == "input_down":
            self._move_vertical(1)
        else:
            self._actions.append(action)

    def _move_vertical(self, delta: int) -> None:
        lines = self.text.split("\n")
        line, column = self._line_column_for_cursor(lines)
        target_line = line + delta
        if target_line < 0 or target_line >= len(lines):
            return
        self.cursor = self._cursor_for_line_column(
            lines,
            target_line,
            min(column, len(lines[target_line])),
        )

    def _line_column_for_cursor(self, lines: list[str]) -> tuple[int, int]:
        remaining = self.cursor
        for line_number, line in enumerate(lines):
            if remaining <= len(line):
                return line_number, remaining
            remaining -= len(line) + 1
        return len(lines) - 1, len(lines[-1])

    def _cursor_for_line_column(
        self,
        lines: list[str],
        line_number: int,
        column: int,
    ) -> int:
        offset = sum(len(line) + 1 for line in lines[:line_number])
        return offset + column


async def run_tui(root: Path, session_id: str | None) -> None:
    """Run the friendly Feather terminal UI.

    Args:
        root: Repository root.
        session_id: Optional existing session ID.
    """

    console = Console()
    runtime = await FeatherRuntime.create(root)
    active_session_id: str | None = None

    try:
        agent = runtime.build_agent("lead")
        active_session_id = session_id or await agent.create_session()
        printer = TuiEventPrinter(
            console,
            session_id=active_session_id,
            agent_name=agent.config.name,
            auto_refresh=True,
        )
        runtime.set_session_event_handler(active_session_id, printer)
        await runtime.start_background_services()

        input_queue = runtime.input_queue
        stop_event = asyncio.Event()
        pending_answer: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        busy_event = asyncio.Event()
        awaiting_event = asyncio.Event()
        new_run_queue: asyncio.Queue[str] = asyncio.Queue()
        input_buffer = TuiInputBuffer()
        run_task: asyncio.Task[Any] | None = None

        async def _refresh_monitor() -> None:
            assert active_session_id is not None
            depth = await input_queue.depth(active_session_id)
            pending = await input_queue.peek(active_session_id)
            printer.set_queue_snapshot(depth, tuple(pending))
            live = await runtime.subagent_registry.snapshot()
            printer.set_running_agents(
                tuple(
                    f"{entry.agent_name} {entry.session_id[:8]}: "
                    f"{preview_inline(entry.task_text, limit=64)}"
                    for entry in live
                )
            )

        def _interrupt_run() -> None:
            if run_task is not None and not run_task.done():
                run_task.cancel()

        async def _stdin_reader() -> None:
            nonlocal run_task
            while not stop_event.is_set():
                try:
                    line = await _read_tui_line(
                        console,
                        printer,
                        stop_event,
                        input_buffer,
                        on_interrupt=_interrupt_run,
                    )
                except (EOFError, KeyboardInterrupt):
                    stop_event.set()
                    return
                text = line.strip()
                if not text:
                    printer.refresh()
                    _print_tui_prompt(printer)
                    continue
                if text in {"/exit", "/quit"}:
                    stop_event.set()
                    return
                if text == "/queue":
                    await _refresh_monitor()
                    printer.refresh()
                    _print_tui_prompt(printer)
                    continue
                if text in {"/agents", "/agent"}:
                    await _refresh_monitor()
                    agents = _format_running_agents(printer.state.running_agents)
                    printer.record_activity("Running agents", agents, style="cyan")
                    printer.refresh()
                    _print_tui_prompt(printer)
                    continue
                if text in {"/tools", "/activity"}:
                    printer.record_activity(
                        "Tool history",
                        "Recent tool calls are shown inline in the conversation.",
                        style="cyan",
                    )
                    printer.refresh()
                    _print_tui_prompt(printer)
                    continue
                if text in {"/help", "/?"}:
                    printer.record_activity(
                        "Commands",
                        "/queue, /agents, /tools, /exit",
                        style="cyan",
                    )
                    printer.refresh()
                    _print_tui_prompt(printer)
                    continue

                if awaiting_event.is_set():
                    printer.record_user_message(text)
                    try:
                        pending_answer.put_nowait(text)
                        awaiting_event.clear()
                    except asyncio.QueueFull:
                        assert active_session_id is not None
                        await input_queue.enqueue(active_session_id, text)
                        await _refresh_monitor()
                    printer.refresh()
                    _print_tui_prompt(printer)
                    continue

                if busy_event.is_set():
                    assert active_session_id is not None
                    ok = await input_queue.enqueue(active_session_id, text)
                    if ok:
                        await _refresh_monitor()
                    printer.refresh()
                    _print_tui_prompt(printer)
                    continue

                printer.record_user_message(text)
                printer.mark_running()
                printer.refresh()
                await new_run_queue.put(text)

        async def _agent_driver() -> None:
            nonlocal run_task
            assert active_session_id is not None
            while not stop_event.is_set():
                get_task = asyncio.create_task(new_run_queue.get())
                stop_task = asyncio.create_task(stop_event.wait())
                await asyncio.wait(
                    {get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if stop_event.is_set():
                    get_task.cancel()
                    return
                stop_task.cancel()
                user_message = get_task.result()

                busy_event.set()
                printer.mark_running()
                printer.refresh()
                try:
                    if user_message == _INBOX_WAKE:
                        try:
                            run_task = asyncio.create_task(
                                agent.resume_on_inbox(
                                    active_session_id,
                                    printer,
                                )
                            )
                            result = await run_task
                            run_task = None
                        except asyncio.CancelledError:
                            _record_agent_interrupted(printer)
                            run_task = None
                            continue
                        except Exception as exc:  # noqa: BLE001
                            run_task = None
                            _record_agent_error(printer, exc)
                            logging.getLogger(__name__).exception(
                                "tui.agent_driver.resume_on_inbox_crashed session_id=%s",
                                active_session_id,
                            )
                            continue
                        if result is None:
                            printer.mark_idle()
                            printer.refresh()
                            _print_tui_prompt(printer)
                            continue
                    else:
                        try:
                            run_task = asyncio.create_task(
                                agent.run(active_session_id, user_message, printer)
                            )
                            result = await run_task
                            run_task = None
                        except asyncio.CancelledError:
                            _record_agent_interrupted(printer)
                            run_task = None
                            continue
                        except Exception as exc:  # noqa: BLE001
                            run_task = None
                            _record_agent_error(printer, exc)
                            logging.getLogger(__name__).exception(
                                "tui.agent_driver.run_crashed session_id=%s",
                                active_session_id,
                            )
                            continue

                    printer.finish_turn()
                    await _refresh_monitor()
                    if result.status == AgentOutcome.COMPLETED:
                        printer.mark_idle()
                    printer.refresh()
                    _print_tui_prompt(printer)

                    while (
                        result.status == AgentOutcome.AWAITING_USER
                        and result.question is not None
                        and not stop_event.is_set()
                    ):
                        awaiting_event.set()
                        printer.refresh()
                        _print_tui_prompt(printer)
                        answer_task = asyncio.create_task(pending_answer.get())
                        stop_task = asyncio.create_task(stop_event.wait())
                        await asyncio.wait(
                            {answer_task, stop_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if stop_event.is_set():
                            answer_task.cancel()
                            awaiting_event.clear()
                            return
                        stop_task.cancel()
                        answer = answer_task.result()
                        try:
                            run_task = asyncio.create_task(
                                agent.run(active_session_id, answer, printer)
                            )
                            result = await run_task
                            run_task = None
                        except asyncio.CancelledError:
                            _record_agent_interrupted(printer)
                            run_task = None
                            break
                        printer.finish_turn()
                        await _refresh_monitor()
                        if result.status == AgentOutcome.COMPLETED:
                            printer.mark_idle()
                        printer.refresh()
                        _print_tui_prompt(printer)
                finally:
                    busy_event.clear()
                    awaiting_event.clear()

        async def _inbox_watcher() -> None:
            assert active_session_id is not None
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        stop_event.wait(), timeout=_INBOX_POLL_INTERVAL_SECONDS
                    )
                    return
                except asyncio.TimeoutError:
                    pass
                if busy_event.is_set() or awaiting_event.is_set():
                    continue
                try:
                    has_pending = await agent.has_pending_inbox(active_session_id)
                except Exception:  # noqa: BLE001
                    continue
                if has_pending and new_run_queue.empty():
                    await new_run_queue.put(_INBOX_WAKE)

        with _RawTerminal():
            with Live(
                printer.render(),
                console=console,
                refresh_per_second=8,
                screen=True,
            ) as live:
                printer.bind_live(live)
                printer.refresh()
                reader_task = asyncio.create_task(
                    _stdin_reader(), name="tui-stdin-reader"
                )
                driver_task = asyncio.create_task(
                    _agent_driver(), name="tui-agent-driver"
                )
                watcher_task = asyncio.create_task(
                    _inbox_watcher(), name="tui-inbox-watcher"
                )
                try:
                    await asyncio.wait(
                        {reader_task, driver_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    stop_event.set()
                    for task in (reader_task, driver_task, watcher_task):
                        if not task.done():
                            task.cancel()
                    for task in (reader_task, driver_task, watcher_task):
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):  # noqa: BLE001
                            pass
                    printer.bind_live(None)
    finally:
        if active_session_id is not None:
            runtime.set_session_event_handler(active_session_id, None)
        await runtime.close()


def summarize_user_text(text: str) -> str:
    """Return a display-safe summary for user-entered text.

    Args:
        text: Raw input text.

    Returns:
        Direct preview for short input, or a pasted-content marker for large
        multi-line input.
    """

    line_count = text.count("\n") + 1 if text else 0
    if len(text) >= _PASTE_CHAR_THRESHOLD or line_count >= _PASTE_LINE_THRESHOLD:
        return f"[Pasted content: {len(text):,} chars, {line_count:,} lines]"
    return preview_inline(text)


def preview_inline(text: str | None, *, limit: int = _INLINE_LIMIT) -> str:
    """Collapse text to one short line for inline display."""

    if not text:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


def preview_detail(text: str | None, *, limit: int = _DETAIL_LIMIT, lines: int = 8) -> str:
    """Return a compact multi-line preview for tool details."""

    if not text:
        return ""
    stripped = text.strip()
    selected_lines = stripped.splitlines()[:lines]
    preview = "\n".join(selected_lines)
    truncated = len(stripped.splitlines()) > lines
    if len(preview) > limit:
        preview = preview[:limit].rstrip()
        truncated = True
    if truncated and not preview.endswith("..."):
        preview += "..."
    return preview


def _visible_conversation_items(
    items: list[TuiItem],
    *,
    max_lines: int,
    scroll_offset: int,
) -> tuple[list[TuiItem], int, int]:
    if not items:
        return [], 0, 0
    end_index = max(1, len(items) - max(0, scroll_offset))
    source = items[:end_index]
    used = 0
    visible_reversed: list[TuiItem] = []
    for item in reversed(source):
        item_lines = _conversation_item_line_count(item)
        if visible_reversed and used + item_lines > max_lines:
            break
        visible_reversed.append(item)
        used += item_lines
        if used >= max_lines:
            break
    visible = list(reversed(visible_reversed))
    hidden_older = max(0, end_index - len(visible))
    hidden_newer = max(0, len(items) - end_index)
    return visible, hidden_older, hidden_newer


def _conversation_item_line_count(item: TuiItem) -> int:
    return 2 + (item.text.count("\n") + 1 if item.text else 0)


def _render_input_with_cursor(text: str, cursor: int) -> Text:
    if len(text) >= _PASTE_CHAR_THRESHOLD or text.count("\n") + 1 >= _PASTE_LINE_THRESHOLD:
        summary = summarize_user_text(text)
        rendered = Text(summary, style="white")
        rendered.append("|", style="bold white")
        return rendered
    safe_cursor = max(0, min(len(text), cursor))
    before = text[:safe_cursor]
    after = text[safe_cursor:]
    rendered = Text(before, style="white")
    rendered.append("|", style="bold white")
    rendered.append(after, style="white")
    return rendered


async def _read_console_input(console: Console, prompt: str) -> str:
    return await asyncio.to_thread(console.input, prompt)


async def _read_tui_line(
    console: Console,
    printer: TuiEventPrinter,
    stop_event: asyncio.Event,
    input_buffer: TuiInputBuffer,
    on_interrupt: Callable[[], None] | None = None,
) -> str:
    if not sys.stdin.isatty():
        line = await _read_console_input(console, "")
        printer.set_input_text("")
        printer.refresh()
        return line

    while not stop_event.is_set():
        chunk = await _read_stdin_chunk()
        input_buffer.feed(chunk)
        for action in input_buffer.pop_actions():
            _apply_tui_action(printer, action, on_interrupt=on_interrupt)
        printer.set_input_text(input_buffer.text, input_buffer.cursor)
        printer.refresh()
        line = input_buffer.pop_line()
        if line is not None:
            return line
    return ""


def _apply_tui_action(
    printer: TuiEventPrinter,
    action: str,
    *,
    on_interrupt: Callable[[], None] | None = None,
) -> None:
    if action == "page_up":
        printer.scroll_conversation(_CONVERSATION_SCROLL_STEP)
    elif action == "page_down":
        printer.scroll_conversation(-_CONVERSATION_SCROLL_STEP)
    elif action == "home":
        printer.scroll_conversation_home()
    elif action == "end":
        printer.scroll_conversation_end()
    elif action == "interrupt" and on_interrupt is not None:
        on_interrupt()


async def _read_stdin_chunk() -> str:
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    future: asyncio.Future[bytes] = loop.create_future()

    def _ready() -> None:
        if future.done():
            return
        try:
            future.set_result(os.read(fd, 4096))
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)

    loop.add_reader(fd, _ready)
    try:
        data = await future
    finally:
        loop.remove_reader(fd)
    if not data:
        raise EOFError
    return data.decode("utf-8", errors="replace")


class _RawTerminal:
    """Temporarily disable terminal echo for footer-owned input rendering."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_attrs: list[Any] | None = None

    def __enter__(self) -> _RawTerminal:
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        attrs = termios.tcgetattr(self._fd)
        attrs[3] &= ~(termios.ICANON | termios.ECHO)
        attrs[6][termios.VMIN] = 1
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self._fd, termios.TCSADRAIN, attrs)
        sys.stdout.write("\x1b[?2004h")
        sys.stdout.flush()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is None:
            return
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
        if self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)


def _print_tui_prompt(printer: TuiEventPrinter) -> None:
    printer.print_prompt()


def _record_agent_error(printer: TuiEventPrinter, exc: Exception) -> None:
    printer.finish_turn()
    printer._append_activity(
        TuiItem(
            title="Agent error",
            text=f"{type(exc).__name__}: {exc}",
            style="red",
        )
    )
    printer.mark_idle()
    printer.refresh()
    _print_tui_prompt(printer)


def _record_agent_interrupted(printer: TuiEventPrinter) -> None:
    printer.finish_turn()
    printer._append_activity(
        TuiItem(
            title="Interrupted",
            text="Esc pressed; active run cancelled.",
            style="yellow",
        )
    )
    printer.mark_idle()
    printer.refresh()
    _print_tui_prompt(printer)


def _format_tool_payload(tool_name: str | None, payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    if tool_name == "bash":
        command = str(payload.get("command") or "").strip()
        return preview_inline(command, limit=120)
    if tool_name == "read_file":
        return preview_inline(str(payload.get("path") or ""), limit=120)
    if tool_name == "grep":
        return preview_inline(
            str(payload.get("pattern") or payload.get("q") or ""), limit=120
        )
    if tool_name == "load_skill":
        return preview_inline(str(payload.get("skill_name") or ""), limit=120)
    if tool_name == "spawn_agent":
        return preview_inline(str(payload.get("agent_name") or ""), limit=120)
    if tool_name in {"web_search", "parallel_search"}:
        queries = payload.get("search_queries")
        if isinstance(queries, list) and queries:
            return f"{len(queries)} queries"
        return preview_inline(str(payload.get("objective") or "search"), limit=120)
    return _payload_hint(payload)


def _format_tool_result(tool_name: str | None, text: str | None) -> str:
    if not text:
        return ""
    if text.startswith("Tool `") and " failed:" in text:
        return preview_inline(text.partition(" failed:")[2].strip(), limit=120)
    if tool_name == "bash":
        code = _line_value(text, "exit_code")
        return f"exit {code}" if code is not None else "finished"
    if tool_name == "load_skill":
        name = _backtick_value(text) or _line_value(text, "skill")
        return f"{name} loaded" if name else "loaded"
    if tool_name == "spawn_agent":
        name = _backtick_value(text) or "agent"
        session_id = _line_value(text, "session_id")
        suffix = f" {session_id[:8]}" if session_id else ""
        return f"{name}{suffix}"
    if tool_name in _TASK_TOOL_NAMES:
        return preview_detail(text, limit=240, lines=5)
    if tool_name in {"web_search", "parallel_search"}:
        match = re.search(r"\bresults=(\d+)\b", text)
        result_count = f"{match.group(1)} results" if match else "search complete"
        titles = _numbered_result_titles(text)
        if titles:
            return f"{result_count}: {'; '.join(titles[:3])}"
        return result_count
    if tool_name == "read_file":
        line_count = re.search(r"\((\d+) lines?\)", text)
        if line_count:
            return f"{line_count.group(1)} lines"
        return "read complete"
    if tool_name == "grep":
        match_count = re.search(r"\b(\d+)\s+matches?\b", text, re.IGNORECASE)
        if match_count:
            return f"{match_count.group(1)} matches"
        return "search complete"
    return "completed"


def _format_tool_error(tool_name: str | None, text: str | None) -> str:
    if not text:
        return "failed without details"
    if text.startswith("Tool `") and " failed:" in text:
        reason = text.partition(" failed:")[2].strip()
        return preview_inline(reason, limit=180) or "failed without details"
    if tool_name == "bash":
        code = _line_value(text, "exit_code")
        stderr = _section_value(text, "stderr")
        stdout = _section_value(text, "stdout")
        detail = stderr or stdout or text
        prefix = f"exit {code}" if code else "failed"
        return f"{prefix}: {preview_inline(detail, limit=160)}"
    return preview_inline(text, limit=180) or "failed without details"


def _payload_hint(payload: dict[str, Any]) -> str:
    for key in (
        "path",
        "file_path",
        "command",
        "pattern",
        "query",
        "skill_name",
        "agent_name",
        "objective",
        "title",
        "task_id",
        "name",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return preview_inline(value, limit=120)
    return ""


def _numbered_result_titles(text: str) -> list[str]:
    titles: list[str] = []
    for line in text.splitlines():
        match = re.match(r"\s*\d+\.\s+(.+)", line)
        if not match:
            continue
        titles.append(preview_inline(match.group(1), limit=60))
    return titles


def _tool_started_title(tool_name: str | None) -> str:
    pretty = _pretty_tool_name(tool_name)
    if tool_name == "load_skill":
        return "Loading skill"
    if tool_name == "read_file":
        return "Reading file"
    if tool_name == "grep":
        return "Running grep"
    if tool_name in {"web_search", "parallel_search"}:
        return "Searching web"
    if tool_name == "recall_memory":
        return "Searching memory"
    if tool_name == "spawn_agent":
        return "Spawning sub-agent"
    if tool_name in _TASK_TOOL_NAMES:
        return f"Running {_pretty_tool_title(tool_name)}"
    return f"Running {pretty}"


def _tool_finished_title(tool_name: str | None) -> str:
    pretty = _pretty_tool_name(tool_name)
    if tool_name == "load_skill":
        return "Loaded skill"
    if tool_name == "read_file":
        return "Read file"
    if tool_name == "grep":
        return "Ran grep"
    if tool_name in {"web_search", "parallel_search"}:
        return "Searched web"
    if tool_name == "recall_memory":
        return "Searched memory"
    if tool_name == "spawn_agent":
        return "Spawned sub-agent"
    if tool_name in _TASK_TOOL_NAMES:
        return f"Ran {_pretty_tool_title(tool_name)}"
    return f"Ran {pretty}"


def _failed_tool_title(tool_name: str | None, text: str | None) -> str | None:
    if not text:
        return None
    if text.startswith("Tool `") and " failed:" in text:
        if tool_name in _TASK_TOOL_NAMES:
            return f"{_pretty_tool_title(tool_name)} failed"
        return f"{_pretty_tool_name(tool_name).capitalize()} failed"
    if tool_name == "bash":
        for line in text.splitlines():
            if not line.startswith("exit_code:"):
                continue
            raw_code = line.partition(":")[2].strip()
            if raw_code and raw_code != "0":
                return f"Bash failed, exit {raw_code}"
            return None
    return None


def _pretty_tool_name(tool_name: str | None) -> str:
    return (tool_name or "tool").replace("_", " ")


def _pretty_tool_title(tool_name: str | None) -> str:
    return _pretty_tool_name(tool_name).title()


def _line_value(text: str, key: str) -> str | None:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            value = line.partition(":")[2].strip()
            return value or None
    return None


def _section_value(text: str, key: str) -> str | None:
    marker = f"{key}:"
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line != marker:
            continue
        values: list[str] = []
        for value_line in lines[index + 1 :]:
            if re.match(r"^[a-z_]+:", value_line):
                break
            values.append(value_line)
        collapsed = " ".join(part.strip() for part in values if part.strip())
        return collapsed or None
    return None


def _backtick_value(text: str) -> str | None:
    match = re.search(r"`([^`]+)`", text)
    return match.group(1) if match else None


def _event_title(kind: str) -> str:
    return kind.replace("_", " ").capitalize()


def _system_event_style(kind: str) -> str:
    if kind.endswith("failed"):
        return "red"
    if kind.endswith("finished") or kind.endswith("completed"):
        return "green"
    return "yellow"


def _status_style(status: str) -> str:
    if status == "awaiting user":
        return "magenta"
    if status == "running":
        return "yellow"
    return "green"


def _conversation_label_style(title: str) -> str:
    if title == "You":
        return "bold cyan"
    if title.startswith("Lead"):
        return "bold green"
    if title in {"Feather", "Sub-agent", "Queued input"}:
        return "bold grey70"
    return "bold white"


def _conversation_body_style(title: str) -> str:
    if title in {"Feather", "Sub-agent", "Queued input"}:
        return "grey70"
    return "bold white"


def _format_running_agents(agents: tuple[str, ...]) -> str:
    if not agents:
        return ""
    return ", ".join(agents)


def _indent_lines(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines()) + "\n"


def _prefix_detail(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    first, *rest = lines
    rendered = [f"  -> {first}"]
    rendered.extend(f"     {line}" for line in rest)
    return "\n".join(rendered) + "\n"


def add_tui_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the `feather tui` subcommand on an argparse parser."""

    parser = subparsers.add_parser("tui", help="Run the friendly Feather TUI.")
    parser.add_argument(
        "--session-id",
        dest="tui_session_id",
        help="Resume an existing session ID.",
        default=None,
    )
