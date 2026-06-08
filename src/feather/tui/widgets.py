"""Composer + slash-command dropdown widgets for the Textual TUI."""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual import events
from textual.app import App
from textual.widgets import RichLog, Static, TextArea

from feather.tui.slash_commands import (
    SlashCommand,
    SlashCommandRegistry,
    parse_slash_input,
)


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
        # Bare ←/→ switch leads, but only when the composer is empty so that
        # cursor movement within typed text is never hijacked.
        if event.key in ("left", "right") and not self.text:
            app = self.app
            if event.key == "left":
                app.action_lead_prev()  # type: ignore[attr-defined]
            else:
                app.action_lead_next()  # type: ignore[attr-defined]
            event.prevent_default()
            event.stop()
            return
        await super()._on_key(event)


