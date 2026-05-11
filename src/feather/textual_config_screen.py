"""Textual modal for /config — interactive config editor.

The modal owns no business logic; every read/write call routes
through ``feather.config_service.ConfigService``.

Layout::

  ┌── /config ────────────────────────────────────────────┐
  │ [App]   Lead                                          │  ← #config-tabs
  ├──────────────┬────────────────────────────────────────┤
  │ subsection 1 │ (form rows)                            │
  │▶subsection 2 │                                        │
  │ subsection 3 │                                        │
  ├──────────────┴────────────────────────────────────────┤
  │ <status>      s=save d=diff r=reset esc=close          │  ← #config-footer
  └────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from feather.config_schema import ConfigField, REGISTRY, ReloadClass, Scope, lookup
from feather.config_service import ConfigService


@dataclass(slots=True, frozen=True)
class _TabSpec:
    """One top-level tab in the config modal.

    Attributes:
        label: Human-readable tab title.
        section_prefix: Dotted prefix that filters REGISTRY entries shown
            in this tab (e.g. ``"app."`` or ``"agents.Lead."``).
    """

    label: str
    section_prefix: str


def _discover_tabs(service: ConfigService) -> list[_TabSpec]:
    """Build the ordered tab list from the registry.

    The ``App`` tab is always first. Each unique agent name found in
    registry entries with ``scope == AGENT`` produces one trailing tab.

    Args:
        service: Config service (unused currently, reserved for future
            per-agent filtering).

    Returns:
        Ordered list of :class:`_TabSpec`.
    """

    del service  # reserved
    tabs: list[_TabSpec] = [_TabSpec(label="App", section_prefix="app.")]
    seen: list[str] = []
    for f in REGISTRY:
        if f.scope is Scope.AGENT:
            name = f.path.split(".")[1]
            if name not in seen:
                seen.append(name)
    for name in seen:
        tabs.append(_TabSpec(label=name, section_prefix=f"agents.{name}."))
    return tabs


class ConfigScreen(ModalScreen[None]):
    """Interactive modal for editing Feather configuration.

    All reads/writes route through :class:`~feather.config_service.ConfigService`.
    Changes accumulate in ``_dirty`` and are only persisted when the user
    presses ``s``.

    Args:
        service: Config service to delegate to.
        runtime: Optional :class:`~feather.runtime.FeatherRuntime` used to
            call ``apply_config_change`` after a successful save.
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("left", "prev_tab", "← tab", show=True),
        Binding("right", "next_tab", "→ tab", show=True),
        Binding("up", "section_prev", "↑ section", show=True),
        Binding("down", "section_next", "↓ section", show=True),
        Binding("enter", "edit_field", "Edit", show=True),
        Binding("s", "save", "Save", show=True),
        Binding("d", "diff", "Diff", show=True),
        Binding("r", "reset", "Reset", show=True),
        Binding("tab", "field_next", "Next field", show=False, priority=True),
        Binding("shift+tab", "field_prev", "Prev field", show=False, priority=True),
        Binding("y", "confirm_self_repair", "Confirm", show=False),
    ]

    DEFAULT_CSS = """
    ConfigScreen {
        align: center middle;
    }
    #config-root {
        width: 90%;
        height: 90%;
        border: round $accent;
        background: $surface;
    }
    #config-tabs {
        height: 1;
        padding: 0 1;
        background: $primary 10%;
    }
    #config-body {
        height: 1fr;
    }
    #config-sidebar {
        width: 20;
        border-right: solid $accent 30%;
        padding: 0 1;
    }
    #config-form {
        width: 1fr;
        padding: 0 1;
        overflow-y: scroll;
    }
    #config-footer {
        height: 1;
        padding: 0 1;
        background: $primary 10%;
    }
    #config-inline-editor {
        dock: bottom;
        height: 3;
        border: solid $accent;
    }
    """

    def __init__(
        self,
        *,
        service: ConfigService,
        runtime: Any | None = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._runtime = runtime
        self._tabs = _discover_tabs(service)
        self._active_tab_index: int = 0
        self._active_section_index: int = 0
        self._active_field_index: int = 0
        self._dirty: dict[str, Any] = {}
        self._self_repair_confirmed: bool = False
        self._confirm_close: bool = False
        self._pending_edit: ConfigField | None = None

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Build the modal widget tree."""

        with Vertical(id="config-root"):
            yield Static(self._render_tab_bar(), id="config-tabs")
            with Horizontal(id="config-body"):
                yield Static(self._render_sidebar(), id="config-sidebar")
                yield Static(self._render_form(), id="config-form")
            yield Static(self._render_footer(), id="config-footer")

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_tab_bar(self) -> str:
        """Render the top tab bar with the active tab highlighted."""

        parts: list[str] = []
        for i, tab in enumerate(self._tabs):
            label = tab.label
            if i == self._active_tab_index:
                parts.append(f"[reverse]{label}[/reverse]")
            else:
                parts.append(label)
        return "   ".join(parts)

    def _subsections(self) -> list[str]:
        """Return unique subsection labels under the active tab.

        For ``app.*`` tabs: returns the second dotted segment
        (e.g. ``compaction``, ``openai``).  For ``agents.<name>.*``
        tabs: returns the third segment for nested fields, or the
        synthetic label ``agent`` for leaf fields directly under the
        agent root.

        Returns:
            Ordered, deduplicated list of subsection names.
        """

        prefix = self._tabs[self._active_tab_index].section_prefix
        out: list[str] = []
        for f in REGISTRY:
            if not f.path.startswith(prefix):
                continue
            tail = f.path[len(prefix):]
            label = tail.split(".", 1)[0] if "." in tail else "agent"
            if label not in out:
                out.append(label)
        return out

    def _render_sidebar(self) -> str:
        """Render the left-hand subsection list with a cursor marker."""

        sections = self._subsections()
        if not sections:
            return "(no fields)"
        # Clamp cursor in case the tab switch changed section count.
        self._active_section_index = self._active_section_index % len(sections)
        parts: list[str] = []
        for i, name in enumerate(sections):
            marker = "▶" if i == self._active_section_index else " "
            parts.append(f"{marker} {name}")
        return "\n".join(parts)

    def _render_form(self) -> str:
        """Render the right-hand form rows for the active subsection."""

        sections = self._subsections()
        if not sections:
            return "(no fields)"
        active_section = sections[self._active_section_index % len(sections)]
        fields = self._fields_in_section()
        rows: list[str] = []
        for idx, f in enumerate(fields):
            cv = self._service.get(f.path)
            badge_src = f"[{cv.source.value}]"
            badge_rl = f"[{f.reload.value}]"
            dirty_badge = " [DIRTY]" if f.path in self._dirty else ""
            cursor_marker = "▶" if idx == (self._active_field_index % max(1, len(fields))) else " "
            current = self._dirty.get(f.path, cv.current)
            rows.append(
                f"{cursor_marker} {f.path}   {badge_src}{dirty_badge}   {badge_rl}\n"
                f"   ▸ {current!r}\n"
                f"   {f.description}"
            )
        return "\n\n".join(rows) or "(no fields)"

    def _render_footer(self) -> str:
        """Render the footer status line."""

        dirty = len(self._dirty)
        return f"{dirty} dirty   s=save  d=diff  r=reset  esc=close"

    def _fields_in_section(self) -> list[ConfigField]:
        """Return registry fields for the currently active subsection.

        Returns:
            Ordered list of :class:`~feather.config_schema.ConfigField`.
        """

        sections = self._subsections()
        if not sections:
            return []
        prefix = self._tabs[self._active_tab_index].section_prefix
        active_section = sections[self._active_section_index % len(sections)]
        result: list[ConfigField] = []
        for f in REGISTRY:
            if not f.path.startswith(prefix):
                continue
            tail = f.path[len(prefix):]
            section = tail.split(".", 1)[0] if "." in tail else "agent"
            if section == active_section:
                result.append(f)
        return result

    def _refresh_body(self) -> None:
        """Re-render sidebar and form in place."""

        self.query_one("#config-sidebar", Static).update(self._render_sidebar())
        self.query_one("#config-form", Static).update(self._render_form())

    # ------------------------------------------------------------------
    # Tab navigation
    # ------------------------------------------------------------------

    def action_prev_tab(self) -> None:
        """Move to the previous tab, wrapping around."""

        self._active_tab_index = (self._active_tab_index - 1) % len(self._tabs)
        self._active_section_index = 0
        self._active_field_index = 0
        self.query_one("#config-tabs", Static).update(self._render_tab_bar())
        self._refresh_body()

    def action_next_tab(self) -> None:
        """Move to the next tab, wrapping around."""

        self._active_tab_index = (self._active_tab_index + 1) % len(self._tabs)
        self._active_section_index = 0
        self._active_field_index = 0
        self.query_one("#config-tabs", Static).update(self._render_tab_bar())
        self._refresh_body()

    # ------------------------------------------------------------------
    # Section navigation
    # ------------------------------------------------------------------

    def action_section_prev(self) -> None:
        """Move the section cursor up one row."""

        sections = self._subsections()
        if sections:
            self._active_section_index = (
                self._active_section_index - 1
            ) % len(sections)
            self._active_field_index = 0
            self._refresh_body()

    def action_section_next(self) -> None:
        """Move the section cursor down one row."""

        sections = self._subsections()
        if sections:
            self._active_section_index = (
                self._active_section_index + 1
            ) % len(sections)
            self._active_field_index = 0
            self._refresh_body()

    # ------------------------------------------------------------------
    # Field navigation
    # ------------------------------------------------------------------

    def action_field_next(self) -> None:
        """Move the field cursor forward one row."""

        fields = self._fields_in_section()
        if fields:
            self._active_field_index = (self._active_field_index + 1) % len(fields)
            self._refresh_body()

    def action_field_prev(self) -> None:
        """Move the field cursor backward one row."""

        fields = self._fields_in_section()
        if fields:
            self._active_field_index = (self._active_field_index - 1) % len(fields)
            self._refresh_body()

    # ------------------------------------------------------------------
    # Inline editor
    # ------------------------------------------------------------------

    def action_edit_field(self) -> None:
        """Open the inline Input widget for the focused field."""

        fields = self._fields_in_section()
        if not fields:
            return
        field = fields[self._active_field_index % len(fields)]
        # Don't open a second editor if one is already mounted.
        existing = self.query("#config-inline-editor")
        if existing:
            return
        self._pending_edit = field
        editor = Input(
            placeholder=f"new value for {field.path}",
            id="config-inline-editor",
        )
        self.mount(editor)
        editor.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle submission of the inline editor.

        Args:
            event: Textual ``Input.Submitted`` message.
        """

        if event.input.id != "config-inline-editor":
            return
        field = self._pending_edit
        if field is None:
            event.input.remove()
            return

        validate = self._service.validate(field.path, event.value)
        if not validate.ok:
            self.query_one("#config-footer", Static).update(
                f"INVALID: {validate.error}   esc=cancel"
            )
            event.input.remove()
            self._pending_edit = None
            return

        self._dirty[field.path] = validate.coerced
        self._pending_edit = None
        event.input.remove()
        self._refresh_body()
        self.query_one("#config-footer", Static).update(self._render_footer())

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def action_save(self) -> None:
        """Write all dirty fields through ConfigService and show a banner."""

        if not self._dirty:
            self.query_one("#config-footer", Static).update(
                "no dirty fields  esc=close"
            )
            return

        # self_repair.enabled requires a y-confirm first.
        if "app.self_repair.enabled" in self._dirty and not self._self_repair_confirmed:
            self.query_one("#config-footer", Static).update(
                "self_repair.enabled change is RESTART-APP. Press 'y' to confirm, esc to cancel."
            )
            return

        live: list[str] = []
        next_turn: list[str] = []
        restart_lead: list[str] = []
        restart_app: list[str] = []
        errors: list[str] = []

        for path, value in list(self._dirty.items()):
            force = path == "app.self_repair.enabled"
            result = self._service.set(path, value, force=force)
            if not result.ok:
                errors.append(f"{path}: {result.error}")
                continue
            del self._dirty[path]
            field_def = lookup(path)
            if field_def is not None:
                bucket = {
                    ReloadClass.LIVE: live,
                    ReloadClass.NEXT_TURN: next_turn,
                    ReloadClass.RESTART_LEAD: restart_lead,
                    ReloadClass.RESTART_APP: restart_app,
                }[field_def.reload]
                bucket.append(path)

        parts: list[str] = []
        if live:
            parts.append(f"{len(live)} live")
        if next_turn:
            parts.append(f"{len(next_turn)} next-turn")
        if restart_lead:
            parts.append(f"{len(restart_lead)} needs restart-lead")
        if restart_app:
            parts.append(f"{len(restart_app)} needs full restart")
        if errors:
            parts.append(f"{len(errors)} errors")

        self.query_one("#config-footer", Static).update(
            "Saved: " + (", ".join(parts) or "nothing") + "   esc=close"
        )
        self._refresh_body()

        # Schedule apply_config_change for immediately applicable fields.
        applied_paths = list(live) + list(next_turn)
        if applied_paths and self._runtime is not None:
            runtime = self._runtime
            footer_widget = self.query_one("#config-footer", Static)

            async def _apply() -> None:
                outcome = await runtime.apply_config_change(applied_paths)
                msg_parts: list[str] = []
                if outcome.applied:
                    msg_parts.append(f"applied: {', '.join(outcome.applied)}")
                if outcome.needs_restart_lead:
                    msg_parts.append(
                        f"restart-lead: {', '.join(outcome.needs_restart_lead)}"
                    )
                if outcome.needs_restart_app:
                    msg_parts.append(
                        f"restart-app: {', '.join(outcome.needs_restart_app)}"
                    )
                footer_widget.update(
                    " | ".join(msg_parts) or "no changes applied"
                )

            self.app.run_worker(_apply(), exclusive=False)

    def action_confirm_self_repair(self) -> None:
        """Set the self_repair confirmed flag and re-invoke save."""

        self._self_repair_confirmed = True
        self.action_save()

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def action_diff(self) -> None:
        """Show a diff view of pending dirty edits and persisted overrides."""

        lines: list[str] = []
        for path, value in self._dirty.items():
            current = self._service.get(path).current
            lines.append(f"PENDING {path}: {current!r} → {value!r}")
        for path, (old, new) in sorted(self._service.diff().items()):
            lines.append(f"PERSISTED {path}: {old!r} → {new!r}")
        body = "\n".join(lines) or "(no overrides or pending edits)"
        self.query_one("#config-form", Static).update(body)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def action_reset(self) -> None:
        """Reset the focused field to its default (removes overlay)."""

        fields = self._fields_in_section()
        if not fields:
            return
        field = fields[self._active_field_index % len(fields)]
        self._dirty.pop(field.path, None)
        self._service.reset(field.path)
        self._refresh_body()
        self.query_one("#config-footer", Static).update(
            f"reset {field.path}   esc=close"
        )

    # ------------------------------------------------------------------
    # Close / Esc
    # ------------------------------------------------------------------

    def action_close(self) -> None:
        """Close the modal, requiring a second Esc when there are dirty fields."""

        if self._dirty and not self._confirm_close:
            self._confirm_close = True
            self.query_one("#config-footer", Static).update(
                f"{len(self._dirty)} dirty — press Esc again to discard"
            )
            return
        self.dismiss(None)

    def on_key(self, event: Any) -> None:  # type: ignore[override]
        """Reset the close-confirm flag on any non-Esc key press.

        Args:
            event: Textual ``Key`` event.
        """

        if getattr(event, "key", None) != "escape":
            self._confirm_close = False


__all__ = ("ConfigScreen",)
