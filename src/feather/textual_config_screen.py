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

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from feather.config_schema import (
    ConfigField,
    INHERIT_SENTINEL,
    REGISTRY,
    ReloadClass,
    Scope,
    WidgetHint,
    hint_for,
    lookup,
)
from feather.config_service import ConfigService
from feather.models_catalog import ModelCatalog, load_catalog


class _FocusableContainer(Container, can_focus=True):
    """Container root that accepts focus so screen bindings receive keys."""


class _ChoicePicker(OptionList):
    """Inline vertical picker widget for BOOLEAN toggles and DROPDOWN choices.

    Renders ``choices`` one per line; ↑/↓ (and ←/→ for symmetry with bool
    toggles) move the cursor and Textual's :class:`OptionList` scrolls
    the viewport automatically when the list exceeds ``max-height``.
    Enter posts a :class:`Picked` message containing the selected value;
    Esc bubbles up to :meth:`ConfigScreen.action_close`, which cancels
    the in-flight edit by removing this widget.

    Reused for both BOOLEAN fields (choices=``("false", "true")``) and
    DROPDOWN fields (choices=``field.enum or field.choices``) so the
    modal only has one code path for constrained input.

    We extend :class:`OptionList` rather than :class:`Static` because
    Static's renderable-measurement caps virtual-size at the widget's
    allotted height — so a 31-row list inside ``max-height: 16`` never
    triggers ``overflow-y: auto``, and the cursor (and its ``▶`` marker)
    silently drops below the visible region on long catalogs.
    OptionList scrolls the highlighted row into view as a built-in.

    Attributes:
        _choices: Tuple of selectable string values (in display order),
            preserved for tests and the controller's commit path.
    """

    BINDINGS = [
        # Left/right behave the same as up/down so bool toggles
        # (conceptually horizontal) feel natural and so arrow keys
        # don't bubble up to the screen's tab-switch bindings.
        Binding("left", "cursor_up", show=False),
        Binding("right", "cursor_down", show=False),
    ]

    DEFAULT_CSS = """
    _ChoicePicker {
        dock: bottom;
        height: auto;
        /* Cap viewport at 14 content rows + 2 border = 16; longer lists
           scroll, shorter lists collapse to their natural size. */
        max-height: 16;
        border: solid $accent;
        padding: 0 1;
    }
    """

    class Picked(Message):
        """Posted on Enter; carries the picked value as a string."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def __init__(
        self,
        *,
        choices: tuple[str, ...],
        current: str | None,
        id: str | None = None,
    ) -> None:
        if not choices:
            raise ValueError("_ChoicePicker requires at least one choice")
        # If ``current`` isn't in the suggested choices (e.g. a custom model
        # name set via /config set), prepend it so the cursor starts on the
        # existing value — otherwise the user could press Enter without
        # navigating and silently overwrite their own setting.
        if current is not None and current not in choices:
            choices = (current,) + choices
        self._choices: tuple[str, ...] = choices
        try:
            initial_index = choices.index(current) if current is not None else 0
        except ValueError:  # pragma: no cover — current is now guaranteed in choices
            initial_index = 0
        self._initial_index = initial_index
        super().__init__(*[Option(c) for c in choices], id=id)

    def on_mount(self) -> None:
        # Place the cursor on the initial selection. Setting ``highlighted``
        # also scrolls that row into view, which matters when the persisted
        # value lives below the visible viewport on long catalogs.
        self.highlighted = self._initial_index

    @property
    def _index(self) -> int:
        """Current cursor index — proxy onto :attr:`OptionList.highlighted`.

        Preserved as a public attribute name for the test suite, which
        asserts cursor position via ``picker._index``.
        """

        return self.highlighted if self.highlighted is not None else 0

    @_index.setter
    def _index(self, value: int) -> None:
        self.highlighted = value

    def _on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        """Forward OptionList's selection event as a :class:`Picked` message.

        The screen controller listens for :class:`_ChoicePicker.Picked`
        rather than the OptionList event so the rest of the modal does not
        need to know which underlying widget the picker is built from.
        """

        event.stop()
        self.post_message(self.Picked(self._choices[event.option_index]))


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
        Binding("escape", "close", "Close", priority=True),
        Binding("left", "prev_tab", "← tab", show=True),
        Binding("right", "next_tab", "→ tab", show=True),
        Binding("up", "section_prev", "↑ section", show=True),
        Binding("down", "section_next", "↓ section", show=True),
        Binding("enter,return", "edit_field", "Edit", show=True),
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
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $primary 10%;
    }
    /* Type-specific so the ChoicePicker (a Static) can still size itself
       via its own DEFAULT_CSS (height: auto, max-height: 12). Without the
       Input qualifier this rule forced the picker to 3 rows, cropping the
       second choice in the bool picker and clipping long dropdowns. */
    Input#config-inline-editor {
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
        model_catalog: ModelCatalog | None = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._runtime = runtime
        # Model capability catalog drives dynamic agent.model choices and
        # (in Commit 1) field gating. Loaded eagerly so we don't re-read
        # YAML on every picker open. Tests can inject a custom catalog.
        self._catalog: ModelCatalog = model_catalog or load_catalog(
            paths=service.paths
        )
        self._tabs = _discover_tabs(service)
        self._active_tab_index: int = 0
        self._active_section_index: int = 0
        self._active_field_index: int = 0
        self._dirty: dict[str, Any] = {}
        self._self_repair_confirmed: bool = False
        self._confirm_close: bool = False
        self._pending_edit: ConfigField | None = None
        self._apply_in_flight: bool = False
        # Saved footer text restored when the inline editor is dismissed.
        self._saved_footer_text: str = self._render_footer()

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Build the modal widget tree."""

        with _FocusableContainer(id="config-root"):
            yield Static(self._render_tab_bar(), id="config-tabs")
            with Horizontal(id="config-body"):
                yield Static(self._render_sidebar(), id="config-sidebar")
                yield Static(self._render_form(), id="config-form")
            yield Static(self._render_footer(), id="config-footer")

    def on_mount(self) -> None:
        """Focus the modal root so Enter/arrow/Esc bindings fire here, not in
        the background TUI's composer Input.

        ``Vertical(can_focus=True)`` makes ``#config-root`` a real focus
        target; without it, focus would stay on the background composer and
        modal keys would never arrive.
        """

        try:
            self.query_one("#config-root", Container).focus()
        except Exception:  # noqa: BLE001 — focus failure is non-fatal
            pass

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_tab_bar(self) -> str:
        """Render the top tab bar with the active tab highlighted."""

        active = self._active_tab_index
        return "   ".join(
            f"[reverse]{tab.label}[/reverse]" if i == active else tab.label
            for i, tab in enumerate(self._tabs)
        )

    @staticmethod
    def _section_label(path: str, prefix: str) -> str:
        """Return the synthetic subsection label for ``path`` under ``prefix``.

        Leaves directly under the prefix (no further dot) collapse to the
        synthetic label ``"agent"``; nested entries return the first
        dotted segment after the prefix.
        """

        tail = path[len(prefix):]
        return tail.split(".", 1)[0] if "." in tail else "agent"

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
            label = self._section_label(f.path, prefix)
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
        active = self._active_section_index
        return "\n".join(
            f"{'▶' if i == active else ' '} {name}"
            for i, name in enumerate(sections)
        )

    def _render_form(self) -> str:
        """Render the right-hand form rows for the active subsection.

        Each row carries a ``[N/A]`` badge for fields the currently-resolved
        model doesn't accept, plus a ``[DIRTY]`` badge for pending edits.
        """

        if not self._subsections():
            return "(no fields)"
        fields = self._fields_in_section()
        rows: list[str] = []
        for idx, f in enumerate(fields):
            cv = self._service.get(f.path)
            badge_src = f"[{cv.source.value}]"
            badge_rl = f"[{f.reload.value}]"
            dirty_badge = " [DIRTY]" if f.path in self._dirty else ""
            editable, _ = self._can_edit_field(f)
            na_badge = "" if editable else " [N/A]"
            cursor_marker = "▶" if idx == (self._active_field_index % max(1, len(fields))) else " "
            current = self._dirty.get(f.path, cv.current)
            # Render None (or the queued inherit sentinel) for agent
            # provider/model fields as "(inherit)" so the form line tells
            # the user what's actually going to happen — instead of the
            # confusing `None`.
            display = self._display_value(f, current)
            rows.append(
                f"{cursor_marker} {f.path}   {badge_src}{dirty_badge}{na_badge}   {badge_rl}\n"
                f"   ▸ {display}\n"
                f"   {f.description}"
            )
        return "\n\n".join(rows) or "(no fields)"

    def _display_value(self, field: ConfigField, current: Any) -> str:
        """Pretty-print ``current`` for a form row.

        Fields that support an "inherit from layer below" semantics
        (agent provider/model, memory.operations.<op> provider/model)
        render ``None`` and the pending ``INHERIT_SENTINEL`` as
        ``(inherit)`` so the row reads unambiguously instead of the
        confusing ``None``. Everything else falls back to ``repr()``.
        """

        if current == INHERIT_SENTINEL:
            return INHERIT_SENTINEL
        if current is None and self._supports_inherit(field.path):
            return INHERIT_SENTINEL
        return repr(current)

    @staticmethod
    def _supports_inherit(path: str) -> bool:
        """Whether ``path`` semantically supports an "inherit" value of None."""

        if path.startswith("agents.") and (
            path.endswith(".provider") or path.endswith(".model")
        ):
            return True
        if path.startswith("app.memory.operations.") and (
            path.endswith(".provider") or path.endswith(".model")
        ):
            return True
        return False

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
        return [
            f
            for f in REGISTRY
            if f.path.startswith(prefix)
            and self._section_label(f.path, prefix) == active_section
        ]

    def _refresh_body(self) -> None:
        """Re-render sidebar and form in place."""

        self.query_one("#config-sidebar", Static).update(self._render_sidebar())
        self.query_one("#config-form", Static).update(self._render_form())

    def _set_footer(self, text: str) -> None:
        """Update the footer status line in place."""

        self.query_one("#config-footer", Static).update(text)

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

    async def action_edit_field(self) -> None:
        """Open the right editor widget for the focused field.

        Dispatches by ``field.widget`` so each type gets a constrained
        widget that makes invalid values impossible to enter:

        * TOGGLE → inline :class:`_ChoicePicker` with ``("false", "true")``
        * DROPDOWN → inline :class:`_ChoicePicker` populated from
          ``field.enum`` (strict) or ``field.choices`` (suggestions)
        * NUMERIC / TEXT → free-text :class:`~textual.widgets.Input` with a
          range hint shown in the placeholder
        * SENSITIVE_READONLY → refused; footer explains the env-var indirection
        * LIST_EDITOR → refused; footer points at ``/config set``

        Async because Textual's ``mount()`` returns an ``AwaitMount`` that
        must be awaited before the new widget is in the DOM — calling
        ``.focus()`` synchronously after a sync ``mount()`` focuses an
        un-mounted widget and the user's keystrokes then hit the screen
        bindings instead of the editor.
        """

        fields = self._fields_in_section()
        if not fields:
            return
        field = fields[self._active_field_index % len(fields)]
        # Don't open a second editor if one is already mounted.
        if self.query("#config-inline-editor"):
            return

        # Refuse editing for read-only / not-yet-supported widget types.
        if field.widget is WidgetHint.SENSITIVE_READONLY:
            self._set_footer(
                f"{field.path} is sensitive — set via the env var "
                f"(value here is the env-var name, not the secret itself)."
            )
            return
        if field.widget is WidgetHint.LIST_EDITOR:
            self._set_footer(
                f"{field.path}: list editing not yet in modal — use "
                f"/config set {field.path} a,b,c"
            )
            return

        # Capability gating: refuse fields the resolved model doesn't accept.
        editable, reason = self._can_edit_field(field)
        if not editable:
            self._set_footer(f"N/A: {reason}")
            return

        self._pending_edit = field
        footer_static = self.query_one("#config-footer", Static)
        self._saved_footer_text = self._render_footer()
        config_root = self.query_one("#config-root", Container)

        if field.widget in (WidgetHint.TOGGLE, WidgetHint.DROPDOWN):
            choices = self._picker_choices_for(field)
            current = self._dirty.get(field.path, self._service.get(field.path).current)
            if field.widget is WidgetHint.TOGGLE:
                current_str = self._bool_to_choice(current)
            elif current == INHERIT_SENTINEL or (
                current is None and self._supports_inherit(field.path)
            ):
                # None or pending inherit on an inherit-supporting field →
                # land cursor on the sentinel option so Enter without
                # navigating keeps "inherit".
                current_str = INHERIT_SENTINEL
            elif current is None:
                # Non-inherit DROPDOWN with no current value: leave cursor
                # at the top so the user picks explicitly.
                current_str = ""
            else:
                current_str = str(current)
            picker = _ChoicePicker(
                choices=choices,
                current=current_str,
                id="config-inline-editor",
            )
            footer_static.update(
                f"editing {field.path}  ↑↓=choose  Enter=save  Esc=cancel"
            )
            footer_static.display = False
            await config_root.mount(picker)
            self.call_after_refresh(picker.focus)
            return

        # Default: free-text Input. NUMERIC and TEXT both land here; the
        # hint differentiates ranges/help in the placeholder. For
        # temperature fields, the model's catalog-declared range takes
        # precedence over the generic validator-derived hint — gpt-4o
        # accepts 0-2 while claude-opus-4-7 caps at 1.
        placeholder = f"new value for {field.path}"
        hint = self._model_aware_hint(field) or hint_for(field)
        if hint:
            placeholder += f"  ({hint})"
        editor = Input(placeholder=placeholder, id="config-inline-editor")
        footer_static.update(f"editing {field.path}  Enter=save  Esc=cancel")
        footer_static.display = False
        await config_root.mount(editor)
        # Defer focus to the next refresh so the Input is fully wired into
        # the focus chain before claiming focus.
        self.call_after_refresh(editor.focus)

    @staticmethod
    def _bool_to_choice(value: Any) -> str:
        """Coerce a boolean (or stringy) value to the picker's display string."""

        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value).strip().lower()
        return "true" if text in ("true", "yes", "on", "1") else "false"

    def _picker_choices_for(self, field: ConfigField) -> tuple[str, ...]:
        """Return the picker's display choices for ``field``.

        Resolution order:

        * ``TOGGLE`` → always ``("false", "true")``.
        * ``agents.<name>.model`` → ``(INHERIT_SENTINEL,) + catalog.slugs_for(resolved_provider)``.
          The agent's resolved provider is its own ``provider`` field if
          set (or pending in dirty), else ``app.active_provider``.
        * ``app.memory.operations.<op>.model`` → same shape, but the
          resolved provider is the op's own ``.provider`` (dirty > persisted)
          falling back to ``app.active_provider``.
        * Other ``DROPDOWN`` → ``field.enum`` (strict) or ``field.choices``.
        """

        if field.widget is WidgetHint.TOGGLE:
            return ("false", "true")
        if self._is_agent_model_field(field.path):
            agent_name = field.path.split(".")[1]
            provider = self._resolved_agent_provider(agent_name)
            catalog_key = self._provider_to_catalog_key(provider)
            return (INHERIT_SENTINEL, *self._catalog.slugs_for(catalog_key))
        if self._is_op_model_field(field.path):
            op_name = field.path.split(".")[3]
            provider = self._resolved_op_provider(op_name)
            catalog_key = self._provider_to_catalog_key(provider)
            return (INHERIT_SENTINEL, *self._catalog.slugs_for(catalog_key))
        return field.enum or field.choices or ()

    @staticmethod
    def _is_op_model_field(path: str) -> bool:
        """Match ``app.memory.operations.<op>.model`` exactly."""

        parts = path.split(".")
        return (
            len(parts) == 5
            and parts[0] == "app"
            and parts[1] == "memory"
            and parts[2] == "operations"
            and parts[4] == "model"
        )

    def _resolved_op_provider(self, op_name: str) -> str:
        """Return the provider for ``app.memory.operations.<op_name>``.

        Order: dirty edit on ``.provider`` → persisted → app.active_provider.
        Inherit-sentinel dirty bypasses persisted (same reasoning as
        :meth:`_resolved_agent_provider`).
        """

        provider_path = f"app.memory.operations.{op_name}.provider"
        dirty = self._dirty.get(provider_path)
        if dirty == INHERIT_SENTINEL:
            return self._active_provider_fallback()
        if dirty:
            return str(dirty).strip().lower()
        try:
            persisted = self._service.get(provider_path).current
        except KeyError:
            persisted = None
        if persisted:
            return str(persisted).strip().lower()
        return self._active_provider_fallback()

    @staticmethod
    def _is_agent_model_field(path: str) -> bool:
        """Match ``agents.<name>.model`` exactly (excludes ``temperature`` etc)."""

        parts = path.split(".")
        return (
            len(parts) == 3
            and parts[0] == "agents"
            and parts[2] == "model"
        )

    def _resolved_agent_provider(self, agent_name: str) -> str:
        """Return the provider this agent will use after resolution.

        Order:
          * Pending dirty edit on ``agents.<name>.provider``:
            - the inherit sentinel BYPASSES the persisted layer entirely
              and goes straight to ``app.active_provider`` — otherwise
              switching an agent to (inherit) inside the modal would
              still see the stale persisted provider until save.
            - any other value wins.
          * Persisted ``agents.<name>.provider``.
          * ``app.active_provider`` fallback.
        """

        provider_path = f"agents.{agent_name}.provider"
        dirty = self._dirty.get(provider_path)
        if dirty == INHERIT_SENTINEL:
            return self._active_provider_fallback()
        if dirty:
            return str(dirty).strip().lower()
        try:
            persisted = self._service.get(provider_path).current
        except KeyError:
            persisted = None
        if persisted:
            return str(persisted).strip().lower()
        return self._active_provider_fallback()

    def _active_provider_fallback(self) -> str:
        """Read ``app.active_provider`` defensively."""

        try:
            return (
                str(self._service.get("app.active_provider").current).strip().lower()
            )
        except KeyError:
            return "openai"

    @staticmethod
    def _provider_to_catalog_key(provider: str) -> str:
        """Map an app-level provider key to the metadata catalog's vendor key.

        ``app.active_provider`` uses ``claude`` for Anthropic; the metadata
        catalog uses ``anthropic`` to match the SDK's vendor name.
        """

        return "anthropic" if provider == "claude" else provider

    # ------------------------------------------------------------------
    # Capability-driven field gating
    # ------------------------------------------------------------------

    def _resolve_field_capability(
        self, field: ConfigField
    ) -> Any:  # ModelCapability | None
        """Return the :class:`~feather.models_catalog.ModelCapability` that
        applies to ``field``, or ``None`` when the field isn't model-scoped.

        Lookup rules:

        * ``app.<openai|openrouter|claude>.<knob>`` → capability of the model
          currently configured at ``app.<provider>.model``.
        * ``agents.<name>.<knob>`` → capability of the agent's resolved model:
          its own ``agents.<name>.model`` (dirty edit wins over persisted),
          falling back to the resolved provider's ``app.<provider>.model``
          if the agent inherits.
        * Anything else (paths, log levels, scheduler knobs, etc.) → ``None``
          (no model-scoped capability means no gating).
        """

        path = field.path
        if path.startswith("app."):
            parts = path.split(".")
            if len(parts) >= 3 and parts[1] in ("openai", "openrouter", "claude"):
                provider = parts[1]
                # Honour pending dirty edits on app.<provider>.model so a
                # user switching the model in the same modal session sees
                # capability gating update immediately (otherwise dirtying
                # gpt-4o would still leave temperature [N/A] while the
                # persisted gpt-5-mini still wins).
                model = self._resolved_app_model(provider)
                if model:
                    return self._catalog.capability(
                        self._provider_to_catalog_key(provider), str(model)
                    )
            return None
        if path.startswith("agents."):
            parts = path.split(".")
            if len(parts) < 3:
                return None
            agent_name = parts[1]
            provider = self._resolved_agent_provider(agent_name)
            catalog_key = self._provider_to_catalog_key(provider)
            model = self._resolved_agent_model(agent_name, provider)
            if model:
                return self._catalog.capability(catalog_key, str(model))
        return None

    def _resolved_agent_model(self, agent_name: str, provider: str) -> str | None:
        """Resolve which model this agent will use.

        Order: dirty edit on ``agents.<name>.model`` → persisted →
        ``app.<provider>.model`` inherit fallback. Inherit sentinel
        bypasses persisted and goes straight to the provider default —
        otherwise dirtying (inherit) while a persisted model still exists
        would resolve to the stale persisted slug until save.

        The provider-level fallback ALSO honours a pending dirty edit on
        ``app.<provider>.model`` so that switching the app model and
        re-evaluating an inheriting agent's capability happens in one
        modal session.
        """

        model_path = f"agents.{agent_name}.model"
        dirty = self._dirty.get(model_path)
        if dirty == INHERIT_SENTINEL:
            return self._resolved_app_model(provider)
        if dirty:
            return str(dirty)
        try:
            persisted = self._service.get(model_path).current
        except KeyError:
            persisted = None
        if persisted:
            return str(persisted)
        return self._resolved_app_model(provider)

    def _resolved_app_model(self, provider: str) -> str | None:
        """Return ``app.<provider>.model`` honouring pending dirty edits.

        Used by every capability lookup whose answer depends on which
        model the provider is currently pointing at — including dirty
        edits that haven't been saved yet.
        """

        app_model_path = f"app.{provider}.model"
        dirty = self._dirty.get(app_model_path)
        if dirty and dirty != INHERIT_SENTINEL:
            return str(dirty)
        try:
            persisted = self._service.get(app_model_path).current
        except KeyError:
            return None
        return str(persisted) if persisted else None

    def _can_edit_field(self, field: ConfigField) -> tuple[bool, str | None]:
        """Return ``(editable, reason)``. When not editable, ``reason``
        is a one-liner suitable for the footer."""

        cap = self._resolve_field_capability(field)
        if cap is None:
            return (True, None)
        if self._catalog.can_edit(field.path, cap):
            return (True, None)
        return (False, self._explain_disabled(field.path, cap))

    def _model_aware_hint(self, field: ConfigField) -> str | None:
        """Return a hint derived from the resolved model's capability, or
        ``None`` when the field isn't model-scoped or the model imposes no
        tighter constraint than the validator already does.

        Currently only :attr:`ModelCapability.temperature_range` overrides
        the generic ``hint_for`` value — chat models accept 0.0-2.0,
        Claude caps at 1.0, mistral at 1.5.
        """

        if not field.path.endswith(".temperature"):
            return None
        cap = self._resolve_field_capability(field)
        if cap is None or cap.temperature_range is None:
            return None
        low, high = cap.temperature_range
        return f"{low}-{high}"

    @staticmethod
    def _explain_disabled(path: str, cap: Any) -> str:
        """Short human reason why ``path`` is N/A for ``cap``."""

        slug = getattr(cap, "slug", "this model")
        if path.endswith(".temperature"):
            return f"{slug} ignores temperature"
        if path.endswith(".parallel_tool_calls"):
            return f"{slug} does not support parallel tool calls"
        if path.endswith(".reasoning.effort"):
            return f"{slug} does not accept reasoning.effort"
        if path.endswith(".reasoning.summary"):
            return f"{slug} does not accept reasoning.summary"
        if path.endswith(".thinking.budget_tokens"):
            return f"{slug} is adaptive-only; budget_tokens not applicable"
        if path.endswith(".thinking.type"):
            return f"{slug} does not support extended thinking"
        return f"{slug} does not support this field"

    @on(_ChoicePicker.Picked)
    def _handle_choice_picked(self, event: _ChoicePicker.Picked) -> None:
        """Handle a value selected via the inline :class:`_ChoicePicker`.

        Bound via ``@on`` rather than ``on_<message>`` so the handler name
        doesn't have to match Textual's camel-to-snake convention for a
        leading-underscore class name (which would otherwise produce
        ``on__choice_picker_picked`` with a double underscore).

        Mirrors :meth:`on_input_submitted` but skips the Input-specific
        plumbing. Validates the picked value through the service, marks the
        field dirty on success, and restores the footer in all cases.

        The ``INHERIT_SENTINEL`` is treated as a special "clear the
        overlay" instruction — stored in ``_dirty`` as the literal
        sentinel string so :meth:`action_save` can call
        :meth:`ConfigService.reset` for it instead of ``set``.
        """

        field = self._pending_edit
        try:
            picker = self.query_one("#config-inline-editor", _ChoicePicker)
        except Exception:  # noqa: BLE001 — picker may already be gone
            picker = None
        if field is None:
            if picker is not None:
                picker.remove()
            self._restore_footer()
            return

        # Inherit sentinel: stage the reset without going through validate
        # (the empty value would fail enum-strict validation, and the
        # sentinel string isn't a real config value anyway). Restricted
        # to fields that semantically support inherit — otherwise a
        # future schema mistake that lets the sentinel into a strict-enum
        # field's choices would silently delete the overlay key.
        if event.value == INHERIT_SENTINEL and self._supports_inherit(field.path):
            if picker is not None:
                picker.remove()
            self._dirty[field.path] = INHERIT_SENTINEL
            self._pending_edit = None
            self._restore_footer()
            self._refresh_body()
            self._set_footer(self._render_footer())
            return

        validate = self._service.validate(field.path, event.value)
        if picker is not None:
            picker.remove()
        if not validate.ok:
            self._pending_edit = None
            self._restore_footer()
            self._set_footer(f"INVALID: {validate.error}   esc=cancel")
            return

        self._dirty[field.path] = validate.coerced
        self._pending_edit = None
        self._restore_footer()
        self._refresh_body()
        self._set_footer(self._render_footer())

    def _restore_footer(self) -> None:
        """Re-show the footer Static after the inline editor is dismissed."""

        footer_static = self.query_one("#config-footer", Static)
        footer_static.display = True
        footer_static.update(self._saved_footer_text)

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
            self._restore_footer()
            return

        validate = self._service.validate(field.path, event.value)
        if not validate.ok:
            event.input.remove()
            self._pending_edit = None
            self._restore_footer()
            self._set_footer(f"INVALID: {validate.error}   esc=cancel")
            return

        self._dirty[field.path] = validate.coerced
        self._pending_edit = None
        event.input.remove()
        self._restore_footer()
        self._refresh_body()
        self._set_footer(self._render_footer())

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def action_save(self) -> None:
        """Write all dirty fields through ConfigService and show a banner."""

        if self._apply_in_flight:
            self._set_footer("apply in progress — wait for previous save to complete")
            return

        if not self._dirty:
            self._set_footer("no dirty fields  esc=close")
            return

        # self_repair.enabled requires a y-confirm first.
        if "app.self_repair.enabled" in self._dirty and not self._self_repair_confirmed:
            self._set_footer(
                "self_repair.enabled change is RESTART-APP. Press 'y' to confirm, esc to cancel."
            )
            return

        buckets: dict[ReloadClass, list[str]] = {rc: [] for rc in ReloadClass}
        errors: list[str] = []

        for path, value in list(self._dirty.items()):
            force = path == "app.self_repair.enabled"
            if value == INHERIT_SENTINEL:
                # Inherit sentinel maps to "remove this key from the
                # overlay so the layer below provides the value". For
                # agent fields, that layer is the agent YAML's default
                # (which the loader treats null as "inherit from app");
                # for app fields it's the packaged default.
                result = self._service.reset(path)
            else:
                result = self._service.set(path, value, force=force)
            if not result.ok:
                errors.append(f"{path}: {result.error}")
                continue
            del self._dirty[path]
            field_def = lookup(path)
            if field_def is not None:
                buckets[field_def.reload].append(path)

        # Banner segments in deterministic display order.
        labels: list[tuple[ReloadClass, str]] = [
            (ReloadClass.LIVE, "live"),
            (ReloadClass.NEXT_TURN, "next-turn"),
            (ReloadClass.RESTART_LEAD, "needs restart-lead"),
            (ReloadClass.RESTART_APP, "needs full restart"),
        ]
        parts = [f"{len(buckets[rc])} {label}" for rc, label in labels if buckets[rc]]
        if errors:
            parts.append(f"{len(errors)} errors")

        self._set_footer(
            "Saved: " + (", ".join(parts) or "nothing") + "   esc=close"
        )
        self._refresh_body()

        # Schedule apply_config_change for immediately applicable fields.
        applied_paths = buckets[ReloadClass.LIVE] + buckets[ReloadClass.NEXT_TURN]
        if applied_paths and self._runtime is not None:
            self._apply_in_flight = True
            runtime = self._runtime

            async def _apply() -> None:
                try:
                    try:
                        outcome = await runtime.apply_config_change(applied_paths)
                    except Exception as exc:  # noqa: BLE001 - surface apply errors to user
                        self._set_footer(f"apply error: {type(exc).__name__}: {exc}")
                        return
                    sections: list[tuple[str, list[str]]] = [
                        ("applied", outcome.applied),
                        ("restart-lead", outcome.needs_restart_lead),
                        ("restart-app", outcome.needs_restart_app),
                    ]
                    msg_parts = [
                        f"{label}: {', '.join(paths)}"
                        for label, paths in sections
                        if paths
                    ]
                    self._set_footer(" | ".join(msg_parts) or "no changes applied")
                finally:
                    self._apply_in_flight = False

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
        self._set_footer(f"reset {field.path}   esc=close")

    # ------------------------------------------------------------------
    # Close / Esc
    # ------------------------------------------------------------------

    def action_close(self) -> None:
        """Close the modal, requiring a second Esc when there are dirty fields.

        If an inline edit is in progress, the first Esc cancels the edit
        (removes the Input widget and clears ``_pending_edit``) and returns
        without dismissing the modal.  The second Esc then follows the normal
        dirty-confirm + dismiss path.
        """

        # Cancel an in-flight inline edit first, if any. The editor widget is
        # either a free-text Input or a _ChoicePicker — query by id (not type)
        # so both are removed identically.
        if self._pending_edit is not None:
            existing = self.query("#config-inline-editor")
            for widget in existing:
                widget.remove()
            self._pending_edit = None
            self._restore_footer()
            self._set_footer("edit cancelled  esc=close")
            return

        if self._dirty and not self._confirm_close:
            self._confirm_close = True
            self._set_footer(
                f"{len(self._dirty)} dirty — press Esc again to discard"
            )
            return
        self.dismiss(None)

    async def on_key(self, event: Any) -> None:  # type: ignore[override]
        """Catch Enter/Return as a fallback + reset close-confirm flag.

        Some terminals + Textual focus states do not deliver Enter to the
        screen's BINDINGS table — explicitly handle it here so editing always
        works, while Enter inside the inline ``Input`` widget continues to
        flow to ``on_input_submitted`` (the Input has focus and consumes the
        key before this handler fires).
        """

        key = getattr(event, "key", None)
        if key != "escape":
            self._confirm_close = False
        if key in ("enter", "return") and self._pending_edit is None:
            await self.action_edit_field()
            try:
                event.stop()
            except Exception:  # noqa: BLE001
                pass


__all__ = ("ConfigScreen",)
