# Phase 2 — Config Modal (Textual UI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the interactive Textual modal that wraps the Phase 1 `ConfigService` — top tabs for `App` and `Lead`, sidebar of subsections, scrollable form, per-class save banner, restart-lead prompt, and the `self_repair.enabled` force-confirm carve-out.

**Architecture:** A new `feather.textual_config_screen.ConfigScreen(ModalScreen)` pushed by the `/config` slash handler. The modal owns no business logic — every read, validate, write, and apply flows through the existing `ConfigService` and `FeatherRuntime` methods. Tabs and sections are derived from the registry at mount time.

**Tech Stack:** Textual 0.x (already a project dep), Python 3.12+, pytest with Textual's pilot harness.

**Worktree:** Same as Phase 0/1 — `/home/dev/feather_v2/.worktrees/config-tui` on `feature/config-tui`. Phase 1 must be merged or complete before starting Phase 2.

**Workflow reminder:** Each task is TDD. Phase wraps with simplify + red-team review (Tasks 14–15).

---

### Task 1: ConfigScreen skeleton — mount, top tabs, footer

**Files:**
- Create: `src/feather/textual_config_screen.py`
- Create: `tests/test_textual_config_screen.py`

- [ ] **Step 1: Failing test — modal mounts and shows the App tab**

```python
"""Tests for the /config Textual modal."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App
from textual.pilot import Pilot

from feather.config import load_app_config
from feather.config_service import ConfigService
from feather.paths import FeatherPaths
from feather.textual_config_screen import ConfigScreen


class _Host(App):
    def __init__(self, service: ConfigService) -> None:
        super().__init__()
        self._service = service

    async def on_mount(self) -> None:
        await self.push_screen(ConfigScreen(service=self._service))


@pytest.fixture
def service(tmp_path: Path) -> ConfigService:
    paths = FeatherPaths(project_root=tmp_path / "proj", home=tmp_path / "global")
    paths.ensure_global_dirs()
    paths.ensure_project_dirs()
    cfg = load_app_config(paths.project_root, paths=paths)
    return ConfigService(paths=paths, app_config=cfg)


async def test_modal_mounts_and_shows_app_tab(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        tabs = pilot.app.query_one("#config-tabs")
        assert "App" in str(tabs.render())
        assert "Lead" in str(tabs.render())


async def test_modal_footer_shows_keybindings(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        footer = pilot.app.query_one("#config-footer")
        body = str(footer.render())
        for keyword in ("save", "diff", "reset", "esc"):
            assert keyword in body.lower(), f"footer missing {keyword!r}"
```

- [ ] **Step 2: Run — expect ImportError**

Run: `uv run pytest tests/test_textual_config_screen.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the skeleton**

```python
"""Textual modal for /config — interactive config editor.

The modal owns no business logic; every read/write call routes
through ``feather.config_service.ConfigService``.

Layout:

  ┌── /config ────────────────────────────────────────────┐
  │ [App]   Lead                                          │ ← #config-tabs
  ├──────────────┬────────────────────────────────────────┤
  │ subsection 1 │ (form rows)                            │
  │▶subsection 2 │                                        │
  │ subsection 3 │                                        │
  ├──────────────┴────────────────────────────────────────┤
  │ <status>      s=save d=diff r=reset esc=close          │ ← #config-footer
  └────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from feather.config_schema import REGISTRY, Scope
from feather.config_service import ConfigService


@dataclass(slots=True, frozen=True)
class _TabSpec:
    label: str
    section_prefix: str  # e.g. "app." or "agents.Lead."


def _discover_tabs(service: ConfigService) -> list[_TabSpec]:
    """Build the tab list from the registry.

    App tab is always first. Each unique agent name in the registry
    produces one trailing tab.
    """

    tabs = [_TabSpec(label="App", section_prefix="app.")]
    agent_names: list[str] = []
    for f in REGISTRY:
        if f.scope is Scope.AGENT:
            name = f.path.split(".")[1]
            if name not in agent_names:
                agent_names.append(name)
    for name in agent_names:
        tabs.append(_TabSpec(label=name, section_prefix=f"agents.{name}."))
    return tabs


class ConfigScreen(ModalScreen[None]):
    """Modal screen for editing Feather configuration."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("left", "prev_tab", "Prev tab"),
        Binding("right", "next_tab", "Next tab"),
        Binding("s", "save", "Save"),
        Binding("d", "diff", "Diff"),
        Binding("r", "reset", "Reset"),
    ]

    DEFAULT_CSS = """
    ConfigScreen {
        align: center middle;
    }
    #config-root {
        width: 90%;
        height: 90%;
        border: round $accent;
    }
    #config-tabs {
        height: 1;
        padding: 0 1;
        background: $primary 10%;
    }
    #config-body {
        height: 1fr;
    }
    #config-footer {
        height: 1;
        padding: 0 1;
        background: $primary 10%;
    }
    """

    def __init__(self, *, service: ConfigService) -> None:
        super().__init__()
        self._service = service
        self._tabs = _discover_tabs(service)
        self._active_tab_index = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="config-root"):
            yield Static(self._render_tab_bar(), id="config-tabs")
            with VerticalScroll(id="config-body"):
                yield Static(self._render_body(), id="config-body-content")
            yield Static(
                "0 dirty   s=save  d=diff  r=reset  esc=close", id="config-footer"
            )

    def _render_tab_bar(self) -> str:
        parts: list[str] = []
        for i, tab in enumerate(self._tabs):
            label = tab.label
            parts.append(f"[reverse]{label}[/reverse]" if i == self._active_tab_index else label)
        return "   ".join(parts)

    def _render_body(self) -> str:
        # Phase 2.1 — placeholder. Task 2 swaps this for the sidebar+form.
        prefix = self._tabs[self._active_tab_index].section_prefix
        return f"{prefix} fields will render here."

    def action_close(self) -> None:
        self.dismiss(None)

    def action_prev_tab(self) -> None:
        self._active_tab_index = (self._active_tab_index - 1) % len(self._tabs)
        self.query_one("#config-tabs", Static).update(self._render_tab_bar())
        self.query_one("#config-body-content", Static).update(self._render_body())

    def action_next_tab(self) -> None:
        self._active_tab_index = (self._active_tab_index + 1) % len(self._tabs)
        self.query_one("#config-tabs", Static).update(self._render_tab_bar())
        self.query_one("#config-body-content", Static).update(self._render_body())

    def action_save(self) -> None: ...  # Task 6
    def action_diff(self) -> None: ...  # Task 9
    def action_reset(self) -> None: ...  # Task 10
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_textual_config_screen.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_textual_config_screen.py src/feather/textual_config_screen.py
git commit -m "Add ConfigScreen modal skeleton with tab bar + footer"
```

---

### Task 2: Tab cycling via arrow keys

**Files:**
- Modify: `tests/test_textual_config_screen.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_textual_config_screen.py`:

```python
async def test_arrow_right_cycles_tabs(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        tabs = pilot.app.query_one("#config-tabs", Static)
        assert "[reverse]App[/reverse]" in str(tabs.render())

        await pilot.press("right")

        assert "[reverse]Lead[/reverse]" in str(tabs.render())

        await pilot.press("right")  # wraps back to App

        assert "[reverse]App[/reverse]" in str(tabs.render())


async def test_arrow_left_cycles_backwards(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        tabs = pilot.app.query_one("#config-tabs", Static)

        await pilot.press("left")  # wraps to Lead

        assert "[reverse]Lead[/reverse]" in str(tabs.render())
```

Add to the imports: `from textual.widgets import Static`.

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_textual_config_screen.py -v`
Expected: green (the skeleton already cycles).

- [ ] **Step 3: Commit if any minor fixes**

```bash
git add tests/test_textual_config_screen.py
git commit -m "Cover tab cycling via arrow keys"
```

---

### Task 3: Sidebar + form layout inside the active tab

**Files:**
- Modify: `src/feather/textual_config_screen.py`
- Modify: `tests/test_textual_config_screen.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_textual_config_screen.py`:

```python
async def test_app_tab_shows_subsection_sidebar(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        sidebar = pilot.app.query_one("#config-sidebar")
        body = str(sidebar.render())
        # All app.<top-level> sections should appear.
        for expected in (
            "compaction", "scheduler", "self_repair",
            "openai", "openrouter", "claude", "parallel", "memory",
        ):
            assert expected in body, f"sidebar missing {expected!r}"


async def test_lead_tab_sidebar_is_minimal(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        await pilot.press("right")  # Lead tab
        sidebar = pilot.app.query_one("#config-sidebar")
        body = str(sidebar.render()).lower()
        # The Lead tab groups under "agent" and "reasoning".
        assert "reasoning" in body
        assert "agent" in body
```

- [ ] **Step 2: Run — expect failure (no sidebar yet)**

Run: `uv run pytest tests/test_textual_config_screen.py -v`
Expected: 2 new failures.

- [ ] **Step 3: Implement sidebar + form layout**

Replace `compose` and add helpers:

```python
    def compose(self) -> ComposeResult:
        with Vertical(id="config-root"):
            yield Static(self._render_tab_bar(), id="config-tabs")
            with Horizontal(id="config-body"):
                yield Static(self._render_sidebar(), id="config-sidebar")
                yield Static(self._render_form(), id="config-form")
            yield Static(self._render_footer(), id="config-footer")

    def _subsections(self) -> list[str]:
        """Return unique subsection labels under the active tab.

        For ``app.*``: returns top-level child segments (compaction,
        scheduler, openai, …).
        For ``agents.<name>.*``: returns top-level subsegments under
        the agent (e.g. ``reasoning``) plus a synthetic ``agent`` for
        leaves directly under the agent root.
        """

        prefix = self._tabs[self._active_tab_index].section_prefix
        out: list[str] = []
        for f in REGISTRY:
            if not f.path.startswith(prefix):
                continue
            tail = f.path[len(prefix):]
            head = tail.split(".", 1)[0]
            # When the leaf sits directly under the prefix, lump it as "agent"
            # (only relevant on the agent tabs where leaves like `personality`
            # have no second segment).
            if "." in tail:
                label = head
            else:
                label = "agent"
            if label not in out:
                out.append(label)
        return out

    def _render_sidebar(self) -> str:
        sections = self._subsections()
        if not sections:
            return "(no fields)"
        cursor = getattr(self, "_active_section_index", 0)
        self._active_section_index = cursor % max(1, len(sections))
        parts: list[str] = []
        for i, name in enumerate(sections):
            marker = "▶" if i == self._active_section_index else " "
            parts.append(f"{marker} {name}")
        return "\n".join(parts)

    def _render_form(self) -> str:
        sections = self._subsections()
        if not sections:
            return "(no fields)"
        prefix = self._tabs[self._active_tab_index].section_prefix
        active_section = sections[self._active_section_index]
        rows: list[str] = []
        for f in REGISTRY:
            if not f.path.startswith(prefix):
                continue
            tail = f.path[len(prefix):]
            section = tail.split(".", 1)[0] if "." in tail else "agent"
            if section != active_section:
                continue
            value = self._service.get(f.path)
            badge_src = f"[{value.source.value}]"
            badge_rl = f"[{f.reload.value}]"
            rows.append(
                f"{f.path}   {badge_src}   {badge_rl}\n"
                f"   ▸ {value.current!r}\n"
                f"   {f.description}"
            )
        return "\n\n".join(rows) or "(no fields)"

    def _render_footer(self) -> str:
        dirty = len(getattr(self, "_dirty", {}))
        return f"{dirty} dirty   s=save  d=diff  r=reset  esc=close"
```

Initialise `_dirty` in `__init__`:

```python
        self._dirty: dict[str, object] = {}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_textual_config_screen.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_textual_config_screen.py src/feather/textual_config_screen.py
git commit -m "ConfigScreen: subsection sidebar + per-section form rendering"
```

---

### Task 4: Sidebar cursor via up/down

**Files:**
- Modify: `src/feather/textual_config_screen.py`
- Modify: `tests/test_textual_config_screen.py`

- [ ] **Step 1: Failing test**

```python
async def test_arrow_down_moves_section_cursor(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        sidebar = pilot.app.query_one("#config-sidebar", Static)
        before = str(sidebar.render()).split("\n")
        first_active = before.index(next(line for line in before if line.startswith("▶")))

        await pilot.press("down")

        after = str(sidebar.render()).split("\n")
        new_active = after.index(next(line for line in after if line.startswith("▶")))
        assert new_active == first_active + 1
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_textual_config_screen.py -v`
Expected: FAIL — no Up/Down binding.

- [ ] **Step 3: Add Up/Down bindings + actions**

Add to `BINDINGS`:

```python
        Binding("up", "section_prev", "Prev section"),
        Binding("down", "section_next", "Next section"),
```

Add the actions:

```python
    def action_section_prev(self) -> None:
        sections = self._subsections()
        if sections:
            self._active_section_index = (self._active_section_index - 1) % len(sections)
            self._refresh_body()

    def action_section_next(self) -> None:
        sections = self._subsections()
        if sections:
            self._active_section_index = (self._active_section_index + 1) % len(sections)
            self._refresh_body()

    def _refresh_body(self) -> None:
        self.query_one("#config-sidebar", Static).update(self._render_sidebar())
        self.query_one("#config-form", Static).update(self._render_form())
```

And update `action_prev_tab` / `action_next_tab` to reset the section cursor and call `_refresh_body()`:

```python
    def action_prev_tab(self) -> None:
        self._active_tab_index = (self._active_tab_index - 1) % len(self._tabs)
        self._active_section_index = 0
        self.query_one("#config-tabs", Static).update(self._render_tab_bar())
        self._refresh_body()

    def action_next_tab(self) -> None:
        self._active_tab_index = (self._active_tab_index + 1) % len(self._tabs)
        self._active_section_index = 0
        self.query_one("#config-tabs", Static).update(self._render_tab_bar())
        self._refresh_body()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_textual_config_screen.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_textual_config_screen.py src/feather/textual_config_screen.py
git commit -m "ConfigScreen: section cursor via up/down arrows"
```

---

### Task 5: Field editing via Enter

**Files:**
- Modify: `src/feather/textual_config_screen.py`
- Modify: `tests/test_textual_config_screen.py`

- [ ] **Step 1: Failing test — Enter on a field marks it dirty**

```python
async def test_enter_on_field_opens_editor_and_marks_dirty(
    service: ConfigService,
) -> None:
    async with _Host(service).run_test() as pilot:
        # Navigate to the Provider section (first under App).
        # Press Enter to focus the first field. Type a value. Submit.
        await pilot.press("enter")  # opens editor (Phase 2.5 — modal popup or inline)

        # The simplest implementation: render an Input widget below the
        # focused row; submission writes to self._dirty.
        await pilot.press("c", "l", "a", "u", "d", "e", "enter")

        screen = pilot.app.screen
        assert "app.active_provider" in screen._dirty
        assert screen._dirty["app.active_provider"] == "claude"
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_textual_config_screen.py::test_enter_on_field_opens_editor_and_marks_dirty -v`
Expected: FAIL.

- [ ] **Step 3: Implement a focused-field cursor + inline editor**

Add `_active_field_index` and Enter binding:

```python
        self._active_field_index: int = 0
```

Add `Binding("enter", "edit_field", "Edit")` to BINDINGS.

Add `action_edit_field` that selects the focused field and updates `_dirty`:

```python
    def _fields_in_section(self) -> list[Any]:
        sections = self._subsections()
        if not sections:
            return []
        prefix = self._tabs[self._active_tab_index].section_prefix
        active_section = sections[self._active_section_index]
        return [
            f
            for f in REGISTRY
            if f.path.startswith(prefix)
            and (
                ("." in f.path[len(prefix):] and f.path[len(prefix):].split(".", 1)[0] == active_section)
                or ("." not in f.path[len(prefix):] and active_section == "agent")
            )
        ]

    def action_edit_field(self) -> None:
        from textual.widgets import Input

        fields = self._fields_in_section()
        if not fields:
            return
        field = fields[self._active_field_index % len(fields)]
        editor = Input(placeholder=f"new value for {field.path}", id="config-inline-editor")
        self.mount(editor)
        editor.focus()

        async def _on_submit(value: str) -> None:
            validate = self._service.validate(field.path, value)
            if not validate.ok:
                self.query_one("#config-footer", Static).update(
                    f"INVALID: {validate.error}   esc=cancel"
                )
                return
            self._dirty[field.path] = validate.coerced
            self._refresh_body()
            self.query_one("#config-footer", Static).update(self._render_footer())
            editor.remove()

        editor.on_input_submitted = lambda event: self.app.call_later(
            _on_submit(event.value)
        )
```

(The exact Textual API for handling Input submission may differ — refer to current Textual docs or use a Message handler. The intent is: on submit, validate via ConfigService, store in `_dirty`, refresh.)

Update `_render_form` to show dirty values overlaid:

```python
            current = (
                self._dirty[f.path]
                if f.path in self._dirty
                else value.current
            )
            dirty_badge = " [DIRTY]" if f.path in self._dirty else ""
            rows.append(
                f"{f.path}   {badge_src}{dirty_badge}   {badge_rl}\n"
                f"   ▸ {current!r}\n"
                f"   {f.description}"
            )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_textual_config_screen.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_textual_config_screen.py src/feather/textual_config_screen.py
git commit -m "ConfigScreen: inline editor on Enter, dirty tracking"
```

---

### Task 6: Save flow — commit dirty fields + show per-class banner

**Files:**
- Modify: `src/feather/textual_config_screen.py`
- Modify: `tests/test_textual_config_screen.py`

- [ ] **Step 1: Failing test**

```python
async def test_save_invokes_set_for_each_dirty_field(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        screen._dirty["app.active_provider"] = "claude"

        await pilot.press("s")

        overlay = service.paths.global_config_dir / "app.yaml"
        assert "claude" in overlay.read_text(encoding="utf-8")
        # Saved fields clear from dirty.
        assert "app.active_provider" not in screen._dirty


async def test_save_banner_shows_reload_class_counts(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        screen._dirty["app.active_provider"] = "claude"        # NEXT_TURN
        screen._dirty["app.compaction.trigger_ratio"] = 0.5    # LIVE
        screen._dirty["app.claude.request_timeout_seconds"] = 200.0  # RESTART_LEAD

        await pilot.press("s")

        footer = str(pilot.app.query_one("#config-footer", Static).render())
        assert "1 live" in footer.lower() or "applied" in footer.lower()
        assert "restart-lead" in footer.lower()
```

(`test_save_banner_shows_reload_class_counts` skips `apply_config_change` since the modal can't reach a real runtime in this fixture — the production wiring is added in Task 8. For now, the test only checks the SAVE wrote to disk and a banner reflecting bucket counts.)

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_textual_config_screen.py -v -k save`
Expected: failures.

- [ ] **Step 3: Implement `action_save` (modal-side only — runtime apply in Task 8)**

```python
    def action_save(self) -> None:
        if not self._dirty:
            self.query_one("#config-footer", Static).update("no dirty fields  esc=close")
            return

        from feather.config_schema import ReloadClass, lookup

        live: list[str] = []
        next_turn: list[str] = []
        restart_lead: list[str] = []
        restart_app: list[str] = []
        errors: list[str] = []
        for path, value in list(self._dirty.items()):
            result = self._service.set(path, value)
            if not result.ok:
                errors.append(f"{path}: {result.error}")
                continue
            del self._dirty[path]
            field_def = lookup(path)
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_textual_config_screen.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_textual_config_screen.py src/feather/textual_config_screen.py
git commit -m "ConfigScreen: save flow with per-class banner"
```

---

### Task 7: `self_repair.enabled` force-confirm carve-out

**Files:**
- Modify: `src/feather/config_service.py`
- Modify: `src/feather/config_slash.py`
- Modify: `src/feather/textual_config_screen.py`
- Modify: `tests/test_config_service.py`, `tests/test_config_slash.py`

- [ ] **Step 1: Failing test in ConfigService**

Append to `tests/test_config_service.py`:

```python
def test_set_self_repair_without_force_refuses(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.set("app.self_repair.enabled", True)

    assert not result.ok
    assert "force" in (result.error or "").lower()


def test_set_self_repair_with_force_succeeds(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.set("app.self_repair.enabled", True, force=True)

    assert result.ok
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_config_service.py -v -k self_repair`
Expected: failures.

- [ ] **Step 3: Implement the carve-out**

In `ConfigService.set` add a `force: bool = False` kwarg and reject `app.self_repair.enabled` unless force is set:

```python
    def set(
        self,
        dotted: str,
        value: Any,
        *,
        scope: PathScope = PathScope.GLOBAL,
        force: bool = False,
    ) -> WriteResult:
        if dotted == "app.self_repair.enabled" and not force:
            return WriteResult(
                ok=False,
                path=dotted,
                error=(
                    "self_repair.enabled change requires a full TUI restart "
                    "and may corrupt mid-session worker state. Re-run with "
                    "--force to acknowledge."
                ),
            )
        # ... existing body ...
```

- [ ] **Step 4: Plumb `--force` through the slash dispatcher**

In `feather/config_slash.py::_parse_scope`, add a `force` flag:

```python
def _parse_scope(rest: list[str]) -> tuple[PathScope, bool, list[str]]:
    scope = PathScope.GLOBAL
    force = False
    remaining: list[str] = []
    for token in rest:
        if token == "--global":
            scope = PathScope.GLOBAL
        elif token == "--project":
            scope = PathScope.PROJECT
        elif token == "--force":
            force = True
        else:
            remaining.append(token)
    return scope, force, remaining
```

Update `_cmd_set` to pass `force` through:

```python
def _cmd_set(service: ConfigService, rest: list[str]) -> ConfigCommandResult:
    scope, force, positional = _parse_scope(rest)
    ...
    write = service.set(path, value, scope=scope, force=force)
    ...
```

(Apply the same triple-unpack change to `_cmd_reset`.)

- [ ] **Step 5: Modal force-confirm prompt**

In `textual_config_screen.action_save`, before iterating `self._dirty`, if `app.self_repair.enabled` is present, push a confirmation prompt:

```python
        if "app.self_repair.enabled" in self._dirty:
            from feather.config_schema import lookup

            field_def = lookup("app.self_repair.enabled")
            # Open a small modal — for Phase 2 we keep it as an inline
            # banner asking the user to press 'y' to confirm.
            if not getattr(self, "_self_repair_confirmed", False):
                self.query_one("#config-footer", Static).update(
                    "self_repair.enabled change is RESTART-APP. Press 'y' to confirm, esc to cancel."
                )
                return
```

Add a `_self_repair_confirmed` toggle bound to `y`:

```python
        Binding("y", "confirm_self_repair", "Confirm", show=False),
```

```python
    def action_confirm_self_repair(self) -> None:
        self._self_repair_confirmed = True
        self.action_save()
```

When saving the self_repair field, pass `force=True`:

```python
            result = self._service.set(
                path,
                value,
                force=(path == "app.self_repair.enabled"),
            )
```

- [ ] **Step 6: Update tests for the new triple-unpack and force flag**

Update `tests/test_config_slash.py::test_set_with_project_flag` and similar — re-run all `/config` slash tests after this change.

Append to `tests/test_config_slash.py`:

```python
def test_set_self_repair_without_force_refuses(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "set app.self_repair.enabled true")

    assert not result.ok
    assert "force" in result.body.lower()


def test_set_self_repair_with_force(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "set app.self_repair.enabled true --force")

    assert result.ok
```

- [ ] **Step 7: Run all the suites**

Run: `uv run pytest tests/test_config_service.py tests/test_config_slash.py tests/test_textual_config_screen.py -v`
Expected: green.

- [ ] **Step 8: Commit**

```bash
git add src/feather/config_service.py src/feather/config_slash.py src/feather/textual_config_screen.py tests/test_config_service.py tests/test_config_slash.py
git commit -m "Carve out self_repair.enabled: require --force / modal y-confirm"
```

---

### Task 8: Wire runtime.apply_config_change after save

**Files:**
- Modify: `src/feather/textual_config_screen.py`
- Modify: `src/feather/textual_tui.py`
- Modify: `tests/test_textual_config_screen.py`

- [ ] **Step 1: Failing test (with a fake runtime)**

```python
async def test_save_calls_apply_config_change(service: ConfigService) -> None:
    applied: list[list[str]] = []

    class _FakeRuntime:
        async def apply_config_change(self, paths):
            applied.append(list(paths))
            from feather.runtime import ConfigApplyResult

            return ConfigApplyResult(
                applied=list(paths), needs_restart_lead=[], needs_restart_app=[]
            )

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        screen._runtime = _FakeRuntime()
        screen._dirty["app.active_provider"] = "claude"

        await pilot.press("s")
        # Pilot's event loop runs the worker scheduled by action_save.
        await pilot.pause()

        assert applied == [["app.active_provider"]]
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/test_textual_config_screen.py::test_save_calls_apply_config_change -v`
Expected: FAIL.

- [ ] **Step 3: Wire it**

Add `runtime` to ConfigScreen's constructor:

```python
    def __init__(
        self, *, service: ConfigService, runtime: Any | None = None
    ) -> None:
        super().__init__()
        self._service = service
        self._runtime = runtime
        self._tabs = _discover_tabs(service)
        self._active_tab_index = 0
        self._active_section_index = 0
        self._active_field_index = 0
        self._dirty: dict[str, object] = {}
        self._self_repair_confirmed = False
```

At the bottom of `action_save`, after the banner is rendered, schedule `apply_config_change`:

```python
        applied_paths = list(live) + list(next_turn)
        if applied_paths and self._runtime is not None:
            async def _apply() -> None:
                result = await self._runtime.apply_config_change(applied_paths)
                msg = []
                if result.applied:
                    msg.append(f"applied: {', '.join(result.applied)}")
                if result.needs_restart_lead:
                    msg.append(
                        f"restart-lead: {', '.join(result.needs_restart_lead)}"
                    )
                if result.needs_restart_app:
                    msg.append(
                        f"restart-app: {', '.join(result.needs_restart_app)}"
                    )
                self.query_one("#config-footer", Static).update(
                    " | ".join(msg) or "no changes applied"
                )

            self.app.run_worker(_apply(), exclusive=False)
```

In `textual_tui.py::_cmd_config`, when args is empty, push the ConfigScreen:

```python
        if not args.strip():
            from feather.config_service import ConfigService
            from feather.textual_config_screen import ConfigScreen

            service = ConfigService(
                paths=self._paths,
                app_config=self._runtime.config,
            )
            self.push_screen(ConfigScreen(service=service, runtime=self._runtime))
            return
        # ... existing headless dispatch below ...
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_textual_config_screen.py tests/test_textual_tui.py -v`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_textual_config_screen.py src/feather/textual_config_screen.py src/feather/textual_tui.py
git commit -m "ConfigScreen: trigger runtime.apply_config_change after save"
```

---

### Task 9: Diff popup (`d`)

**Files:**
- Modify: `src/feather/textual_config_screen.py`
- Modify: `tests/test_textual_config_screen.py`

- [ ] **Step 1: Failing test**

```python
async def test_diff_key_shows_dirty_and_overlay(service: ConfigService) -> None:
    service.set("app.active_provider", "claude")
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        screen._dirty["app.openai.temperature"] = 0.3

        await pilot.press("d")

        body = str(pilot.app.query_one("#config-form-content").render())
        assert "app.active_provider" in body
        assert "app.openai.temperature" in body
```

(Adapt selectors to the actual widget IDs the implementation uses.)

- [ ] **Step 2: Implement `action_diff`**

```python
    def action_diff(self) -> None:
        from feather.config_service import ValueSource

        lines: list[str] = []
        # Pending (dirty) edits
        for path, value in self._dirty.items():
            current = self._service.get(path).current
            lines.append(f"PENDING {path}: {current!r} → {value!r}")
        # Persisted overrides (global vs default)
        for path, (old, new) in sorted(self._service.diff().items()):
            lines.append(f"PERSISTED {path}: {old!r} → {new!r}")
        body = "\n".join(lines) or "(no overrides or pending edits)"
        self.query_one("#config-form", Static).update(body)
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_textual_config_screen.py src/feather/textual_config_screen.py
git commit -m "ConfigScreen: 'd' shows dirty + persisted diff"
```

---

### Task 10: Reset focused field (`r`)

**Files:**
- Modify: `src/feather/textual_config_screen.py`
- Modify: `tests/test_textual_config_screen.py`

- [ ] **Step 1: Failing test**

```python
async def test_reset_focused_field(service: ConfigService) -> None:
    service.set("app.active_provider", "claude")
    async with _Host(service).run_test() as pilot:
        # Navigate to the provider section + focus the active_provider field.
        # In the current implementation that's index 0 of the provider
        # subsection — adapt if the registry order changes.
        await pilot.press("r")

        # The persisted override is gone now.
        assert "app.active_provider" not in service.diff()
```

- [ ] **Step 2: Implement `action_reset`**

```python
    def action_reset(self) -> None:
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
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_textual_config_screen.py src/feather/textual_config_screen.py
git commit -m "ConfigScreen: 'r' resets the focused field"
```

---

### Task 11: Esc with dirty prompt

**Files:**
- Modify: `src/feather/textual_config_screen.py`
- Modify: `tests/test_textual_config_screen.py`

- [ ] **Step 1: Failing test**

```python
async def test_esc_with_dirty_prompts_to_confirm(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        screen._dirty["app.active_provider"] = "claude"

        await pilot.press("escape")
        # First Esc shows a confirm prompt; modal still open.
        assert isinstance(pilot.app.screen, ConfigScreen)

        await pilot.press("escape")
        # Second Esc actually dismisses.
        assert not isinstance(pilot.app.screen, ConfigScreen)
```

- [ ] **Step 2: Implement**

Replace `action_close`:

```python
    def action_close(self) -> None:
        if self._dirty and not getattr(self, "_confirm_close", False):
            self._confirm_close = True
            self.query_one("#config-footer", Static).update(
                f"{len(self._dirty)} dirty — press Esc again to discard"
            )
            return
        self.dismiss(None)
```

Reset `_confirm_close` whenever the user makes a non-Esc keystroke (so two Esc presses must be consecutive):

```python
    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key != "escape":
            self._confirm_close = False
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_textual_config_screen.py src/feather/textual_config_screen.py
git commit -m "ConfigScreen: Esc requires double-press when dirty"
```

---

### Task 12: Tab-internal field cursor (`tab`, `shift+tab`)

**Files:**
- Modify: `src/feather/textual_config_screen.py`
- Modify: `tests/test_textual_config_screen.py`

- [ ] **Step 1: Failing test**

```python
async def test_tab_cycles_field_focus(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert screen._active_field_index == 0

        await pilot.press("tab")

        assert screen._active_field_index == 1
```

- [ ] **Step 2: Implement**

Add bindings + actions:

```python
        Binding("tab", "field_next", "Next field"),
        Binding("shift+tab", "field_prev", "Prev field"),
```

```python
    def action_field_next(self) -> None:
        fields = self._fields_in_section()
        if fields:
            self._active_field_index = (self._active_field_index + 1) % len(fields)
            self._refresh_body()

    def action_field_prev(self) -> None:
        fields = self._fields_in_section()
        if fields:
            self._active_field_index = (self._active_field_index - 1) % len(fields)
            self._refresh_body()
```

Update `_render_form` to show a `▶` marker on the focused field index.

- [ ] **Step 3: Commit**

```bash
git add tests/test_textual_config_screen.py src/feather/textual_config_screen.py
git commit -m "ConfigScreen: field cursor via Tab / Shift+Tab"
```

---

### Task 13: Full Phase 2 test suite

**Files:**
- (verification)

- [ ] **Step 1: Run the modal tests**

Run: `uv run pytest tests/test_textual_config_screen.py -v`
Expected: every test green.

- [ ] **Step 2: Run the whole suite**

Run: `uv run pytest -x -q 2>&1 | tail -25`
Expected: all phases green.

---

### Task 14: Simplify pass

- [ ] **Step 1: Dispatch simplifier**

Invoke `code-simplifier:code-simplifier`. Brief:

> Simplify the new `src/feather/textual_config_screen.py` for the config modal. Focus on: removing duplicated `_render_*` boilerplate, collapsing repetitive `query_one` calls, removing comments that just restate what the next line does, normalising the section/field cursor logic. Do NOT touch the `BINDINGS` list (the keymap is intentional).

- [ ] **Step 2: Re-run modal suite + full**

Run: `uv run pytest tests/test_textual_config_screen.py tests/ -x -q 2>&1 | tail -10`
Expected: green.

- [ ] **Step 3: Commit if changes**

```bash
git add -p
git commit -m "Simplify ConfigScreen per code-simplifier pass"
```

---

### Task 15: Red-team review

- [ ] **Step 1: Dispatch reviewer**

Invoke `superpowers:code-reviewer`. Brief:

> Red-team review of Phase 2 (Textual config modal) against `docs/superpowers/specs/2026-05-11-config-tui-design.md`. Hunt:
>
> 1. **Race between save + composer:** ConfigScreen is a ModalScreen; the underlying composer is unmounted while open. If `apply_config_change` rebuilds the lead agent mid-modal, does the worker (in worker-mode) handle simultaneous in-flight `run` envelopes correctly? What happens if the user presses Esc and then sends a message before the rebuild ack arrives?
> 2. **Keyboard ergonomics:** Does the modal swallow Esc, Tab, arrows in a way that breaks navigation? Are there any chord conflicts with existing TUI keybindings (composer Tab, conversation Page Up/Down)?
> 3. **`self_repair` carve-out completeness:** The `force` flag is required for ConfigService.set, the slash dispatcher passes it, and the modal has a `y`-confirm. Is there a code path that smuggles a value through (e.g. direct write through `config_writer.write_yaml_value` bypassing ConfigService.set)? If yes, that's a blocker.
> 4. **Stale `app_config`:** The modal captures `self._service` once at mount. If a reload happens elsewhere (worker pushes back) while the modal is open, the modal's view becomes stale. Does the modal need a refresh hook?
> 5. **Source-badge accuracy:** Project-vs-global resolution order is project > global > default. Verify the modal's badge reflects that ordering (test it with a value present in both the project file and the global overlay).
> 6. **Memory of dirty state on Esc:** After a discard, are there any references (worker tasks, async timers) still holding the discarded values?
>
> Report blocking vs nit. Under 500 words.

- [ ] **Step 2: Address every BLOCKING finding**

Each blocker: failing test → fix → re-run all Phase 2 tests → commit `Address red-team finding: <summary>`.

- [ ] **Step 3: Push**

```bash
git push origin feature/config-tui
```

---

## Phase 2 self-review checklist

- [ ] ConfigScreen skeleton with tabs + footer (Task 1)
- [ ] Tab cycling via arrows (Task 2)
- [ ] Sidebar + form layout per tab (Task 3)
- [ ] Section cursor via Up/Down (Task 4)
- [ ] Inline editor on Enter + dirty tracking (Task 5)
- [ ] Save flow with per-class banner (Task 6)
- [ ] `self_repair.enabled` force-confirm at all three layers (Task 7)
- [ ] Save triggers `runtime.apply_config_change` (Task 8)
- [ ] Diff popup (Task 9)
- [ ] Reset focused field (Task 10)
- [ ] Esc dirty-confirm (Task 11)
- [ ] Field cursor via Tab/Shift-Tab (Task 12)
- [ ] Full suite green (Task 13)
- [ ] Simplify pass (Task 14)
- [ ] Red-team review with blockers addressed (Task 15)
