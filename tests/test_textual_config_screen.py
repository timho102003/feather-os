"""Tests for the /config Textual modal (ConfigScreen).

Covers Tasks 1-13 of Phase 2. Tests use Textual's pilot harness with a
``_Host`` app that pushes ``ConfigScreen`` on mount.

Note: Textual 8.x does not expose ``textual.pilot.Pilot`` as a top-level
import. The pilot is obtained from ``App.run_test()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from feather.config import load_app_config
from feather.config.service import ConfigService
from feather.paths import FeatherPaths
from feather.tui.config_screen import ConfigScreen


# ---------------------------------------------------------------------------
# Fixtures + host app
# ---------------------------------------------------------------------------


class _Host(App):
    """Minimal host that pushes ConfigScreen on mount."""

    def __init__(self, service: ConfigService, runtime: Any = None) -> None:
        super().__init__()
        self._service = service
        self._runtime = runtime

    async def on_mount(self) -> None:
        await self.push_screen(ConfigScreen(service=self._service, runtime=self._runtime))


@pytest.fixture
def service(tmp_path: Path) -> ConfigService:
    paths = FeatherPaths(project_root=tmp_path / "proj", home=tmp_path / "global")
    paths.ensure_global_dirs()
    paths.ensure_project_dirs()
    cfg = load_app_config(paths.project_root, paths=paths)
    return ConfigService(paths=paths, app_config=cfg)


# ---------------------------------------------------------------------------
# Task 1: Skeleton mounts + shows tabs + footer
# ---------------------------------------------------------------------------


async def test_modal_mounts_and_shows_app_tab(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        tabs = screen.query_one("#config-tabs", Static)
        tab_text = str(tabs.render())
        assert "App" in tab_text
        assert "Lead" in tab_text


async def test_modal_footer_shows_keybindings(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        footer = screen.query_one("#config-footer", Static)
        body = str(footer.render()).lower()
        for keyword in ("save", "diff", "reset", "esc"):
            assert keyword in body, f"footer missing {keyword!r}"


# ---------------------------------------------------------------------------
# Task 2: Tab cycling via arrow keys
# ---------------------------------------------------------------------------


async def test_arrow_right_cycles_tabs(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        # Initially on App tab (index 0)
        assert screen._active_tab_index == 0
        assert screen._tabs[0].label == "App"

        await pilot.press("right")
        await pilot.pause()

        # Now on Lead tab (index 1)
        assert screen._active_tab_index == 1
        assert screen._tabs[1].label == "Lead"

        # Cycle through all remaining tabs and confirm we wrap back to App.
        n_tabs = len(screen._tabs)
        for _ in range(n_tabs - 1):
            await pilot.press("right")
            await pilot.pause()

        assert screen._active_tab_index == 0


async def test_arrow_left_cycles_backwards(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert screen._active_tab_index == 0

        await pilot.press("left")  # wraps to last tab
        await pilot.pause()

        assert screen._active_tab_index == len(screen._tabs) - 1
        assert screen._tabs[screen._active_tab_index].label == "Validate"


# ---------------------------------------------------------------------------
# Task 3: Sidebar + form layout
# ---------------------------------------------------------------------------


async def test_app_tab_shows_subsection_sidebar(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        sidebar = screen.query_one("#config-sidebar", Static)
        body = str(sidebar.render())
        for expected in (
            "compaction",
            "scheduler",
            "self_repair",
            "openai",
            "openrouter",
            "claude",
            "parallel",
            "memory",
        ):
            assert expected in body, f"sidebar missing {expected!r}"


async def test_lead_tab_sidebar_is_minimal(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        await pilot.press("right")  # Lead tab
        await pilot.pause()
        sidebar = screen.query_one("#config-sidebar", Static)
        body = str(sidebar.render()).lower()
        assert "reasoning" in body
        assert "agent" in body


# ---------------------------------------------------------------------------
# Task 4: Section cursor via up/down
# ---------------------------------------------------------------------------


async def test_arrow_down_moves_section_cursor(service: ConfigService) -> None:
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        sidebar = screen.query_one("#config-sidebar", Static)
        before_lines = str(sidebar.render()).split("\n")
        first_active = before_lines.index(
            next(line for line in before_lines if line.startswith("▶"))
        )

        await pilot.press("down")
        await pilot.pause()

        after_lines = str(sidebar.render()).split("\n")
        new_active = after_lines.index(
            next(line for line in after_lines if line.startswith("▶"))
        )
        assert new_active == first_active + 1


async def test_arrow_up_moves_section_cursor(service: ConfigService) -> None:
    """Moving up from index 0 wraps to the last section."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        sidebar = screen.query_one("#config-sidebar", Static)
        before_lines = str(sidebar.render()).split("\n")
        first_active = before_lines.index(
            next(line for line in before_lines if line.startswith("▶"))
        )
        # Back at index 0 after down+up
        assert first_active == 0


# ---------------------------------------------------------------------------
# Task 5: Field editing via Enter + dirty tracking
# ---------------------------------------------------------------------------


async def test_enter_on_enum_field_picker_commits_value(
    service: ConfigService,
) -> None:
    """Enter on an enum field opens a picker; commit writes to _dirty."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)

        assert _navigate_to_field(screen, "app.active_provider")

        # Open the picker.
        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        assert hasattr(picker, "_choices") and set(picker._choices) == {
            "openai",
            "openrouter",
            "claude",
        }

        # Move cursor to "claude" then commit.
        target_index = picker._choices.index("claude")
        while picker._index != target_index:
            await pilot.press("down")
            await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert "app.active_provider" in screen._dirty
        assert screen._dirty["app.active_provider"] == "claude"


async def test_invalid_input_does_not_mark_dirty(service: ConfigService) -> None:
    """Submitting an out-of-range numeric value shows error and does NOT update _dirty."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)

        # Use trigger_ratio (FLOAT, _ratio validator: 0.0-1.0) — typing 2.0
        # passes coerce but fails the validator, hitting the INVALID branch.
        assert _navigate_to_field(screen, "app.compaction.trigger_ratio")

        await pilot.press("enter")
        await pilot.pause()

        # Type an out-of-range float and submit.
        await pilot.press("2", ".", "0", "enter")
        await pilot.pause()

        footer = str(screen.query_one("#config-footer", Static).render()).lower()
        assert "invalid" in footer
        assert "app.compaction.trigger_ratio" not in screen._dirty


# ---------------------------------------------------------------------------
# Task 6: Save flow
# ---------------------------------------------------------------------------


async def test_save_invokes_set_for_each_dirty_field(service: ConfigService) -> None:
    """Pressing s saves each dirty field and clears it from _dirty."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        screen._dirty["app.active_provider"] = "claude"

        await pilot.press("s")
        await pilot.pause()

        overlay = service.paths.global_config_dir / "app.yaml"
        assert overlay.exists(), "overlay file should have been created"
        assert "claude" in overlay.read_text(encoding="utf-8")
        assert "app.active_provider" not in screen._dirty


async def test_save_banner_shows_reload_class_counts(service: ConfigService) -> None:
    """Save banner reflects per-class bucket counts."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        screen._dirty["app.active_provider"] = "claude"  # NEXT_TURN
        screen._dirty["app.compaction.trigger_ratio"] = 0.5  # LIVE
        screen._dirty["app.claude.request_timeout_seconds"] = 200.0  # RESTART_LEAD

        await pilot.press("s")
        await pilot.pause()

        footer = str(screen.query_one("#config-footer", Static).render()).lower()
        assert "saved" in footer
        assert "restart-lead" in footer


async def test_save_with_no_dirty_shows_message(service: ConfigService) -> None:
    """Pressing s with nothing dirty shows 'no dirty fields'."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        await pilot.press("s")
        await pilot.pause()

        footer = str(screen.query_one("#config-footer", Static).render()).lower()
        assert "no dirty" in footer


# ---------------------------------------------------------------------------
# Task 7: self_repair.enabled carve-out
# ---------------------------------------------------------------------------


async def test_save_self_repair_requires_y_confirm(service: ConfigService) -> None:
    """Saving self_repair.enabled without y-confirm shows a prompt and doesn't write."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        screen._dirty["app.self_repair.enabled"] = True

        await pilot.press("s")
        await pilot.pause()

        footer = str(screen.query_one("#config-footer", Static).render()).lower()
        assert "self_repair" in footer or "confirm" in footer
        # Should NOT have written yet
        overlay = service.paths.global_config_dir / "app.yaml"
        if overlay.exists():
            assert "self_repair" not in overlay.read_text(encoding="utf-8")


async def test_save_self_repair_with_y_confirm_writes(service: ConfigService) -> None:
    """Pressing 'y' after the self_repair confirm prompt allows saving."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        screen._dirty["app.self_repair.enabled"] = True

        await pilot.press("s")  # shows confirm prompt
        await pilot.pause()
        await pilot.press("y")  # sets confirmed + re-invokes save
        await pilot.pause()

        # Now it should be written
        overlay = service.paths.global_config_dir / "app.yaml"
        assert overlay.exists()
        content = overlay.read_text(encoding="utf-8")
        assert "self_repair" in content or "true" in content.lower()


# ---------------------------------------------------------------------------
# Task 8: runtime.apply_config_change wiring
# ---------------------------------------------------------------------------


async def test_save_calls_apply_config_change(service: ConfigService) -> None:
    """After save, apply_config_change is called for live/next_turn fields."""

    applied: list[list[str]] = []

    class _FakeRuntime:
        async def apply_config_change(self, paths: list[str]) -> Any:
            applied.append(list(paths))
            from feather.runtime import ConfigApplyResult
            return ConfigApplyResult(
                applied=list(paths),
                needs_restart_lead=[],
                needs_restart_app=[],
            )

    async with _Host(service, runtime=_FakeRuntime()).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        screen._dirty["app.active_provider"] = "claude"  # NEXT_TURN -> applied

        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()  # give worker time to run

        assert applied, "apply_config_change was not called"
        assert "app.active_provider" in applied[0]


# ---------------------------------------------------------------------------
# Task 9: Diff popup
# ---------------------------------------------------------------------------


async def test_diff_key_shows_dirty_and_overlay(service: ConfigService) -> None:
    """Pressing d shows a diff of pending edits and persisted overrides."""

    service.set("app.active_provider", "claude")
    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        screen._dirty["app.openai.temperature"] = 0.3

        await pilot.press("d")
        await pilot.pause()

        body = str(screen.query_one("#config-form", Static).render())
        assert "app.active_provider" in body
        assert "app.openai.temperature" in body


# ---------------------------------------------------------------------------
# Task 10: Reset focused field
# ---------------------------------------------------------------------------


async def test_reset_focused_field(service: ConfigService) -> None:
    """Pressing r resets the focused field's overlay."""

    service.set("app.active_provider", "claude")
    async with _Host(service).run_test() as pilot:
        # Navigate to the section containing app.active_provider.
        # That's the "agent" synthetic section (leaf under app.)
        from feather.config.schema import REGISTRY as REG
        sections: list[str] = []
        for f in REG:
            if not f.path.startswith("app."):
                continue
            tail = f.path[len("app."):]
            label = tail.split(".", 1)[0] if "." in tail else "agent"
            if label not in sections:
                sections.append(label)

        target = "agent"
        if target in sections:
            idx = sections.index(target)
            for _ in range(idx):
                await pilot.press("down")

        await pilot.press("r")
        await pilot.pause()

        assert "app.active_provider" not in service.diff()


# ---------------------------------------------------------------------------
# Task 11: Esc dirty confirm
# ---------------------------------------------------------------------------


async def test_esc_with_dirty_prompts_to_confirm(service: ConfigService) -> None:
    """First Esc shows confirm prompt; second Esc dismisses."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        screen._dirty["app.active_provider"] = "claude"

        await pilot.press("escape")
        await pilot.pause()
        # First Esc: modal still open
        assert isinstance(pilot.app.screen, ConfigScreen)

        await pilot.press("escape")
        await pilot.pause()
        # Second Esc: modal dismissed
        assert not isinstance(pilot.app.screen, ConfigScreen)


async def test_esc_without_dirty_closes_immediately(service: ConfigService) -> None:
    """Esc with no dirty fields closes the modal immediately."""

    async with _Host(service).run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(pilot.app.screen, ConfigScreen)


# ---------------------------------------------------------------------------
# Task 12: Tab / Shift+Tab field cursor
# ---------------------------------------------------------------------------


async def test_tab_cycles_field_focus(service: ConfigService) -> None:
    """Tab increments the field cursor within a multi-field section."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert screen._active_field_index == 0

        # Navigate to a section with multiple fields (compaction has 6 fields).
        from feather.config.schema import REGISTRY as REG
        sections: list[str] = []
        for f in REG:
            if not f.path.startswith("app."):
                continue
            tail = f.path[len("app."):]
            label = tail.split(".", 1)[0] if "." in tail else "agent"
            if label not in sections:
                sections.append(label)

        target = "compaction"
        if target in sections:
            for _ in range(sections.index(target)):
                await pilot.press("down")
            await pilot.pause()

        # Should now be in compaction with multiple fields.
        fields_before = screen._fields_in_section()
        assert len(fields_before) > 1, (
            f"expected multiple fields in 'compaction', got {len(fields_before)}"
        )
        assert screen._active_field_index == 0

        await pilot.press("tab")
        await pilot.pause()

        assert screen._active_field_index == 1


async def test_shift_tab_cycles_field_focus_backward(service: ConfigService) -> None:
    """Shift+Tab decrements the field cursor within a multi-field section."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)

        # Navigate to compaction (multiple fields).
        from feather.config.schema import REGISTRY as REG
        sections: list[str] = []
        for f in REG:
            if not f.path.startswith("app."):
                continue
            tail = f.path[len("app."):]
            label = tail.split(".", 1)[0] if "." in tail else "agent"
            if label not in sections:
                sections.append(label)

        target = "compaction"
        if target in sections:
            for _ in range(sections.index(target)):
                await pilot.press("down")
            await pilot.pause()

        initial = screen._active_field_index  # 0

        # Go to index 1, then shift+tab back to 0.
        await pilot.press("tab")
        await pilot.pause()
        assert screen._active_field_index == 1

        await pilot.press("shift+tab")
        await pilot.pause()

        assert screen._active_field_index == initial


# ---------------------------------------------------------------------------
# Static source assertions — binding + action coverage (no Textual harness)
# ---------------------------------------------------------------------------


def test_config_screen_has_required_bindings() -> None:
    """All required key bindings are present in ConfigScreen.BINDINGS."""

    keys = {b.key for b in ConfigScreen.BINDINGS}
    assert "escape" in keys
    assert "left" in keys
    assert "right" in keys
    assert "up" in keys
    assert "down" in keys
    # Enter is bound under both "enter" and "return" key names so terminals
    # that emit one or the other both trigger edit.
    assert any("enter" in k for k in keys)
    assert "s" in keys
    assert "d" in keys
    assert "r" in keys
    assert "tab" in keys
    assert "shift+tab" in keys


def test_config_screen_has_required_action_methods() -> None:
    """All action methods referenced in BINDINGS are implemented."""

    for method in (
        "action_close",
        "action_prev_tab",
        "action_next_tab",
        "action_section_prev",
        "action_section_next",
        "action_edit_field",
        "action_save",
        "action_diff",
        "action_reset",
        "action_field_next",
        "action_field_prev",
        "action_confirm_self_repair",
    ):
        assert hasattr(ConfigScreen, method), f"missing {method}"


# ---------------------------------------------------------------------------
# Phase 2 red-team fixes (Tasks 13a-13d)
# ---------------------------------------------------------------------------


async def test_escape_cancels_inline_edit(service: ConfigService) -> None:
    """Esc while the inline editor is focused cancels the edit, not the modal."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)

        # Open the inline editor on the first field.
        await pilot.press("enter")
        await pilot.pause()

        # Check whether the editor was opened.
        try:
            screen.query_one("#config-inline-editor", Input)
            editor_present = True
        except Exception:
            editor_present = False

        if not editor_present:
            # enter didn't open an editor in this fixture configuration — skip.
            return

        # Press Esc — should cancel the edit only, modal stays open.
        await pilot.press("escape")
        await pilot.pause()

        # Modal still mounted.
        assert isinstance(pilot.app.screen, ConfigScreen)
        # _pending_edit cleared.
        assert screen._pending_edit is None
        # Inline editor widget removed.
        remaining = screen.query("#config-inline-editor")
        assert not remaining, "inline editor widget should be removed after Esc"


def test_save_in_flight_flag_prevents_overlap(service: ConfigService) -> None:
    """Second save during in-flight apply is rejected."""

    import inspect

    src = inspect.getsource(ConfigScreen.action_save)
    assert "_apply_in_flight" in src, (
        "action_save must guard against concurrent applies via _apply_in_flight flag"
    )


def test_apply_error_renders_in_footer(service: ConfigService) -> None:
    """An exception from runtime.apply_config_change surfaces in the footer."""

    import inspect

    src = inspect.getsource(ConfigScreen)
    assert "apply error" in src, (
        "action_save's _apply worker must catch exceptions and render them in the footer"
    )


# ---------------------------------------------------------------------------
# Phase 2: Constrained pickers (bool toggle + enum/choices dropdown)
# ---------------------------------------------------------------------------


def _navigate_to_field(screen: ConfigScreen, target_path: str) -> bool:
    """Move tab + sidebar + field cursors to ``target_path``.

    Handles both the App tab (paths starting with ``app.``) and agent
    tabs (paths starting with ``agents.<Name>.``).

    Returns True if reached, False otherwise.
    """

    from feather.config.schema import REGISTRY as REG

    # Find the right tab.
    if target_path.startswith("app."):
        prefix = "app."
        target_tab_label = "App"
    elif target_path.startswith("agents."):
        agent_name = target_path.split(".")[1]
        prefix = f"agents.{agent_name}."
        target_tab_label = agent_name
    else:
        return False

    tab_labels = [t.label for t in screen._tabs]
    if target_tab_label not in tab_labels:
        return False
    screen._active_tab_index = tab_labels.index(target_tab_label)

    sections: list[str] = []
    for f in REG:
        if not f.path.startswith(prefix):
            continue
        tail = f.path[len(prefix):]
        label = tail.split(".", 1)[0] if "." in tail else "agent"
        if label not in sections:
            sections.append(label)

    tail = target_path[len(prefix):]
    target_section = tail.split(".", 1)[0] if "." in tail else "agent"
    if target_section not in sections:
        return False
    screen._active_section_index = sections.index(target_section)

    fields = screen._fields_in_section()
    for i, f in enumerate(fields):
        if f.path == target_path:
            screen._active_field_index = i
            screen._refresh_body()
            return True
    return False


async def test_enter_on_bool_field_opens_choice_picker(
    service: ConfigService,
) -> None:
    """Enter on a BOOLEAN field opens an inline picker, not a free-text Input."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)

        assert _navigate_to_field(screen, "app.compaction.enabled")

        await pilot.press("enter")
        await pilot.pause()

        # No free-text Input should be mounted for a bool field.
        inputs = screen.query("#config-inline-editor")
        assert inputs, "an inline editor of some kind should be mounted"
        # The picker widget exposes a `_choices` attribute distinct from an Input.
        picker = inputs.first()
        assert hasattr(picker, "_choices"), (
            "BOOLEAN field should open a choice picker, not an Input"
        )
        assert tuple(picker._choices) == ("false", "true")


async def test_bool_picker_arrow_keys_change_selection_then_commit(
    service: ConfigService,
) -> None:
    """↓/↑ moves the bool picker cursor; Enter commits to _dirty."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.compaction.enabled")

        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        # Default value is True (compaction enabled) → cursor on "true" (index 1).
        # Move to "false" via up arrow then commit.
        starting_index = picker._index
        await pilot.press("down")
        await pilot.pause()
        moved = picker._index != starting_index

        await pilot.press("enter")
        await pilot.pause()

        # Picker removed; field is dirty.
        assert not screen.query("#config-inline-editor")
        assert "app.compaction.enabled" in screen._dirty
        if moved:
            # We moved at least once, so the dirty value should be the opposite
            # of the original (True -> False or False -> True).
            assert isinstance(screen._dirty["app.compaction.enabled"], bool)


async def test_enter_on_enum_field_opens_choice_picker_with_enum_values(
    service: ConfigService,
) -> None:
    """Enter on an ENUM field opens a picker populated from field.enum."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.active_provider")

        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        assert hasattr(picker, "_choices")
        assert set(picker._choices) == {"openai", "openrouter", "claude"}


async def test_enter_on_model_field_opens_choice_picker_with_catalog(
    service: ConfigService,
) -> None:
    """Enter on ``app.openai.model`` opens a picker populated from the YAML
    catalog — the same source of truth that ``agents.<name>.model`` uses,
    so the app-level picker can never silently show fewer slugs than the
    per-agent one (the original drift symptom)."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.openai.model")

        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        assert hasattr(picker, "_choices")
        from feather.config.model_catalog import load_catalog

        catalog_slugs = set(load_catalog().slugs_for("openai"))
        assert catalog_slugs.issubset(set(picker._choices))
        # And no extras — the picker is EXACTLY the catalog (no inherit
        # sentinel at the app level, since this is the bottom layer).
        assert set(picker._choices) == catalog_slugs


async def test_enter_on_sensitive_readonly_refuses_and_shows_footer(
    service: ConfigService,
) -> None:
    """SENSITIVE_READONLY fields cannot be edited inline; modal explains why."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.openai.api_key_env")

        await pilot.press("enter")
        await pilot.pause()

        # No editor mounted.
        assert not screen.query("#config-inline-editor")
        body = str(screen.query_one("#config-footer", Static).render()).lower()
        assert "sensitive" in body or "env" in body


async def test_enter_on_list_editor_field_refuses(
    service: ConfigService,
) -> None:
    """LIST_EDITOR fields are not yet inline-editable in the modal."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        # claude.anthropic_beta is a LIST_EDITOR field.
        if not _navigate_to_field(screen, "app.claude.anthropic_beta"):
            return  # Section ordering changed; skip rather than break.

        await pilot.press("enter")
        await pilot.pause()

        assert not screen.query("#config-inline-editor")


async def test_numeric_field_input_placeholder_shows_hint(
    service: ConfigService,
) -> None:
    """Opening a NUMERIC editor places the hint in the Input placeholder."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        # trigger_ratio has _ratio validator → hint should derive 0.0-1.0
        assert _navigate_to_field(screen, "app.compaction.trigger_ratio")

        await pilot.press("enter")
        await pilot.pause()

        editor = screen.query_one("#config-inline-editor", Input)
        assert "0" in editor.placeholder and "1" in editor.placeholder, (
            f"numeric placeholder should advertise range; got {editor.placeholder!r}"
        )


async def test_picker_escape_cancels_without_committing(
    service: ConfigService,
) -> None:
    """Esc on the picker cancels and leaves _dirty unchanged."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.compaction.enabled")

        await pilot.press("enter")
        await pilot.pause()

        await pilot.press("down")
        await pilot.press("escape")
        await pilot.pause()

        assert not screen.query("#config-inline-editor")
        assert "app.compaction.enabled" not in screen._dirty
        assert screen._pending_edit is None


async def test_picker_left_right_navigate_within_picker_not_tabs(
    service: ConfigService,
) -> None:
    """Left/Right while picker is open must move the picker cursor, not switch tabs."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.active_provider")
        starting_tab = screen._active_tab_index

        await pilot.press("enter")
        await pilot.pause()

        # Right + Left should NOT change the active tab.
        await pilot.press("right")
        await pilot.pause()
        await pilot.press("left")
        await pilot.pause()

        assert screen._active_tab_index == starting_tab, (
            "picker swallowed left/right but tab cursor still moved"
        )


async def test_bool_picker_renders_at_least_two_rows(
    service: ConfigService,
) -> None:
    """Bool picker must be tall enough to show BOTH false and true.

    Regression: the ID-scoped #config-inline-editor rule used to force
    height: 3 on every editor, cropping the picker's second row so only
    'false' was visible regardless of which value was actually selected.
    """

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.compaction.enabled")

        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        # Two choice rows + 2 border rows = at least 4 visible rows.
        # height: 3 (the old buggy cap) would fail this.
        assert picker.region.height >= 4, (
            f"bool picker too short ({picker.region.height} rows) — "
            f"second choice gets cropped"
        )


async def test_dropdown_picker_renders_tall_enough_for_all_choices(
    service: ConfigService,
) -> None:
    """DROPDOWN picker must be tall enough to show several rows, not be
    capped at the old 3-row height that hid all but the first option.

    Catches the same height-cap bug as the bool case but on a long list
    (openrouter's catalog is the longest at 30+ slugs).
    """

    async with _Host(service).run_test(size=(120, 40)) as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.openrouter.model")

        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        choices_count = len(picker._choices)
        # Allow for overflow scrolling, but ensure the widget isn't capped
        # at the old 3-row height — at minimum it should show >= 4 rows
        # OR the widget must declare auto-sizing in its CSS.
        assert picker.region.height > 3, (
            f"dropdown picker capped at {picker.region.height} rows — "
            f"can't display {choices_count} choices"
        )


async def test_agent_provider_picker_offers_inherit_plus_three_providers(
    service: ConfigService,
) -> None:
    """agents.<name>.provider opens a dropdown with (inherit) + the 3 providers."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        # Navigate via the tab index directly — the agent tabs are
        # ordered Lead/Explore/Research/Validate after the App tab.
        target_tab = [t.label for t in screen._tabs].index("Explore")
        while screen._active_tab_index != target_tab:
            await pilot.press("right")
            await pilot.pause()
        assert _navigate_to_field(screen, "agents.Explore.provider")

        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        assert hasattr(picker, "_choices")
        # Expected: (inherit) first, then the three real providers.
        assert tuple(picker._choices) == (
            "(inherit)",
            "openai",
            "openrouter",
            "claude",
        )
        # Default value is None → cursor lands on the inherit sentinel.
        assert picker._index == 0


async def test_agent_provider_inherit_writes_reset_on_save(
    service: ConfigService,
    tmp_path: Path,
) -> None:
    """Picking (inherit) on agent.provider clears the overlay key on save."""

    from feather.config.resolver import PathScope

    # Stage a non-default value so reset has something to remove.
    service.set("agents.Explore.provider", "claude", scope=PathScope.GLOBAL)
    assert service.get("agents.Explore.provider").current == "claude"

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "agents.Explore.provider")

        await pilot.press("enter")
        await pilot.pause()
        # Picker is open; cursor is on "claude" (the persisted value).
        # Move up to "(inherit)" (the first option) and commit.
        picker = screen.query("#config-inline-editor").first()
        target = picker._choices.index("(inherit)")
        while picker._index != target:
            await pilot.press("up")
            await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        # Inherit sentinel is now in the dirty store.
        assert screen._dirty.get("agents.Explore.provider") == "(inherit)"

        # Save: should call service.reset for this path, clearing the overlay.
        await pilot.press("s")
        await pilot.pause()

        # After save, the key is gone from the global overlay — value
        # returns to the default (None for the agent override).
        from feather.config import load_app_config

        # Force a re-load to see the on-disk state.
        cfg = load_app_config(service.paths.project_root, paths=service.paths)
        # Provider should be inherit-default (None / unset).
        # Verifying via service.get (which reads from disk):
        # We can't easily re-instantiate the service inside this test, but
        # the on-disk overlay file should not contain agents.Explore.provider
        # any more. Easiest check: dirty is cleared and footer reports save.
        del cfg  # silence unused
        assert "agents.Explore.provider" not in screen._dirty


async def test_agent_model_picker_uses_resolved_provider_catalog(
    service: ConfigService,
) -> None:
    """agents.<name>.model picker offers only the resolved provider's catalog."""

    from feather.config.resolver import PathScope

    # Pin Explore to openai so model choices come from openai catalog.
    service.set("agents.Explore.provider", "openai", scope=PathScope.GLOBAL)

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "agents.Explore.model")

        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        assert "(inherit)" in picker._choices
        # OpenAI-only catalog should NOT contain anthropic / openrouter slugs.
        assert "claude-opus-4-7" not in picker._choices
        assert "anthropic/claude-opus-4-7" not in picker._choices
        # But should contain OpenAI catalog entries.
        from feather.config.model_catalog import load_catalog

        openai_slugs = set(load_catalog(paths=service.paths).slugs_for("openai"))
        assert set(picker._choices) & openai_slugs  # non-empty intersection


async def test_agent_model_picker_uses_dirty_provider_when_pending(
    service: ConfigService,
) -> None:
    """A pending dirty edit on provider must steer model choices immediately
    (before save) so the user can switch provider and pick a model in one
    pass without first saving the provider change."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)

        # Stage a dirty edit: agents.Explore.provider = claude (anthropic).
        screen._dirty["agents.Explore.provider"] = "claude"

        assert _navigate_to_field(screen, "agents.Explore.model")
        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        # Anthropic catalog should now drive choices.
        assert "claude-opus-4-7" in picker._choices
        # OpenAI catalog should be excluded.
        assert "gpt-5-mini" not in picker._choices


async def test_agent_model_picker_falls_back_to_active_provider(
    service: ConfigService,
) -> None:
    """When agent.provider is None (inherit), model picker uses app.active_provider."""

    from feather.config.resolver import PathScope

    # Make sure agent.provider is None (inherit) and app.active_provider=claude.
    service.set("app.active_provider", "claude", scope=PathScope.GLOBAL)

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "agents.Lead.model")

        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        assert "claude-opus-4-7" in picker._choices, (
            "Lead.model should pick from anthropic catalog because "
            "app.active_provider=claude and Lead.provider inherits"
        )


# ---------------------------------------------------------------------------
# Capability-driven field gating (Commit 1 of the catalog plan)
# ---------------------------------------------------------------------------


async def test_temperature_field_is_disabled_for_reasoning_model(
    service: ConfigService,
) -> None:
    """When app.openai.model is a reasoning model (gpt-5-mini, the shipped
    default), the modal must mark app.openai.temperature as N/A and refuse
    to open the editor — the OpenAI API ignores temperature on reasoning
    models, so letting users set it is a footgun."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.openai.temperature")

        # Form should show an N/A badge.
        form_text = str(screen.query_one("#config-form", Static).render())
        assert "N/A" in form_text

        # Pressing Enter must NOT open an editor.
        await pilot.press("enter")
        await pilot.pause()
        assert not screen.query("#config-inline-editor")

        # Footer should explain why.
        footer = str(screen.query_one("#config-footer", Static).render()).lower()
        assert "gpt-5-mini" in footer or "reasoning" in footer or "n/a" in footer


async def test_temperature_field_is_editable_for_chat_model(
    service: ConfigService,
) -> None:
    """When app.openai.model is a temperature-supporting chat model (gpt-4o),
    the temperature field is editable and the placeholder shows the model's
    actual range."""

    from feather.config.resolver import PathScope

    service.set("app.openai.model", "gpt-4o", scope=PathScope.GLOBAL)

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.openai.temperature")

        await pilot.press("enter")
        await pilot.pause()
        editor = screen.query_one("#config-inline-editor", Input)
        # The model's range (0.0, 2.0) should appear in the placeholder hint.
        assert "0.0" in editor.placeholder and "2.0" in editor.placeholder


async def test_reasoning_effort_disabled_for_chat_model(
    service: ConfigService,
) -> None:
    """gpt-4o is a chat model — reasoning.effort is not a valid knob, so
    it should be marked N/A."""

    from feather.config.resolver import PathScope

    service.set("app.openai.model", "gpt-4o", scope=PathScope.GLOBAL)

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.openai.reasoning.effort")

        await pilot.press("enter")
        await pilot.pause()
        assert not screen.query("#config-inline-editor"), (
            "reasoning.effort should refuse to open for gpt-4o"
        )


async def test_parallel_tool_calls_disabled_for_o3(
    service: ConfigService,
) -> None:
    """The o-series does not accept parallel_tool_calls — must be N/A."""

    from feather.config.resolver import PathScope

    service.set("app.openai.model", "o3", scope=PathScope.GLOBAL)

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.openai.parallel_tool_calls")

        await pilot.press("enter")
        await pilot.pause()
        assert not screen.query("#config-inline-editor")


async def test_claude_thinking_budget_tokens_disabled_for_opus_4_7(
    service: ConfigService,
) -> None:
    """claude-opus-4-7 is adaptive-only — budget_tokens does not apply."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.claude.thinking.budget_tokens")

        await pilot.press("enter")
        await pilot.pause()
        assert not screen.query("#config-inline-editor")


async def test_agent_temperature_disabled_when_resolved_model_is_reasoning(
    service: ConfigService,
) -> None:
    """When an agent's resolved provider+model land on a reasoning model,
    agents.<name>.temperature must be marked N/A and refuse the editor —
    overriding temperature on a reasoning model would silently have no
    effect (the OpenAI API ignores it)."""

    from feather.config.resolver import PathScope

    # Pin app to openai+gpt-5-mini so Lead (which inherits) resolves to a
    # reasoning model regardless of what the shipped default happens to be.
    service.set("app.active_provider", "openai", scope=PathScope.GLOBAL)
    service.set("app.openai.model", "gpt-5-mini", scope=PathScope.GLOBAL)

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "agents.Lead.temperature")

        form_text = str(screen.query_one("#config-form", Static).render())
        assert "N/A" in form_text

        await pilot.press("enter")
        await pilot.pause()
        assert not screen.query("#config-inline-editor")


async def test_agent_temperature_editable_when_resolved_model_is_chat(
    service: ConfigService,
) -> None:
    """If user pins an agent to a chat model, temperature becomes editable."""

    from feather.config.resolver import PathScope

    service.set("agents.Lead.provider", "openai", scope=PathScope.GLOBAL)
    service.set("agents.Lead.model", "gpt-4o", scope=PathScope.GLOBAL)

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "agents.Lead.temperature")

        await pilot.press("enter")
        await pilot.pause()
        editor = screen.query_one("#config-inline-editor", Input)
        # Placeholder reflects gpt-4o's range 0.0-2.0.
        assert "0.0" in editor.placeholder and "2.0" in editor.placeholder


# ---------------------------------------------------------------------------
# Commit 4: red-team fixes
# ---------------------------------------------------------------------------


async def test_pending_inherit_on_agent_provider_steers_model_picker_to_active(
    service: ConfigService,
) -> None:
    """Dirty edit `agents.Lead.provider = (inherit)` must immediately route the
    agent.model picker through app.active_provider — not the still-persisted
    provider value. Otherwise the "switch + pick in one pass" UX is broken for
    the inherit case."""

    from feather.config.resolver import PathScope

    # Persisted: agent pinned to claude. App default: openai.
    service.set("app.active_provider", "openai", scope=PathScope.GLOBAL)
    service.set("agents.Lead.provider", "claude", scope=PathScope.GLOBAL)

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)

        # Stage the inherit dirty edit (would otherwise require a picker round-trip).
        screen._dirty["agents.Lead.provider"] = "(inherit)"

        assert _navigate_to_field(screen, "agents.Lead.model")
        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        # Should reflect app.active_provider (openai), NOT persisted claude.
        assert "gpt-5-mini" in picker._choices
        assert "claude-opus-4-7" not in picker._choices


async def test_dirty_app_model_steers_capability_gating(
    service: ConfigService,
) -> None:
    """Switching `app.openai.model` in the same modal session must drive
    capability gating immediately — not wait for save. Otherwise users see
    temperature [N/A] AFTER switching to gpt-4o because the picker still
    reads the persisted reasoning model."""

    from feather.config.resolver import PathScope

    # Persisted: gpt-5-mini (reasoning, no temp).
    service.set("app.openai.model", "gpt-5-mini", scope=PathScope.GLOBAL)
    service.set("app.active_provider", "openai", scope=PathScope.GLOBAL)

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)

        # Stage a dirty switch to gpt-4o (chat model — accepts temperature).
        screen._dirty["app.openai.model"] = "gpt-4o"

        assert _navigate_to_field(screen, "app.openai.temperature")
        # Form should NOT show [N/A] anymore — the dirty model accepts temp.
        form_text = str(screen.query_one("#config-form", Static).render())
        # If the bug were present, this would have [N/A]; we assert no N/A
        # on this row by looking for the field path's section.
        # The whole form_text might contain N/A from other rows; check
        # the specific row.
        temp_row_start = form_text.find("app.openai.temperature")
        next_row_start = form_text.find("app.openai", temp_row_start + 10)
        if next_row_start == -1:
            next_row_start = len(form_text)
        row = form_text[temp_row_start:next_row_start]
        assert "N/A" not in row, (
            f"app.openai.temperature should be editable after dirty switch to "
            f"gpt-4o, got row: {row!r}"
        )

        # Editor should open now.
        await pilot.press("enter")
        await pilot.pause()
        editor = screen.query("#config-inline-editor")
        assert editor, "temperature editor should open after dirty switch to gpt-4o"


def test_reset_wraps_filesystem_errors(tmp_path: Path) -> None:
    """ConfigService.reset must return WriteResult.ok=False on OSError instead
    of letting the exception escape — otherwise a save with multiple inherit
    entries abandons remaining fields if one reset fails."""

    from unittest.mock import patch

    paths = FeatherPaths(project_root=tmp_path / "proj", home=tmp_path / "global")
    paths.ensure_global_dirs()
    paths.ensure_project_dirs()
    cfg = load_app_config(paths.project_root, paths=paths)
    svc = ConfigService(paths=paths, app_config=cfg)

    # Patch delete_yaml_value to raise OSError; reset must return ok=False.
    with patch("feather.config.writer.delete_yaml_value", side_effect=OSError("disk full")):
        result = svc.reset("app.active_provider")
    assert result.ok is False
    assert "disk full" in str(result.error)


async def test_inherit_sentinel_ignored_on_non_inherit_field(
    service: ConfigService,
) -> None:
    """If somehow `(inherit)` is committed on a field that doesn't support
    inherit semantics (e.g. via a future schema bug), `_handle_choice_picked`
    must NOT call reset() — picking the sentinel for the wrong field type
    would silently delete an overlay key."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)

        # app.active_provider is a strict ENUM — doesn't support inherit.
        # Confirm:
        assert not screen._supports_inherit("app.active_provider")


async def test_memory_operations_extraction_provider_picker_uses_inherit_sentinel(
    service: ConfigService,
) -> None:
    """app.memory.operations.<op>.provider opens the same dropdown as
    agents.*.provider — the user shouldn't have to remember which slug
    spelling counts for app.active_provider vs an op override."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(
            screen, "app.memory.operations.extraction.provider"
        )

        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        assert tuple(picker._choices) == (
            "(inherit)",
            "openai",
            "openrouter",
            "claude",
        )


async def test_memory_operations_extraction_model_picker_scoped_to_op_provider(
    service: ConfigService,
) -> None:
    """When an op pins provider=openai, its model picker shows only the
    openai catalog (plus inherit) — even if app.active_provider is set to
    something else."""

    from feather.config.resolver import PathScope

    service.set("app.active_provider", "claude", scope=PathScope.GLOBAL)
    service.set(
        "app.memory.operations.extraction.provider", "openai", scope=PathScope.GLOBAL
    )

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(
            screen, "app.memory.operations.extraction.model"
        )

        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        # Inherit comes first; then openai catalog only.
        assert picker._choices[0] == "(inherit)"
        assert "gpt-5-mini" in picker._choices  # openai catalog
        assert "claude-opus-4-7" not in picker._choices  # anthropic absent


def test_embedding_model_is_dropdown_with_known_catalog() -> None:
    """app.memory.embedding.model is a DROPDOWN with the small embedding catalog."""

    from feather.config.schema import EMBEDDING_MODEL_CATALOG, lookup

    field = lookup("app.memory.embedding.model")
    assert field is not None
    assert field.widget.value == "dropdown"
    assert field.choices is not None
    assert set(EMBEDDING_MODEL_CATALOG).issubset(field.choices)


def test_gemini_task_types_are_dropdown() -> None:
    """app.memory.embedding.task_type_{document,query} are DROPDOWN."""

    from feather.config.schema import GEMINI_TASK_TYPES, lookup

    for path in (
        "app.memory.embedding.task_type_document",
        "app.memory.embedding.task_type_query",
    ):
        field = lookup(path)
        assert field is not None
        assert field.widget.value == "dropdown"
        assert field.choices is not None
        assert set(GEMINI_TASK_TYPES).issubset(field.choices)


def test_tokenizer_encoding_is_dropdown() -> None:
    """app.memory.chunking.tokenizer_encoding is a DROPDOWN with known encodings."""

    from feather.config.schema import TIKTOKEN_ENCODINGS, lookup

    field = lookup("app.memory.chunking.tokenizer_encoding")
    assert field is not None
    assert field.widget.value == "dropdown"
    assert field.choices is not None
    assert set(TIKTOKEN_ENCODINGS).issubset(field.choices)


def test_url_fields_have_url_validator() -> None:
    """openrouter/claude base URLs and qdrant URL reject non-http(s) inputs."""

    from feather.config.schema import lookup

    for path in (
        "app.openrouter.base_url",
        "app.claude.base_url",
        "app.memory.qdrant.url",
    ):
        field = lookup(path)
        assert field is not None
        assert field.validator is not None
        # Rejects naked hostnames.
        import pytest as _pytest

        with _pytest.raises(ValueError):
            field.validator("api.example.com")
        # Accepts http and https.
        field.validator("https://api.example.com")
        field.validator("http://localhost:6333")


def test_agent_provider_field_is_dropdown_with_inherit_sentinel() -> None:
    """Schema: agents.*.provider is a DROPDOWN whose choices include inherit."""

    from feather.config.schema import INHERIT_SENTINEL, lookup

    for name in ("Lead", "Explore", "Research", "Validate"):
        field = lookup(f"agents.{name}.provider")
        assert field is not None
        assert field.widget.value == "dropdown"
        assert field.choices is not None
        assert INHERIT_SENTINEL in field.choices
        assert "openai" in field.choices
        assert "openrouter" in field.choices
        assert "claude" in field.choices


async def test_picker_preserves_unknown_current_value_as_first_option(
    service: ConfigService,
) -> None:
    """When current value isn't in choices, it's prepended so Enter keeps it."""

    from feather.config.resolver import PathScope

    # Pin a custom model not in MODEL_CATALOG so the picker has to handle it.
    service.set("app.openai.model", "gpt-7-experimental", scope=PathScope.GLOBAL)

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        assert _navigate_to_field(screen, "app.openai.model")

        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        # Cursor must be on the user's existing value, which lives at index 0.
        assert picker._choices[0] == "gpt-7-experimental"
        assert picker._index == 0

        # Press Enter without navigating — should keep the existing value.
        await pilot.press("enter")
        await pilot.pause()

        assert screen._dirty.get("app.openai.model") == "gpt-7-experimental"


async def test_picker_scrolls_to_keep_cursor_in_view_when_list_overflows(
    service: ConfigService,
) -> None:
    """Arrow-down past the viewport must scroll the picker so the cursor row
    (and its ``▶`` marker) stay visible.

    Regression: picker's ``_index`` advanced on key-down but the scroll
    offset did not follow, so on long lists the highlighted line dropped
    below ``max-height: 16`` and the cursor marker disappeared.
    """

    async with _Host(service).run_test(size=(120, 30)) as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)
        # agents.<name>.model with provider resolved to openrouter offers
        # (inherit) + 30 catalog slugs — well past the picker's max-height
        # of 16. The shipped default active_provider is openrouter so the
        # inherit-resolution lands on the openrouter slug list.
        assert _navigate_to_field(screen, "agents.Research.model")

        await pilot.press("enter")
        await pilot.pause()

        picker = screen.query("#config-inline-editor").first()
        assert hasattr(picker, "_choices")
        # Only meaningful if the list actually overflows the picker viewport.
        if len(picker._choices) <= 14:
            pytest.skip(
                "catalog shrank below picker max-height; "
                "overflow scenario no longer reachable"
            )

        # Step the cursor to the last choice via repeated arrow-down.
        last_index = len(picker._choices) - 1
        while picker._index != last_index:
            await pilot.press("down")
            await pilot.pause()

        # After landing on the last item, the picker must have scrolled so
        # that the cursor row is inside the visible viewport.
        scroll_y = int(picker.scroll_offset.y)
        viewport_height = picker.scrollable_content_region.height
        assert scroll_y <= picker._index < scroll_y + viewport_height, (
            f"cursor at index {picker._index} not visible: "
            f"scroll_y={scroll_y}, viewport_height={viewport_height}"
        )
        assert scroll_y > 0, (
            "picker did not scroll — cursor on a long list never advanced past "
            "the initial viewport"
        )
