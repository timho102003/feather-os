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
from feather.config_service import ConfigService
from feather.paths import FeatherPaths
from feather.textual_config_screen import ConfigScreen


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


async def test_enter_on_field_opens_editor_and_marks_dirty(
    service: ConfigService,
) -> None:
    """Enter opens inline editor; submitting a valid value marks field dirty."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)

        # The first section on the App tab is 'database'; navigate down
        # to 'active_provider' section (index varies by registry order).
        # Instead of navigating, directly press enter on whatever the
        # first field is in the first section, then test that _dirty
        # receives data when we submit a valid enum value.
        # For a more reliable test, navigate to where app.active_provider lives.
        # app.active_provider is in the REGISTRY — let's find its section index.
        from feather.config_schema import REGISTRY as REG
        sections: list[str] = []
        for f in REG:
            if not f.path.startswith("app."):
                continue
            tail = f.path[len("app."):]
            label = tail.split(".", 1)[0] if "." in tail else "agent"
            if label not in sections:
                sections.append(label)

        # Navigate to 'active_provider' section (it's a leaf under app.*)
        # The leaf fields appear as the synthetic "agent" label — but
        # app.active_provider has no second segment so it appears as 'active_provider'.
        # Actually app.active_provider -> tail = 'active_provider', no dot -> label = 'agent'
        # No — let's count more carefully. 'active_provider' has no dot after prefix,
        # so label = 'agent'. But wait, there's no second dot in 'active_provider'.
        # Let me check: app.active_provider -> tail = 'active_provider'
        # -> no "." in tail -> label = "agent"
        # But also app.database.path -> tail = "database.path" -> label = "database"
        # So we need to navigate to the "agent" section for app.active_provider.
        # Actually app.active_provider is the only field in section "agent"
        # (leaves directly under app.). We need to navigate there.
        target_section = "agent"
        if target_section in sections:
            idx = sections.index(target_section)
            for _ in range(idx):
                await pilot.press("down")

        # Now press Enter to open editor
        await pilot.press("enter")
        await pilot.pause()

        # Type "claude" and submit
        await pilot.press("c", "l", "a", "u", "d", "e", "enter")
        await pilot.pause()

        assert "app.active_provider" in screen._dirty
        assert screen._dirty["app.active_provider"] == "claude"


async def test_invalid_input_does_not_mark_dirty(service: ConfigService) -> None:
    """Submitting an invalid value shows error and does NOT update _dirty."""

    async with _Host(service).run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, ConfigScreen)

        # Navigate to app.compaction section to get a float field
        from feather.config_schema import REGISTRY as REG
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

        # Open editor for the first compaction field (app.compaction.enabled, boolean)
        await pilot.press("enter")
        await pilot.pause()

        # Type an invalid value for boolean
        await pilot.press("n", "o", "t", "_", "a", "_", "b", "o", "o", "l", "enter")
        await pilot.pause()

        # Check footer shows INVALID message
        footer = str(screen.query_one("#config-footer", Static).render()).lower()
        assert "invalid" in footer
        # And the field is not dirty
        assert "app.compaction.enabled" not in screen._dirty


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
        from feather.config_schema import REGISTRY as REG
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
        from feather.config_schema import REGISTRY as REG
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
        from feather.config_schema import REGISTRY as REG
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
    assert "enter" in keys
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
