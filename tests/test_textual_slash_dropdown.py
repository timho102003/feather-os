"""Tests for the Textual slash-command dropdown integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from feather.tui.slash_commands import (
    SlashCommand,
    SlashCommandRegistry,
    default_registry,
)
from feather.tui.app import (
    FeatherTextualApp,
    SlashCommandDropdown,
    render_dropdown_text,
    render_help_text,
)


def _registry() -> SlashCommandRegistry:
    return SlashCommandRegistry(
        [
            SlashCommand(name="help", summary="Show help", aliases=("?",)),
            SlashCommand(name="exit", summary="Leave the TUI", aliases=("quit",)),
            SlashCommand(name="clear", summary="Clear transcript"),
            SlashCommand(name="copy", summary="Copy transcript"),
        ]
    )


def test_dropdown_starts_closed_and_empty() -> None:
    dropdown = SlashCommandDropdown(_registry())

    assert dropdown.is_open is False
    assert dropdown.selected_command is None
    assert dropdown.matches == ()


def test_update_for_no_slash_closes_dropdown() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("hello world")

    assert dropdown.is_open is False
    assert dropdown.matches == ()


def test_update_for_bare_slash_lists_all_commands() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("/")

    assert dropdown.is_open is True
    assert {cmd.name for cmd in dropdown.matches} == {"help", "exit", "clear", "copy"}
    assert dropdown.selected_command is not None
    assert dropdown.selected_command.name == "help"


def test_update_for_prefix_filters_matches() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("/c")

    assert dropdown.is_open is True
    assert {cmd.name for cmd in dropdown.matches} == {"clear", "copy"}


def test_update_for_command_with_args_closes_dropdown() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("/help me")

    assert dropdown.is_open is False


def test_update_for_unknown_prefix_keeps_dropdown_open_with_empty_state() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("/zzz")

    # Stays open so the user gets feedback that nothing matched, but with
    # zero matches and no selectable command.
    assert dropdown.is_open is True
    assert dropdown.matches == ()
    assert dropdown.selected_command is None


def test_move_selection_wraps_around() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("/c")
    first = dropdown.selected_command

    dropdown.move_selection(1)
    second = dropdown.selected_command

    dropdown.move_selection(1)
    third = dropdown.selected_command

    assert first is not None and second is not None and third is not None
    assert first.name != second.name
    assert third.name == first.name  # wraps after 2 entries


def test_move_selection_negative_wraps_to_last() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("/c")
    dropdown.move_selection(-1)

    last = dropdown.selected_command
    assert last is not None
    assert last.name == "copy"


def test_move_selection_is_safe_when_no_matches() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("/zzz")

    dropdown.move_selection(1)  # must not raise

    assert dropdown.selected_command is None


def test_dropdown_text_highlights_selected_command() -> None:
    registry = _registry()
    cmds = registry.match("c")  # clear, copy
    text = render_dropdown_text(cmds, selected_index=1)

    assert "clear" in text.plain
    assert "copy" in text.plain
    # Selection marker appears once and is on the second row (copy).
    plain_lines = text.plain.splitlines()
    selection_lines = [line for line in plain_lines if "›" in line]
    assert len(selection_lines) == 1
    assert "copy" in selection_lines[0]


def test_dropdown_text_renders_empty_state_when_no_matches() -> None:
    text = render_dropdown_text((), selected_index=0)

    assert "No commands match" in text.plain


def test_help_text_lists_every_registered_command() -> None:
    registry = default_registry()
    text = render_help_text(registry)

    plain = text.plain
    for cmd in registry.all():
        assert f"/{cmd.name}" in plain
        assert cmd.summary in plain


def test_help_text_renders_each_category_header_only_once() -> None:
    """Regression M3: interleaved categories used to print duplicate headers."""

    registry = SlashCommandRegistry(
        [
            SlashCommand(name="a", summary="x", category="info"),
            SlashCommand(name="b", summary="x", category="view"),
            SlashCommand(name="c", summary="x", category="info"),
        ]
    )
    text = render_help_text(registry)
    plain = text.plain

    # Each category header appears exactly once.
    headers = [line for line in plain.splitlines() if line.startswith("[") and line.endswith("]")]
    assert headers.count("[info]") == 1
    assert headers.count("[view]") == 1
    # Every command still appears under its category.
    for name in ("a", "b", "c"):
        assert f"/{name}" in plain


def test_app_holds_a_default_slash_registry() -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")

    assert app.slash_registry is not None
    names = {cmd.name for cmd in app.slash_registry.all()}
    assert {"help", "exit", "quit"}.intersection(names) == {"help", "exit"}


def test_app_dispatch_unknown_slash_writes_warning(monkeypatch) -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    recorded: list[tuple[str, str, str]] = []

    def fake_marker(title: str, text: str = "", *, style: str = "grey70") -> None:
        recorded.append((title, text, style))

    monkeypatch.setattr(app, "_write_marker", fake_marker)

    handled = app._dispatch_slash_input("/banana")

    assert handled is True
    assert recorded
    assert "Unknown command" in recorded[0][0]
    assert "/banana" in recorded[0][1]
    assert recorded[0][2] in {"yellow", "red"}


def test_app_dispatch_known_command_invokes_handler(monkeypatch) -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    called: list[tuple[str, str]] = []

    def fake_help(args: str) -> None:
        called.append(("help", args))

    app._slash_handlers["help"] = fake_help
    # Mark help as accepting args so the dispatcher does not emit a
    # "ignored extra text" marker (which would query the DOM).
    app._slash_handlers_accepts_args.add("help")
    handled = app._dispatch_slash_input("/help me")

    assert handled is True
    assert called == [("help", "me")]


def test_app_dispatch_returns_false_for_non_slash_text() -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    handled = app._dispatch_slash_input("just talking")

    assert handled is False


def test_app_help_handler_writes_help_block(monkeypatch) -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    written: list[tuple[str, str]] = []

    def fake_write(title: str, body: str, **_: object) -> None:
        written.append((title, body))

    monkeypatch.setattr(app, "_write_conversation", fake_write)

    app._cmd_help("")

    assert written
    title, body = written[0]
    assert "Slash" in title or "Commands" in title or "Help" in title
    for cmd in app.slash_registry.all():
        assert f"/{cmd.name}" in body


def test_app_clear_handler_resets_transcript(monkeypatch) -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    from feather.tui.app import _ConversationBlock

    app._conversation_blocks = [
        _ConversationBlock(
            title="You",
            body="hi",
            label_style="bold cyan",
            body_style="white",
        )
    ]
    app._transcript_blocks = ["You\n  hi"]
    app._assistant_parts = ["streaming bit"]

    rendered: list[int] = []

    def fake_render() -> None:
        rendered.append(1)

    written: list[tuple[str, str]] = []

    def fake_marker(title: str, text: str = "", **_: object) -> None:
        written.append((title, text))

    monkeypatch.setattr(app, "_render_conversation", fake_render)
    monkeypatch.setattr(app, "_write_marker", fake_marker)

    app._cmd_clear("")

    assert app._conversation_blocks == []
    assert app._transcript_blocks == []
    assert app._assistant_parts == []
    assert rendered  # transcript was redrawn
    assert written and "Cleared" in written[0][0]


def test_qdrant_unknown_subcommand_writes_warning(monkeypatch) -> None:
    """`/qdrant <bogus>` writes a yellow marker, no Docker call."""

    app = FeatherTextualApp(root=Path("."), session_id="s1")
    spawned: list = []
    monkeypatch.setattr(app, "_spawn_async_command", spawned.append)
    written: list[tuple[str, str, str]] = []

    def fake_marker(title: str, text: str = "", *, style: str = "grey70") -> None:
        written.append((title, text, style))

    monkeypatch.setattr(app, "_write_marker", fake_marker)

    app._cmd_qdrant("banana split")

    assert spawned == []
    assert written
    assert "unknown subcommand" in written[0][1]
    assert written[0][2] == "yellow"


def test_qdrant_help_subcommand_writes_help_block(monkeypatch) -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    written: list[tuple[str, str]] = []
    spawned: list = []
    monkeypatch.setattr(app, "_spawn_async_command", spawned.append)
    monkeypatch.setattr(
        app,
        "_write_conversation",
        lambda title, body, **_: written.append((title, body)),
    )

    app._cmd_qdrant("help")

    assert spawned == []
    assert written
    title, body = written[0]
    assert "qdrant" in title
    assert "/qdrant status" in body
    assert "/qdrant start" in body
    assert "/qdrant stop" in body
    assert "/qdrant remove" in body


def test_qdrant_status_subcommand_dispatches_async(monkeypatch) -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    spawned: list = []
    monkeypatch.setattr(app, "_spawn_async_command", spawned.append)
    # Empty args defaults to status.
    app._cmd_qdrant("")
    app._cmd_qdrant("status")
    app._cmd_qdrant("start")
    app._cmd_qdrant("stop")
    app._cmd_qdrant("remove")
    assert len(spawned) == 5


# ---- Compose-context detection + gating ------------------------------------


def test_resolve_qdrant_url_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)

    from feather.tui.app import _resolve_qdrant_url

    url, source = _resolve_qdrant_url()
    assert url == "http://localhost:6333"
    assert source == "default"


def test_resolve_qdrant_url_reports_env_source(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")

    from feather.tui.app import _resolve_qdrant_url

    url, source = _resolve_qdrant_url()
    assert url == "https://qdrant.example"
    assert source == "env"


def test_is_compose_managed_false_when_env_unset(monkeypatch, tmp_path) -> None:
    """Default bare-metal: no /.dockerenv, no QDRANT_URL → not compose."""

    monkeypatch.delenv("QDRANT_URL", raising=False)
    # Force the dockerenv probe to miss by patching Path checks via the
    # filesystem state — the check is `Path('/.dockerenv').exists()`,
    # which on a normal dev host is False. We rely on that being the
    # case in the test environment; the assert below documents the
    # contract.
    from feather.tui.app import _is_compose_managed_qdrant
    from pathlib import Path as _Path

    if _Path("/.dockerenv").exists():
        import pytest

        pytest.skip("test host is itself inside a container; cannot exercise this branch")

    assert _is_compose_managed_qdrant() is False


def test_is_compose_managed_true_when_env_points_at_remote_host(
    monkeypatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")

    from feather.tui.app import _is_compose_managed_qdrant
    from pathlib import Path as _Path

    if _Path("/.dockerenv").exists():
        # /.dockerenv would short-circuit True regardless; that's still
        # a valid compose context.
        assert _is_compose_managed_qdrant() is True
    else:
        assert _is_compose_managed_qdrant() is True


def test_is_compose_managed_false_for_localhost_env(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    from feather.tui.app import _is_compose_managed_qdrant
    from pathlib import Path as _Path

    if _Path("/.dockerenv").exists():
        import pytest

        pytest.skip("test host is itself inside a container")

    assert _is_compose_managed_qdrant() is False


def test_qdrant_start_in_compose_context_writes_friendly_marker(
    monkeypatch,
) -> None:
    """Lifecycle commands in compose context defer to compose, no docker call."""

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    written: list[tuple[str, str, str]] = []

    def fake_marker(title: str, text: str = "", *, style: str = "grey70") -> None:
        written.append((title, text, style))

    monkeypatch.setattr(app, "_write_marker", fake_marker)
    # Sanity: the test must NEVER call onboarding helpers if the
    # compose path is correctly taken.
    import feather.onboarding as ob

    monkeypatch.setattr(
        ob,
        "ensure_local_qdrant_container",
        lambda *a, **k: pytest_fail("docker call leaked"),
    )
    import asyncio

    asyncio.run(app._qdrant_start_async())

    assert written
    assert "managed by docker compose" in written[0][1]
    assert "docker compose start qdrant" in written[0][1]


def test_qdrant_stop_in_compose_context_writes_friendly_marker(
    monkeypatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    written: list[tuple[str, str, str]] = []

    def fake_marker(title: str, text: str = "", *, style: str = "grey70") -> None:
        written.append((title, text, style))

    monkeypatch.setattr(app, "_write_marker", fake_marker)

    import asyncio

    asyncio.run(app._qdrant_stop_async())

    assert written
    assert "docker compose stop qdrant" in written[0][1]


def test_qdrant_remove_in_compose_context_writes_friendly_marker(
    monkeypatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    written: list[tuple[str, str, str]] = []

    def fake_marker(title: str, text: str = "", *, style: str = "grey70") -> None:
        written.append((title, text, style))

    monkeypatch.setattr(app, "_write_marker", fake_marker)

    import asyncio

    asyncio.run(app._qdrant_remove_async())

    assert written
    assert "docker compose down qdrant" in written[0][1]


def pytest_fail(msg: str) -> None:  # tiny helper to keep test imports compact
    import pytest

    pytest.fail(msg)


def test_app_onboard_handler_clears_markers_and_exits(
    tmp_path, monkeypatch
) -> None:
    """`/onboard` removes the completion markers and exits the TUI.

    Next ``feather tui`` run will then re-trigger the wizard because
    ``maybe_run_onboarding`` checks for both ``onboarded.json`` and
    ``user.md``.
    """

    feather_dir = tmp_path / ".feather"
    feather_dir.mkdir(parents=True)
    onboarded = feather_dir / "onboarded.json"
    user_md = feather_dir / "user.md"
    onboarded.write_text("{\"completed\": true}", encoding="utf-8")
    user_md.write_text("name: Alice\n", encoding="utf-8")

    app = FeatherTextualApp(root=tmp_path, session_id="s1")
    written: list[tuple[str, str]] = []
    exited = {"called": False}

    def fake_write(title: str, body: str, **_: object) -> None:
        written.append((title, body))

    def fake_exit(*_a, **_k) -> None:
        exited["called"] = True

    monkeypatch.setattr(app, "_write_conversation", fake_write)
    monkeypatch.setattr(app, "exit", fake_exit)

    app._cmd_onboard("")

    assert not onboarded.exists()
    assert not user_md.exists()
    assert exited["called"] is True
    assert written
    title, body = written[0]
    assert "Onboard" in title
    assert "feather tui" in body or "feather onboard" in body


def test_app_onboard_handler_is_safe_when_markers_absent(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / ".feather").mkdir(parents=True)

    app = FeatherTextualApp(root=tmp_path, session_id="s1")
    written: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app,
        "_write_conversation",
        lambda title, body, **_: written.append((title, body)),
    )
    monkeypatch.setattr(app, "exit", lambda *a, **k: None)

    app._cmd_onboard("")

    assert written
    body = written[0][1]
    # Friendly message even with nothing to delete.
    assert "no completion markers were present" in body or "Onboarding markers cleared" in body


def test_dropdown_close_resets_state() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("/c")
    assert dropdown.is_open is True

    dropdown.close()

    assert dropdown.is_open is False
    assert dropdown.matches == ()
    assert dropdown.selected_command is None


def test_dropdown_select_returns_selected_command_text() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("/he")

    cmd = dropdown.selected_command
    assert cmd is not None and cmd.name == "help"
    assert dropdown.completion_text() == "/help "


def test_dropdown_completion_text_is_none_without_selection() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("/zzz")

    assert dropdown.completion_text() is None


def test_dropdown_does_not_open_for_text_without_leading_slash_after_strip() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("\nhello")
    assert dropdown.is_open is False


def test_dropdown_opens_for_lstripped_slash_input() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("   /he")

    assert dropdown.is_open is True
    cmd = dropdown.selected_command
    assert cmd is not None and cmd.name == "help"


def test_dropdown_preserves_selection_when_match_set_unchanged() -> None:
    """Regression N4: typing a refining char should not reset selection."""

    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("/c")
    dropdown.move_selection(1)
    second = dropdown.selected_command
    assert second is not None

    # Re-update with a query that produces an identical match list.
    dropdown.update_for_text("/c")
    assert dropdown.selected_command is not None
    assert dropdown.selected_command.name == second.name


def test_dropdown_resets_selection_when_match_set_changes() -> None:
    dropdown = SlashCommandDropdown(_registry())
    dropdown.update_for_text("/c")
    dropdown.move_selection(1)

    # New query => new match set => reset to top.
    dropdown.update_for_text("/h")
    assert dropdown.selected_command is not None
    assert dropdown.selected_command.name == "help"


def test_dispatch_warns_when_command_has_unexpected_args(monkeypatch) -> None:
    """Regression M1: pasted multi-line body must not silently vanish."""

    from feather.tui.app import FeatherTextualApp

    app = FeatherTextualApp(root=Path("."), session_id="s1")
    recorded: list[tuple[str, str, str]] = []

    def fake_marker(title: str, text: str = "", *, style: str = "grey70") -> None:
        recorded.append((title, text, style))

    monkeypatch.setattr(app, "_write_marker", fake_marker)
    monkeypatch.setattr(app, "_write_conversation", lambda *a, **k: None)

    handled = app._dispatch_slash_input("/help\nactual question text")

    assert handled is True
    # A warning marker mentions the discarded extra content.
    discard_markers = [
        entry for entry in recorded if "ignored" in entry[1].lower() or "discarded" in entry[1].lower()
    ]
    assert discard_markers, f"expected discard warning, got {recorded!r}"
    assert "actual question text" in discard_markers[0][1]


def test_dispatch_does_not_warn_when_handler_consumes_args(monkeypatch) -> None:
    from feather.tui.app import FeatherTextualApp

    app = FeatherTextualApp(root=Path("."), session_id="s1")
    consumed: list[str] = []

    def fake_handler(args: str) -> None:
        consumed.append(args)

    app._slash_handlers["help"] = fake_handler
    # Mark help as accepting args explicitly:
    app._slash_handlers_accepts_args.add("help")

    recorded: list[tuple[str, str, str]] = []

    def fake_marker(title: str, text: str = "", *, style: str = "grey70") -> None:
        recorded.append((title, text, style))

    monkeypatch.setattr(app, "_write_marker", fake_marker)

    handled = app._dispatch_slash_input("/help my question")
    assert handled is True
    assert consumed == ["my question"]
    # No "ignored" warning when the handler consumes args.
    assert not any("ignored" in entry[1].lower() for entry in recorded)


def test_dispatch_writes_awaiting_reminder_when_paused(monkeypatch) -> None:
    """Regression N7: slash command while awaiting must remind the user."""

    from feather.tui.app import FeatherTextualApp

    app = FeatherTextualApp(root=Path("."), session_id="s1")
    app._awaiting_event.set()
    recorded: list[tuple[str, str, str]] = []

    def fake_marker(title: str, text: str = "", *, style: str = "grey70") -> None:
        recorded.append((title, text, style))

    monkeypatch.setattr(app, "_write_marker", fake_marker)
    monkeypatch.setattr(app, "_write_conversation", lambda *a, **k: None)
    # ``_cmd_clear`` calls ``_render_conversation`` which queries the DOM.
    monkeypatch.setattr(app, "_render_conversation", lambda: None)

    handled = app._dispatch_slash_input("/clear")

    assert handled is True
    awaiting_markers = [
        entry for entry in recorded if "awaiting" in entry[1].lower() or "still" in entry[1].lower()
    ]
    assert awaiting_markers, f"expected awaiting reminder, got {recorded!r}"


def test_app_suppresses_change_event_after_tab_completion() -> None:
    """Regression C1: Tab autocomplete must not re-open the dropdown."""

    from feather.tui.app import FeatherTextualApp

    app = FeatherTextualApp(root=Path("."), session_id="s1")
    # Simulate the composer text-area state after Tab handler ran.
    dropdown = SlashCommandDropdown(default_registry(), id="slash_dropdown")
    app._slash_dropdown_widget = dropdown
    dropdown.update_for_text("/he")
    # Tab handler closes dropdown and arms suppression.
    dropdown.close()
    app._slash_suppress_next_change = True

    # The Changed event posted by ``self.text = "/help "`` arrives now.
    class _FakeTextArea:
        id = "composer"
        text = "/help "

    class _FakeChanged:
        text_area = _FakeTextArea()

    app.on_text_area_changed(_FakeChanged())  # type: ignore[arg-type]

    assert dropdown.is_open is False
    # Suppression flag is consumed (one-shot).
    assert app._slash_suppress_next_change is False


def test_app_subsequent_change_after_tab_still_updates(monkeypatch) -> None:
    from feather.tui.app import FeatherTextualApp

    app = FeatherTextualApp(root=Path("."), session_id="s1")
    dropdown = SlashCommandDropdown(default_registry(), id="slash_dropdown")
    app._slash_dropdown_widget = dropdown
    app._slash_suppress_next_change = True

    class _FakeTextArea:
        id = "composer"
        text = "/help "

    class _FakeChanged:
        text_area = _FakeTextArea()

    # First change consumes the flag.
    app.on_text_area_changed(_FakeChanged())  # type: ignore[arg-type]
    assert app._slash_suppress_next_change is False

    # User now types something. The next change should drive the dropdown
    # normally — suppression must not stick.
    _FakeTextArea.text = "/h"
    app.on_text_area_changed(_FakeChanged())  # type: ignore[arg-type]
    assert dropdown.is_open is True


def _wide_registry(count: int) -> SlashCommandRegistry:
    return SlashCommandRegistry(
        [SlashCommand(name=f"cmd{i}", summary=f"summary {i}") for i in range(count)]
    )


def test_dropdown_viewport_keeps_selection_visible_when_moving_down() -> None:
    """User reported: arrow-down past the visible window stops updating display."""

    dropdown = SlashCommandDropdown(_wide_registry(12))
    dropdown.update_for_text("/")
    visible = dropdown.max_visible_rows
    assert visible >= 4 and visible < 12

    # Move selection past the bottom of the initial window.
    for _ in range(visible):
        dropdown.move_selection(1)

    # The selected entry must be inside the rendered window.
    assert dropdown.first_visible_index <= dropdown._selected
    assert dropdown._selected < dropdown.first_visible_index + visible


def test_dropdown_viewport_keeps_selection_visible_when_moving_up() -> None:
    dropdown = SlashCommandDropdown(_wide_registry(12))
    dropdown.update_for_text("/")
    visible = dropdown.max_visible_rows

    # Jump to the last entry by wrapping upward.
    dropdown.move_selection(-1)
    assert dropdown._selected == 11
    assert dropdown.first_visible_index <= 11
    assert 11 < dropdown.first_visible_index + visible


def test_dropdown_viewport_resets_when_match_set_changes() -> None:
    dropdown = SlashCommandDropdown(_wide_registry(12))
    dropdown.update_for_text("/")
    for _ in range(dropdown.max_visible_rows):
        dropdown.move_selection(1)
    assert dropdown.first_visible_index > 0

    # New text → new match set → window resets.
    dropdown.update_for_text("/cmd1")  # cmd1, cmd10, cmd11
    assert dropdown.first_visible_index == 0


def test_dropdown_render_includes_overflow_hints() -> None:
    matches = SlashCommandRegistry(
        [SlashCommand(name=f"x{i}", summary="x") for i in range(8)]
    ).match("")
    text = render_dropdown_text(
        matches,
        selected_index=4,
        first_visible=2,
        max_visible=3,
    )

    plain = text.plain
    # "↑ N more" hint above the window when first_visible > 0.
    assert "more above" in plain or "↑" in plain
    # "↓ N more" hint below when window does not reach the end.
    assert "more below" in plain or "↓" in plain
    # Only the windowed entries are listed.
    assert "x2" in plain and "x3" in plain and "x4" in plain
    # Entries outside the window are not listed.
    assert "x0" not in plain and "x7" not in plain


def test_dropdown_render_no_hints_when_window_covers_all() -> None:
    matches = SlashCommandRegistry(
        [SlashCommand(name=f"x{i}", summary="x") for i in range(3)]
    ).match("")
    text = render_dropdown_text(
        matches,
        selected_index=1,
        first_visible=0,
        max_visible=6,
    )

    plain = text.plain
    assert "more above" not in plain and "more below" not in plain
    assert "↑" not in plain and "↓" not in plain


def test_default_registry_handlers_cover_every_command() -> None:
    """Nit 1: every default command must have a bound handler."""

    from feather.tui.app import FeatherTextualApp

    app = FeatherTextualApp(root=Path("."), session_id="s1")
    for cmd in app.slash_registry.all():
        assert cmd.name.lower() in app._slash_handlers, (
            f"no handler bound for /{cmd.name}"
        )
        for alias in cmd.aliases:
            assert alias.lower() in app._slash_handlers, (
                f"no handler bound for alias /{alias} of /{cmd.name}"
            )
