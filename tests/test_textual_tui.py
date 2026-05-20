"""Tests for the Textual TUI render helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from textual.geometry import Region

from feather.models import RuntimeEvent, TaskRecord, TaskStatus
from feather.textual_tui import (
    ComposerTextArea,
    FeatherTextualApp,
    build_transcript_text,
    build_header_text,
    build_work_text,
    format_transcript_block,
    format_task_row,
    format_agent_message_event,
    is_exit_command,
    normalize_pasted_attachment_text,
    region_contains_point,
    summarize_agent_message_update,
    summarize_user_input_for_display,
    _mouse_enabled,
)


def test_header_text_includes_status_context_queue_and_active_tool() -> None:
    text = build_header_text(
        agent_name="Lead",
        status="running",
        context_ratio=0.42,
        queue_depth=2,
        active_tool="web_search",
        session_id="session-1",
    )

    plain = text.plain
    assert "Feather" in plain
    assert "Lead" in plain
    assert "running" in plain
    assert "ctx 42%" in plain
    assert "queued 2" in plain
    assert "active web_search" in plain
    assert "lead session session-1" in plain


def test_work_text_renders_queued_queries_and_future_tasks() -> None:
    text = build_work_text(
        queue_depth=2,
        queued_messages=("first queued", "second queued"),
        running_agents=("research abc123: compare options",),
        task_rows=("RUN   research   abc123   compare options",),
        task_updates=("started research abc123",),
    )

    plain = text.plain
    assert "Queued queries" in plain
    assert "1. first queued" in plain
    assert "2. second queued" in plain
    assert "Tasks" in plain
    assert "RUN   research   abc123   compare options" in plain
    assert "Live sub-agents" in plain
    assert "research abc123: compare options" in plain
    assert "Recent task updates" in plain
    assert "started research abc123" in plain
    assert plain.index("Live sub-agents") < plain.index("Tasks")


def test_user_input_display_summarizes_dropped_attachments(tmp_path: Path) -> None:
    """The TUI transcript should show compact placeholders for dropped files."""

    image = tmp_path / "image.png"
    document = tmp_path / "paper.pdf"
    image.write_bytes(b"png")
    document.write_bytes(b"%PDF-1.4")

    assert summarize_user_input_for_display(
        f"compare {image} {document}",
        tmp_path,
    ) == "compare\n[image #1] [File #1]"


def test_pasted_external_file_path_becomes_explicit_file_uri(tmp_path: Path) -> None:
    """Terminal file drops from outside the workspace should attach cleanly."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "Downloads" / "paper.pdf"
    outside.parent.mkdir()
    outside.write_bytes(b"%PDF-1.4")

    normalized = normalize_pasted_attachment_text(str(outside), workspace)

    assert normalized == outside.resolve().as_uri()
    assert summarize_user_input_for_display(f"read {normalized}", workspace) == (
        "read\n[File #1]"
    )


def test_pasted_prose_with_path_is_not_rewritten(tmp_path: Path) -> None:
    """Only paste/drop payloads that are just paths should be normalized."""

    outside = tmp_path / "paper.pdf"
    outside.write_bytes(b"%PDF-1.4")

    assert normalize_pasted_attachment_text(f"read {outside}", tmp_path) == (
        f"read {outside}"
    )


def test_work_text_empty_state_is_calm() -> None:
    text = build_work_text(
        queue_depth=0,
        queued_messages=(),
        running_agents=(),
    )

    assert text.plain == (
        "No queued queries, tracked tasks, live sub-agents, or recent task updates."
    )


def test_agent_message_event_renders_full_body_from_payload() -> None:
    body = "final report\n" + ("A" * 500)
    event = RuntimeEvent(
        kind="agent_message_received",
        text="research (session-123): 1 message(s), 513 chars\n    preview...",
        payload={
            "from_agent_name": "research",
            "from_session_id": "session-123",
            "count": 1,
            "total_chars": len(body),
            "bodies": [body],
        },
    )

    rendered = format_agent_message_event(event)

    assert "research session" in rendered
    assert body in rendered
    assert "preview..." not in rendered


def test_agent_message_update_marks_subagent_completed() -> None:
    event = RuntimeEvent(
        kind="agent_message_received",
        payload={
            "from_agent_name": "research",
            "from_session_id": "abcdef12-3456",
            "count": 1,
            "total_chars": 1987,
        },
    )

    assert summarize_agent_message_update(event) == (
        "completed research abcdef12 (1 message, 1987 chars)"
    )


def test_task_row_renders_status_agent_session_and_blocker() -> None:
    task = TaskRecord(
        id="task-1",
        plan_id=None,
        parent_task_id=None,
        title="Research prices",
        description="",
        success_criteria="",
        required_outputs=[],
        status=TaskStatus.BLOCKED_NEEDS_INPUT,
        responsible_agent_name="research",
        responsible_session_id="abcdef12-3456",
        lead_session_id="lead",
        blocked_question="Include rowhouses?",
        blocked_correlation_id="corr",
        error=None,
        created_at="now",
        updated_at="now",
    )

    row = format_task_row(task)

    assert "BLOCK" in row
    assert "research" in row
    assert "abcdef12" in row
    assert "Include rowhouses?" in row


def test_task_row_marks_running_task_live_when_registry_has_session() -> None:
    task = TaskRecord(
        id="task-1",
        plan_id=None,
        parent_task_id=None,
        title="Research prices",
        description="",
        success_criteria="",
        required_outputs=[],
        status=TaskStatus.RUNNING,
        responsible_agent_name="research",
        responsible_session_id="abcdef12-3456",
        lead_session_id="lead",
        blocked_question=None,
        blocked_correlation_id=None,
        error=None,
        created_at="now",
        updated_at="now",
    )

    row = format_task_row(task, live_sessions=frozenset({"abcdef12-3456"}))

    assert row.startswith("LIVE")
    assert "research" in row
    assert "Research prices" in row


def test_textual_app_can_be_constructed_without_runtime_side_effects() -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")

    assert app._requested_session_id == "s1"


def test_textual_app_renders_lead_deltas_as_they_stream(monkeypatch) -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    rendered = 0

    def render() -> None:
        nonlocal rendered
        rendered += 1

    monkeypatch.setattr(app, "_update_header", lambda: None)
    monkeypatch.setattr(app, "_render_conversation", render)

    app._handle_runtime_event(RuntimeEvent(kind="assistant_text_delta", text="hello"))

    assert app._assistant_parts == ["hello"]
    assert rendered == 1


def test_finishing_streamed_turn_clears_stream_before_final_write(monkeypatch) -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    app._agent = SimpleNamespace(config=SimpleNamespace(name="Lead"))
    app._assistant_parts = ["hello", " world"]
    recorded: list[tuple[str, str]] = []

    def write(title: str, body: str, **_: object) -> None:
        assert app._assistant_parts == []
        recorded.append((title, body))

    monkeypatch.setattr(app, "_write_conversation", write)

    app._finish_assistant_turn()

    assert recorded == [("Lead", "hello world")]


def test_tick_spinner_skips_render_when_nothing_is_streaming(monkeypatch) -> None:
    """An idle session must pay almost nothing for the 10 fps spinner timer:
    no conversation repaint, no frame advance. Only the early-out check
    runs."""

    app = FeatherTextualApp(root=Path("."), session_id="s1")
    rendered = 0

    def render() -> None:
        nonlocal rendered
        rendered += 1

    monkeypatch.setattr(app, "_render_conversation", render)
    assert app._assistant_parts == []
    starting_frame = app._spinner_frame

    app._tick_spinner()
    app._tick_spinner()
    app._tick_spinner()

    assert rendered == 0
    assert app._spinner_frame == starting_frame


def test_tick_spinner_advances_frame_and_renders_when_streaming(monkeypatch) -> None:
    """When ``_assistant_parts`` has content the tick must advance the
    frame counter AND repaint the conversation so the spinner glyph
    cycles even while text deltas are paused (model mid-reasoning)."""

    from feather.textual_tui import _SPINNER_FRAMES

    app = FeatherTextualApp(root=Path("."), session_id="s1")
    rendered = 0

    def render() -> None:
        nonlocal rendered
        rendered += 1

    monkeypatch.setattr(app, "_render_conversation", render)
    app._assistant_parts = ["streaming text"]
    app._spinner_frame = 0

    for _ in range(len(_SPINNER_FRAMES) + 2):
        app._tick_spinner()

    # Every tick repainted exactly once.
    assert rendered == len(_SPINNER_FRAMES) + 2
    # The frame counter cycles modulo the frame set; after N+2 ticks
    # starting from 0 we land on (N+2) % N == 2.
    assert app._spinner_frame == 2


def test_write_conversation_updates_render_state(monkeypatch) -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")
    rendered = 0

    def render() -> None:
        nonlocal rendered
        rendered += 1

    monkeypatch.setattr(app, "_render_conversation", render)

    app._write_conversation("Lead", "Hello", label_style="bold green")

    assert app._conversation_blocks[-1].title == "Lead"
    assert app._conversation_blocks[-1].body == "Hello"
    assert app._transcript_blocks[-1] == "Lead\n  Hello"
    assert rendered == 1


def test_textual_app_uses_composer_that_handles_mouse_wheel() -> None:
    app = FeatherTextualApp(root=Path("."), session_id="s1")

    nodes = list(app.compose())

    assert any(isinstance(node, ComposerTextArea) for node in nodes)


def test_transcript_helpers_render_plain_copyable_text() -> None:
    block = format_transcript_block("Lead", "Line one\nLine two")

    assert block == "Lead\n  Line one\n  Line two"
    assert build_transcript_text(("You\n  Hi", block, "")) == (
        "You\n  Hi\n\nLead\n  Line one\n  Line two"
    )


def test_region_contains_point_accepts_mouse_event_coordinates() -> None:
    region = Region(10, 20, 30, 5)

    assert region_contains_point(region, 10.8, 20.2)
    assert region_contains_point(region, 39, 24)
    assert not region_contains_point(region, 40, 24)


def test_textual_app_does_not_advertise_ctrl_q_exit_binding() -> None:
    keys = {binding.key for binding in FeatherTextualApp.BINDINGS}

    assert "ctrl+q" not in keys


def test_textual_app_has_work_scroll_bindings() -> None:
    keys = {binding.key for binding in FeatherTextualApp.BINDINGS}

    assert "shift+pageup" in keys
    assert "shift+pagedown" in keys


def test_mouse_reporting_defaults_on_and_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv("FEATHER_TUI_MOUSE", raising=False)
    assert _mouse_enabled()

    monkeypatch.setenv("FEATHER_TUI_MOUSE", "0")
    assert not _mouse_enabled()

    monkeypatch.setenv("FEATHER_TUI_MOUSE", "false")
    assert not _mouse_enabled()

    monkeypatch.setenv("FEATHER_TUI_MOUSE", "1")
    assert _mouse_enabled()


def test_textual_tui_cmd_config_apply_handles_exceptions() -> None:
    """_cmd_config's _apply worker must catch exceptions and surface them."""

    import inspect

    src = inspect.getsource(FeatherTextualApp._cmd_config)
    assert "apply error" in src or "try:" in src, (
        "_cmd_config's _apply worker must catch exceptions from apply_config_change"
    )


def test_app_priority_bindings_muted_under_modal_screen(monkeypatch) -> None:
    """App-level priority Enter/Esc bindings must be disabled while a
    ModalScreen is active; otherwise they preempt the modal's own bindings
    and the user cannot edit fields or close the modal.

    Regression test for the bug where `/config` opened the modal but Enter
    never opened the inline editor and Esc never closed the modal — both
    keys were being eaten by the App's priority bindings.
    """

    from unittest.mock import MagicMock

    from textual.screen import ModalScreen, Screen

    app = FeatherTextualApp(root=Path("/tmp"))

    # The actions that must be silenced under a modal include at minimum the
    # two that block the config TUI (submit, interrupt).
    assert "submit" in FeatherTextualApp._ACTIONS_MUTED_UNDER_MODAL
    assert "interrupt" in FeatherTextualApp._ACTIONS_MUTED_UNDER_MODAL

    # When the active screen is a ModalScreen, check_action returns False
    # for the muted actions — which tells Textual to skip the binding.
    fake_modal = MagicMock(spec=ModalScreen)
    monkeypatch.setattr(
        FeatherTextualApp, "screen", property(lambda self: fake_modal)
    )
    assert app.check_action("submit", ()) is False
    assert app.check_action("interrupt", ()) is False

    # When the active screen is a non-modal Screen, the actions are NOT
    # muted (check_action returns None → default behavior, which the parent
    # class's check_action returns).
    fake_screen = MagicMock(spec=Screen)
    monkeypatch.setattr(
        FeatherTextualApp, "screen", property(lambda self: fake_screen)
    )
    assert app.check_action("submit", ()) is not False
    assert app.check_action("interrupt", ()) is not False


def test_app_unrelated_actions_not_muted_under_modal(monkeypatch) -> None:
    """Actions outside the mute set (e.g. copy_selection_or_transcript)
    must not be disabled when a modal is on top — Ctrl+C still copies."""

    from unittest.mock import MagicMock

    from textual.screen import ModalScreen

    app = FeatherTextualApp(root=Path("/tmp"))
    fake_modal = MagicMock(spec=ModalScreen)
    monkeypatch.setattr(
        FeatherTextualApp, "screen", property(lambda self: fake_modal)
    )

    # copy_selection_or_transcript is not in the mute set
    assert app.check_action("copy_selection_or_transcript", ()) is not False
