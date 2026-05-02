"""Tests for the Rich-based Feather TUI renderer."""

from __future__ import annotations

import io

from rich.console import Console

from feather.models import RuntimeEvent
from feather.tui import (
    TuiEventPrinter,
    TuiInputBuffer,
    _apply_tui_action,
    summarize_user_text,
)


def _printer() -> tuple[TuiEventPrinter, io.StringIO]:
    """Build a TUI printer backed by an in-memory console."""

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    return TuiEventPrinter(console, session_id="s1", agent_name="Lead"), buf


def test_assistant_deltas_are_coalesced_into_one_turn() -> None:
    printer, _ = _printer()

    printer(RuntimeEvent(kind="assistant_text_delta", text="hello "))
    printer(RuntimeEvent(kind="assistant_text_delta", text="world"))
    printer.finish_turn()

    assert printer.state.status == "idle"
    assert printer.state.transcript[-1].title == "Lead"
    assert printer.state.transcript[-1].text == "hello world"


def test_usage_updates_context_percent_in_rendered_header() -> None:
    printer, buf = _printer()

    printer(RuntimeEvent(kind="usage_updated", payload={"usage_ratio": 0.374}))
    printer.refresh()

    output = buf.getvalue()
    assert "ctx 37%" in output


def test_tool_events_render_as_concise_conversation_markers() -> None:
    printer, _ = _printer()

    printer(RuntimeEvent(kind="tool_started", tool_name="grep", payload={"q": "x"}))
    printer(RuntimeEvent(kind="tool_finished", tool_name="grep", text="A" * 500))

    assert printer.state.transcript[0].title == "Feather"
    assert printer.state.transcript[0].text == "Running grep · x"
    assert printer.state.transcript[-1].title == "Feather"
    assert printer.state.transcript[-1].text == "Ran grep · search complete"
    assert "A" * 500 not in printer.state.transcript[-1].text


def test_web_search_marker_is_concise_not_raw_json() -> None:
    printer, _ = _printer()

    printer(
        RuntimeEvent(
            kind="tool_started",
            tool_name="web_search",
            payload={
                "objective": "Find detailed information.",
                "search_queries": ["a", "b", "c"],
            },
        )
    )
    printer(
        RuntimeEvent(
            kind="tool_finished",
            tool_name="web_search",
            text=(
                "objective: Find detailed information.\n"
                "mode=fast results=6\n"
                "1. First authoritative hit\n"
                "2. Second useful hit\n"
                "3. Third useful hit"
            ),
        )
    )

    assert printer.state.transcript[0].text == "Searching web · 3 queries"
    assert (
        printer.state.transcript[1].text
        == "Searched web · 6 results: First authoritative hit; Second useful hit; Third useful hit"
    )
    assert "objective:" not in printer.state.transcript[1].text


def test_spawn_agent_marker_shows_short_agent_and_session() -> None:
    printer, _ = _printer()

    printer(
        RuntimeEvent(
            kind="tool_started",
            tool_name="spawn_agent",
            payload={"agent_name": "research", "task": "Goal: do a long task"},
        )
    )
    printer(
        RuntimeEvent(
            kind="tool_finished",
            tool_name="spawn_agent",
            text=(
                "Sub-agent `research` spawned.\n"
                "session_id: d6c476f5-33b8-412f-852a-765a370804ca\n"
                "correlation_id: abc\n"
                "pid: 123"
            ),
        )
    )

    assert printer.state.transcript[0].text == "Spawning sub-agent · research"
    assert printer.state.transcript[1].text == "Spawned sub-agent · research d6c476f5"
    assert "correlation_id" not in printer.state.transcript[1].text


def test_task_tool_marker_shows_title_case_and_detail() -> None:
    printer, _ = _printer()

    printer(
        RuntimeEvent(
            kind="tool_started",
            tool_name="task_create",
            payload={"title": "Research prices"},
        )
    )
    printer(
        RuntimeEvent(
            kind="tool_finished",
            tool_name="task_create",
            text=(
                "Created task.\n"
                "id: task-1\n"
                "title: Research prices\n"
                "status: queued\n"
                "plan_id: -\n"
                "lead_session_id: lead"
            ),
        )
    )

    assert printer.state.transcript[0].text == "Running Task Create · Research prices"
    assert printer.state.transcript[1].text.startswith("Ran Task Create · Created task.")
    assert "id: task-1" in printer.state.transcript[1].text


def test_ask_user_tool_events_are_not_rendered_as_activity_noise() -> None:
    printer, _ = _printer()

    printer(
        RuntimeEvent(
            kind="tool_started",
            tool_name="ask_user",
            payload={"question": "Pick one of these long options."},
        )
    )
    printer(
        RuntimeEvent(
            kind="tool_finished",
            tool_name="ask_user",
            text="User input required: Pick one.",
        )
    )

    assert printer.state.transcript == []


def test_failed_tool_marker_includes_error_reason() -> None:
    printer, _ = _printer()

    printer(
        RuntimeEvent(
            kind="tool_finished",
            tool_name="read_file",
            text="Tool `read_file` failed: file does not exist: missing.md",
        )
    )

    assert printer.state.transcript[-1].title == "Feather"
    assert printer.state.transcript[-1].text == (
        "Read file failed · file does not exist: missing.md"
    )


def test_failed_bash_marker_includes_exit_code_and_stderr() -> None:
    printer, _ = _printer()

    printer(
        RuntimeEvent(
            kind="tool_finished",
            tool_name="bash",
            text=(
                "exit_code: 2\n"
                "cwd: .\n"
                "command: pytest\n"
                "stdout:\n"
                "some stdout\n"
                "stderr:\n"
                "usage error"
            ),
        )
    )

    assert printer.state.transcript[-1].title == "Feather"
    assert printer.state.transcript[-1].text == "Bash failed, exit 2 · exit 2: usage error"


def test_subagent_message_is_not_truncated_in_conversation() -> None:
    printer, _ = _printer()
    message = "sub-agent report " + ("A" * 500)

    printer(RuntimeEvent(kind="agent_message_received", text=message))

    assert printer.state.transcript[-1].title == "Sub-agent"
    assert printer.state.transcript[-1].text == message
    assert not printer.state.transcript[-1].text.endswith("...")


def test_queue_snapshot_updates_status_panel() -> None:
    printer, buf = _printer()

    printer.set_queue_snapshot(3, ("first", "second", "third"))
    printer.refresh()

    output = buf.getvalue()
    assert "Queued / Future Work" in output
    assert "queued 3" in output
    assert "first" in output
    assert "third" in output


def test_running_agents_show_in_footer() -> None:
    printer, buf = _printer()

    printer.set_running_agents(("explorer abc123", "validator def456"))
    printer.refresh()

    output = buf.getvalue()
    assert "Future tasks" in output
    assert "explorer abc123" in output
    assert "validator def456" in output


def test_awaiting_user_event_marks_status_and_transcript() -> None:
    printer, _ = _printer()

    printer(RuntimeEvent(kind="awaiting_user", text="Approve?"))

    assert printer.state.status == "awaiting user"
    assert printer.state.transcript[-1].title == "Lead asks"
    assert printer.state.transcript[-1].text == "Approve?"


def test_answer_collapses_previous_question_but_keeps_user_answer() -> None:
    printer, _ = _printer()

    printer(RuntimeEvent(kind="awaiting_user", text="Pick one of several long options."))
    printer.record_user_message("1")

    assert printer.state.transcript[-2].title == "Lead asks"
    assert printer.state.transcript[-2].text == "Asked for clarification."
    assert printer.state.transcript[-1].title == "You"
    assert printer.state.transcript[-1].text == "1"


def test_user_message_is_not_truncated_in_conversation() -> None:
    printer, _ = _printer()
    message = "help me research " + ("x" * 500)

    printer.record_user_message(message)

    assert printer.state.transcript[-1].title == "You"
    assert printer.state.transcript[-1].text == message
    assert not printer.state.transcript[-1].text.endswith("...")


def test_render_has_no_separate_activity_section() -> None:
    printer, buf = _printer()

    printer(RuntimeEvent(kind="tool_started", tool_name="grep", payload={"q": "x"}))
    printer.refresh()

    output = buf.getvalue()
    assert "Activity" not in output
    assert "Conversation" in output
    assert "Queued / Future Work" in output
    assert "Input" in output


def test_conversation_render_is_bounded_but_state_keeps_full_transcript() -> None:
    printer, buf = _printer()

    for index in range(40):
        printer.record_user_message(f"message {index}")
    printer.refresh()

    output = buf.getvalue()
    assert len(printer.state.transcript) == 40
    assert "message 39" in output
    assert "message 0" not in output
    assert "older items above" in output
    assert "Input" in output


def test_conversation_can_scroll_to_older_items() -> None:
    printer, buf = _printer()

    for index in range(40):
        printer.record_user_message(f"message {index}")
    printer.scroll_conversation_home()
    printer.refresh()

    output = buf.getvalue()
    assert "message 0" in output
    assert "newer items below" in output
    assert "Input" in output


def test_conversation_scroll_end_returns_to_latest_items() -> None:
    printer, buf = _printer()

    for index in range(40):
        printer.record_user_message(f"message {index}")
    printer.scroll_conversation_home()
    printer.scroll_conversation_end()
    printer.refresh()

    output = buf.getvalue()
    assert "message 39" in output
    assert "message 0" not in output


def test_prompt_is_rendered_only_in_footer() -> None:
    printer, buf = _printer()

    printer.refresh()
    printer.print_prompt()

    assert buf.getvalue().count("you>") == 1


def test_footer_renders_current_input_buffer() -> None:
    printer, buf = _printer()

    printer.set_input_text("hello there")
    printer.refresh()

    output = buf.getvalue()
    assert "you>" in output
    assert "hello there|" in output


def test_footer_renders_cursor_inside_input_buffer() -> None:
    printer, buf = _printer()

    printer.set_input_text("hello", cursor=2)
    printer.refresh()

    assert "he|llo" in buf.getvalue()


def test_footer_shows_cursor_when_input_is_empty() -> None:
    printer, buf = _printer()

    printer.refresh()

    output = buf.getvalue()
    assert "you> |" in output


def test_footer_does_not_append_empty_agent_status_to_input_line() -> None:
    printer, buf = _printer()

    printer.set_input_text("help me research chatgpt")
    printer.refresh()

    output = buf.getvalue()
    assert "help me research chatgpt|" in output
    assert "no sub-agents running" not in output


def test_conversation_panel_uses_bold_white_text() -> None:
    printer, _ = _printer()

    printer.record_user_message("hello")
    printer(RuntimeEvent(kind="assistant_text_delta", text="hi"))
    printer.finish_turn()
    panel = printer._render_transcript()
    rendered = panel.renderable

    styles = [str(span.style) for span in rendered.spans]
    assert "bold cyan" in styles
    assert "bold green" in styles
    assert "bold white" in styles


def test_input_buffer_handles_backspace_and_enter() -> None:
    buffer = TuiInputBuffer()

    buffer.feed("abc\x7f\n")

    assert buffer.text == ""
    assert buffer.pop_line() == "ab"
    assert buffer.pop_line() is None


def test_input_buffer_handles_left_and_right_arrows() -> None:
    buffer = TuiInputBuffer()

    buffer.feed("abc\x1b[D\x1b[DZ\x1b[C!")

    assert buffer.text == "aZb!c"
    assert buffer.cursor == 4


def test_input_buffer_reports_scroll_actions() -> None:
    buffer = TuiInputBuffer()

    buffer.feed("\x1b[5~\x1b[6~\x1b[H\x1b[F")

    assert buffer.pop_actions() == ("page_up", "page_down", "home", "end")
    assert buffer.pop_actions() == ()


def test_input_buffer_reports_escape_interrupt_action() -> None:
    buffer = TuiInputBuffer()

    buffer.feed("\x1b")

    assert buffer.pop_actions() == ("interrupt",)


def test_interrupt_action_invokes_callback() -> None:
    printer, _ = _printer()
    called = False

    def interrupt() -> None:
        nonlocal called
        called = True

    _apply_tui_action(printer, "interrupt", on_interrupt=interrupt)

    assert called


def test_input_buffer_handles_up_and_down_arrows_in_multiline_text() -> None:
    buffer = TuiInputBuffer()

    buffer.feed("\x1b[200~abc\ndefg\nhi\x1b[201~\x1b[A!")
    assert buffer.text == "abc\nde!fg\nhi"
    assert buffer.cursor == len("abc\nde!")

    buffer.feed("\x1b[B?")
    assert buffer.text == "abc\nde!fg\nhi?"
    assert buffer.cursor == len("abc\nde!fg\nhi?")


def test_input_buffer_backspace_respects_cursor_position() -> None:
    buffer = TuiInputBuffer()

    buffer.feed("abc\x1b[D\x7f")

    assert buffer.text == "ac"
    assert buffer.cursor == 1


def test_input_buffer_preserves_bracketed_paste_newlines() -> None:
    buffer = TuiInputBuffer()

    buffer.feed("\x1b[200~line 1\nline 2\x1b[201~\n")

    assert buffer.pop_line() == "line 1\nline 2"


def test_large_pasted_user_text_gets_display_summary_only() -> None:
    text = "\n".join(f"line {i}" for i in range(80))

    assert summarize_user_text(text) == "[Pasted content: 629 chars, 80 lines]"


def test_regular_user_text_is_previewed_directly() -> None:
    assert summarize_user_text("ship the tui") == "ship the tui"
