"""Tests for the CLI event printer display logic."""

from __future__ import annotations

import io

from rich.console import Console

from feather.cli import CliEventPrinter, _truncate_tool_output
from feather.models import RuntimeEvent


def _printer() -> tuple[CliEventPrinter, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=240)
    return CliEventPrinter(console, "Lead"), buf


def test_truncate_tool_output_returns_short_text_unchanged() -> None:
    assert _truncate_tool_output("hello world") == "hello world"


def test_truncate_tool_output_collapses_whitespace_and_caps_at_200() -> None:
    long_text = "x" * 500
    truncated = _truncate_tool_output(long_text)
    assert truncated.endswith("...")
    assert len(truncated) == 203  # 200 chars + ellipsis


def test_truncate_tool_output_collapses_newlines() -> None:
    assert _truncate_tool_output("line1\nline2\n\nline3") == "line1 line2 line3"


def test_tool_finished_is_rendered_truncated() -> None:
    printer, buf = _printer()
    printer(
        RuntimeEvent(
            kind="tool_finished",
            tool_name="web_search",
            text="A" * 500,
        )
    )
    output = buf.getvalue()
    assert "tool> web_search: " in output
    assert "..." in output
    # Full 500-char payload must not leak into the display.
    assert "A" * 500 not in output


def test_assistant_header_without_usage_has_no_ctx_suffix() -> None:
    printer, buf = _printer()
    printer(RuntimeEvent(kind="assistant_text_delta", text="hi"))
    printer.finish_turn()
    assert "Lead> hi" in buf.getvalue()
    assert "ctx:" not in buf.getvalue()


def test_assistant_header_includes_ctx_percent_after_usage_event() -> None:
    printer, buf = _printer()
    printer(RuntimeEvent(kind="usage_updated", payload={"usage_ratio": 0.37}))
    printer(RuntimeEvent(kind="assistant_text_delta", text="hi"))
    printer.finish_turn()
    assert "Lead (ctx: 37%)> hi" in buf.getvalue()


def test_usage_event_updates_for_subsequent_turns() -> None:
    printer, buf = _printer()
    printer(RuntimeEvent(kind="usage_updated", payload={"usage_ratio": 0.1}))
    printer(RuntimeEvent(kind="assistant_text_delta", text="first"))
    printer.finish_turn()
    printer(RuntimeEvent(kind="usage_updated", payload={"usage_ratio": 0.82}))
    printer(RuntimeEvent(kind="assistant_text_delta", text="second"))
    printer.finish_turn()
    output = buf.getvalue()
    assert "Lead (ctx: 10%)> first" in output
    assert "Lead (ctx: 82%)> second" in output


def test_usage_event_clamps_out_of_range_ratios() -> None:
    printer, buf = _printer()
    printer(RuntimeEvent(kind="usage_updated", payload={"usage_ratio": 1.4}))
    printer(RuntimeEvent(kind="assistant_text_delta", text="x"))
    printer.finish_turn()
    assert "Lead (ctx: 100%)> x" in buf.getvalue()
