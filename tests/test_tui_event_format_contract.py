"""Pin the shared tool-event formatter contract exposed by ``feather.tui``.

``feather.tui`` (the ex-``tui.py`` Rich dashboard module) exposes a set of
underscore-private formatters/constants that ``feather.tui.app`` (the Textual
TUI) and the split ``feather.tui.render`` / ``feather.tui.drill`` modules import
across the package boundary. They have no ``__all__`` declaring them public, so
a rename in ``feather.tui`` would otherwise *silently* break the Textual TUI's
imports. This test makes that break loud: rename a formatter and this fails.
"""

from __future__ import annotations

import importlib

import pytest

# The cross-module formatter/constant contract that feather.tui.app + render +
# drill rely on. Keep in sync with the `from feather.tui import (...)` blocks.
_CONTRACT = (
    "_TASK_TOOL_NAMES",
    "_INBOX_WAKE",
    "_event_title",
    "_failed_tool_title",
    "_format_tool_error",
    "_format_tool_payload",
    "_format_tool_result",
    "_indent_lines",
    "_status_style",
    "_system_event_style",
    "_tool_finished_title",
    "_tool_started_title",
    "preview_inline",
)


@pytest.mark.parametrize("name", _CONTRACT)
def test_feather_tui_exposes_shared_formatter(name: str) -> None:
    tui = importlib.import_module("feather.tui")
    assert hasattr(tui, name), f"feather.tui lost shared formatter: {name}"


def test_split_tui_modules_consume_the_contract() -> None:
    """The split app/render/drill modules import without breaking the contract."""

    for mod in ("feather.tui.app", "feather.tui.render", "feather.tui.drill"):
        assert importlib.import_module(mod) is not None
    # render.py must stay App-free (no FeatherTextualApp leak) so it is the
    # cheap, fast unit-test surface the split was designed to create.
    render = importlib.import_module("feather.tui.render")
    assert not hasattr(render, "FeatherTextualApp")
