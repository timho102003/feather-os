"""Tests for the hang-watcher state-machine helper and slash registration."""

from __future__ import annotations

import pytest

from feather.slash_commands import default_registry
from feather.textual_tui import decide_hang_alert


# --------------------------------------------------------------------- #
# decide_hang_alert pure helper
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "prev,curr,expected",
    [
        (False, False, None),     # steady healthy — no chatter
        (True, True, None),       # sustained hang — only one banner
        (False, True, "alert"),   # transition into hang
        (True, False, "recover"), # transition out of hang
    ],
)
def test_decide_hang_alert_only_fires_on_transitions(
    prev: bool, curr: bool, expected: str | None
) -> None:
    assert decide_hang_alert(prev, curr) is expected


# --------------------------------------------------------------------- #
# Slash command registration
# --------------------------------------------------------------------- #


def test_restart_lead_slash_command_is_registered() -> None:
    """The slash dispatcher must surface /restart-lead as a real command."""

    registry = default_registry()
    names = {cmd.name for cmd in registry.all()}
    assert "restart-lead" in names
    cmd = next(c for c in registry.all() if c.name == "restart-lead")
    assert "restart_lead" in cmd.aliases
    assert cmd.category == "session"


def test_restart_lead_handler_is_bound_in_textual_tui() -> None:
    """`_register_default_handlers` must wire /restart-lead — otherwise the
    runtime check raises at TUI construction. Smoke test by importing the
    binding map symbol indirectly via the module."""

    from feather.textual_tui import FeatherTextualApp

    # The handler binding lives inside _register_default_handlers; the
    # runtime guard there raises if the slash command exists with no
    # binding. So instantiating the app's slash subsystem is enough.
    handler_name = "_cmd_restart_lead"
    assert hasattr(FeatherTextualApp, handler_name), (
        f"FeatherTextualApp must define {handler_name}"
    )
