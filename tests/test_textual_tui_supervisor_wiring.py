"""Verify the TUI wires the runtime<->supervisor reload bridge."""

from __future__ import annotations

import inspect

from feather import textual_tui


def test_start_lead_worker_supervisor_calls_attach_supervisor() -> None:
    """Worker-mode reload requires runtime.attach_supervisor."""

    source = inspect.getsource(textual_tui.FeatherTextualApp._start_lead_worker_supervisor)
    assert "attach_supervisor" in source, (
        "FeatherTextualApp._start_lead_worker_supervisor must call "
        "self._runtime.attach_supervisor so /config reloads reach the worker."
    )


def test_tui_detaches_supervisor_before_shutdown() -> None:
    """Tear-down releases the supervisor reference."""

    full_source = inspect.getsource(textual_tui)
    assert "detach_supervisor" in full_source, (
        "FeatherTextualApp must call self._runtime.detach_supervisor before "
        "tearing down the supervisor."
    )
