"""Tests for the ``lead_in_subprocess`` flag on FeatherRuntime.start_background_services.

When the flag is True (TUI's worker-mode is active), the **messaging
service** must NOT start — its inbound queue is the in-process
``UserInputQueue`` which the worker can't see. The **cron scheduler**
runs in both modes: it routes through the agent_messages mailbox
(which is process-shared via SQLite), so the worker's existing
``resume_on_inbox`` path picks up cron-triggered turns naturally with
no race on session state. The sub-agent reaper always runs because
sub-agents are spawned by tools, not by background services.
"""

from __future__ import annotations

from typing import Any

from feather.core.session.input_queue import UserInputQueue
from feather.runtime import FeatherRuntime


class _Recorder:
    """Stand-in for a service exposing ``start`` / ``stop`` — records starts."""

    def __init__(self) -> None:
        self.start_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        pass


def _build_runtime_with_recorders() -> tuple[
    FeatherRuntime, _Recorder, _Recorder, _Recorder
]:
    """Bypass ``FeatherRuntime.create`` and patch in three recorder services."""

    runtime = FeatherRuntime.__new__(FeatherRuntime)
    cron, messaging, reaper = _Recorder(), _Recorder(), _Recorder()
    runtime._cron_scheduler = cron  # type: ignore[attr-defined]
    runtime._messaging_service = messaging  # type: ignore[attr-defined]
    runtime._subagent_reaper = reaper  # type: ignore[attr-defined]
    runtime._input_queue = UserInputQueue()  # type: ignore[attr-defined]
    return runtime, cron, messaging, reaper


async def test_lead_in_subprocess_true_skips_only_messaging() -> None:
    """In worker mode the messaging router is paused; cron now runs.

    Cron previously raced the worker on session state because it built
    its own in-process BaseAgent. Now it routes through the
    agent_messages mailbox, so it's safe in both modes.
    """

    runtime, cron, messaging, reaper = _build_runtime_with_recorders()
    await runtime.start_background_services(lead_in_subprocess=True)

    assert cron.start_calls == 1, (
        "cron must start in worker mode — mailbox routing makes it safe"
    )
    assert messaging.start_calls == 0, (
        "messaging must not start when lead is in subprocess — its "
        "inbound UserInputQueue lives in this process and the worker "
        "can't see it"
    )
    assert reaper.start_calls == 1, (
        "subagent reaper must still start — sub-agents are spawned by tools, "
        "not by background services, so they don't race the worker."
    )


async def test_lead_in_subprocess_false_starts_all_services() -> None:
    """Default behavior (in-process lead) must start all three services."""

    runtime, cron, messaging, reaper = _build_runtime_with_recorders()
    await runtime.start_background_services()  # default: lead_in_subprocess=False

    assert cron.start_calls == 1
    assert messaging.start_calls == 1
    assert reaper.start_calls == 1


async def test_lead_in_subprocess_default_is_in_process() -> None:
    """Calling without the flag must NOT skip cron and messaging — the flag's
    default is the safe (long-standing) in-process behavior."""

    runtime, cron, messaging, _ = _build_runtime_with_recorders()
    await runtime.start_background_services()
    assert cron.start_calls == 1
    assert messaging.start_calls == 1


# Suppress pyright's "Any unused" warning — used in type aliases above.
_ = Any
