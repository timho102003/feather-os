"""Tests for the ``lead_in_subprocess`` flag on FeatherRuntime.start_background_services.

When the flag is True (TUI's worker-mode is active), the cron scheduler
and messaging service must NOT start. Both build their own in-process
``BaseAgent`` and would race the worker on the shared ``sessions`` row,
silently corrupting ``last_response_id`` / ``pending_inputs`` /
``messages.sequence``. The sub-agent reaper must still start because
sub-agents are spawned by tools, not by background services.
"""

from __future__ import annotations

from typing import Any

from feather.core.input_queue import UserInputQueue
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


async def test_lead_in_subprocess_true_skips_cron_and_messaging() -> None:
    """When the lead is out-of-process, cron + messaging must not start."""

    runtime, cron, messaging, reaper = _build_runtime_with_recorders()
    await runtime.start_background_services(lead_in_subprocess=True)

    assert cron.start_calls == 0, "cron must not start when lead is in subprocess"
    assert messaging.start_calls == 0, (
        "messaging must not start when lead is in subprocess"
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
