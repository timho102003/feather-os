"""Real-subprocess smoke tests for the lead worker entry script.

The orchestration logic in :class:`feather.core.lead_worker_core.WorkerCore`
and the supervisor's drain logic are covered by in-memory unit tests.
This module bounds the script-level "is the entry point even loadable
under ``python -m``" risk that those unit tests cannot — argparse,
imports, and the asyncio-stream wiring all live below the unit-test
seam and can break in ways pure-Python tests cannot detect.

We deliberately do NOT spawn a worker that talks to an LLM here; that
would require a fake provider, a real config, and the full FeatherRuntime
boot. Subsequent steps in the worker roadmap will add that integration
test once self-repair (which actually drives full runs through the
worker) lands.
"""

from __future__ import annotations

import asyncio
import sys

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX subprocess pipes used by the worker entry are not reliable on Windows CI.",
)


async def test_lead_worker_entry_module_loads_under_python_m() -> None:
    """``python -m feather.lead_worker_entry --help`` must exit 0.

    Catches argparse misconfiguration and import-time failures (eg a
    circular import or a stale module path) at the wire boundary before
    a real worker spawn can mask them.
    """

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "feather.lead_worker_entry",
        "--help",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    assert proc.returncode == 0, (
        f"--help should exit 0; got {proc.returncode}\n"
        f"stdout: {stdout.decode(errors='replace')[:400]}\n"
        f"stderr: {stderr.decode(errors='replace')[:400]}"
    )
    text = stdout.decode("utf-8", errors="replace")
    # Argparse should advertise every flag the supervisor supplies.
    for flag in ("--session-id", "--root", "--heartbeat-interval", "--agent-name"):
        assert flag in text, f"--help missing {flag}"


async def test_lead_worker_entry_rejects_non_positive_heartbeat() -> None:
    """argparse-level validator rejects ``--heartbeat-interval 0`` upfront."""

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "feather.lead_worker_entry",
        "--session-id",
        "doesnotmatter",
        "--root",
        ".",
        "--heartbeat-interval",
        "0",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    # argparse exits 2 on type-validator failure.
    assert proc.returncode == 2
    assert b"heartbeat-interval must be > 0" in stderr
