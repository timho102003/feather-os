"""Tests for the shared subprocess plumbing in ``core/ipc/process.py``.

Real ``sys.executable -c`` children exercise the spawn/drain/terminate
paths end-to-end (pipe behaviour and POSIX signal escalation can't be
faithfully faked). Each test is bounded well under 5s.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from feather.core.ipc.process import (
    PipedProcess,
    cancel_drainers,
    drain_stream,
    spawn_piped_process,
    terminate_process,
)
from feather.core.ipc.subprocess_env import subprocess_env_with_home


def _argv(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _env() -> dict[str, str]:
    return subprocess_env_with_home()


async def test_spawn_captures_stdout_and_stderr(tmp_path) -> None:
    """Both pipes are drained into their buffers by the spawned drainers."""

    code = (
        "import sys; "
        "sys.stdout.write('hello-out'); sys.stdout.flush(); "
        "sys.stderr.write('hello-err'); sys.stderr.flush()"
    )
    piped = await spawn_piped_process(
        _argv(code),
        cwd=str(tmp_path),
        env=_env(),
        stdin=asyncio.subprocess.DEVNULL,
        name="cap",
    )
    rc = await piped.process.wait()
    # Two drainers (stdout + stderr) when capture_stdout defaults to True.
    assert len(piped.drainers) == 2
    await asyncio.gather(*piped.drainers)
    assert rc == 0
    assert bytes(piped.stdout_buffer) == b"hello-out"
    assert bytes(piped.stderr_buffer) == b"hello-err"


async def test_capture_stdout_false_leaves_stdout_readable(tmp_path) -> None:
    """With capture_stdout=False the caller line-reads stdout itself; only
    stderr is drained (the lead-worker supervisor pattern)."""

    code = (
        "import sys; "
        "sys.stdout.write('line-one\\n'); sys.stdout.flush(); "
        "sys.stderr.write('err-side'); sys.stderr.flush()"
    )
    piped = await spawn_piped_process(
        _argv(code),
        cwd=str(tmp_path),
        env=_env(),
        stdin=asyncio.subprocess.PIPE,
        capture_stdout=False,
        name="nocap",
    )
    # Exactly one drainer (stderr only).
    assert len(piped.drainers) == 1
    assert piped.process.stdout is not None
    raw = await piped.process.stdout.readline()
    assert raw == b"line-one\n"
    rc = await piped.process.wait()
    await asyncio.gather(*piped.drainers)
    assert rc == 0
    # stdout was read by us, never copied into the buffer.
    assert bytes(piped.stdout_buffer) == b""
    assert bytes(piped.stderr_buffer) == b"err-side"


async def test_stderr_cap_bounds_buffer_but_drains_pipe(tmp_path) -> None:
    """The load-bearing hardening: a child that floods stderr beyond the cap
    must NOT deadlock — the drainer keeps reading the pipe while the buffer
    stops growing at ``max_bytes``."""

    cap = 64 * 1024
    # Write ~2 MiB to stderr, then exit 0. If the pipe weren't drained past
    # the cap, the child would block on a full OS pipe buffer forever.
    code = (
        "import sys; "
        "sys.stderr.buffer.write(b'x' * (2 * 1024 * 1024)); "
        "sys.stderr.flush(); "
        "sys.exit(0)"
    )
    piped = await spawn_piped_process(
        _argv(code),
        cwd=str(tmp_path),
        env=_env(),
        stdin=asyncio.subprocess.DEVNULL,
        stderr_max_bytes=cap,
        name="flood",
    )
    rc = await asyncio.wait_for(piped.process.wait(), timeout=5.0)
    await asyncio.wait_for(asyncio.gather(*piped.drainers), timeout=5.0)
    assert rc == 0
    assert len(piped.stderr_buffer) == cap


async def test_drain_stream_without_cap_reads_everything(tmp_path) -> None:
    """Uncapped drain_stream copies every byte to EOF."""

    payload = b"a" * 100_000
    code = (
        "import sys; "
        f"sys.stdout.buffer.write(b'a' * {len(payload)}); sys.stdout.flush()"
    )
    process = await asyncio.create_subprocess_exec(
        *_argv(code),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=str(tmp_path),
        env=_env(),
    )
    buffer = bytearray()
    await asyncio.wait_for(drain_stream(process.stdout, buffer), timeout=5.0)
    await process.wait()
    assert bytes(buffer) == payload


async def test_terminate_process_sigterm_path(tmp_path) -> None:
    """A cooperative child exits on SIGTERM well inside the term timeout."""

    code = "import time; time.sleep(30)"
    process = await asyncio.create_subprocess_exec(
        *_argv(code),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=str(tmp_path),
        env=_env(),
    )
    await asyncio.wait_for(terminate_process(process), timeout=5.0)
    assert process.returncode is not None


async def test_terminate_process_kill_escalation(tmp_path) -> None:
    """A child that ignores SIGTERM is escalated to SIGKILL after the
    (short) term timeout, and then reaped."""

    code = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    process = await asyncio.create_subprocess_exec(
        *_argv(code),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=str(tmp_path),
        env=_env(),
    )
    await asyncio.wait_for(
        terminate_process(process, term_timeout=0.3), timeout=5.0
    )
    assert process.returncode is not None


async def test_terminate_process_idempotent_on_exited_child(tmp_path) -> None:
    """Calling terminate_process on an already-dead child is a quiet no-op,
    even twice."""

    code = "import sys; sys.exit(0)"
    process = await asyncio.create_subprocess_exec(
        *_argv(code),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=str(tmp_path),
        env=_env(),
    )
    await process.wait()
    assert process.returncode is not None
    # Must not raise on a reaped child, and must stay idempotent.
    await terminate_process(process)
    await terminate_process(process)


async def test_cancel_drainers_swallows_cancellation(tmp_path) -> None:
    """cancel_drainers cancels a still-running drainer and returns without
    raising the CancelledError."""

    code = "import time; time.sleep(30)"
    process = await asyncio.create_subprocess_exec(
        *_argv(code),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=str(tmp_path),
        env=_env(),
    )
    buffer = bytearray()
    drainer = asyncio.create_task(drain_stream(process.stdout, buffer))
    await asyncio.sleep(0)  # let the drainer block on read()
    assert not drainer.done()
    await asyncio.wait_for(cancel_drainers((drainer,)), timeout=5.0)
    assert drainer.done()
    # Clean up the sleeping child.
    await terminate_process(process)


async def test_piped_process_defaults() -> None:
    """PipedProcess gives independent default buffers and an empty drainer
    tuple (guards the mutable-default contract)."""

    a = PipedProcess(process=None)  # type: ignore[arg-type]
    b = PipedProcess(process=None)  # type: ignore[arg-type]
    a.stdout_buffer.extend(b"x")
    assert bytes(b.stdout_buffer) == b""
    assert a.drainers == ()
