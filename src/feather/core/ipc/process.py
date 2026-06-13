"""Shared subprocess spawn/drain/terminate plumbing.

One canonical implementation for the three places Feather runs piped child
processes (sub-agent spawn, lead-worker spawn, runtime shutdown sweep), so
pipe-deadlock prevention and the SIGTERM->SIGKILL escalation cannot drift.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field

__all__ = (
    "PipedProcess",
    "cancel_drainers",
    "drain_stream",
    "spawn_piped_process",
    "terminate_process",
)


async def drain_stream(
    stream: asyncio.StreamReader | None,
    buffer: bytearray,
    *,
    max_bytes: int | None = None,
) -> None:
    """Read ``stream`` to EOF into ``buffer``; never raises into the loop.

    With ``max_bytes`` the buffer stops growing but the stream keeps being
    consumed, so a chatty child cannot deadlock on a full pipe while the
    parent caps its memory.
    """

    if stream is None:
        return
    try:
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                return
            if max_bytes is None:
                buffer.extend(chunk)
            elif len(buffer) < max_bytes:
                buffer.extend(chunk[: max_bytes - len(buffer)])
    except Exception:  # noqa: BLE001 — drainers must never raise
        return


@dataclass(slots=True)
class PipedProcess:
    """A spawned child plus its capture buffers and drainer tasks."""

    process: asyncio.subprocess.Process
    stdout_buffer: bytearray = field(default_factory=bytearray)
    stderr_buffer: bytearray = field(default_factory=bytearray)
    drainers: tuple[asyncio.Task[None], ...] = ()


async def spawn_piped_process(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    stdin: int,
    capture_stdout: bool = True,
    stderr_max_bytes: int | None = None,
    name: str = "process",
) -> PipedProcess:
    """Spawn ``argv`` with piped stdout/stderr and start drainer tasks.

    ``capture_stdout=False`` leaves stdout undrained for line-oriented
    protocol readers (the lead-worker supervisor) — only stderr gets a
    drainer in that mode.
    """

    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )
    piped = PipedProcess(process=process)
    drainers: list[asyncio.Task[None]] = []
    if capture_stdout:
        drainers.append(
            asyncio.create_task(
                drain_stream(process.stdout, piped.stdout_buffer),
                name=f"{name}-stdout",
            )
        )
    drainers.append(
        asyncio.create_task(
            drain_stream(
                process.stderr, piped.stderr_buffer, max_bytes=stderr_max_bytes
            ),
            name=f"{name}-stderr",
        )
    )
    piped.drainers = tuple(drainers)
    return piped


async def terminate_process(
    process: asyncio.subprocess.Process,
    *,
    term_timeout: float = 2.0,
    kill_timeout: float = 2.0,
) -> None:
    """SIGTERM -> wait -> SIGKILL -> wait escalation; idempotent and quiet.

    Callers that need to log a still-alive child check ``returncode`` after
    this returns (it stays ``None`` only when even SIGKILL's wait timed out).
    """

    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=term_timeout)
        return
    except asyncio.TimeoutError:
        pass
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=kill_timeout)


async def cancel_drainers(drainers: tuple[asyncio.Task[None], ...]) -> None:
    """Cancel-and-await drainer tasks, swallowing their exit."""

    for drainer in drainers:
        if not drainer.done():
            drainer.cancel()
        with contextlib.suppress(BaseException):
            await drainer
