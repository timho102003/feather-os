"""Subprocess entry point for the lead worker (the "worker pod").

The supervisor (the Textual TUI process, see
:mod:`feather.core.lead_supervisor`) launches this module with::

    python -m feather.lead_worker_entry \\
        --session-id <uuid> --root <repo-root> \\
        [--heartbeat-interval 1.0]

The script is intentionally thin: it parses argv, builds the
:class:`FeatherRuntime`, opens a :class:`WorkerHeartbeatStore`, wires
stdin / stdout into asyncio streams, installs a SIGTERM handler, and
hands off to :class:`feather.core.lead_worker_core.WorkerCore` —
everything interesting lives there so it can be unit-tested without a
real subprocess.

Communication with the supervisor:

* **stdin** — one JSON line per command (see
  :mod:`feather.core.worker_command_codec`).
* **stdout** — one JSON line per ``RuntimeEvent``; control events
  (kinds prefixed with ``_`` like ``_run_complete``) carry agent-run
  outcome metadata back to the supervisor.
* **SQLite ``worker_heartbeats`` table** — liveness signal the
  supervisor polls to detect hangs.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Callable

from feather.core.lead_worker_core import WorkerCore
from feather.runtime import FeatherRuntime
from feather.storage.worker_heartbeat_store import WorkerHeartbeatStore

logger = logging.getLogger(__name__)

_DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 1.0


def _positive_float(raw: str) -> float:
    """Argparse type that rejects non-positive heartbeat intervals upfront."""

    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"heartbeat-interval must be > 0 (got {value})"
        )
    return value


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="feather-lead-worker",
        description="Feather lead-agent worker subprocess entry point.",
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="Lead session id (pre-assigned by the supervisor).",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Feather project root (passed to FeatherRuntime.create).",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=_positive_float,
        default=_DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        help=(
            "Seconds between worker_heartbeats refreshes. Must be > 0. "
            f"Default {_DEFAULT_HEARTBEAT_INTERVAL_SECONDS}."
        ),
    )
    parser.add_argument(
        "--agent-name",
        default="lead",
        help='Catalog name of the worker agent (default "lead").',
    )
    return parser.parse_args(argv)


async def _stdin_lines() -> AsyncIterator[str]:
    """Yield decoded UTF-8 lines from this process's stdin, async-friendly.

    Uses :func:`asyncio.connect_read_pipe` so the loop never blocks on
    ``sys.stdin.readline``; the iterator naturally exits on EOF, which
    the worker treats as an implicit shutdown.
    """

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    while True:
        raw = await reader.readline()
        if not raw:
            return
        yield raw.decode("utf-8", errors="replace")


def _stdout_event_sink(line: str) -> None:
    """Write one encoded event line to stdout and flush immediately."""

    sys.stdout.write(line)
    sys.stdout.write("\n")
    sys.stdout.flush()


async def _run_async(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    runtime = await FeatherRuntime.create(root)
    try:
        agent = runtime.build_agent(args.agent_name)
        await agent.ensure_session_with_id(args.session_id)

        db_path = Path(runtime.config.database.path)
        if not db_path.is_absolute():
            db_path = (root / db_path).resolve()
        heartbeat_store = WorkerHeartbeatStore(db_path)
        await heartbeat_store.initialize()
        try:
            core = WorkerCore(
                agent=agent,
                input_queue=runtime.input_queue,
                heartbeat_store=heartbeat_store,
                session_id=args.session_id,
                pid=os.getpid(),
                heartbeat_interval=args.heartbeat_interval,
                command_source=_stdin_lines(),
                event_sink=_stdout_event_sink,
                runtime=runtime,
            )
            _install_sigterm_handler(core.request_shutdown)
            logger.info(
                "lead_worker started session_id=%s pid=%s heartbeat=%.2fs",
                args.session_id,
                os.getpid(),
                args.heartbeat_interval,
            )
            await core.run()
        finally:
            await heartbeat_store.close()
    finally:
        await runtime.close()
    return 0


def _install_sigterm_handler(on_signal: Callable[[], None]) -> None:
    """Install SIGTERM (and SIGINT) handlers that trip a clean shutdown.

    Best-effort: falls back to plain ``signal.signal`` registration when
    asyncio's loop-aware variant isn't available (e.g. Windows).
    """

    def _handle(*_args: object) -> None:
        on_signal()

    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, _handle)
        loop.add_signal_handler(signal.SIGINT, _handle)
    except (NotImplementedError, RuntimeError):
        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGINT, _handle)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m feather.lead_worker_entry``."""

    args = _parse_args(argv if argv is not None else sys.argv[1:])
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    # Set PYTHONUNBUFFERED so any nested subprocesses we spawn inherit
    # unbuffered stdio. Our own per-line flush in ``_stdout_event_sink``
    # is what keeps the supervisor from blocking on a half-buffered pipe.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
