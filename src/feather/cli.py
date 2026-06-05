"""Simple streaming CLI for the first Feather slice.

The CLI runs two concurrent coroutines for one session:

- A **stdin reader** that always awaits the next line from the terminal.
- An **agent driver** that awaits ``agent.run`` whenever there is work to do.

Dispatch rules for each line from stdin:

- ``/exit`` / ``/quit``: stop the CLI.
- ``/queue``: show currently pending (queued-but-not-injected) messages.
- ``AWAITING_USER`` + non-empty line: delivered as the answer to the agent's
  outstanding question.
- Agent idle + non-empty line: start a new ``agent.run``.
- Agent running + non-empty line: push to the per-session ``UserInputQueue``
  so ``BaseAgent.run_loop`` can inject it between turns.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console

from feather.models import AgentOutcome, RuntimeEvent
from feather.onboarding import maybe_run_onboarding
from feather.runtime import FeatherRuntime


_TOOL_OUTPUT_DISPLAY_LIMIT = 200

# Sentinel enqueued by the inbox watcher to wake the driver without a
# user-supplied message. Chosen as a string that cannot collide with a
# real input line (inputs are stripped before being enqueued).
_INBOX_WAKE = "\x00__inbox_wake__\x00"
_INBOX_POLL_INTERVAL_SECONDS = 0.5


def _truncate_tool_output(text: str | None) -> str:
    """Collapse newlines and cap tool output at the CLI display limit."""

    if not text:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _TOOL_OUTPUT_DISPLAY_LIMIT:
        return collapsed
    return collapsed[:_TOOL_OUTPUT_DISPLAY_LIMIT] + "..."


class CliEventPrinter:
    """Render runtime events in a simple terminal-friendly format."""

    def __init__(self, console: Console, agent_name: str) -> None:
        self._console = console
        self._agent_name = agent_name
        self._stream_open = False
        self._latest_usage_ratio: float | None = None

    def _agent_header(self) -> str:
        """Format the streamed assistant-turn header, with context-% if known."""

        if self._latest_usage_ratio is None:
            return f"[bold cyan]{self._agent_name}[/bold cyan]> "
        pct = max(0, min(100, round(self._latest_usage_ratio * 100)))
        return f"[bold cyan]{self._agent_name}[/bold cyan] [dim](ctx: {pct}%)[/dim]> "

    def __call__(self, event: RuntimeEvent) -> None:
        """Render one runtime event.

        Args:
            event: Runtime event to render.
        """

        if event.kind == "assistant_text_delta":
            if not self._stream_open:
                # The stdin reader is always-on: by the time the agent
                # starts streaming, the next "you> " prompt has usually
                # already been printed on a fresh line by ``console.input``
                # (waiting for the user to queue another message). If we
                # just print "Lead> …" now, it either collides with that
                # prompt (same line) or leaves a dangling "you> " above
                # the response. Rewind the cursor to column 0 and clear
                # the current line so the agent header overwrites the
                # dangling prompt cleanly.
                #
                # Bypassing rich here is intentional — rich doesn't know
                # about readline's cursor state, so its own ``print``
                # layer won't help. Raw ANSI control codes are the right
                # fit for this one interaction.
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                self._console.print(self._agent_header(), end="")
                self._stream_open = True
            self._console.print(event.text or "", end="", markup=False, highlight=False)
            return

        if event.kind == "usage_updated":
            ratio = (event.payload or {}).get("usage_ratio")
            if isinstance(ratio, (int, float)):
                self._latest_usage_ratio = float(ratio)
            return

        self.finish_turn()
        if event.kind == "tool_started":
            self._console.print(
                f"[grey50]tool> {event.tool_name} {event.payload or {}}[/grey50]",
                highlight=False,
            )
        elif event.kind == "tool_finished":
            truncated = _truncate_tool_output(event.text)
            self._console.print(
                f"[grey50]tool> {event.tool_name}: {truncated}[/grey50]",
                highlight=False,
            )
        elif event.kind == "awaiting_user":
            self._console.print(
                f"[bold magenta]{self._agent_name} asks[/bold magenta]> {event.text}",
                highlight=False,
            )
        elif event.kind == "user_message_injected":
            self._console.print(
                f"[dim italic]injected> {event.text}[/dim italic]",
                highlight=False,
            )
        elif event.kind == "agent_message_received":
            self._console.print(
                f"[magenta]inbox>[/magenta] {event.text}",
                highlight=False,
            )
        elif event.kind == "compaction_started":
            self._console.print(f"[yellow]system[/yellow]> {event.text}", highlight=False)
        elif event.kind == "compaction_finished":
            self._console.print(f"[green]system[/green]> {event.text}", highlight=False)
        elif event.kind == "compaction_failed":
            self._console.print(f"[red]system[/red]> {event.text}", highlight=False)
        elif event.kind == "scheduled_task_triggered":
            self._console.print(f"[yellow]system[/yellow]> {event.text}", highlight=False)
        elif event.kind == "scheduled_task_completed":
            self._console.print(f"[green]system[/green]> {event.text}", highlight=False)
        elif event.kind == "scheduled_task_failed":
            self._console.print(f"[red]system[/red]> {event.text}", highlight=False)

    def finish_turn(self) -> None:
        """Close the current streamed line and redisplay a 'you> ' prompt.

        After the stream-open path rewinds with ``\\r\\033[K`` to overwrite
        the dangling ``you> `` prompt with ``Lead> …``, the stdin reader's
        ``console.input`` call is still blocked — but its on-screen prompt
        was erased, so if we just printed a newline here the user would
        type into an unprefixed line. Writing ``you> `` directly to stdout
        gives them a visible prompt to type against. The reader's next
        ``console.input`` iteration will reprint its own ``you> `` which
        the next stream-open will again rewind over, so there's no
        permanent double-prompt.
        """

        if self._stream_open:
            self._console.print()
            self._stream_open = False
            _print_user_prompt()


async def _read_console_input(console: Console, prompt: str) -> str:
    """Read terminal input without blocking the event loop.

    ``prompt`` is typically ``""`` — the CLI prints a styled ``you> ``
    prompt explicitly (a) at startup, and (b) after each streamed agent
    response (in :meth:`CliEventPrinter.finish_turn`). Printing it
    pre-emptively from the reader would cause a dangling prompt to
    appear above the in-progress ``Lead> …`` output.
    """

    return await asyncio.to_thread(console.input, prompt)


def _print_user_prompt() -> None:
    """Write the styled 'you> ' prompt directly to stdout."""

    sys.stdout.write("\033[1;32myou>\033[0m ")
    sys.stdout.flush()


async def run_cli(
    root: Path, session_id: str | None, paths: object = None
) -> None:
    """Run the Feather terminal session.

    Args:
        root: Repository root.
        session_id: Optional existing session ID.
    """

    console = Console()
    runtime = await FeatherRuntime.create(root, paths=paths)
    active_session_id: str | None = None

    try:
        lead_name = runtime.default_lead_name
        agent = runtime.build_agent(lead_name)

        active_session_id = session_id or await agent.create_session()
        console.print(f"[bold]Feather[/bold] session: {active_session_id}")
        printer = CliEventPrinter(console, agent.config.name)
        runtime.set_session_event_handler(active_session_id, printer)
        await runtime.start_background_services()

        input_queue = runtime.input_queue
        stop_event = asyncio.Event()
        # Pending answer for the model's ask_user question; set by the stdin
        # reader when the agent is paused AWAITING_USER.
        pending_answer: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        # "Busy" mirrors whether an agent.run is currently in flight, so the
        # stdin reader knows whether to start a run, deliver an answer, or
        # enqueue.
        busy_event = asyncio.Event()
        # AWAITING_USER gate — set by the driver when the agent paused, cleared
        # once the answer has been consumed.
        awaiting_event = asyncio.Event()
        # Fresh user messages that should start a *new* run (not an injection).
        # The special sentinel ``_INBOX_WAKE`` is used by the inbox watcher
        # to tell the driver "the agent has mail — drain it, don't expect
        # any new user text."
        new_run_queue: asyncio.Queue[str] = asyncio.Queue()

        async def _stdin_reader() -> None:
            """Always-on terminal reader; dispatches lines by current state.

            The prompt is printed elsewhere: once at startup (just before
            the tasks spawn) and once after each streamed agent response
            (:meth:`CliEventPrinter.finish_turn`). Leaving the prompt out
            of :func:`_read_console_input` avoids the dangling-prompt
            flicker that appears when the reader pre-emptively prints a
            new ``you> `` before the agent has started responding.
            """

            while not stop_event.is_set():
                try:
                    line = await _read_console_input(console, "")
                except (EOFError, KeyboardInterrupt):
                    stop_event.set()
                    return
                except Exception as exc:  # noqa: BLE001
                    # Anything else — UnicodeDecodeError, terminal
                    # reconfiguration, readline glitch with multi-byte
                    # input, etc. — previously propagated through the
                    # task and got silently swallowed by the shutdown
                    # except clause. Print it, reprint the prompt, and
                    # keep reading so the CLI doesn't die on a single
                    # bad character.
                    console.print(
                        f"[red]input error[/red]: {type(exc).__name__}: {exc}",
                        highlight=False,
                    )
                    _print_user_prompt()
                    continue
                text = line.strip()
                if not text:
                    # User pressed Enter on an empty prompt — reprint so
                    # the next prompt is visible, don't just silently
                    # swallow the keystroke.
                    _print_user_prompt()
                    continue
                if text in {"/exit", "/quit"}:
                    stop_event.set()
                    return
                if text == "/queue":
                    depth = await input_queue.depth(active_session_id or "")
                    pending = await input_queue.peek(active_session_id or "")
                    console.print(f"[dim]queued ({depth}):[/dim]", highlight=False)
                    for i, msg in enumerate(pending, 1):
                        console.print(
                            f"[dim]  {i}. {_truncate_tool_output(msg)}[/dim]",
                            highlight=False,
                        )
                    continue

                if awaiting_event.is_set():
                    # The driver is blocked on an ask_user answer; this line
                    # is that answer, not a queue injection.
                    try:
                        pending_answer.put_nowait(text)
                        awaiting_event.clear()
                    except asyncio.QueueFull:
                        # A previous answer is already queued; enqueue as a
                        # normal injection so it isn't lost.
                        assert active_session_id is not None
                        await input_queue.enqueue(active_session_id, text)
                        console.print(
                            f"[dim]queued> {_truncate_tool_output(text)}[/dim]",
                            highlight=False,
                        )
                    continue

                if busy_event.is_set():
                    assert active_session_id is not None
                    ok = await input_queue.enqueue(active_session_id, text)
                    if ok:
                        console.print(
                            f"[dim]queued> {_truncate_tool_output(text)}[/dim]",
                            highlight=False,
                        )
                    continue

                # Idle: kick off a brand new run.
                await new_run_queue.put(text)

        async def _agent_driver() -> None:
            """Drive agent.run whenever new-run requests arrive."""

            assert active_session_id is not None
            nonlocal agent
            while not stop_event.is_set():
                get_task = asyncio.create_task(new_run_queue.get())
                stop_task = asyncio.create_task(stop_event.wait())
                await asyncio.wait(
                    {get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if stop_event.is_set():
                    get_task.cancel()
                    return
                stop_task.cancel()
                user_message = get_task.result()

                # /config saves with NEXT_TURN reload class call
                # ``runtime.rebuild_agent("lead")`` which swaps
                # ``runtime._agents["lead"]`` underneath us. Refresh the
                # local handle now so the next agent.run uses the new
                # provider/model — otherwise the captured ``agent`` keeps
                # talking to the pre-save provider for the rest of the
                # session.
                try:
                    agent = runtime.get_agent(lead_name)
                except KeyError:  # pragma: no cover — lead is built at startup
                    pass

                busy_event.set()
                try:
                    if user_message == _INBOX_WAKE:
                        # Inbox-driven resume — no new user text. Returns
                        # None if the inbox got drained by a concurrent
                        # path between the watcher's check and the lock
                        # acquisition; treat that as a no-op.
                        try:
                            result = await agent.resume_on_inbox(
                                active_session_id, printer
                            )
                        except Exception as exc:  # noqa: BLE001
                            printer.finish_turn()
                            console.print(
                                f"[red]agent error[/red]: "
                                f"{type(exc).__name__}: {exc}",
                                highlight=False,
                            )
                            logging.getLogger(__name__).exception(
                                "cli.agent_driver.resume_on_inbox_crashed "
                                "session_id=%s",
                                active_session_id,
                            )
                            _print_user_prompt()
                            continue
                        printer.finish_turn()
                        if result is None:
                            continue
                    else:
                        try:
                            result = await agent.run(
                                active_session_id, user_message, printer
                            )
                        except Exception as exc:  # noqa: BLE001
                            printer.finish_turn()
                            console.print(
                                f"[red]agent error[/red]: "
                                f"{type(exc).__name__}: {exc}",
                                highlight=False,
                            )
                            logging.getLogger(__name__).exception(
                                "cli.agent_driver.run_crashed session_id=%s",
                                active_session_id,
                            )
                            _print_user_prompt()
                            continue
                        printer.finish_turn()

                    while (
                        result.status == AgentOutcome.AWAITING_USER
                        and result.question is not None
                        and not stop_event.is_set()
                    ):
                        awaiting_event.set()
                        answer_task = asyncio.create_task(pending_answer.get())
                        stop_task = asyncio.create_task(stop_event.wait())
                        await asyncio.wait(
                            {answer_task, stop_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if stop_event.is_set():
                            answer_task.cancel()
                            awaiting_event.clear()
                            return
                        stop_task.cancel()
                        answer = answer_task.result()
                        result = await agent.run(active_session_id, answer, printer)
                        printer.finish_turn()
                finally:
                    busy_event.clear()
                    awaiting_event.clear()

        async def _inbox_watcher() -> None:
            """Poll the agent's inbox and wake the driver when messages arrive.

            Sub-agent final reports and scheduled-task triggers post into
            the agent_message_store asynchronously; without this watcher
            the lead would have those reports sitting in its inbox with
            no one to pick them up until the user typed something. The
            watcher only enqueues a wake sentinel when the driver is
            idle — otherwise the running turn's own inbox drain will pick
            messages up at its top-of-loop.
            """

            assert active_session_id is not None
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        stop_event.wait(), timeout=_INBOX_POLL_INTERVAL_SECONDS
                    )
                    return  # stop_event fired
                except asyncio.TimeoutError:
                    pass
                if busy_event.is_set() or awaiting_event.is_set():
                    continue
                try:
                    has_pending = await agent.has_pending_inbox(active_session_id)
                except Exception:  # noqa: BLE001
                    continue
                if not has_pending:
                    continue
                # Only enqueue a wake if we haven't already done so on a
                # prior tick that hasn't been consumed yet.
                if new_run_queue.empty():
                    await new_run_queue.put(_INBOX_WAKE)

        # Print the first prompt explicitly — the reader uses an empty
        # prompt string now and relies on this + ``finish_turn`` to show
        # the "you> " prefix at the right times.
        _print_user_prompt()
        reader_task = asyncio.create_task(_stdin_reader(), name="cli-stdin-reader")
        driver_task = asyncio.create_task(_agent_driver(), name="cli-agent-driver")
        watcher_task = asyncio.create_task(_inbox_watcher(), name="cli-inbox-watcher")
        try:
            await asyncio.wait(
                {reader_task, driver_task}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            stop_event.set()
            for task in (reader_task, driver_task, watcher_task):
                if not task.done():
                    task.cancel()
            # Swallow cancellation so shutdown continues cleanly.
            for task in (reader_task, driver_task, watcher_task):
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
    finally:
        if active_session_id is not None:
            runtime.set_session_event_handler(active_session_id, None)
        await runtime.close()


def main() -> None:
    """CLI entrypoint.

    Walks up from the working directory looking for an existing
    ``.feather/`` project. When none is found the CLI runs in
    "global-only" mode where session state lives under ``~/.feather``.

    Subcommands:

    * ``feather`` (default) → Textual TUI.
    * ``feather tui`` → same, explicit.
    * ``feather cli`` → the older streaming Rich console loop.
    * ``feather init`` → create ``./.feather/`` here and register it.
    * ``feather init-memory`` → spin a local Qdrant container.
    * ``feather stop-memory`` → stop the container, marker preserved.
    * ``feather remove-memory [--purge]`` → remove container + marker.
    * ``feather onboard [--force]`` → run the first-run wizard.
    """

    from feather import __version__ as _feather_version
    from feather.paths import FeatherPaths

    parser = argparse.ArgumentParser(prog="feather", description="Feather agent OS CLI")
    parser.add_argument("--version", action="version", version=f"feather {_feather_version}")
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Override project root detection (skip the .feather walk-up).",
    )
    parser.add_argument(
        "--session-id", help="Resume an existing session ID.", default=None
    )
    parser.add_argument(
        "--skip-onboarding",
        action="store_true",
        help="Skip the first-run wizard.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "init",
        help="Create ./.feather/ in the current directory and register the project.",
    )

    init_mem = subparsers.add_parser(
        "init-memory",
        help="Start a local Qdrant container for long-term memory.",
    )
    init_mem.add_argument("--port", type=int, default=6333, help="Host port for Qdrant REST API.")

    subparsers.add_parser(
        "stop-memory",
        help="Stop the local Qdrant container; marker is preserved.",
    )

    rm_mem = subparsers.add_parser(
        "remove-memory",
        help="Stop and remove the Qdrant container; delete the marker.",
    )
    rm_mem.add_argument(
        "--purge",
        action="store_true",
        help="Also delete the persistent docker volume (irreversible).",
    )

    onboard_parser = subparsers.add_parser(
        "onboard", help="Run the first-run onboarding wizard."
    )
    onboard_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the wizard even when onboarding was already completed.",
    )

    subparsers.add_parser(
        "cli",
        help="Use the older streaming Rich CLI instead of the Textual TUI.",
    )
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the web/API server (full agent experience over HTTP + WebSocket).",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port.")

    from feather.tui import add_tui_subparser

    add_tui_subparser(subparsers)
    args = parser.parse_args()

    # Resolve paths: --project beats walk-up; walk-up beats global-only fallback.
    if args.project is not None:
        paths = FeatherPaths.for_project(args.project.resolve())
    else:
        paths = FeatherPaths.detect()

    # Memory-management subcommands don't need a project.
    if args.command == "init":
        if paths.project_root is None:
            paths = FeatherPaths.for_project(Path.cwd().resolve(), home=paths.global_root)
        from feather.cli_commands import init_project
        sys.exit(init_project(paths))
    if args.command == "init-memory":
        from feather.cli_commands import init_memory
        sys.exit(init_memory(paths))
    if args.command == "stop-memory":
        from feather.cli_commands import stop_memory
        sys.exit(stop_memory(paths))
    if args.command == "remove-memory":
        from feather.cli_commands import remove_memory
        sys.exit(remove_memory(paths, purge=args.purge))

    # Below here the legacy code path still uses Path.cwd() — paths
    # propagation into the runtime lands in Phase 5 along with the
    # onboarding-wizard rework. Until then, behavior is unchanged for
    # existing in-repo dev workflows.
    cwd = paths.project_root or Path.cwd()

    # Offer the legacy → global migration once per project. Skipped
    # silently when nothing legacy exists or when the breadcrumb says
    # we already prompted.
    from feather.migration import maybe_migrate

    maybe_migrate(paths)

    if args.command == "onboard":
        asyncio.run(maybe_run_onboarding(cwd, force=args.force, paths=paths))
        return
    if args.command == "cli":
        # Opt-in: the streaming Rich CLI loop. Bare `feather` lands on
        # the Textual TUI now; this subcommand is for users (and tests)
        # who specifically want the older one-shot console flow.
        asyncio.run(
            maybe_run_onboarding(cwd, skip=args.skip_onboarding, paths=paths)
        )
        asyncio.run(run_cli(cwd, args.session_id, paths=paths))
        return
    if args.command == "serve":
        try:
            import uvicorn

            from feather.api.server import create_app
        except ImportError:
            print(
                "The web server needs the optional API extra. Install it with:\n"
                "  uv pip install 'feather-agent-os[api]'   (or: uv sync --extra api)"
            )
            sys.exit(1)
        app = create_app(cwd, provider_factory=None)
        uvicorn.run(app, host=args.host, port=args.port)
        return
    # Default + explicit `tui`: launch the Textual TUI.
    from feather.textual_tui import run_textual_tui

    tui_session_id = getattr(args, "tui_session_id", None) or args.session_id
    asyncio.run(
        maybe_run_onboarding(cwd, skip=args.skip_onboarding, paths=paths)
    )
    asyncio.run(run_textual_tui(cwd, tui_session_id))
