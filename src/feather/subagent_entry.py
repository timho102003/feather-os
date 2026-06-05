"""Subprocess entry point used by the lead-agent `spawn_agent` tool.

The lead launches this module with `python -m feather.subagent_entry`. This
process builds its own :class:`FeatherRuntime`, runs a single agent turn with
the dispatched task prompt, and emits exactly one marker-wrapped JSON envelope
on stdout so the parent can reliably parse the result regardless of any
library stderr chatter.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from feather.core.agent.catalog import AgentCatalog
from feather.providers.base import BaseLLMProvider
from feather.runtime import FeatherRuntime
from feather.subagent_protocol import RESULT_BEGIN, RESULT_END

logger = logging.getLogger(__name__)

ProviderFactory = Callable[[Any], BaseLLMProvider]

_NON_DISPATCHABLE_NAMES = frozenset({"lead"})


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the subprocess invocation arguments."""

    parser = argparse.ArgumentParser(
        prog="feather-subagent",
        description="Feather sub-agent subprocess entry point.",
    )
    parser.add_argument(
        "--agent-name",
        required=True,
        help="Catalog name of the sub-agent (YAML filename without extension).",
    )
    parser.add_argument(
        "--task-file",
        required=True,
        help="Path to a UTF-8 text file containing the dispatched task prompt.",
    )
    parser.add_argument(
        "--parent-session",
        required=True,
        help="Parent session id that dispatched this sub-agent.",
    )
    parser.add_argument(
        "--parent-agent-name",
        required=False,
        default="lead",
        help="Parent agent name that dispatched this sub-agent.",
    )
    parser.add_argument(
        "--session-id",
        required=False,
        default=None,
        help=(
            "Pre-assigned session id for this sub-agent. When omitted, the "
            "sub-agent mints a fresh session id. Use the pre-assigned form "
            "so the parent can address this child via send_message before "
            "the child has finished starting up."
        ),
    )
    parser.add_argument(
        "--correlation-id",
        required=False,
        default=None,
        help=(
            "Correlation id to stamp on the final-report agent_message sent "
            "to the parent. Omitted when the parent is not correlating."
        ),
    )
    parser.add_argument(
        "--task-id",
        required=False,
        default=None,
        help="Durable task id associated with this sub-agent run.",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Feather repository root.",
    )
    parser.add_argument(
        "--keep-task-file",
        action="store_true",
        help="Retain the task file after reading (useful for tests).",
    )
    return parser.parse_args(argv)


def _read_task(task_file: Path, *, delete: bool) -> str:
    """Load the dispatched task prompt and (optionally) remove the staging file."""

    text = task_file.read_text(encoding="utf-8").strip()
    if delete:
        try:
            task_file.unlink()
        except FileNotFoundError:
            pass
    if not text:
        raise ValueError("Sub-agent task file is empty.")
    return text


def _emit_envelope(envelope: dict[str, object]) -> None:
    """Write the single marker-wrapped JSON envelope to stdout."""

    payload = json.dumps(envelope, ensure_ascii=False)
    sys.stdout.write(f"\n{RESULT_BEGIN}\n{payload}\n{RESULT_END}\n")
    sys.stdout.flush()


async def run_subagent_async(
    *,
    agent_name: str,
    task_text: str,
    parent_session_id: str,
    root: Path,
    session_id: str | None = None,
    parent_agent_name: str = "lead",
    correlation_id: str | None = None,
    task_id: str | None = None,
    provider_factory: ProviderFactory | None = None,
) -> dict[str, object]:
    """Bootstrap the runtime, run one sub-agent turn, and return the envelope body.

    The function is intentionally side-effect-light: it does not print the
    envelope itself so it can be reused from unit tests. ``emit`` callers
    should serialize the returned mapping themselves. Tests can inject a
    ``provider_factory`` to swap in a fake LLM provider.

    When ``session_id`` is supplied, the sub-agent uses it verbatim (the
    parent pre-assigned the id so it can address the child via
    ``send_message`` before the child has even finished starting). When
    ``session_id`` is None, the sub-agent mints a fresh id.
    """

    if not AgentCatalog.is_valid_name(agent_name):
        raise ValueError(f"Invalid sub-agent name: {agent_name!r}")
    if agent_name in _NON_DISPATCHABLE_NAMES:
        raise ValueError(f"Sub-agent name `{agent_name}` is not dispatchable.")

    runtime = await FeatherRuntime.create(root, provider_factory=provider_factory)
    try:
        agent = runtime.build_agent(agent_name)
        if session_id is None:
            effective_session_id = await agent.create_session()
        else:
            effective_session_id = await agent.ensure_session_with_id(session_id)
        logger.info(
            "subagent started agent_name=%s parent_session_id=%s session_id=%s correlation_id=%s",
            agent_name,
            parent_session_id,
            effective_session_id,
            correlation_id,
        )
        # Frame the task with parent context so the sub-agent knows where
        # to address replies when it calls send_message.
        framed_task = (
            f"<parent_session_id>{parent_session_id}</parent_session_id>\n"
            f"<parent_agent_name>{parent_agent_name}</parent_agent_name>\n"
            + (
                f"<correlation_id>{correlation_id}</correlation_id>\n"
                if correlation_id else ""
            )
            + (f"<task_id>{task_id}</task_id>\n" if task_id else "")
            + f"<task>\n{task_text}\n</task>"
        )
        result = await agent.run(effective_session_id, framed_task, None)
        # Wasted-spawn detection: a research/explore/validate sub-agent
        # that completed without ever calling a tool produced only an
        # acknowledgement, not the work it was dispatched to do. Flag
        # that loudly in the envelope so the lead's inbox shows a real
        # failure rather than a "completed" message whose body is just
        # "Understood, I will begin…".
        work_roles = {"research", "explore", "validate"}
        error_note: str | None = None
        status_value = result.status.value
        if (
            status_value == "completed"
            and agent.config.role in work_roles
            and result.total_tool_calls == 0
        ):
            status_value = "failed"
            error_note = (
                "wasted spawn: sub-agent role=%s exited without calling any tools. "
                "The body is an acknowledgement, not a report. Either re-spawn "
                "with a sharper task, bump the sub-agent's reasoning effort, or "
                "do the work inline." % agent.config.role
            )
            logger.warning(
                "subagent wasted-spawn role=%s session_id=%s output_chars=%s",
                agent.config.role,
                effective_session_id,
                len(result.assistant_text or ""),
            )
        return {
            "status": status_value,
            "agent_name": agent_name,
            "role": agent.config.role,
            "session_id": effective_session_id,
            "parent_session_id": parent_session_id,
            "parent_agent_name": parent_agent_name,
            "correlation_id": correlation_id,
            "task_id": task_id,
            "assistant_text": result.assistant_text,
            "question": result.question,
            "error": error_note,
            "total_tool_calls": result.total_tool_calls,
        }
    finally:
        await runtime.close()


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m feather.subagent_entry`."""

    args = _parse_args(list(argv if argv is not None else sys.argv[1:]))
    root = Path(args.root).resolve()
    task_file = Path(args.task_file)

    try:
        task_text = _read_task(task_file, delete=not args.keep_task_file)
    except Exception as exc:  # noqa: BLE001
        _emit_envelope(
            {
                "status": "failed",
                "agent_name": args.agent_name,
                "session_id": None,
                "parent_session_id": args.parent_session,
                "assistant_text": "",
                "question": None,
                "error": f"failed to load task file: {exc}",
            }
        )
        return 1

    try:
        envelope = asyncio.run(
            run_subagent_async(
                agent_name=args.agent_name,
                task_text=task_text,
                parent_session_id=args.parent_session,
                session_id=args.session_id,
                parent_agent_name=args.parent_agent_name,
                correlation_id=args.correlation_id,
                task_id=args.task_id,
                root=root,
            )
        )
        _emit_envelope(envelope)
        status = str(envelope.get("status") or "")
        if status != "completed":
            # Log the envelope-reported failure so it's visible in
            # .feather/logs/feather.log — the stdout envelope is only
            # ever read by the parent process, not the human operator.
            logger.error(
                "subagent run ended with non-completed status agent=%s "
                "session_id=%s status=%s error=%s",
                args.agent_name,
                envelope.get("session_id"),
                status,
                envelope.get("error"),
            )
        return 0 if status == "completed" else 1
    except Exception as exc:  # noqa: BLE001
        # Log the full traceback before packaging for the envelope so
        # the operator can see what blew up without having to parse the
        # stdout JSON out of the reaper's "final report" inbox message.
        logger.exception(
            "subagent run crashed agent=%s parent_session=%s error=%s",
            args.agent_name,
            args.parent_session,
            exc,
        )
        _emit_envelope(
            {
                "status": "failed",
                "agent_name": args.agent_name,
                "session_id": None,
                "parent_session_id": args.parent_session,
                "assistant_text": "",
                "question": None,
                "error": f"{exc.__class__.__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return 1


if __name__ == "__main__":
    # Best-effort: keep stdout clean even if a nested library tries to chatter.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
