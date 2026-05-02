"""Lead-only tool that terminates one running sub-agent subprocess.

Pairs with :class:`feather.tools.spawn_agent_tool.SpawnAgentTool`. When
``spawn_agent`` returns, the lead is tracking a correlation_id that will
eventually be closed by either:

- the sub-agent finishing on its own (reaper posts the final report), or
- the lead calling ``terminate_agent`` (this tool).

Either way, the lead ends up with one inbox message in-reply-to that
correlation_id. No dangling "pending spawn" state.

Design:

- Lookup the child in the :class:`SubagentRegistry`. If absent, either
  the child already exited (reaper handled it) or the session_id is
  wrong. Either way there is nothing to terminate — return a descriptive
  no-op message.
- If the child is registered but ``returncode is not None`` it has
  already exited; the reaper will post the real report shortly. Don't
  post a synthetic "terminated" message — that would be a lie.
- If the child is alive: SIGTERM → 2 s wait → SIGKILL; cancel the stdout
  / stderr drainers; post a single agent_message to the PARENT's inbox
  with ``in_reply_to=correlation_id`` so the lead's correlation tracking
  closes cleanly; then drop from the registry.

Race with the reaper: we remove from the registry *before* posting the
termination message. The reaper's ``snapshot()`` takes a copy under a
lock, so a reaper iteration that already saw the entry will attempt its
own delivery. To keep it to ONE final message we set
``LiveSubagent.correlation_id = None`` before removal so the reaper's
``_deliver_final_report`` path is still safe to call but won't double-
post — and we're the one who already posted. See the tests for the
interleavings this covers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from feather.core.subagent_registry import SubagentRegistry
from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.storage.agent_message_store import AgentMessageStore
from feather.tools.base import BaseTool

logger = logging.getLogger(__name__)

_MAX_REASON_CHARS = 500


class TerminateAgentTool(BaseTool):
    """Forcefully terminate a running sub-agent subprocess."""

    name = "terminate_agent"
    description = (
        "Terminate a running sub-agent subprocess. Use when the plan has "
        "changed, the sub-agent appears stuck, or the user has redirected "
        "the work. The termination arrives in your inbox as the final reply "
        "for that spawn's correlation_id, so your correlation bookkeeping "
        "closes cleanly. If the sub-agent already finished, this is a no-op."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": (
                    "session_id of the sub-agent to terminate. This is the "
                    "value returned by `spawn_agent` for that child."
                ),
            },
            "reason": {
                "type": ["string", "null"],
                "description": (
                    "Short explanation of why the sub-agent is being "
                    "terminated. Included in the termination message."
                ),
            },
        },
        "required": ["session_id", "reason"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        registry: SubagentRegistry,
        agent_message_store: AgentMessageStore,
        parent_agent_name: str,
    ) -> None:
        self._registry = registry
        self._message_store = agent_message_store
        self._parent_agent_name = parent_agent_name

    def get_prompt(self) -> str:
        return (
            "- `terminate_agent`: stop a running sub-agent identified by its "
            "`session_id`. Use when the plan changed, the sub-agent is stuck, "
            "or the user cancelled the work. You will receive the termination "
            "as the final reply for that spawn's correlation_id in your inbox."
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        raw_sid = arguments.get("session_id")
        if not isinstance(raw_sid, str) or not raw_sid.strip():
            raise ValueError("terminate_agent `session_id` must be a non-empty string.")
        session_id = raw_sid.strip()
        raw_reason = arguments.get("reason")
        reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
        if len(reason) > _MAX_REASON_CHARS:
            reason = reason[:_MAX_REASON_CHARS] + "..."

        live = await self._registry.get(session_id)
        if live is None:
            logger.info(
                "terminate_agent no registry entry parent=%s target_session=%s",
                context.session_id,
                session_id,
            )
            return ToolExecutionResult(
                output=(
                    f"Sub-agent `{session_id}` is not registered as live. "
                    "It either already exited (its final report will arrive "
                    "via the reaper) or the session_id is wrong. Nothing to "
                    "terminate."
                )
            )

        if live.parent_session_id != context.session_id:
            # Defense-in-depth: the lead can only terminate children
            # spawned *from its own session*. Without this, another lead
            # session (if ever supported) could revoke a sibling's work.
            raise ValueError(
                f"terminate_agent refused: sub-agent `{session_id}` was "
                f"spawned by a different session (parent_session_id="
                f"{live.parent_session_id}, current={context.session_id})."
            )

        if live.process.returncode is not None:
            # Already exited; let the reaper deliver the real report.
            logger.info(
                "terminate_agent skip (already exited) parent=%s target_session=%s rc=%s",
                context.session_id,
                session_id,
                live.process.returncode,
            )
            return ToolExecutionResult(
                output=(
                    f"Sub-agent `{live.agent_name}` (session_id={session_id}) "
                    f"has already exited (returncode={live.process.returncode}). "
                    "Its real final report will arrive via the reaper shortly."
                )
            )

        # Kill sequence: SIGTERM → 2 s wait → SIGKILL. Tracks what
        # SubagentReaper / CLI shutdown do.
        correlation_id = live.correlation_id
        # Clear correlation on the live entry so if the reaper's
        # snapshot already caught this child, its delivery path does
        # NOT re-attach in_reply_to — our termination message is
        # canonical for this correlation_id.
        live.correlation_id = None
        # Remove from the registry first so the reaper's next poll
        # won't pick it up and deliver a second final report.
        removed = await self._registry.remove(session_id)
        if removed is None:
            # Concurrent remover (e.g. reaper) already claimed it.
            # Restore correlation_id so logs remain coherent.
            live.correlation_id = correlation_id
            return ToolExecutionResult(
                output=(
                    f"Sub-agent `{live.agent_name}` (session_id={session_id}) "
                    "was already being reaped; its real report is on the way."
                )
            )

        proc = live.process
        terminate_note = "terminate signal sent"
        try:
            proc.terminate()
        except ProcessLookupError:
            terminate_note = "process already gone"
        else:
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    terminate_note = "SIGKILL after SIGTERM timeout"
                except ProcessLookupError:
                    terminate_note = "process disappeared before SIGKILL"
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    terminate_note = "process did not exit after SIGKILL"

        # Cancel the background stream drainers.
        for drainer in live.drainers:
            if not drainer.done():
                drainer.cancel()
            try:
                await drainer
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        # Post ONE termination message to the parent's inbox so the
        # correlation_id closes cleanly.
        body_lines = [
            f"Sub-agent `{live.agent_name}` (session_id={session_id}) terminated by "
            f"{self._parent_agent_name} ({terminate_note}).",
        ]
        if reason:
            body_lines.append(f"reason: {reason}")
        body_lines.append(
            "Any in-progress work has been discarded. Remove this sub-agent "
            "from your active tracking."
        )
        try:
            await self._message_store.send(
                from_session_id=session_id,
                from_agent_name=live.agent_name,
                to_session_id=live.parent_session_id,
                to_agent_name=live.parent_agent_name,
                body="\n".join(body_lines),
                in_reply_to=correlation_id,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "terminate_agent failed to post termination message parent=%s target_session=%s",
                context.session_id,
                session_id,
            )

        logger.info(
            "terminate_agent completed parent=%s target_session=%s note=%s",
            context.session_id,
            session_id,
            terminate_note,
        )
        return ToolExecutionResult(
            output=(
                f"Sub-agent `{live.agent_name}` (session_id={session_id}) "
                f"terminated: {terminate_note}. Termination message delivered "
                "to your inbox."
            )
        )
