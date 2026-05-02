"""Background task that reaps exited sub-agent subprocesses.

The non-blocking :class:`SpawnAgentTool` leaves the lead free to continue
its loop. That means something else has to:

- notice when a sub-agent subprocess exits,
- read its stdout (the marker-wrapped JSON envelope),
- post the final report to the parent's inbox as one last
  :class:`AgentMessage`,
- drop the entry from the registry so the child's resources can be
  released.

That "something else" is the reaper. It runs as a single asyncio task
that periodically snapshots the registry, inspects each live
``asyncio.subprocess.Process``, and handles any that have exited. The
poll cadence is short (0.25 s by default) — the reaper is watching
event-loop state, not disk, so the wake cost is negligible.

Delivery contract:

- If the envelope's status is ``completed``: post the sub-agent's
  ``assistant_text`` as a ``body`` with ``in_reply_to=correlation_id``.
  The parent inbox drain will then mark the original ``spawn_agent``
  correlation as responded.
- If the envelope is missing or status is not ``completed``: post a
  short diagnostic body that still sets ``in_reply_to``, so the parent
  can't deadlock waiting for a reply that'll never come.

Shutdown: :meth:`stop` signals the background task, cancels the next
sleep, and waits. Any still-running children are NOT killed here — that
is :class:`FeatherRuntime.shutdown`'s responsibility.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from feather.core.subagent_registry import LiveSubagent, SubagentRegistry
from feather.models import TaskOutputKind, TaskRunStatus, TaskStatus
from feather.storage.agent_message_store import AgentMessageStore
from feather.storage.task_store import TaskStore
from feather.tools.spawn_agent_tool import extract_envelope

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL = 0.25
_MAX_REPORT_CHARS = 8000
_STDERR_TAIL_CHARS = 800


class SubagentReaper:
    """Watch live sub-agent subprocesses and deliver their final reports."""

    def __init__(
        self,
        *,
        registry: SubagentRegistry,
        agent_message_store: AgentMessageStore,
        task_store: TaskStore | None = None,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL,
    ) -> None:
        self._registry = registry
        self._message_store = agent_message_store
        self._task_store = task_store
        self._poll_interval = poll_interval_seconds
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background poll loop. Idempotent."""

        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="subagent-reaper")
        logger.info("subagent_reaper started")

    async def stop(self) -> None:
        """Request shutdown and await the task."""

        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("subagent_reaper did not stop in 5s; cancelling")
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._task = None
        logger.info("subagent_reaper stopped")

    async def run_once(self) -> int:
        """Run one reap pass immediately (for tests / forced tick)."""

        return await self._reap_exited()

    async def _run(self) -> None:
        """Background loop: poll, reap, sleep."""

        while not self._stop_event.is_set():
            try:
                await self._reap_exited()
            except Exception:  # noqa: BLE001
                logger.exception("subagent_reaper tick failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._poll_interval
                )
            except asyncio.TimeoutError:
                continue

    async def _reap_exited(self) -> int:
        """Inspect all live subprocesses; deliver+cleanup any that exited.

        Claim ordering: ``registry.remove`` is the ownership gate. If a
        concurrent ``terminate_agent`` already removed the entry, our
        ``remove`` returns ``None`` and we skip delivery entirely — the
        terminator has already posted the canonical final message. This
        prevents the double-inbox-post race.
        """

        live_list = await self._registry.snapshot()
        reaped = 0
        for snapshot in live_list:
            if snapshot.process.returncode is None:
                continue
            claimed = await self._registry.remove(snapshot.session_id)
            if claimed is None:
                # terminate_agent (or another reaper) beat us to it.
                continue
            await self._deliver_final_report(claimed)
            reaped += 1
        return reaped

    async def _deliver_final_report(self, live: LiveSubagent) -> None:
        """Parse the child's stdout and post the final message to parent's inbox."""

        returncode = live.process.returncode or 0
        # Wait for the drainer tasks to finish so we see every byte the child
        # wrote before exit. The drainers are bounded — they stop at EOF —
        # and `returncode is not None` means both pipes will close shortly.
        for drainer in live.drainers:
            try:
                await asyncio.wait_for(drainer, timeout=2.0)
            except asyncio.TimeoutError:
                drainer.cancel()
                try:
                    await drainer
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                # Any other drainer failure is already swallowed by
                # _drain_stream, but belt-and-braces.
                pass
        stdout_text = bytes(live.stdout_buffer).decode("utf-8", errors="replace")
        stderr_text = bytes(live.stderr_buffer).decode("utf-8", errors="replace")
        envelope = extract_envelope(stdout_text)
        task_note = await self._update_task_state(
            live=live,
            envelope=envelope,
            returncode=returncode,
        )
        body = self._render_body(
            live=live,
            envelope=envelope,
            returncode=returncode,
            stderr_text=stderr_text,
        )
        if task_note:
            body = f"{task_note}\n\n{body}"
        try:
            await self._message_store.send(
                from_session_id=live.session_id,
                from_agent_name=live.agent_name,
                to_session_id=live.parent_session_id,
                to_agent_name=live.parent_agent_name,
                body=body,
                in_reply_to=live.correlation_id,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "subagent_reaper failed to post final report session_id=%s",
                live.session_id,
            )
            return
        logger.info(
            "subagent_reaper posted final report session_id=%s parent=%s/%s status=%s",
            live.session_id,
            live.parent_agent_name,
            live.parent_session_id,
            (envelope or {}).get("status") if envelope else "unknown",
        )

    async def _update_task_state(
        self,
        *,
        live: LiveSubagent,
        envelope: dict[str, Any] | None,
        returncode: int,
    ) -> str:
        """Update durable task/run rows and return a parent-facing note."""

        if self._task_store is None or live.task_id is None:
            return ""
        envelope_status = (
            str(envelope.get("status") or "unknown") if envelope is not None else "unknown"
        )
        error = str((envelope or {}).get("error") or "") or None
        run_status = TaskRunStatus.EXITED if returncode == 0 else TaskRunStatus.CRASHED
        if live.task_run_id is not None:
            await self._task_store.finish_run(
                live.task_run_id,
                status=run_status,
                exit_code=returncode,
                envelope_status=envelope_status,
                error=error,
            )

        try:
            task = await self._task_store.get_task(live.task_id)
        except ValueError:
            return f"Task tracking: missing task_id={live.task_id}"

        if task.status == TaskStatus.BLOCKED_NEEDS_INPUT:
            return (
                f"Task tracking: task_id={task.id} status=blocked_needs_input. "
                "Lead/user input is required before this task can continue."
            )
        if envelope is None:
            await self._task_store.update_task(
                task.id,
                status=TaskStatus.FAILED,
                error="Sub-agent exited with no parseable result envelope.",
            )
            return f"Task tracking: task_id={task.id} status=failed."
        if envelope_status != "completed":
            await self._task_store.update_task(
                task.id,
                status=TaskStatus.FAILED,
                error=error or f"Sub-agent envelope status={envelope_status}.",
            )
            return f"Task tracking: task_id={task.id} status=failed."
        outputs = await self._task_store.list_outputs(task.id)
        final_outputs = [output for output in outputs if output.is_final]
        if final_outputs:
            status = (
                TaskStatus.COMPLETED_WITH_ARTIFACTS
                if any(output.path for output in final_outputs)
                else TaskStatus.COMPLETED_WITH_REPORT
            )
            await self._task_store.update_task(task.id, status=status, error=None)
            return f"Task tracking: task_id={task.id} status={status.value}."

        assistant_text = str(envelope.get("assistant_text") or "").strip()
        if assistant_text:
            await self._task_store.add_output(
                task_id=task.id,
                kind=TaskOutputKind.REPORT,
                path=None,
                content=assistant_text,
                summary="Final assistant report captured by sub-agent reaper.",
                created_by_session_id=live.session_id,
                validated=True,
                is_final=True,
            )
            await self._task_store.update_task(
                task.id,
                status=TaskStatus.COMPLETED_WITH_REPORT,
                error=None,
            )
            return (
                f"Task tracking: task_id={task.id} "
                "status=completed_with_report."
            )

        await self._task_store.update_task(
            task.id,
            status=TaskStatus.FAILED,
            error="Sub-agent exited without a final task_output.",
        )
        return (
            f"Task tracking: task_id={task.id} status=failed. "
            "The sub-agent exited without a final task_output."
        )

    def _render_body(
        self,
        *,
        live: LiveSubagent,
        envelope: dict[str, Any] | None,
        returncode: int,
        stderr_text: str,
    ) -> str:
        """Format the final message body delivered to the parent's inbox."""

        if envelope is None:
            tail = _tail(stderr_text, _STDERR_TAIL_CHARS)
            return (
                f"Sub-agent `{live.agent_name}` (session_id={live.session_id}) "
                f"exited (code={returncode}) with no parseable result envelope. "
                f"stderr tail: {tail or '(empty)'}"
            )
        status = str(envelope.get("status") or "unknown")
        assistant_text = str(envelope.get("assistant_text") or "").strip()
        if len(assistant_text) > _MAX_REPORT_CHARS:
            assistant_text = (
                assistant_text[:_MAX_REPORT_CHARS]
                + f"\n...(truncated to {_MAX_REPORT_CHARS} chars)"
            )
        if status == "completed":
            return (
                f"Sub-agent `{live.agent_name}` completed (session_id={live.session_id}).\n\n"
                f"{assistant_text or '(no final text)'}"
            )
        error = str(envelope.get("error") or "")
        lines = [
            f"Sub-agent `{live.agent_name}` did NOT complete (status={status}, session_id={live.session_id}).",
        ]
        if error:
            lines.append(f"error: {error}")
        if assistant_text:
            lines.extend(["", "Partial:", assistant_text])
        return "\n".join(lines)


def _tail(text: str, limit: int) -> str:
    """Return the last ``limit`` characters with whitespace collapsed."""

    if not text:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return "..." + collapsed[-limit:]
