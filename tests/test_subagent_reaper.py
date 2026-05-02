"""Tests for the SubagentReaper."""

from __future__ import annotations

import json
from pathlib import Path

from feather.core.subagent_reaper import SubagentReaper
from feather.core.subagent_registry import LiveSubagent, SubagentRegistry
from feather.models import TaskOutputKind, TaskStatus
from feather.storage.agent_message_store import AgentMessageStore
from feather.storage.task_store import TaskStore
from feather.subagent_protocol import RESULT_BEGIN, RESULT_END


class _FakeProc:
    """Minimal subprocess stand-in (the reaper reads buffers, not pipes)."""

    def __init__(self, *, returncode: int | None) -> None:
        self.returncode = returncode


def _envelope(**overrides: object) -> bytes:
    body = {
        "status": overrides.get("status", "completed"),
        "agent_name": overrides.get("agent_name", "engineer-custom"),
        "role": overrides.get("role", "custom"),
        "session_id": overrides.get("session_id", "child-sess"),
        "parent_session_id": overrides.get("parent_session_id", "lead-sess"),
        "assistant_text": overrides.get("assistant_text", "final report"),
        "question": None,
        "error": overrides.get("error"),
    }
    return f"noise\n{RESULT_BEGIN}\n{json.dumps(body)}\n{RESULT_END}\n".encode()


async def test_reaper_delivers_successful_envelope(tmp_path: Path) -> None:
    store = AgentMessageStore(tmp_path / "feather.db")
    await store.initialize()
    registry = SubagentRegistry()
    try:
        proc = _FakeProc(returncode=0)
        live = LiveSubagent(
            session_id="child-sess",
            agent_name="engineer-custom",
            parent_session_id="lead-sess",
            parent_agent_name="Lead",
            process=proc,  # type: ignore[arg-type]
            task_text="build X",
            correlation_id="corr-abc",
            stdout_buffer=bytearray(_envelope(assistant_text="done")),
        )
        await registry.register(live)

        reaper = SubagentReaper(registry=registry, agent_message_store=store)
        reaped = await reaper.run_once()
        assert reaped == 1
        # Inbox of parent now has one message with the final report.
        inbox = await store.inbox(
            to_session_id="lead-sess", to_agent_name="Lead"
        )
        assert len(inbox) == 1
        assert "engineer-custom" in inbox[0].body
        assert "done" in inbox[0].body
        assert inbox[0].in_reply_to == "corr-abc"
        # Registry is empty now.
        assert await registry.snapshot() == []
    finally:
        await store.close()


async def test_reaper_skips_still_running_processes(tmp_path: Path) -> None:
    store = AgentMessageStore(tmp_path / "feather.db")
    await store.initialize()
    registry = SubagentRegistry()
    try:
        proc = _FakeProc(returncode=None)
        live = LiveSubagent(
            session_id="child",
            agent_name="engineer",
            parent_session_id="lead",
            parent_agent_name="Lead",
            process=proc,  # type: ignore[arg-type]
            task_text="",
        )
        await registry.register(live)
        reaper = SubagentReaper(registry=registry, agent_message_store=store)
        assert await reaper.run_once() == 0
        assert len(await registry.snapshot()) == 1
    finally:
        await store.close()


async def test_reaper_handles_missing_envelope(tmp_path: Path) -> None:
    store = AgentMessageStore(tmp_path / "feather.db")
    await store.initialize()
    registry = SubagentRegistry()
    try:
        proc = _FakeProc(returncode=1)
        live = LiveSubagent(
            session_id="c",
            agent_name="engineer",
            parent_session_id="lead",
            parent_agent_name="Lead",
            process=proc,  # type: ignore[arg-type]
            task_text="",
            correlation_id="corr-x",
            stdout_buffer=bytearray(b"no markers here"),
            stderr_buffer=bytearray(b"something broke"),
        )
        await registry.register(live)
        reaper = SubagentReaper(registry=registry, agent_message_store=store)
        await reaper.run_once()
        inbox = await store.inbox(
            to_session_id="lead", to_agent_name="Lead"
        )
        assert len(inbox) == 1
        assert "no parseable result envelope" in inbox[0].body
        assert inbox[0].in_reply_to == "corr-x"
    finally:
        await store.close()


async def test_reaper_handles_failed_envelope(tmp_path: Path) -> None:
    store = AgentMessageStore(tmp_path / "feather.db")
    await store.initialize()
    registry = SubagentRegistry()
    try:
        proc = _FakeProc(returncode=1)
        live = LiveSubagent(
            session_id="c",
            agent_name="engineer",
            parent_session_id="lead",
            parent_agent_name="Lead",
            process=proc,  # type: ignore[arg-type]
            task_text="",
            correlation_id="corr-y",
            stdout_buffer=bytearray(
                _envelope(
                    status="failed",
                    error="prompt overflowed",
                    assistant_text="partial",
                )
            ),
        )
        await registry.register(live)
        reaper = SubagentReaper(registry=registry, agent_message_store=store)
        await reaper.run_once()
        inbox = await store.inbox(
            to_session_id="lead", to_agent_name="Lead"
        )
        assert len(inbox) == 1
        body = inbox[0].body
        assert "did NOT complete" in body
        assert "prompt overflowed" in body
        assert "partial" in body
    finally:
        await store.close()


async def test_reaper_captures_completed_envelope_text_as_final_report(
    tmp_path: Path,
) -> None:
    store = AgentMessageStore(tmp_path / "feather.db")
    task_store = TaskStore(tmp_path / "feather.db")
    await store.initialize()
    await task_store.initialize()
    registry = SubagentRegistry()
    try:
        task = await task_store.create_task(
            lead_session_id="lead",
            title="Needs final output",
            responsible_agent_name="engineer",
            responsible_session_id="child",
            status=TaskStatus.RUNNING,
        )
        run = await task_store.create_run(
            task_id=task.id,
            session_id="child",
            agent_name="engineer",
            pid=123,
        )
        live = LiveSubagent(
            session_id="child",
            agent_name="engineer",
            parent_session_id="lead",
            parent_agent_name="Lead",
            process=_FakeProc(returncode=0),  # type: ignore[arg-type]
            task_text="",
            correlation_id="corr",
            task_id=task.id,
            task_run_id=run.id,
            stdout_buffer=bytearray(_envelope(assistant_text="I started but need more.")),
        )
        await registry.register(live)

        reaper = SubagentReaper(
            registry=registry,
            agent_message_store=store,
            task_store=task_store,
        )
        await reaper.run_once()

        updated = await task_store.get_task(task.id)
        outputs = await task_store.list_outputs(task.id)
        assert updated.status == TaskStatus.COMPLETED_WITH_REPORT
        assert updated.error is None
        assert len(outputs) == 1
        assert outputs[0].kind == TaskOutputKind.REPORT
        assert outputs[0].is_final
        assert outputs[0].content == "I started but need more."
        inbox = await store.inbox(to_session_id="lead", to_agent_name="Lead")
        assert "status=completed_with_report" in inbox[0].body
    finally:
        await task_store.close()
        await store.close()


async def test_reaper_marks_completed_envelope_without_any_output_failed(
    tmp_path: Path,
) -> None:
    store = AgentMessageStore(tmp_path / "feather.db")
    task_store = TaskStore(tmp_path / "feather.db")
    await store.initialize()
    await task_store.initialize()
    registry = SubagentRegistry()
    try:
        task = await task_store.create_task(
            lead_session_id="lead",
            title="No final text",
            responsible_agent_name="engineer",
            responsible_session_id="child",
            status=TaskStatus.RUNNING,
        )
        run = await task_store.create_run(
            task_id=task.id,
            session_id="child",
            agent_name="engineer",
            pid=123,
        )
        live = LiveSubagent(
            session_id="child",
            agent_name="engineer",
            parent_session_id="lead",
            parent_agent_name="Lead",
            process=_FakeProc(returncode=0),  # type: ignore[arg-type]
            task_text="",
            correlation_id="corr",
            task_id=task.id,
            task_run_id=run.id,
            stdout_buffer=bytearray(_envelope(assistant_text="")),
        )
        await registry.register(live)

        reaper = SubagentReaper(
            registry=registry,
            agent_message_store=store,
            task_store=task_store,
        )
        await reaper.run_once()

        updated = await task_store.get_task(task.id)
        assert updated.status == TaskStatus.FAILED
        assert "final task_output" in (updated.error or "")
        inbox = await store.inbox(to_session_id="lead", to_agent_name="Lead")
        assert "status=failed" in inbox[0].body
    finally:
        await task_store.close()
        await store.close()


async def test_reaper_keeps_blocked_task_blocked(tmp_path: Path) -> None:
    store = AgentMessageStore(tmp_path / "feather.db")
    task_store = TaskStore(tmp_path / "feather.db")
    await store.initialize()
    await task_store.initialize()
    registry = SubagentRegistry()
    try:
        task = await task_store.create_task(
            lead_session_id="lead",
            title="Blocked",
            responsible_agent_name="engineer",
            responsible_session_id="child",
            status=TaskStatus.RUNNING,
        )
        task = await task_store.update_task(
            task.id,
            status=TaskStatus.BLOCKED_NEEDS_INPUT,
            blocked_question="Need input",
            blocked_correlation_id="question-corr",
        )
        run = await task_store.create_run(
            task_id=task.id,
            session_id="child",
            agent_name="engineer",
            pid=123,
        )
        await registry.register(
            LiveSubagent(
                session_id="child",
                agent_name="engineer",
                parent_session_id="lead",
                parent_agent_name="Lead",
                process=_FakeProc(returncode=0),  # type: ignore[arg-type]
                task_text="",
                correlation_id="corr",
                task_id=task.id,
                task_run_id=run.id,
                stdout_buffer=bytearray(_envelope(assistant_text="waiting")),
            )
        )

        reaper = SubagentReaper(
            registry=registry,
            agent_message_store=store,
            task_store=task_store,
        )
        await reaper.run_once()

        updated = await task_store.get_task(task.id)
        assert updated.status == TaskStatus.BLOCKED_NEEDS_INPUT
        inbox = await store.inbox(to_session_id="lead", to_agent_name="Lead")
        assert "blocked_needs_input" in inbox[0].body
    finally:
        await task_store.close()
        await store.close()


async def test_reaper_marks_task_completed_with_final_artifact(tmp_path: Path) -> None:
    store = AgentMessageStore(tmp_path / "feather.db")
    task_store = TaskStore(tmp_path / "feather.db")
    await store.initialize()
    await task_store.initialize()
    registry = SubagentRegistry()
    try:
        task = await task_store.create_task(
            lead_session_id="lead",
            title="Complete",
            responsible_agent_name="engineer",
            responsible_session_id="child",
            status=TaskStatus.RUNNING,
        )
        await task_store.add_output(
            task_id=task.id,
            kind=TaskOutputKind.ARTIFACT,
            path=".feather/artifacts/outputs/report.md",
            content=None,
            summary="report",
            created_by_session_id="child",
            is_final=True,
        )
        run = await task_store.create_run(
            task_id=task.id,
            session_id="child",
            agent_name="engineer",
            pid=123,
        )
        await registry.register(
            LiveSubagent(
                session_id="child",
                agent_name="engineer",
                parent_session_id="lead",
                parent_agent_name="Lead",
                process=_FakeProc(returncode=0),  # type: ignore[arg-type]
                task_text="",
                correlation_id="corr",
                task_id=task.id,
                task_run_id=run.id,
                stdout_buffer=bytearray(_envelope(assistant_text="done")),
            )
        )

        reaper = SubagentReaper(
            registry=registry,
            agent_message_store=store,
            task_store=task_store,
        )
        await reaper.run_once()

        updated = await task_store.get_task(task.id)
        assert updated.status == TaskStatus.COMPLETED_WITH_ARTIFACTS
    finally:
        await task_store.close()
        await store.close()
