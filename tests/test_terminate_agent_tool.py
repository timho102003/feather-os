"""Tests for the lead-only TerminateAgentTool."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from feather.core.subagents.registry import LiveSubagent, SubagentRegistry
from feather.models import AgentMessageStatus, ToolExecutionContext
from feather.storage.agent_message_store import AgentMessageStore
from feather.tools.terminate_agent_tool import TerminateAgentTool


def _ctx(session_id: str = "lead-sess") -> ToolExecutionContext:
    return ToolExecutionContext(session_id=session_id, agent_name="Lead")


class _FakeProc:
    """Process stand-in whose terminate/kill set returncode."""

    def __init__(
        self,
        *,
        returncode: int | None = None,
        terminate_exits: bool = True,
    ) -> None:
        self.returncode = returncode
        self.terminate_called = False
        self.kill_called = False
        self._terminate_exits = terminate_exits

    def terminate(self) -> None:
        self.terminate_called = True
        if self._terminate_exits and self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.kill_called = True
        if self.returncode is None:
            self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


def _live(
    *,
    session_id: str = "child-sess",
    proc: _FakeProc,
    correlation_id: str | None = "corr-1",
    parent_session_id: str = "lead-sess",
) -> LiveSubagent:
    return LiveSubagent(
        session_id=session_id,
        agent_name="engineer",
        parent_session_id=parent_session_id,
        parent_agent_name="Lead",
        process=proc,  # type: ignore[arg-type]
        task_text="build X",
        correlation_id=correlation_id,
    )


async def _open_store(tmp_path: Path) -> AgentMessageStore:
    store = AgentMessageStore(tmp_path / "feather.db")
    await store.initialize()
    return store


async def test_terminate_alive_subagent_posts_message_and_cleans_registry(
    tmp_path: Path,
) -> None:
    registry = SubagentRegistry()
    store = await _open_store(tmp_path)
    try:
        proc = _FakeProc()
        await registry.register(_live(proc=proc))
        tool = TerminateAgentTool(
            registry=registry,
            agent_message_store=store,
            parent_agent_name="Lead",
        )
        result = await tool.execute(
            {"session_id": "child-sess", "reason": "plan change"},
            _ctx(),
        )
        assert "terminated" in result.output
        assert proc.terminate_called is True
        assert await registry.snapshot() == []
        inbox = await store.inbox(
            to_session_id="lead-sess", to_agent_name="Lead"
        )
        assert len(inbox) == 1
        assert "terminated by Lead" in inbox[0].body
        assert "plan change" in inbox[0].body
        assert inbox[0].in_reply_to == "corr-1"
        assert inbox[0].status == AgentMessageStatus.PENDING
    finally:
        await store.close()


async def test_terminate_on_already_exited_is_noop(tmp_path: Path) -> None:
    registry = SubagentRegistry()
    store = await _open_store(tmp_path)
    try:
        proc = _FakeProc(returncode=0)  # already exited
        await registry.register(_live(proc=proc))
        tool = TerminateAgentTool(
            registry=registry,
            agent_message_store=store,
            parent_agent_name="Lead",
        )
        result = await tool.execute(
            {"session_id": "child-sess", "reason": "late"},
            _ctx(),
        )
        assert "already exited" in result.output
        assert proc.terminate_called is False
        inbox = await store.inbox(
            to_session_id="lead-sess", to_agent_name="Lead"
        )
        # No synthetic termination message posted — reaper owns the truth.
        assert inbox == []
        # Registry entry still present; reaper will remove on its tick.
        assert len(await registry.snapshot()) == 1
    finally:
        await store.close()


async def test_terminate_on_missing_session_returns_no_op(tmp_path: Path) -> None:
    registry = SubagentRegistry()
    store = await _open_store(tmp_path)
    try:
        tool = TerminateAgentTool(
            registry=registry,
            agent_message_store=store,
            parent_agent_name="Lead",
        )
        result = await tool.execute(
            {"session_id": "never-spawned", "reason": None},
            _ctx(),
        )
        assert "not registered as live" in result.output
        # No inbox row.
        assert (
            await store.inbox(
                to_session_id="lead-sess", to_agent_name="Lead"
            )
            == []
        )
    finally:
        await store.close()


async def test_terminate_refuses_cross_parent_session(tmp_path: Path) -> None:
    """A lead calling terminate on a child spawned by a *different* session
    must be rejected — defense-in-depth against cross-session revocation."""

    registry = SubagentRegistry()
    store = await _open_store(tmp_path)
    try:
        await registry.register(
            _live(proc=_FakeProc(), parent_session_id="other-lead-sess")
        )
        tool = TerminateAgentTool(
            registry=registry,
            agent_message_store=store,
            parent_agent_name="Lead",
        )
        with pytest.raises(ValueError, match="spawned by a different session"):
            await tool.execute(
                {"session_id": "child-sess", "reason": None},
                _ctx(),
            )
    finally:
        await store.close()


async def test_terminate_rejects_empty_session_id(tmp_path: Path) -> None:
    registry = SubagentRegistry()
    store = await _open_store(tmp_path)
    try:
        tool = TerminateAgentTool(
            registry=registry,
            agent_message_store=store,
            parent_agent_name="Lead",
        )
        with pytest.raises(ValueError, match="session_id"):
            await tool.execute({"session_id": "   ", "reason": None}, _ctx())
    finally:
        await store.close()


async def test_terminate_escalates_to_kill_on_sigterm_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    """If SIGTERM doesn't cause the process to exit within the wait window,
    the tool must escalate to SIGKILL."""

    registry = SubagentRegistry()
    store = await _open_store(tmp_path)
    try:
        proc = _FakeProc(terminate_exits=False)  # terminate leaves returncode=None
        await registry.register(_live(proc=proc))

        async def fake_wait_for(coro, timeout):
            # Drain the coroutine to avoid "never awaited" warnings.
            try:
                coro.close()
            except Exception:  # noqa: BLE001
                pass
            raise asyncio.TimeoutError

        monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

        tool = TerminateAgentTool(
            registry=registry,
            agent_message_store=store,
            parent_agent_name="Lead",
        )
        result = await tool.execute(
            {"session_id": "child-sess", "reason": "stuck"},
            _ctx(),
        )
        assert proc.terminate_called is True
        assert proc.kill_called is True
        assert (
            "SIGKILL" in result.output or "did not exit after SIGKILL" in result.output
        )
    finally:
        await store.close()


async def test_terminate_is_lead_only_in_factory(tmp_path: Path) -> None:
    """terminate_agent must be gated behind the spawn capability — custom
    YAMLs that list it on a non-spawning agent should have it stripped."""

    from feather.core.agent.factory import _CAPABILITY_GATED_TOOLS

    assert _CAPABILITY_GATED_TOOLS["terminate_agent"] == "can_spawn"
