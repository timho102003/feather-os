"""Race tests between the reaper and terminate_agent over one child."""

from __future__ import annotations

from pathlib import Path

from feather.core.subagents.reaper import SubagentReaper
from feather.core.subagents.registry import LiveSubagent, SubagentRegistry
from feather.models import ToolExecutionContext
from feather.storage.agent_message_store import AgentMessageStore
from feather.subagent_protocol import RESULT_BEGIN, RESULT_END
from feather.tools.terminate_agent_tool import TerminateAgentTool


class _FakeProc:
    def __init__(self, *, returncode: int | None) -> None:
        self.returncode = returncode

    def terminate(self) -> None:
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        if self.returncode is None:
            self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


def _envelope(status: str = "completed", text: str = "done") -> bytes:
    import json

    body = {
        "status": status,
        "agent_name": "slow",
        "role": "custom",
        "session_id": "child-sess",
        "parent_session_id": "lead-sess",
        "assistant_text": text,
        "question": None,
        "error": None,
    }
    return f"{RESULT_BEGIN}\n{json.dumps(body)}\n{RESULT_END}\n".encode()


async def test_terminate_first_blocks_reaper_duplicate_post(tmp_path: Path) -> None:
    """When terminate_agent claims the child first, a later reaper tick
    that finds the same returncode must NOT deliver a second final
    message."""

    store = AgentMessageStore(tmp_path / "feather.db")
    await store.initialize()
    registry = SubagentRegistry()
    try:
        proc = _FakeProc(returncode=None)  # starts alive; terminate sets rc
        live = LiveSubagent(
            session_id="child-sess",
            agent_name="slow",
            parent_session_id="lead-sess",
            parent_agent_name="Lead",
            process=proc,  # type: ignore[arg-type]
            task_text="",
            correlation_id="corr-1",
            # Pre-populate the buffer so if the reaper DID run, it would
            # have something to post. Escapes catching a false pass.
            stdout_buffer=bytearray(_envelope(text="finished-on-own")),
        )
        await registry.register(live)

        # Step 1: Terminate claims the child first.
        tool = TerminateAgentTool(
            registry=registry,
            agent_message_store=store,
            parent_agent_name="Lead",
        )
        await tool.execute(
            {"session_id": "child-sess", "reason": "plan change"},
            ToolExecutionContext(session_id="lead-sess", agent_name="Lead"),
        )

        # Step 2: Now fake the reaper's scenario: the subprocess actually
        # exited before terminate completed. Its returncode is set. The
        # entry is already removed, so the reaper snapshot is empty.
        reaper = SubagentReaper(registry=registry, agent_message_store=store)
        reaped = await reaper.run_once()
        assert reaped == 0

        inbox = await store.inbox(
            to_session_id="lead-sess", to_agent_name="Lead"
        )
        assert len(inbox) == 1, f"expected 1 final message, got {len(inbox)}"
        assert "terminated" in inbox[0].body
        # And critically NOT the reaper's envelope text.
        assert "finished-on-own" not in inbox[0].body
    finally:
        await store.close()


async def test_reaper_first_prevents_later_terminate_duplicate(
    tmp_path: Path,
) -> None:
    """If the reaper already claimed an exited child, a subsequent
    terminate_agent call must see the missing entry and return a no-op
    WITHOUT posting anything."""

    store = AgentMessageStore(tmp_path / "feather.db")
    await store.initialize()
    registry = SubagentRegistry()
    try:
        proc = _FakeProc(returncode=0)  # exited
        live = LiveSubagent(
            session_id="child-sess",
            agent_name="slow",
            parent_session_id="lead-sess",
            parent_agent_name="Lead",
            process=proc,  # type: ignore[arg-type]
            task_text="",
            correlation_id="corr-1",
            stdout_buffer=bytearray(_envelope(text="finished-ok")),
        )
        await registry.register(live)

        reaper = SubagentReaper(registry=registry, agent_message_store=store)
        assert await reaper.run_once() == 1
        # Reaper has posted one message; registry is empty.
        assert await registry.snapshot() == []

        tool = TerminateAgentTool(
            registry=registry,
            agent_message_store=store,
            parent_agent_name="Lead",
        )
        result = await tool.execute(
            {"session_id": "child-sess", "reason": "late"},
            ToolExecutionContext(session_id="lead-sess", agent_name="Lead"),
        )
        assert "not registered as live" in result.output
        inbox = await store.inbox(
            to_session_id="lead-sess", to_agent_name="Lead"
        )
        assert len(inbox) == 1
        assert "finished-ok" in inbox[0].body
    finally:
        await store.close()
