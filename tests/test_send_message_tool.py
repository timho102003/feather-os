"""Tests for the SendMessageTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.models import ToolExecutionContext
from feather.storage.agent_message_store import AgentMessageStore
from feather.tools.send_message_tool import SendMessageTool


async def _open_store(tmp_path: Path) -> AgentMessageStore:
    store = AgentMessageStore(tmp_path / "feather.db")
    await store.initialize()
    return store


async def test_send_message_valid_call_persists_row(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        tool = SendMessageTool(store, from_agent_name="Engineer")
        result = await tool.execute(
            {
                "to_agent_name": "Lead",
                "to_session_id": "lead-sess",
                "body": "status: 50% done",
                "expects_response": False,
                "in_reply_to": None,
            },
            ToolExecutionContext(session_id="eng-sess", agent_name="Engineer"),
        )
        assert "message_id:" in result.output
        assert "to: Lead" in result.output
        inbox = await store.inbox(
            to_session_id="lead-sess", to_agent_name="Lead"
        )
        assert [m.body for m in inbox] == ["status: 50% done"]
        assert inbox[0].from_agent_name == "Engineer"
    finally:
        await store.close()


async def test_expects_response_surfaces_correlation_id(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        tool = SendMessageTool(store, from_agent_name="Lead")
        result = await tool.execute(
            {
                "to_agent_name": "Engineer",
                "to_session_id": "eng-sess",
                "body": "status?",
                "expects_response": True,
                "in_reply_to": None,
            },
            ToolExecutionContext(session_id="lead-sess", agent_name="Lead"),
        )
        assert "correlation_id:" in result.output
    finally:
        await store.close()


async def test_reply_marks_original_responded(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        lead_tool = SendMessageTool(store, from_agent_name="Lead")
        question = await lead_tool.execute(
            {
                "to_agent_name": "Engineer",
                "to_session_id": "eng-sess",
                "body": "ETA?",
                "expects_response": True,
                "in_reply_to": None,
            },
            ToolExecutionContext(session_id="lead-sess", agent_name="Lead"),
        )
        # Pull correlation_id from output.
        cid = next(
            line.split(":", 1)[1].strip()
            for line in question.output.splitlines()
            if line.startswith("correlation_id:")
        )
        eng_tool = SendMessageTool(store, from_agent_name="Engineer")
        await eng_tool.execute(
            {
                "to_agent_name": "Lead",
                "to_session_id": "lead-sess",
                "body": "5 minutes",
                "expects_response": False,
                "in_reply_to": cid,
            },
            ToolExecutionContext(session_id="eng-sess", agent_name="Engineer"),
        )
        paired = await store.get_by_correlation(cid)
        statuses = {m.body: m.status.value for m in paired}
        assert statuses["ETA?"] == "responded"
    finally:
        await store.close()


async def test_rejects_empty_fields(tmp_path: Path) -> None:
    store = await _open_store(tmp_path)
    try:
        tool = SendMessageTool(store, from_agent_name="Engineer")
        with pytest.raises(ValueError):
            await tool.execute(
                {
                    "to_agent_name": "",
                    "to_session_id": "x",
                    "body": "hi",
                    "expects_response": False,
                    "in_reply_to": None,
                },
                ToolExecutionContext(session_id="eng", agent_name="Engineer"),
            )
        with pytest.raises(ValueError):
            await tool.execute(
                {
                    "to_agent_name": "Lead",
                    "to_session_id": "x",
                    "body": "   ",
                    "expects_response": False,
                    "in_reply_to": None,
                },
                ToolExecutionContext(session_id="eng", agent_name="Engineer"),
            )
    finally:
        await store.close()


async def test_self_send_is_rejected(tmp_path: Path) -> None:
    """Sending to one's own inbox is a no-op and should be rejected."""

    store = await _open_store(tmp_path)
    try:
        tool = SendMessageTool(store, from_agent_name="Engineer")
        with pytest.raises(ValueError, match="sender's own inbox"):
            await tool.execute(
                {
                    "to_agent_name": "Engineer",
                    "to_session_id": "eng-sess",
                    "body": "self-talk",
                    "expects_response": False,
                    "in_reply_to": None,
                },
                ToolExecutionContext(session_id="eng-sess", agent_name="Engineer"),
            )
    finally:
        await store.close()
