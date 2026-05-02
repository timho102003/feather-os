"""Tests for the AgentMessageStore SQLite mailbox."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from feather.models import AgentMessageStatus
from feather.storage.agent_message_store import AgentMessageStore


async def _open(tmp_path: Path, *, cap: int = 50) -> AgentMessageStore:
    store = AgentMessageStore(tmp_path / "feather.db", inbox_cap=cap)
    await store.initialize()
    return store


async def test_send_and_inbox_fifo(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        for i in range(3):
            await store.send(
                from_session_id="s-sender",
                from_agent_name="lead",
                to_session_id="s-recv",
                to_agent_name="engineer",
                body=f"msg-{i}",
            )
        inbox = await store.inbox(
            to_session_id="s-recv", to_agent_name="engineer"
        )
        assert [m.body for m in inbox] == ["msg-0", "msg-1", "msg-2"]
        assert all(m.status == AgentMessageStatus.PENDING for m in inbox)
    finally:
        await store.close()


async def test_inbox_isolates_recipients(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await store.send(
            from_session_id="sa",
            from_agent_name="lead",
            to_session_id="sx",
            to_agent_name="engineer",
            body="to engineer",
        )
        await store.send(
            from_session_id="sa",
            from_agent_name="lead",
            to_session_id="sy",
            to_agent_name="designer",
            body="to designer",
        )
        eng = await store.inbox(to_session_id="sx", to_agent_name="engineer")
        des = await store.inbox(to_session_id="sy", to_agent_name="designer")
        assert [m.body for m in eng] == ["to engineer"]
        assert [m.body for m in des] == ["to designer"]
    finally:
        await store.close()


async def test_empty_body_is_rejected(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        with pytest.raises(ValueError):
            await store.send(
                from_session_id="sa",
                from_agent_name="lead",
                to_session_id="sb",
                to_agent_name="engineer",
                body="   ",
            )
    finally:
        await store.close()


async def test_mark_delivered_transitions_status(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        m1 = await store.send(
            from_session_id="sa",
            from_agent_name="lead",
            to_session_id="sb",
            to_agent_name="engineer",
            body="hi",
        )
        assert await store.pending_count(
            to_session_id="sb", to_agent_name="engineer"
        ) == 1
        assert await store.mark_delivered([m1.id]) == 1
        assert await store.pending_count(
            to_session_id="sb", to_agent_name="engineer"
        ) == 0
        # Idempotent: calling again must not mutate.
        assert await store.mark_delivered([m1.id]) == 1  # reports ids attempted
    finally:
        await store.close()


async def test_expects_response_generates_correlation_id(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        m = await store.send(
            from_session_id="sa",
            from_agent_name="lead",
            to_session_id="sb",
            to_agent_name="engineer",
            body="status?",
            expects_response=True,
        )
        assert m.correlation_id is not None
        assert m.expects_response is True
    finally:
        await store.close()


async def test_reply_marks_original_responded(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        # Lead asks engineer for status.
        question = await store.send(
            from_session_id="s-lead",
            from_agent_name="lead",
            to_session_id="s-eng",
            to_agent_name="engineer",
            body="status?",
            expects_response=True,
        )
        cid = question.correlation_id
        assert cid is not None
        # Engineer replies.
        await store.send(
            from_session_id="s-eng",
            from_agent_name="engineer",
            to_session_id="s-lead",
            to_agent_name="lead",
            body="50% done",
            in_reply_to=cid,
        )
        # Original message must be flipped to RESPONDED.
        all_for_corr = await store.get_by_correlation(cid)
        statuses = {m.body: m.status for m in all_for_corr}
        assert statuses["status?"] == AgentMessageStatus.RESPONDED
    finally:
        await store.close()


async def test_reply_marks_delivered_original_responded(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        question = await store.send(
            from_session_id="s-lead",
            from_agent_name="lead",
            to_session_id="s-eng",
            to_agent_name="engineer",
            body="status?",
            expects_response=True,
        )
        cid = question.correlation_id
        assert cid is not None
        await store.mark_delivered([question.id])

        await store.send(
            from_session_id="s-eng",
            from_agent_name="engineer",
            to_session_id="s-lead",
            to_agent_name="lead",
            body="50% done",
            in_reply_to=cid,
        )

        all_for_corr = await store.get_by_correlation(cid)
        statuses = {m.body: m.status for m in all_for_corr}
        assert statuses["status?"] == AgentMessageStatus.RESPONDED
    finally:
        await store.close()


async def test_claim_reply_delivers_only_matching_pending_reply(tmp_path: Path) -> None:
    store = await _open(tmp_path)
    try:
        await store.send(
            from_session_id="s-lead",
            from_agent_name="lead",
            to_session_id="s-eng",
            to_agent_name="engineer",
            body="wrong answer",
            in_reply_to="other-correlation",
        )
        expected = await store.send(
            from_session_id="s-lead",
            from_agent_name="lead",
            to_session_id="s-eng",
            to_agent_name="engineer",
            body="right answer",
            in_reply_to="wanted-correlation",
        )

        claimed = await store.claim_reply(
            to_session_id="s-eng",
            to_agent_name="engineer",
            in_reply_to="wanted-correlation",
        )

        assert claimed is not None
        assert claimed.id == expected.id
        assert claimed.status == AgentMessageStatus.DELIVERED
        remaining = await store.inbox(to_session_id="s-eng", to_agent_name="engineer")
        assert [message.body for message in remaining] == ["wrong answer"]
    finally:
        await store.close()


async def test_inbox_cap_drops_oldest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = await _open(tmp_path, cap=3)
    try:
        caplog.set_level(logging.WARNING)
        for i in range(3):
            await store.send(
                from_session_id="sa",
                from_agent_name="lead",
                to_session_id="sb",
                to_agent_name="engineer",
                body=f"m{i}",
            )
        # 4th insert must drop m0 (oldest).
        await store.send(
            from_session_id="sa",
            from_agent_name="lead",
            to_session_id="sb",
            to_agent_name="engineer",
            body="m3",
        )
        inbox = await store.inbox(
            to_session_id="sb", to_agent_name="engineer"
        )
        assert [m.body for m in inbox] == ["m1", "m2", "m3"]
        assert any("inbox overflow" in rec.message for rec in caplog.records)
    finally:
        await store.close()


async def test_invalid_cap_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AgentMessageStore(tmp_path / "x.db", inbox_cap=0)


async def test_cross_connection_visibility(tmp_path: Path) -> None:
    """Two stores on the same DB file must see each other's writes —
    mirrors the parent-writer / subprocess-reader deployment.
    """

    writer = await _open(tmp_path)
    reader = AgentMessageStore(tmp_path / "feather.db")
    await reader.initialize()
    try:
        await writer.send(
            from_session_id="sa",
            from_agent_name="lead",
            to_session_id="sb",
            to_agent_name="engineer",
            body="cross-process",
        )
        inbox = await reader.inbox(
            to_session_id="sb", to_agent_name="engineer"
        )
        assert [m.body for m in inbox] == ["cross-process"]
    finally:
        await writer.close()
        await reader.close()
