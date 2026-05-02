"""Tests for filesystem-backed attachment persistence."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from feather.models import MessageRole
from feather.storage.attachment_store import AttachmentStore
from feather.storage.session_store import SessionStore


class FakeMemoryService:
    """Capture attachment indexing calls without Qdrant."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def index_attachment(self, record: object, *, content: str) -> None:
        """Record an attempted attachment index."""

        self.calls.append({"record": record, "content": content})


async def test_attachment_store_copies_records_and_indexes_text(
    tmp_path: Path,
) -> None:
    """Dropped files should be copied under .feather/attachments and indexed."""

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    memory = FakeMemoryService()
    attachment_store = AttachmentStore(
        root=tmp_path,
        session_store=session_store,
        memory_service=memory,
    )
    source = tmp_path / "source.txt"
    source.write_text("attachment body", encoding="utf-8")

    try:
        session = await session_store.create_session("Lead")
        message = await session_store.add_message(
            session.id,
            MessageRole.USER,
            "please read\n[File #1]",
        )
        _, pending = attachment_store.discover(f"please read {source}")

        records = await attachment_store.save_pending(
            session_id=session.id,
            message_id=message.id,
            attachments=pending,
        )
        await attachment_store.drain_indexing(timeout_s=1.0)

        assert len(records) == 1
        record = records[0]
        assert record.filepath.startswith(f".feather/attachments/{session.id}/")
        assert (tmp_path / record.filepath).read_text(encoding="utf-8") == (
            "attachment body"
        )
        assert await session_store.list_message_attachments(message.id) == records
        assert memory.calls[0]["record"] == record
        assert memory.calls[0]["content"] == "attachment body"
    finally:
        await session_store.close()


async def test_attachment_store_indexes_in_background(tmp_path: Path) -> None:
    """Qdrant indexing should not block saving provider-ready attachments."""

    class SlowMemoryService(FakeMemoryService):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def index_attachment(self, record: object, *, content: str) -> None:
            self.started.set()
            await self.release.wait()
            await super().index_attachment(record, content=content)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    memory = SlowMemoryService()
    attachment_store = AttachmentStore(
        root=tmp_path,
        session_store=session_store,
        memory_service=memory,
    )
    source = tmp_path / "source.txt"
    source.write_text("attachment body", encoding="utf-8")

    try:
        session = await session_store.create_session("Lead")
        message = await session_store.add_message(
            session.id,
            MessageRole.USER,
            "please read\n[File #1]",
        )
        _, pending = attachment_store.discover(f"please read {source}")

        records = await attachment_store.save_pending(
            session_id=session.id,
            message_id=message.id,
            attachments=pending,
        )
        await asyncio.wait_for(memory.started.wait(), timeout=1.0)

        assert len(records) == 1
        assert memory.calls == []

        memory.release.set()
        await attachment_store.drain_indexing(timeout_s=1.0)

        assert memory.calls[0]["record"] == records[0]
    finally:
        await session_store.close()


async def test_attachment_store_rejects_unsafe_session_storage_segment(
    tmp_path: Path,
) -> None:
    """Caller-supplied session ids must not control attachment directories."""

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    attachment_store = AttachmentStore(root=tmp_path, session_store=session_store)
    source = tmp_path / "source.txt"
    source.write_text("attachment body", encoding="utf-8")

    try:
        session = await session_store.create_session("Lead", session_id="../escape")
        message = await session_store.add_message(
            session.id,
            MessageRole.USER,
            "please read\n[File #1]",
        )
        _, pending = attachment_store.discover(f"please read {source}")

        try:
            await attachment_store.save_pending(
                session_id=session.id,
                message_id=message.id,
                attachments=pending,
            )
        except ValueError as exc:
            assert "Unsafe attachment storage segment" in str(exc)
        else:
            raise AssertionError("unsafe session id was accepted")

        assert await session_store.list_message_attachments(message.id) == []
        assert not (tmp_path / ".feather" / "attachments").exists()
    finally:
        await session_store.close()
