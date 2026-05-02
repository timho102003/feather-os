"""Filesystem-backed storage for chat attachments."""

from __future__ import annotations

import logging
import re
import asyncio
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from feather.attachments import parse_attachment_drops
from feather.models import AttachmentKind, AttachmentRecord, PendingAttachment
from feather.storage.session_store import SessionStore

logger = logging.getLogger(__name__)

_MAX_INDEX_CHARS = 200_000


class AttachmentStore:
    """Copy dropped files into `.feather/attachments` and record metadata."""

    def __init__(
        self,
        *,
        root: Path,
        session_store: SessionStore,
        memory_service: Any | None = None,
        directory: str = ".feather/attachments",
    ) -> None:
        self._root = root.resolve()
        self._session_store = session_store
        self._memory_service = memory_service
        self._directory = directory
        self._index_tasks: set[asyncio.Task[None]] = set()

    @property
    def root(self) -> Path:
        """Return the workspace root."""

        return self._root

    def discover(self, text: str) -> tuple[str, tuple[PendingAttachment, ...]]:
        """Parse dropped files from user text without saving them."""

        draft = parse_attachment_drops(text, root=self._root)
        return draft.text, draft.attachments

    async def save_pending(
        self,
        *,
        session_id: str,
        message_id: str,
        attachments: tuple[PendingAttachment, ...],
        index: bool = True,
    ) -> list[AttachmentRecord]:
        """Copy pending attachments and persist their DB rows."""

        saved: list[AttachmentRecord] = []
        for pending in attachments:
            record = await self._save_one(
                session_id=session_id,
                message_id=message_id,
                pending=pending,
            )
            saved.append(record)
        if index:
            self.schedule_indexing(saved)
        return saved

    def schedule_indexing(self, records: list[AttachmentRecord]) -> None:
        """Schedule best-effort memory indexing for saved attachments."""

        for record in records:
            self._schedule_index(record)

    async def drain_indexing(self, *, timeout_s: float = 5.0) -> None:
        """Wait briefly for in-flight attachment indexing tasks."""

        if not self._index_tasks:
            return
        _, pending = await asyncio.wait(tuple(self._index_tasks), timeout=timeout_s)
        if pending:
            logger.warning(
                "attachment.index.drain_timeout",
                extra={"pending_tasks": len(pending), "timeout_s": timeout_s},
            )

    async def discard_message(self, message_id: str) -> None:
        """Delete one provisional message and unlink any copied attachment files."""

        records = await self._session_store.delete_message_and_attachments(message_id)
        for record in records:
            path = (self._root / record.filepath).resolve()
            try:
                if path.is_file() and _is_under(path, self._root):
                    path.unlink()
            except OSError:
                logger.warning(
                    "attachment.discard_file_failed",
                    extra={
                        "message_id": message_id,
                        "attachment_id": record.id,
                        "filepath": record.filepath,
                    },
                    exc_info=True,
                )

    async def _save_one(
        self,
        *,
        session_id: str,
        message_id: str,
        pending: PendingAttachment,
    ) -> AttachmentRecord:
        source = Path(pending.source_path).resolve()
        if not source.is_file():
            raise ValueError(f"Attachment file no longer exists: {pending.source_path}")
        attachment_id = str(uuid4())
        session_segment = _safe_storage_segment(session_id)
        base_dir = (self._root / self._directory).resolve()
        if not _is_under(base_dir, self._root):
            raise ValueError(
                f"Attachment directory escapes workspace: {self._directory}"
            )
        target_dir = (base_dir / session_segment).resolve()
        if not _is_under(target_dir, base_dir):
            raise ValueError(f"Attachment session path escapes workspace: {session_id}")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{attachment_id}-{_safe_filename(pending.original_name)}"
        copy_task: asyncio.Task[object] | None = None
        try:
            copy_task = asyncio.create_task(
                asyncio.to_thread(shutil.copy2, source, target)
            )
            await asyncio.shield(copy_task)
            filepath = str(target.resolve().relative_to(self._root))
            return await self._session_store.add_attachment(
                attachment_id=attachment_id,
                session_id=session_id,
                message_id=message_id,
                kind=pending.kind,
                mime_type=pending.mime_type,
                original_name=pending.original_name,
                filepath=filepath,
                size_bytes=target.stat().st_size,
            )
        except BaseException:
            if copy_task is not None and not copy_task.done():
                try:
                    await asyncio.shield(copy_task)
                except Exception:
                    pass
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                logger.warning(
                    "attachment.partial_copy_cleanup_failed",
                    extra={"target": str(target)},
                    exc_info=True,
                )
            raise

    def _schedule_index(self, record: AttachmentRecord) -> None:
        if self._memory_service is None:
            return
        task = asyncio.create_task(self._index_best_effort(record))
        self._index_tasks.add(task)
        task.add_done_callback(self._index_tasks.discard)

    async def _index_best_effort(self, record: AttachmentRecord) -> None:
        if self._memory_service is None:
            return
        try:
            content = await asyncio.to_thread(self._indexable_content, record)
            await self._memory_service.index_attachment(record, content=content)
        except Exception:
            logger.exception(
                "attachment.index.failed",
                extra={
                    "session_id": record.session_id,
                    "message_id": record.message_id,
                    "attachment_id": record.id,
                    "filepath": record.filepath,
                },
            )

    def _indexable_content(self, record: AttachmentRecord) -> str:
        path = self._root / record.filepath
        if record.kind == AttachmentKind.IMAGE:
            return (
                f"Image attachment {record.original_name}. "
                f"MIME type {record.mime_type}. Saved at {record.filepath}."
            )
        if record.mime_type == "application/pdf":
            text = _try_extract_pdf_text(path)
            if text:
                return text[:_MAX_INDEX_CHARS]
        if record.mime_type.startswith("text/") or path.suffix.lower() in {
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".py",
            ".js",
            ".ts",
            ".html",
            ".css",
            ".xml",
        }:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = ""
            if text.strip():
                return text[:_MAX_INDEX_CHARS]
        return (
            f"File attachment {record.original_name}. MIME type {record.mime_type}. "
            f"Saved at {record.filepath}."
        )


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip(".-")
    return safe or "attachment"


def _safe_storage_segment(value: str) -> str:
    safe = _safe_filename(value)
    if safe != value:
        raise ValueError(f"Unsafe attachment storage segment: {value}")
    return safe


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _try_extract_pdf_text(path: Path) -> str:
    try:
        from feather.pdf import extract_pdf_text
    except Exception:
        return ""
    try:
        return extract_pdf_text(path, mode="auto", max_chars=_MAX_INDEX_CHARS)
    except Exception:
        return ""
