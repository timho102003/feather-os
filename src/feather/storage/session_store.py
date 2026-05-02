"""SQLite-backed session and message persistence."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite

from feather.models import (
    AttachmentKind,
    AttachmentRecord,
    MessageRole,
    SessionMessage,
    SessionRecord,
    SessionStatus,
)
from feather.storage.schema import initialize_database_schema

_UNSET = object()


class SessionStore:
    """Persist lead-agent session state in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._active_mcp_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create the database and required tables if missing."""

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self._db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA foreign_keys=ON;")
        await self._connection.execute("PRAGMA journal_mode=WAL;")
        await initialize_database_schema(self._connection)
        await self._connection.commit()

    async def close(self) -> None:
        """Close the SQLite connection."""

        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def create_session(
        self,
        agent_name: str,
        *,
        session_id: str | None = None,
    ) -> SessionRecord:
        """Create a new session row.

        Args:
            agent_name: Name of the owning agent.
            session_id: Optional caller-supplied id. When omitted a fresh
                UUID is minted. Used by the subprocess sub-agent path so
                the parent can address the child's inbox before the child
                has finished starting up.

        Returns:
            Newly created session record.
        """

        now = _utc_now()
        effective_id = session_id or str(uuid4())
        await self._execute(
            """
            INSERT INTO sessions (
                id, agent_name, status, last_response_id, loaded_skills, active_mcp_servers, pending_inputs, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                effective_id,
                agent_name,
                SessionStatus.ACTIVE.value,
                None,
                "[]",
                "[]",
                "[]",
                now,
                now,
            ),
        )
        await self._connection.commit()
        return await self.get_session(effective_id)

    async def get_session(self, session_id: str) -> SessionRecord:
        """Fetch one session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session record.
        """

        row = await self._fetchone("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if row is None:
            raise ValueError(f"Unknown session: {session_id}")
        return SessionRecord(
            id=row["id"],
            agent_name=row["agent_name"],
            status=SessionStatus(row["status"]),
            last_response_id=row["last_response_id"],
            loaded_skills=json.loads(row["loaded_skills"]),
            active_mcp_servers=json.loads(row["active_mcp_servers"]),
            pending_inputs=json.loads(row["pending_inputs"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        *,
        file_ref: str | None = None,
        is_compact: bool = False,
    ) -> SessionMessage:
        """Append one message to the session.

        Args:
            session_id: Session identifier.
            role: Message role.
            content: Message content.
            file_ref: Optional file reference for overflow or file-backed content.

        Returns:
            The stored message.
        """

        sequence_row = await self._fetchone(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM messages WHERE session_id = ?",
            (session_id,),
        )
        sequence = int(sequence_row["next_sequence"])
        message_id = str(uuid4())
        now = _utc_now()
        await self._execute(
            """
            INSERT INTO messages (id, session_id, role, content, file_ref, is_compact, sequence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, session_id, role.value, content, file_ref, int(is_compact), sequence, now),
        )
        await self._touch_session(session_id)
        await self._connection.commit()
        return SessionMessage(
            id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            file_ref=file_ref,
            is_compact=is_compact,
            sequence=sequence,
            created_at=now,
        )

    async def update_message_content(self, message_id: str, content: str) -> None:
        """Replace a stored message's display content."""

        await self._execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            (content, message_id),
        )
        await self._connection.commit()

    async def add_attachment(
        self,
        *,
        attachment_id: str,
        session_id: str,
        message_id: str,
        kind: AttachmentKind,
        mime_type: str,
        original_name: str,
        filepath: str,
        size_bytes: int,
    ) -> AttachmentRecord:
        """Persist one file attachment linked to a chat message."""

        now = _utc_now()
        await self._execute(
            """
            INSERT INTO message_attachments (
                id, session_id, message_id, kind, mime_type, original_name, filepath, size_bytes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attachment_id,
                session_id,
                message_id,
                kind.value,
                mime_type,
                original_name,
                filepath,
                int(size_bytes),
                now,
            ),
        )
        await self._connection.commit()
        return AttachmentRecord(
            id=attachment_id,
            session_id=session_id,
            message_id=message_id,
            kind=kind,
            mime_type=mime_type,
            original_name=original_name,
            filepath=filepath,
            size_bytes=int(size_bytes),
            created_at=now,
        )

    async def list_message_attachments(self, message_id: str) -> list[AttachmentRecord]:
        """List attachments for one message in creation order."""

        cursor = await self._connection.execute(
            """
            SELECT * FROM message_attachments
            WHERE message_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (message_id,),
        )
        rows = await cursor.fetchall()
        return [self._attachment_from_row(row) for row in rows]

    async def list_session_attachments(self, session_id: str) -> list[AttachmentRecord]:
        """List attachments for one session in creation order."""

        cursor = await self._connection.execute(
            """
            SELECT * FROM message_attachments
            WHERE session_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [self._attachment_from_row(row) for row in rows]

    async def list_attachments_for_messages(
        self,
        message_ids: list[str],
    ) -> dict[str, list[AttachmentRecord]]:
        """Batch-load attachments grouped by message id."""

        if not message_ids:
            return {}
        placeholders = ",".join("?" for _ in message_ids)
        cursor = await self._connection.execute(
            f"""
            SELECT * FROM message_attachments
            WHERE message_id IN ({placeholders})
            ORDER BY created_at ASC, id ASC
            """,
            tuple(message_ids),
        )
        rows = await cursor.fetchall()
        grouped: dict[str, list[AttachmentRecord]] = {}
        for row in rows:
            record = self._attachment_from_row(row)
            grouped.setdefault(record.message_id, []).append(record)
        return grouped

    async def delete_message_and_attachments(
        self,
        message_id: str,
    ) -> list[AttachmentRecord]:
        """Delete one message plus linked attachment rows.

        Returns:
            Attachment rows that were linked to the deleted message so callers
            can clean up copied files.
        """

        attachments = await self.list_message_attachments(message_id)
        await self._execute(
            "DELETE FROM message_attachments WHERE message_id = ?",
            (message_id,),
        )
        await self._execute("DELETE FROM messages WHERE id = ?", (message_id,))
        await self._connection.commit()
        return attachments

    async def list_messages(self, session_id: str) -> list[SessionMessage]:
        """List messages for one session.

        Args:
            session_id: Session identifier.

        Returns:
            Ordered messages.
        """

        cursor = await self._connection.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY sequence ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            SessionMessage(
                id=row["id"],
                session_id=row["session_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                file_ref=row["file_ref"],
                is_compact=bool(row["is_compact"]),
                sequence=row["sequence"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def list_active_messages(self, session_id: str) -> list[SessionMessage]:
        """List the active context window for one session.

        Active history starts at the latest compact message, if one exists.
        Otherwise it includes the entire session history.
        """

        latest_compact = await self.get_latest_compact_message(session_id)
        if latest_compact is None:
            return await self.list_messages(session_id)

        cursor = await self._connection.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ? AND sequence >= ?
            ORDER BY sequence ASC
            """,
            (session_id, latest_compact.sequence),
        )
        rows = await cursor.fetchall()
        return [
            SessionMessage(
                id=row["id"],
                session_id=row["session_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                file_ref=row["file_ref"],
                is_compact=bool(row["is_compact"]),
                sequence=row["sequence"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def get_latest_compact_message(self, session_id: str) -> SessionMessage | None:
        """Fetch the latest compact summary message for a session, if any."""

        row = await self._fetchone(
            """
            SELECT * FROM messages
            WHERE session_id = ? AND is_compact = 1
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (session_id,),
        )
        if row is None:
            return None
        return SessionMessage(
            id=row["id"],
            session_id=row["session_id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            file_ref=row["file_ref"],
            is_compact=bool(row["is_compact"]),
            sequence=row["sequence"],
            created_at=row["created_at"],
        )

    async def get_non_compact_after(
        self, session_id: str, after_message_id: str | None
    ) -> list[SessionMessage]:
        """Return non-compact messages after the given anchor message.

        Used by the memory subsystem's window builder to determine the next
        10-turn extraction window. The anchor is the ``end_message_id`` of
        the most recently created memory for this session (read from Qdrant).
        Compact summary rows are filtered out so they never enter memory
        extraction.

        Args:
            session_id: Session identifier.
            after_message_id: Anchor message id; messages with ``sequence``
                strictly greater than this id's sequence are returned. When
                ``None``, *or* when the id refers to a row that no longer
                exists locally, all non-compact messages are returned —
                Qdrant may reference a message that has been pruned.

        Returns:
            Non-compact messages in ascending sequence order.
        """

        anchor_sequence = 0
        if after_message_id is not None:
            row = await self._fetchone(
                """
                SELECT sequence FROM messages
                WHERE session_id = ? AND id = ?
                """,
                (session_id, after_message_id),
            )
            if row is not None:
                anchor_sequence = int(row["sequence"])

        cursor = await self._connection.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ? AND sequence > ? AND is_compact = 0
            ORDER BY sequence ASC
            """,
            (session_id, anchor_sequence),
        )
        rows = await cursor.fetchall()
        return [
            SessionMessage(
                id=row["id"],
                session_id=row["session_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                file_ref=row["file_ref"],
                is_compact=bool(row["is_compact"]),
                sequence=row["sequence"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def get_recent_non_compact(
        self, session_id: str, limit: int
    ) -> list[SessionMessage]:
        """Return the latest ``limit`` non-compact messages in ascending order.

        Used by the memory query-builder to gather conversation context for
        the read-path retrieval query. ``limit=0`` returns an empty list.

        Args:
            session_id: Session identifier.
            limit: Maximum number of messages to return; ``0`` allowed.

        Returns:
            Non-compact messages, oldest-first within the slice.
        """

        if limit <= 0:
            return []
        cursor = await self._connection.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ? AND is_compact = 0
            ORDER BY sequence DESC
            LIMIT ?
            """,
            (session_id, int(limit)),
        )
        rows = await cursor.fetchall()
        # Re-order ascending so callers get oldest → newest.
        rows = list(reversed(rows))
        return [
            SessionMessage(
                id=row["id"],
                session_id=row["session_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                file_ref=row["file_ref"],
                is_compact=bool(row["is_compact"]),
                sequence=row["sequence"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def render_history_for_cache(self, session_id: str) -> str:
        """Render session history into a cache-friendly text format.

        Tool messages with file-backed content are rendered as file references
        rather than inline payloads so replay caches stay small.

        Args:
            session_id: Session identifier.

        Returns:
            Rendered history string.
        """

        messages = await self.list_active_messages(session_id)
        attachments_by_message = await self.list_attachments_for_messages(
            [message.id for message in messages]
        )
        lines: list[str] = []
        for message in messages:
            rendered_content = message.content
            role_label = message.role.value
            attachments = attachments_by_message.get(message.id, [])
            if attachments:
                rendered_content = "\n".join(
                    [
                        rendered_content,
                        _render_attachment_history_block(attachments),
                    ]
                ).strip()
            if message.is_compact:
                role_label = f"{role_label}[compact]"
            lines.append(f"{role_label}: {rendered_content}")
        return "\n".join(lines)

    async def append_loaded_skill(self, session_id: str, skill_name: str) -> None:
        """Append one loaded skill if not already present.

        Args:
            session_id: Session identifier.
            skill_name: Exact loaded skill name.
        """

        session = await self.get_session(session_id)
        loaded_skills = list(session.loaded_skills)
        if skill_name not in loaded_skills:
            loaded_skills.append(skill_name)
        await self._update_session_fields(
            session_id,
            loaded_skills=json.dumps(loaded_skills),
        )

    async def append_active_mcp_server(self, session_id: str, server_label: str) -> None:
        """Mark one MCP server active for a session if not already present.

        Args:
            session_id: Session identifier.
            server_label: Configured MCP server label.
        """

        async with self._active_mcp_lock:
            session = await self.get_session(session_id)
            active_mcp_servers = list(session.active_mcp_servers)
            if server_label not in active_mcp_servers:
                active_mcp_servers.append(server_label)
            await self._update_session_fields(
                session_id,
                active_mcp_servers=json.dumps(active_mcp_servers),
            )

    async def update_response_state(
        self,
        session_id: str,
        *,
        last_response_id: str | None | object = _UNSET,
        pending_inputs: list[dict[str, Any]] | object = _UNSET,
        status: SessionStatus | None = None,
    ) -> None:
        """Update the provider cursor and pending inputs.

        Args:
            session_id: Session identifier.
            last_response_id: New provider response ID, if any.
            pending_inputs: Pending input items to send later.
            status: New session status.
        """

        values: dict[str, str | None] = {}
        if last_response_id is not _UNSET:
            values["last_response_id"] = last_response_id
        if pending_inputs is not _UNSET:
            values["pending_inputs"] = json.dumps(pending_inputs)
        if status is not None:
            values["status"] = status.value
        if values:
            await self._update_session_fields(session_id, **values)

    async def _update_session_fields(self, session_id: str, **values: str | None) -> None:
        assignments = ", ".join(f"{key} = ?" for key in values)
        params = list(values.values()) + [_utc_now(), session_id]
        await self._execute(
            f"UPDATE sessions SET {assignments}, updated_at = ? WHERE id = ?",
            tuple(params),
        )
        await self._connection.commit()

    async def _touch_session(self, session_id: str) -> None:
        await self._execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (_utc_now(), session_id),
        )

    async def _execute(self, query: str, params: tuple) -> None:
        if self._connection is None:
            raise RuntimeError("SessionStore.initialize() must be called before use.")
        await self._connection.execute(query, params)

    async def _fetchone(self, query: str, params: tuple) -> aiosqlite.Row | None:
        if self._connection is None:
            raise RuntimeError("SessionStore.initialize() must be called before use.")
        cursor = await self._connection.execute(query, params)
        return await cursor.fetchone()


    @staticmethod
    def _attachment_from_row(row: aiosqlite.Row) -> AttachmentRecord:
        return AttachmentRecord(
            id=row["id"],
            session_id=row["session_id"],
            message_id=row["message_id"],
            kind=AttachmentKind(row["kind"]),
            mime_type=row["mime_type"],
            original_name=row["original_name"],
            filepath=row["filepath"],
            size_bytes=int(row["size_bytes"]),
            created_at=row["created_at"],
        )


def _utc_now() -> str:
    """Return an ISO timestamp in UTC.

    Returns:
        UTC timestamp string.
    """

    return datetime.now(UTC).isoformat()


def _render_attachment_history_block(attachments: list[AttachmentRecord]) -> str:
    lines = ["attachments:"]
    image_index = 0
    file_index = 0
    for attachment in attachments:
        if attachment.kind == AttachmentKind.IMAGE:
            image_index += 1
            label = f"[image #{image_index}]"
        else:
            file_index += 1
            label = f"[File #{file_index}]"
        lines.append(
            "  "
            f"{label} name={json.dumps(attachment.original_name)} "
            f"mime={json.dumps(attachment.mime_type)} "
            f"path={json.dumps(attachment.filepath)}"
        )
    return "\n".join(lines)
