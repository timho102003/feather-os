"""SQLite-backed mailbox for inter-agent messages.

Each agent (lead, sub-agent, custom) has an inbox addressed by
``(to_session_id, to_agent_name)``. Writers append pending messages via
:meth:`AgentMessageStore.send`. Recipients drain their inbox once per
agent-loop iteration via :meth:`AgentMessageStore.inbox`.

The store is deliberately thin — it owns rows on disk, not the in-memory
buffer ``BaseAgent._inbox``. Flow:

  sender.send_message_tool → AgentMessageStore.send (INSERT, status=PENDING)
  recipient.run_loop (per iteration):
      → AgentMessageStore.inbox (SELECT LIMIT 50 ORDER BY created_at)
      → BaseAgent processes and responds
      → AgentMessageStore.mark_delivered (UPDATE status → DELIVERED)
      → if message had in_reply_to, the matched pending message is
        transitioned to RESPONDED via :meth:`mark_responded`

The inbox is bounded per (recipient) pair: when a sender tries to push a
message and the recipient's pending count is already at the cap, the
OLDEST pending message is dropped with a warning (FIFO-bounded). This
matches the user-input queue pattern and prevents a slow recipient from
accumulating unbounded rows.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import aiosqlite

from feather.models import AgentMessage, AgentMessageStatus
from feather.storage.connection import open_store_connection
from feather.storage.schema import initialize_database_schema

logger = logging.getLogger(__name__)

_DEFAULT_INBOX_CAP = 50


class AgentMessageStore:
    """Persist and query the inter-agent message mailbox."""

    def __init__(
        self,
        db_path: Path,
        *,
        inbox_cap: int = _DEFAULT_INBOX_CAP,
    ) -> None:
        if inbox_cap <= 0:
            raise ValueError("inbox_cap must be positive")
        self._db_path = db_path
        self._inbox_cap = inbox_cap
        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the connection and ensure the schema exists."""

        self._connection = await open_store_connection(self._db_path)
        await initialize_database_schema(self._connection)
        await self._connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def send(
        self,
        *,
        from_session_id: str,
        from_agent_name: str,
        to_session_id: str,
        to_agent_name: str,
        body: str,
        expects_response: bool = False,
        correlation_id: str | None = None,
        in_reply_to: str | None = None,
    ) -> AgentMessage:
        """Insert a message into the recipient's inbox (FIFO-bounded)."""

        cleaned_body = body.strip()
        if not cleaned_body:
            raise ValueError("agent_message body must not be empty")
        if not from_session_id or not from_agent_name:
            raise ValueError("from_session_id / from_agent_name must be non-empty")
        if not to_session_id or not to_agent_name:
            raise ValueError("to_session_id / to_agent_name must be non-empty")

        if expects_response and correlation_id is None:
            correlation_id = str(uuid4())

        now = _utc_now()
        message_id = str(uuid4())

        connection = self._require_connection()

        # Single transaction around cap enforcement + insert + optional
        # reply flip so two concurrent writers can't both pass the cap
        # check and silently blow past the FIFO bound.
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await self._enforce_inbox_cap(
                connection, to_session_id, to_agent_name
            )
            await connection.execute(
                """
                INSERT INTO agent_messages (
                    id, from_session_id, from_agent_name,
                    to_session_id, to_agent_name, body,
                    correlation_id, in_reply_to, expects_response,
                    status, created_at, delivered_at, responded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    message_id,
                    from_session_id,
                    from_agent_name,
                    to_session_id,
                    to_agent_name,
                    cleaned_body,
                    correlation_id,
                    in_reply_to,
                    1 if expects_response else 0,
                    AgentMessageStatus.PENDING.value,
                    now,
                ),
            )
            # Reply semantics: flip the ORIGINAL question while it is
            # still awaiting a response. It may already be DELIVERED if
            # the recipient read it before sending the correlated reply.
            if in_reply_to:
                await connection.execute(
                    """
                    UPDATE agent_messages
                       SET status = ?, responded_at = ?
                     WHERE correlation_id = ?
                       AND status IN (?, ?)
                       AND expects_response = 1
                    """,
                    (
                        AgentMessageStatus.RESPONDED.value,
                        now,
                        in_reply_to,
                        AgentMessageStatus.PENDING.value,
                        AgentMessageStatus.DELIVERED.value,
                    ),
                )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

        logger.info(
            "agent_message send from=%s/%s to=%s/%s corr=%s reply=%s expects_response=%s",
            from_agent_name,
            from_session_id,
            to_agent_name,
            to_session_id,
            correlation_id,
            in_reply_to,
            expects_response,
        )
        return AgentMessage(
            id=message_id,
            from_session_id=from_session_id,
            from_agent_name=from_agent_name,
            to_session_id=to_session_id,
            to_agent_name=to_agent_name,
            body=cleaned_body,
            correlation_id=correlation_id,
            in_reply_to=in_reply_to,
            expects_response=expects_response,
            status=AgentMessageStatus.PENDING,
            created_at=now,
            delivered_at=None,
            responded_at=None,
        )

    async def inbox(
        self,
        *,
        to_session_id: str,
        to_agent_name: str,
        limit: int | None = None,
    ) -> list[AgentMessage]:
        """Return pending messages for one recipient, oldest first."""

        effective_limit = self._inbox_cap if limit is None else max(1, int(limit))
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT *
              FROM agent_messages
             WHERE to_session_id = ? AND to_agent_name = ? AND status = ?
             ORDER BY created_at ASC, id ASC
             LIMIT ?
            """,
            (
                to_session_id,
                to_agent_name,
                AgentMessageStatus.PENDING.value,
                effective_limit,
            ),
        )
        rows = await cursor.fetchall()
        return [_row_to_message(row) for row in rows]

    async def mark_delivered(self, message_ids: Iterable[str]) -> int:
        """Flip the status of ``message_ids`` to DELIVERED."""

        ids = [m for m in message_ids if m]
        if not ids:
            return 0
        connection = self._require_connection()
        now = _utc_now()
        placeholders = ",".join("?" for _ in ids)
        await connection.execute(
            f"""
            UPDATE agent_messages
               SET status = ?, delivered_at = ?
             WHERE id IN ({placeholders}) AND status = ?
            """,
            (
                AgentMessageStatus.DELIVERED.value,
                now,
                *ids,
                AgentMessageStatus.PENDING.value,
            ),
        )
        await connection.commit()
        logger.info("agent_message mark_delivered count=%s", len(ids))
        return len(ids)

    async def pending_count(
        self,
        *,
        to_session_id: str,
        to_agent_name: str,
    ) -> int:
        """Return how many PENDING messages are addressed to this recipient."""

        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT COUNT(*)
              FROM agent_messages
             WHERE to_session_id = ? AND to_agent_name = ? AND status = ?
            """,
            (to_session_id, to_agent_name, AgentMessageStatus.PENDING.value),
        )
        row = await cursor.fetchone()
        if row is None:
            return 0
        return int(row[0])

    async def get_by_correlation(
        self, correlation_id: str
    ) -> list[AgentMessage]:
        """Return every message sharing a correlation_id (ordered by created_at)."""

        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT *
              FROM agent_messages
             WHERE correlation_id = ?
             ORDER BY created_at ASC, id ASC
            """,
            (correlation_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_message(row) for row in rows]

    async def claim_reply(
        self,
        *,
        to_session_id: str,
        to_agent_name: str,
        in_reply_to: str,
    ) -> AgentMessage | None:
        """Atomically claim one pending reply addressed to an agent.

        Args:
            to_session_id: Recipient session id.
            to_agent_name: Recipient agent name.
            in_reply_to: Correlation id the reply answers.

        Returns:
            The claimed reply, or ``None`` when no matching pending reply exists.
        """

        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = await connection.execute(
                """
                SELECT *
                  FROM agent_messages
                 WHERE to_session_id = ?
                   AND to_agent_name = ?
                   AND in_reply_to = ?
                   AND status = ?
                 ORDER BY created_at ASC, id ASC
                 LIMIT 1
                """,
                (
                    to_session_id,
                    to_agent_name,
                    in_reply_to,
                    AgentMessageStatus.PENDING.value,
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                await connection.commit()
                return None
            now = _utc_now()
            await connection.execute(
                """
                UPDATE agent_messages
                   SET status = ?, delivered_at = ?
                 WHERE id = ? AND status = ?
                """,
                (
                    AgentMessageStatus.DELIVERED.value,
                    now,
                    row["id"],
                    AgentMessageStatus.PENDING.value,
                ),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

        message = _row_to_message(row)
        message.status = AgentMessageStatus.DELIVERED
        message.delivered_at = now
        logger.info(
            "agent_message claim_reply to=%s/%s reply=%s message_id=%s",
            to_agent_name,
            to_session_id,
            in_reply_to,
            message.id,
        )
        return message

    async def _enforce_inbox_cap(
        self,
        connection: aiosqlite.Connection,
        to_session_id: str,
        to_agent_name: str,
    ) -> None:
        """Drop the oldest pending row if inserting would exceed ``inbox_cap``."""

        cursor = await connection.execute(
            """
            SELECT COUNT(*)
              FROM agent_messages
             WHERE to_session_id = ? AND to_agent_name = ? AND status = ?
            """,
            (to_session_id, to_agent_name, AgentMessageStatus.PENDING.value),
        )
        row = await cursor.fetchone()
        pending = int(row[0]) if row else 0
        if pending < self._inbox_cap:
            return
        # Drop the oldest to make room.
        cursor = await connection.execute(
            """
            SELECT id
              FROM agent_messages
             WHERE to_session_id = ? AND to_agent_name = ? AND status = ?
             ORDER BY created_at ASC, id ASC
             LIMIT 1
            """,
            (to_session_id, to_agent_name, AgentMessageStatus.PENDING.value),
        )
        oldest = await cursor.fetchone()
        if oldest is None:
            return
        oldest_id = str(oldest[0])
        await connection.execute(
            """
            UPDATE agent_messages
               SET status = ?
             WHERE id = ?
            """,
            (AgentMessageStatus.EXPIRED.value, oldest_id),
        )
        logger.warning(
            "agent_message inbox overflow to=%s/%s dropped_id=%s cap=%s",
            to_agent_name,
            to_session_id,
            oldest_id,
            self._inbox_cap,
        )

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError(
                "AgentMessageStore.initialize() must be awaited before use"
            )
        return self._connection


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with microsecond precision."""

    return datetime.now(UTC).isoformat()


def _row_to_message(row: Any) -> AgentMessage:
    """Convert one aiosqlite Row into an :class:`AgentMessage`."""

    return AgentMessage(
        id=str(row["id"]),
        from_session_id=str(row["from_session_id"]),
        from_agent_name=str(row["from_agent_name"]),
        to_session_id=str(row["to_session_id"]),
        to_agent_name=str(row["to_agent_name"]),
        body=str(row["body"]),
        correlation_id=(
            None if row["correlation_id"] is None else str(row["correlation_id"])
        ),
        in_reply_to=(
            None if row["in_reply_to"] is None else str(row["in_reply_to"])
        ),
        expects_response=bool(row["expects_response"]),
        status=AgentMessageStatus(str(row["status"])),
        created_at=str(row["created_at"]),
        delivered_at=(
            None if row["delivered_at"] is None else str(row["delivered_at"])
        ),
        responded_at=(
            None if row["responded_at"] is None else str(row["responded_at"])
        ),
    )
