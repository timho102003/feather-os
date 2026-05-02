"""SQLite persistence for messaging integrations.

Three tables, all created via :mod:`feather.storage.schema`:

- ``messaging_credentials`` — one row per platform with the opaque
  configuration blob (bot tokens, channel secrets, …).
- ``messaging_chats`` — chat-id ↔ session-id mapping. A new mapping is
  inserted the first time we see an unrecognised chat; later messages
  reuse the same Feather session so the agent has continuous context
  with that user.
- ``messaging_inbound_dedup`` — recent native message ids per platform.
  Used to drop duplicate webhook deliveries (Meta and LINE both retry
  on non-200, and LINE explicitly tags retries via
  ``deliveryContext.isRedelivery``).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

from feather.messaging.models import (
    ChatMappingRecord,
    CredentialRecord,
    Platform,
)
from feather.storage.schema import initialize_database_schema

logger = logging.getLogger(__name__)

# How many recent message ids to retain in the dedup table per platform
# before pruning. 1000 covers many minutes at the platforms' rate limits;
# the platforms only retry for a few hours so a small window is enough.
_DEDUP_KEEP_HOURS = 24


class MessagingStore:
    """Persist credentials, chat mappings, and inbound dedup keys.

    Args:
        db_path: Path to the same SQLite database used by other stores.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        """Run schema migrations on the database."""

        async with aiosqlite.connect(self._db_path) as connection:
            await initialize_database_schema(connection)
            await connection.commit()

    # ---- credentials -----------------------------------------------------

    async def save_credentials(
        self,
        platform: Platform,
        config: dict[str, object],
        *,
        enabled: bool = True,
    ) -> None:
        """Insert or replace the credential row for ``platform``."""

        config_json = json.dumps(config, sort_keys=True)
        now = _now_iso()
        async with aiosqlite.connect(self._db_path) as connection:
            await connection.execute(
                """
                INSERT INTO messaging_credentials
                    (platform, config_json, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(platform) DO UPDATE SET
                    config_json = excluded.config_json,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (platform.value, config_json, 1 if enabled else 0, now, now),
            )
            await connection.commit()

    async def load_credentials(
        self, platform: Platform
    ) -> CredentialRecord | None:
        """Return the credential row for ``platform``, or ``None`` if absent."""

        async with aiosqlite.connect(self._db_path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(
                """
                SELECT platform, config_json, enabled, created_at, updated_at
                FROM messaging_credentials WHERE platform = ?
                """,
                (platform.value,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return CredentialRecord(
            platform=Platform(row["platform"]),
            config=json.loads(row["config_json"]),
            enabled=bool(row["enabled"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    async def list_credentials(self) -> list[CredentialRecord]:
        """Return every credential row (used by service start-up)."""

        async with aiosqlite.connect(self._db_path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(
                """
                SELECT platform, config_json, enabled, created_at, updated_at
                FROM messaging_credentials
                """
            )
            rows = await cursor.fetchall()
        records: list[CredentialRecord] = []
        for row in rows:
            try:
                platform = Platform(row["platform"])
            except ValueError:
                # Forward-compatible: unknown platform names persisted by
                # a future version are skipped, not crashed on.
                logger.warning(
                    "messaging.store.unknown_platform platform=%s",
                    row["platform"],
                )
                continue
            records.append(
                CredentialRecord(
                    platform=platform,
                    config=json.loads(row["config_json"]),
                    enabled=bool(row["enabled"]),
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                )
            )
        return records

    async def delete_credentials(self, platform: Platform) -> None:
        """Remove the credential row for ``platform``."""

        async with aiosqlite.connect(self._db_path) as connection:
            await connection.execute(
                "DELETE FROM messaging_credentials WHERE platform = ?",
                (platform.value,),
            )
            await connection.commit()

    # ---- chat mappings --------------------------------------------------

    async def get_chat_mapping(
        self, platform: Platform, chat_id: str
    ) -> ChatMappingRecord | None:
        """Return the mapping for a (platform, chat_id) pair, or ``None``."""

        async with aiosqlite.connect(self._db_path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(
                """
                SELECT platform, chat_id, session_id, display_name, created_at, updated_at
                FROM messaging_chats WHERE platform = ? AND chat_id = ?
                """,
                (platform.value, chat_id),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_mapping(row)

    async def upsert_chat_mapping(
        self,
        platform: Platform,
        chat_id: str,
        session_id: str,
        display_name: str,
    ) -> ChatMappingRecord:
        """Insert a new mapping, or refresh the display name if one exists."""

        now = _now_iso()
        async with aiosqlite.connect(self._db_path) as connection:
            connection.row_factory = aiosqlite.Row
            await connection.execute(
                """
                INSERT INTO messaging_chats
                    (platform, chat_id, session_id, display_name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(platform, chat_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    updated_at = excluded.updated_at
                """,
                (platform.value, chat_id, session_id, display_name, now, now),
            )
            await connection.commit()
            cursor = await connection.execute(
                """
                SELECT platform, chat_id, session_id, display_name, created_at, updated_at
                FROM messaging_chats WHERE platform = ? AND chat_id = ?
                """,
                (platform.value, chat_id),
            )
            row = await cursor.fetchone()
        assert row is not None  # Just inserted.
        return _row_to_mapping(row)

    async def count_chats_for_platform(self, platform: Platform) -> int:
        """Return how many chats are currently mapped for ``platform``."""

        async with aiosqlite.connect(self._db_path) as connection:
            cursor = await connection.execute(
                "SELECT COUNT(*) FROM messaging_chats WHERE platform = ?",
                (platform.value,),
            )
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    # ---- inbound dedup --------------------------------------------------

    async def claim_inbound(
        self, platform: Platform, native_message_id: str
    ) -> bool:
        """Atomically record that ``native_message_id`` has been seen.

        Returns:
            True when the message was new (caller should process it);
            False when it was already in the table (caller should drop
            it as a redelivery).
        """

        now = _now_iso()
        async with aiosqlite.connect(self._db_path) as connection:
            try:
                await connection.execute(
                    """
                    INSERT INTO messaging_inbound_dedup
                        (platform, native_message_id, seen_at)
                    VALUES (?, ?, ?)
                    """,
                    (platform.value, native_message_id, now),
                )
            except aiosqlite.IntegrityError:
                # Already seen — primary-key conflict.
                return False
            await connection.commit()
        return True

    async def release_inbound(
        self, platform: Platform, native_message_id: str
    ) -> None:
        """Remove a dedup row so platform redeliveries can be processed.

        Called by the router when handling raised an exception after
        the row was claimed (review fix M4). Without this, a single
        transient failure (DB busy, agent build error) would
        permanently lose the message because Meta/LINE redeliveries
        would hit the dedup table and be dropped.
        """

        async with aiosqlite.connect(self._db_path) as connection:
            await connection.execute(
                """
                DELETE FROM messaging_inbound_dedup
                WHERE platform = ? AND native_message_id = ?
                """,
                (platform.value, native_message_id),
            )
            await connection.commit()

    async def prune_inbound_older_than(
        self, hours: int = _DEDUP_KEEP_HOURS
    ) -> int:
        """Delete dedup rows older than ``hours``.

        Returns:
            The number of rows pruned (for logging).
        """

        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        async with aiosqlite.connect(self._db_path) as connection:
            cursor = await connection.execute(
                "DELETE FROM messaging_inbound_dedup WHERE seen_at < ?",
                (cutoff,),
            )
            await connection.commit()
            return cursor.rowcount or 0


def _row_to_mapping(row: aiosqlite.Row) -> ChatMappingRecord:
    return ChatMappingRecord(
        platform=Platform(row["platform"]),
        chat_id=str(row["chat_id"]),
        session_id=str(row["session_id"]),
        display_name=str(row["display_name"] or ""),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ("MessagingStore",)
