"""Tests for centralized SQLite schema helpers."""

from pathlib import Path

import aiosqlite

from feather.storage.schema import ColumnSchema, ensure_column, initialize_database_schema


async def test_ensure_column_supports_default_sqlite_rows(tmp_path: Path) -> None:
    """Schema helpers should work even when `row_factory` is not configured."""

    connection = await aiosqlite.connect(tmp_path / "schema.db")
    try:
        await connection.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await ensure_column(connection, "messages", ColumnSchema(name="file_ref", definition="TEXT"))
        cursor = await connection.execute("PRAGMA table_info(messages)")
        rows = await cursor.fetchall()
        columns = {row[1] for row in rows}
        assert "file_ref" in columns
    finally:
        await connection.close()


async def test_initialize_database_schema_adds_all_required_message_columns(tmp_path: Path) -> None:
    """Schema initialization should backfill message migrations and create cron-job tables."""

    connection = await aiosqlite.connect(tmp_path / "schema.db")
    try:
        await connection.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                status TEXT NOT NULL,
                last_response_id TEXT,
                loaded_skills TEXT NOT NULL,
                pending_inputs TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        await connection.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await initialize_database_schema(connection)
        cursor = await connection.execute("PRAGMA table_info(messages)")
        rows = await cursor.fetchall()
        columns = {row[1] for row in rows}
        assert "file_ref" in columns
        assert "is_compact" in columns

        cron_cursor = await connection.execute("PRAGMA table_info(cron_jobs)")
        cron_rows = await cron_cursor.fetchall()
        cron_columns = {row[1] for row in cron_rows}
        assert "next_run_at" in cron_columns
        assert "last_error" in cron_columns

        attachment_cursor = await connection.execute(
            "PRAGMA table_info(message_attachments)"
        )
        attachment_rows = await attachment_cursor.fetchall()
        attachment_columns = {row[1] for row in attachment_rows}
        assert {
            "id",
            "session_id",
            "message_id",
            "kind",
            "mime_type",
            "original_name",
            "filepath",
            "size_bytes",
        }.issubset(attachment_columns)
    finally:
        await connection.close()
