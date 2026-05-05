"""Centralized SQLite schema definitions for Feather storage."""

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite


@dataclass(slots=True, frozen=True)
class ColumnSchema:
    """A required table column and its SQLite definition."""

    name: str
    definition: str


@dataclass(slots=True, frozen=True)
class TableSchema:
    """Schema definition for one SQLite table."""

    name: str
    create_sql: str
    required_columns: tuple[ColumnSchema, ...] = ()
    indexes: tuple[str, ...] = ()


SESSIONS_TABLE = TableSchema(
    name="sessions",
    create_sql="""
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        agent_name TEXT NOT NULL,
        status TEXT NOT NULL,
        last_response_id TEXT,
        loaded_skills TEXT NOT NULL,
        active_mcp_servers TEXT NOT NULL DEFAULT '[]',
        pending_inputs TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        restart_requested_at TEXT,
        restart_reason TEXT
    )
    """,
    required_columns=(
        ColumnSchema(name="active_mcp_servers", definition="TEXT NOT NULL DEFAULT '[]'"),
        # Self-repair flag: written by request_restart tool (worker side),
        # polled by the supervisor to trigger a graceful worker restart
        # on the same session_id. NULL when no restart is pending.
        ColumnSchema(name="restart_requested_at", definition="TEXT"),
        ColumnSchema(name="restart_reason", definition="TEXT"),
    ),
)

MESSAGES_TABLE = TableSchema(
    name="messages",
    create_sql="""
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        file_ref TEXT,
        is_compact INTEGER NOT NULL DEFAULT 0,
        sequence INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """,
    required_columns=(
        ColumnSchema(name="file_ref", definition="TEXT"),
        ColumnSchema(name="is_compact", definition="INTEGER NOT NULL DEFAULT 0"),
    ),
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_messages_session_sequence ON messages(session_id, sequence)",
    ),
)

MESSAGE_ATTACHMENTS_TABLE = TableSchema(
    name="message_attachments",
    create_sql="""
    CREATE TABLE IF NOT EXISTS message_attachments (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        message_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        original_name TEXT NOT NULL,
        filepath TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES sessions(id),
        FOREIGN KEY(message_id) REFERENCES messages(id)
    )
    """,
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_message_attachments_message "
        "ON message_attachments(message_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_message_attachments_session "
        "ON message_attachments(session_id, created_at)",
    ),
)

AGENT_MESSAGES_TABLE = TableSchema(
    name="agent_messages",
    create_sql="""
    CREATE TABLE IF NOT EXISTS agent_messages (
        id TEXT PRIMARY KEY,
        from_session_id TEXT NOT NULL,
        from_agent_name TEXT NOT NULL,
        to_session_id TEXT NOT NULL,
        to_agent_name TEXT NOT NULL,
        body TEXT NOT NULL,
        correlation_id TEXT,
        in_reply_to TEXT,
        expects_response INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        delivered_at TEXT,
        responded_at TEXT
    )
    """,
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_agent_messages_inbox "
        "ON agent_messages(to_session_id, to_agent_name, status, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_agent_messages_correlation "
        "ON agent_messages(correlation_id)",
    ),
)

CRON_JOBS_TABLE = TableSchema(
    name="cron_jobs",
    create_sql="""
    CREATE TABLE IF NOT EXISTS cron_jobs (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        agent_key TEXT NOT NULL,
        name TEXT NOT NULL,
        schedule_type TEXT NOT NULL,
        schedule_value TEXT NOT NULL,
        timezone TEXT NOT NULL,
        prompt TEXT NOT NULL,
        status TEXT NOT NULL,
        last_run_at TEXT,
        next_run_at TEXT,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """,
    required_columns=(
        ColumnSchema(name="last_error", definition="TEXT"),
    ),
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_cron_jobs_status_next_run ON cron_jobs(status, next_run_at)",
        "CREATE INDEX IF NOT EXISTS idx_cron_jobs_session_name ON cron_jobs(session_id, name)",
    ),
)

PLANS_TABLE = TableSchema(
    name="plans",
    create_sql="""
    CREATE TABLE IF NOT EXISTS plans (
        id TEXT PRIMARY KEY,
        filepath TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        status TEXT NOT NULL,
        lead_session_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_plans_lead_status ON plans(lead_session_id, status, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_plans_filepath ON plans(filepath)",
    ),
)

TASKS_TABLE = TableSchema(
    name="tasks",
    create_sql="""
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        plan_id TEXT,
        parent_task_id TEXT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        success_criteria TEXT NOT NULL,
        required_outputs TEXT NOT NULL,
        status TEXT NOT NULL,
        responsible_agent_name TEXT,
        responsible_session_id TEXT,
        lead_session_id TEXT NOT NULL,
        blocked_question TEXT,
        blocked_correlation_id TEXT,
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(plan_id) REFERENCES plans(id),
        FOREIGN KEY(parent_task_id) REFERENCES tasks(id)
    )
    """,
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_tasks_lead_status ON tasks(lead_session_id, status, updated_at)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_plan_status ON tasks(plan_id, status, updated_at)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_responsible_session ON tasks(responsible_session_id)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_blocked_correlation ON tasks(blocked_correlation_id)",
    ),
)

TASK_RUNS_TABLE = TableSchema(
    name="task_runs",
    create_sql="""
    CREATE TABLE IF NOT EXISTS task_runs (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        agent_name TEXT NOT NULL,
        pid INTEGER,
        status TEXT NOT NULL,
        exit_code INTEGER,
        envelope_status TEXT,
        error TEXT,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        FOREIGN KEY(task_id) REFERENCES tasks(id)
    )
    """,
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_task_runs_task_started ON task_runs(task_id, started_at)",
        "CREATE INDEX IF NOT EXISTS idx_task_runs_session_status ON task_runs(session_id, status)",
    ),
)

TASK_OUTPUTS_TABLE = TableSchema(
    name="task_outputs",
    create_sql="""
    CREATE TABLE IF NOT EXISTS task_outputs (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        path TEXT,
        content TEXT,
        summary TEXT NOT NULL,
        created_by_session_id TEXT NOT NULL,
        validated INTEGER NOT NULL DEFAULT 0,
        is_final INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY(task_id) REFERENCES tasks(id)
    )
    """,
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_task_outputs_task_created ON task_outputs(task_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_task_outputs_task_final ON task_outputs(task_id, is_final)",
    ),
)

TASK_EVENTS_TABLE = TableSchema(
    name="task_events",
    create_sql="""
    CREATE TABLE IF NOT EXISTS task_events (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        message TEXT NOT NULL,
        agent_name TEXT,
        session_id TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(task_id) REFERENCES tasks(id)
    )
    """,
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_task_events_task_created ON task_events(task_id, created_at)",
    ),
)

MESSAGING_CREDENTIALS_TABLE = TableSchema(
    name="messaging_credentials",
    create_sql="""
    CREATE TABLE IF NOT EXISTS messaging_credentials (
        platform TEXT PRIMARY KEY,
        config_json TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
)

MESSAGING_CHATS_TABLE = TableSchema(
    name="messaging_chats",
    create_sql="""
    CREATE TABLE IF NOT EXISTS messaging_chats (
        platform TEXT NOT NULL,
        chat_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        display_name TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(platform, chat_id),
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """,
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_messaging_chats_session ON messaging_chats(session_id)",
    ),
)

MESSAGING_INBOUND_DEDUP_TABLE = TableSchema(
    name="messaging_inbound_dedup",
    create_sql="""
    CREATE TABLE IF NOT EXISTS messaging_inbound_dedup (
        platform TEXT NOT NULL,
        native_message_id TEXT NOT NULL,
        seen_at TEXT NOT NULL,
        PRIMARY KEY(platform, native_message_id)
    )
    """,
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_messaging_dedup_seen ON messaging_inbound_dedup(seen_at)",
    ),
)

WORKER_HEARTBEATS_TABLE = TableSchema(
    name="worker_heartbeats",
    create_sql="""
    CREATE TABLE IF NOT EXISTS worker_heartbeats (
        session_id TEXT PRIMARY KEY,
        pid INTEGER NOT NULL,
        status TEXT NOT NULL,
        heartbeat_at TEXT NOT NULL
    )
    """,
)

TABLE_SCHEMAS = (
    SESSIONS_TABLE,
    MESSAGES_TABLE,
    MESSAGE_ATTACHMENTS_TABLE,
    AGENT_MESSAGES_TABLE,
    CRON_JOBS_TABLE,
    PLANS_TABLE,
    TASKS_TABLE,
    TASK_RUNS_TABLE,
    TASK_OUTPUTS_TABLE,
    TASK_EVENTS_TABLE,
    MESSAGING_CREDENTIALS_TABLE,
    MESSAGING_CHATS_TABLE,
    MESSAGING_INBOUND_DEDUP_TABLE,
    WORKER_HEARTBEATS_TABLE,
)


async def initialize_database_schema(connection: aiosqlite.Connection) -> None:
    """Create and migrate the SQLite schema."""

    for table in TABLE_SCHEMAS:
        await connection.execute(table.create_sql)
        for column in table.required_columns:
            await ensure_column(connection, table.name, column)
        for index_sql in table.indexes:
            await connection.execute(index_sql)


async def ensure_column(
    connection: aiosqlite.Connection,
    table_name: str,
    column: ColumnSchema,
) -> None:
    """Add a column to a table if it does not already exist."""

    cursor = await connection.execute(f"PRAGMA table_info({table_name})")
    rows = await cursor.fetchall()
    existing = {_pragma_row_name(row) for row in rows}
    if column.name not in existing:
        await connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column.name} {column.definition}"
        )


def _pragma_row_name(row: aiosqlite.Row | tuple) -> str:
    """Return the column name field from a `PRAGMA table_info` row."""

    if isinstance(row, tuple):
        return str(row[1])
    return str(row["name"])
