"""Persisted record dataclasses + their lifecycle/status enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Supported session message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AttachmentKind(str, Enum):
    """Supported message attachment classes."""

    IMAGE = "image"
    FILE = "file"


class SessionStatus(str, Enum):
    """Supported lifecycle states for a session."""

    ACTIVE = "active"
    AWAITING_USER = "awaiting_user"


class CronScheduleType(str, Enum):
    """Supported schedule encodings for cron jobs."""

    CRON = "cron"
    ONCE = "once"


class CronJobStatus(str, Enum):
    """Supported lifecycle states for cron jobs."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class AgentMessageStatus(str, Enum):
    """Lifecycle states for inter-agent messages."""

    PENDING = "pending"
    DELIVERED = "delivered"
    RESPONDED = "responded"
    EXPIRED = "expired"


class PlanStatus(str, Enum):
    """Lifecycle states for durable plans."""

    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Lifecycle states for durable agent tasks."""

    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED_NEEDS_INPUT = "blocked_needs_input"
    COMPLETED_WITH_REPORT = "completed_with_report"
    COMPLETED_WITH_ARTIFACTS = "completed_with_artifacts"
    COMPLETED_WITHOUT_ARTIFACTS = "completed_without_artifacts"
    FAILED = "failed"
    STOPPED = "stopped"


class TaskRunStatus(str, Enum):
    """Lifecycle states for one subprocess attempt on a task."""

    RUNNING = "running"
    EXITED = "exited"
    CRASHED = "crashed"
    KILLED = "killed"


class WorkerStatus(str, Enum):
    """Self-reported lifecycle states for a lead worker subprocess.

    The worker writes its current state to ``worker_heartbeats`` so the
    supervisor (TUI) can distinguish a clean stop from a hang. ``CRASHED``
    is never set by the worker itself — it is inferred by the supervisor
    when a heartbeat goes stale beyond the staleness threshold.
    """

    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


class TaskOutputKind(str, Enum):
    """Supported task output classes."""

    REPORT = "report"
    ARTIFACT = "artifact"
    SOURCE = "source"
    LOG = "log"


@dataclass(slots=True)
class SkillMetadata:
    """Metadata extracted from a skill file."""

    name: str
    description: str
    path: str
    refs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LoadedSkill:
    """Fully loaded skill content."""

    metadata: SkillMetadata
    content: str


@dataclass(slots=True)
class SessionRecord:
    """Current persisted state for one session."""

    id: str
    agent_name: str
    status: SessionStatus
    last_response_id: str | None
    loaded_skills: list[str]
    active_mcp_servers: list[str]
    pending_inputs: list[dict[str, Any]]
    created_at: str
    updated_at: str


@dataclass(slots=True)
class SessionMessage:
    """One message stored in a local session."""

    id: str
    session_id: str
    role: MessageRole
    content: str
    file_ref: str | None
    is_compact: bool
    sequence: int
    created_at: str


@dataclass(slots=True)
class AttachmentRecord:
    """One file saved for a chat message."""

    id: str
    session_id: str
    message_id: str
    kind: AttachmentKind
    mime_type: str
    original_name: str
    filepath: str
    size_bytes: int
    created_at: str


@dataclass(slots=True, frozen=True)
class PendingAttachment:
    """A local file path detected from user input before it is persisted."""

    source_path: str
    kind: AttachmentKind
    mime_type: str
    original_name: str
    size_bytes: int


@dataclass(slots=True)
class AgentMessage:
    """One inter-agent message stored on the SQLite bus."""

    id: str
    from_session_id: str
    from_agent_name: str
    to_session_id: str
    to_agent_name: str
    body: str
    correlation_id: str | None
    in_reply_to: str | None
    expects_response: bool
    status: AgentMessageStatus
    created_at: str
    delivered_at: str | None
    responded_at: str | None


@dataclass(slots=True)
class CronJobRecord:
    """One persisted scheduled job."""

    id: str
    session_id: str
    agent_key: str
    name: str
    schedule_type: CronScheduleType
    schedule_value: str
    timezone: str
    prompt: str
    status: CronJobStatus
    last_run_at: str | None
    next_run_at: str | None
    last_error: str | None
    created_at: str
    updated_at: str

    def next_run_datetime(self) -> datetime | None:
        """Return the next run timestamp as a parsed datetime when present."""

        if self.next_run_at is None:
            return None
        return datetime.fromisoformat(self.next_run_at)


@dataclass(slots=True)
class PlanRecord:
    """One durable plan tracked by Feather."""

    id: str
    filepath: str
    title: str
    summary: str
    status: PlanStatus
    lead_session_id: str
    created_at: str
    updated_at: str


@dataclass(slots=True)
class TaskRecord:
    """One durable task tracked across agent processes."""

    id: str
    plan_id: str | None
    parent_task_id: str | None
    title: str
    description: str
    success_criteria: str
    required_outputs: list[str]
    status: TaskStatus
    responsible_agent_name: str | None
    responsible_session_id: str | None
    lead_session_id: str
    blocked_question: str | None
    blocked_correlation_id: str | None
    error: str | None
    created_at: str
    updated_at: str


@dataclass(slots=True)
class TaskRunRecord:
    """One process attempt for a task."""

    id: str
    task_id: str
    session_id: str
    agent_name: str
    pid: int | None
    status: TaskRunStatus
    exit_code: int | None
    envelope_status: str | None
    error: str | None
    started_at: str
    ended_at: str | None


@dataclass(slots=True)
class TaskOutputRecord:
    """One persisted report, artifact, source, or log for a task."""

    id: str
    task_id: str
    kind: TaskOutputKind
    path: str | None
    content: str | None
    summary: str
    created_by_session_id: str
    validated: bool
    is_final: bool
    created_at: str


@dataclass(slots=True)
class TaskEventRecord:
    """One append-only task event for monitoring and audit."""

    id: str
    task_id: str
    event_type: str
    message: str
    agent_name: str | None
    session_id: str | None
    created_at: str


@dataclass(slots=True, frozen=True)
class WorkerHeartbeat:
    """One row of ``worker_heartbeats`` — a worker's last self-reported tick."""

    session_id: str
    pid: int
    status: WorkerStatus
    heartbeat_at: datetime


__all__ = (
    "AgentMessage",
    "AgentMessageStatus",
    "AttachmentKind",
    "AttachmentRecord",
    "CronJobRecord",
    "CronJobStatus",
    "CronScheduleType",
    "LoadedSkill",
    "MessageRole",
    "PendingAttachment",
    "PlanRecord",
    "PlanStatus",
    "SessionMessage",
    "SessionRecord",
    "SessionStatus",
    "SkillMetadata",
    "TaskEventRecord",
    "TaskOutputKind",
    "TaskOutputRecord",
    "TaskRecord",
    "TaskRunRecord",
    "TaskRunStatus",
    "TaskStatus",
    "WorkerHeartbeat",
    "WorkerStatus",
)
