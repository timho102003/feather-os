"""Core dataclasses and enums for Feather."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from feather.memory.config import MemoryConfig


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


class AgentOutcome(str, Enum):
    """Terminal outcomes for a single agent run."""

    COMPLETED = "completed"
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
class DatabaseConfig:
    """Database configuration."""

    path: str


@dataclass(slots=True)
class StorageConfig:
    """Filesystem storage configuration."""

    temp_directory: str


@dataclass(slots=True)
class LoggingConfig:
    """Logging configuration."""

    path: str
    level: str = "INFO"


@dataclass(slots=True)
class CompactionConfig:
    """Automatic compaction configuration."""

    enabled: bool
    trigger_ratio: float
    context_window_tokens: int
    model: str | None = None
    max_output_tokens: int = 2000
    temperature: float = 0.2


@dataclass(slots=True)
class SkillsConfig:
    """Skill storage configuration."""

    directory: str


@dataclass(slots=True)
class SchedulerConfig:
    """Background scheduler configuration."""

    enabled: bool
    poll_interval_seconds: float
    failure_retry_seconds: float
    max_due_jobs_per_tick: int


@dataclass(slots=True)
class SelfRepairConfig:
    """Self-repair safety net configuration.

    When ``enabled`` is True (set via the onboarding wizard or by
    flipping the YAML manually), the TUI runs the lead agent in a
    separate worker subprocess so it can detect hangs and let the
    agent reload its own patched code. ``FEATHER_USE_LEAD_WORKER=1``
    in the environment overrides this — handy for one-off testing
    without flipping the persistent config.
    """

    enabled: bool = False


@dataclass(slots=True)
class ReasoningConfig:
    """OpenAI reasoning configuration."""

    effort: str | None = None
    summary: str | None = None


@dataclass(slots=True, frozen=True)
class MCPServerConfig:
    """Configuration for one remote Model Context Protocol server."""

    label: str
    server_url: str | None = None
    server_description: str | None = None
    transport: str = "http"
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    allowed_tools: tuple[str, ...] = ()
    require_approval: str | dict[str, Any] | None = "never"
    providers: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    headers: dict[str, str] = field(default_factory=dict)
    header_envs: dict[str, str] = field(default_factory=dict)
    request_timeout_seconds: float = 30.0


@dataclass(slots=True)
class MCPConfig:
    """Top-level remote MCP server registry."""

    enabled: bool = False
    servers: tuple[MCPServerConfig, ...] = ()


@dataclass(slots=True)
class OpenAIConfig:
    """OpenAI provider configuration."""

    api_key_env: str
    model: str
    max_output_tokens: int
    temperature: float
    parallel_tool_calls: bool
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    store: bool = True
    reasoning: ReasoningConfig | None = None
    stream_idle_timeout_seconds: float = 90.0


@dataclass(slots=True)
class ParallelConfig:
    """Parallel AI web-tools configuration."""

    api_key_env: str
    default_search_mode: str = "fast"
    max_results: int = 5
    inline_full_content_threshold: int = 4000


@dataclass(slots=True)
class OpenRouterTracingConfig:
    """OpenRouter trace-broadcast metadata configuration.

    OpenRouter forwards the request body's ``user``, ``session_id``, and
    ``trace`` fields to every observability destination configured on the
    OpenRouter dashboard (Comet Opik, Langfuse, OTel collectors, etc.).
    This config governs *whether* Feather emits those fields and what
    static metadata to fold into ``trace`` alongside the per-turn agent
    identity.

    Attributes:
        enabled: Master switch. Default off so the stateless byte-on-the-
            wire stays identical for users who haven't opted in.
        user: Optional end-user identifier emitted as the OpenRouter
            top-level ``user`` field (≤128 chars per OpenRouter spec).
            Useful when one operator owns multiple Feather installations
            and wants Opik to bucket traces by human.
        metadata: Static key/value pairs merged into the per-turn ``trace``
            object. Passed through to Opik as both trace + span metadata.
            Clamped to OpenRouter's hard limits at translate time
            (16 keys, 64-char keys, 512-char values) so a misconfigured
            value can never trigger a 400 from OpenRouter.
    """

    enabled: bool = False
    user: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class OpenRouterConfig:
    """OpenRouter (Chat Completions) provider configuration.

    Mirrors ``OpenAIConfig`` in spirit — one dataclass that captures the
    defaults for OpenRouter turns — plus OpenRouter-specific knobs
    (``provider_preferences`` for routing, ``fallback_models`` for
    model-level failover, ``cache_strategy`` for Anthropic/Gemini
    prompt-cache breakpoints).
    """

    api_key_env: str = "OPEN_ROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    http_referer: str | None = None
    app_title: str | None = None
    model: str = "anthropic/claude-sonnet-4.6"
    max_output_tokens: int = 32_000
    temperature: float = 1.0
    parallel_tool_calls: bool = True
    reasoning: ReasoningConfig | None = None
    provider_preferences: dict[str, Any] | None = None
    fallback_models: list[str] | None = None
    cache_strategy: str = "anthropic_breakpoint"
    stream_idle_timeout_seconds: float = 90.0
    request_timeout_seconds: float = 120.0
    max_attempts: int = 3
    supports_multimodal: bool = True
    # Wall-clock cap on a single streamed response. Caps a misbehaving
    # upstream that drips keep-alive bytes inside the idle window forever.
    # 600s is generous — long-thinking models (gpt-5/o3) should still
    # finish well under this; reduce for interactive UX, raise for
    # batch-style workloads.
    max_stream_wall_seconds: float = 600.0
    tracing: OpenRouterTracingConfig | None = None


@dataclass(slots=True)
class ClaudeThinkingConfig:
    """Anthropic extended-thinking configuration.

    Mirrors the ``thinking`` field on the Messages API request body. ``type``
    is ``"enabled"`` (explicit budget), ``"adaptive"`` (model picks its own
    budget — required on Opus 4.7+), or ``"disabled"``. ``budget_tokens`` is
    consulted only when ``type == "enabled"`` and must be smaller than
    ``max_output_tokens`` unless the ``interleaved-thinking-2025-05-14``
    beta header is set.

    When extended thinking is on, Anthropic ignores ``temperature``,
    ``top_p``, and ``top_k`` for the thinking phase, and forbids
    ``tool_choice`` values other than ``auto`` or ``none``. The provider
    enforces these constraints at translate time.
    """

    type: str = "enabled"
    budget_tokens: int | None = None


@dataclass(slots=True)
class ClaudeConfig:
    """Anthropic Claude (Messages API) provider configuration.

    Mirrors :class:`OpenRouterConfig` in spirit — one dataclass captures
    the defaults for Claude turns — plus Claude-specific knobs:
    ``anthropic_version`` (pinned API version date), ``anthropic_beta``
    (extra beta feature flags broadcast as the ``anthropic-beta`` header),
    ``thinking`` (extended-thinking), and ``cache_strategy``
    (prompt-cache breakpoint placement).

    The Messages API is stateless, so the provider replays full history
    on every turn — same posture as :class:`OpenRouterConfig`.
    """

    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str = "https://api.anthropic.com"
    anthropic_version: str = "2023-06-01"
    anthropic_beta: tuple[str, ...] = ()
    model: str = "claude-opus-4-7"
    max_output_tokens: int = 32_000
    temperature: float = 1.0
    parallel_tool_calls: bool = True
    thinking: ClaudeThinkingConfig | None = None
    cache_strategy: str = "anthropic_breakpoint"
    stream_idle_timeout_seconds: float = 90.0
    request_timeout_seconds: float = 120.0
    max_attempts: int = 3
    supports_multimodal: bool = True
    max_stream_wall_seconds: float = 600.0


@dataclass(slots=True)
class AppConfig:
    """Application-wide configuration."""

    database: DatabaseConfig
    storage: StorageConfig
    logging: LoggingConfig
    compaction: CompactionConfig
    skills: SkillsConfig
    scheduler: SchedulerConfig
    openai: OpenAIConfig
    parallel: ParallelConfig | None = None
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    active_provider: str = "openai"
    openrouter: OpenRouterConfig | None = None
    claude: ClaudeConfig | None = None
    self_repair: SelfRepairConfig = field(default_factory=SelfRepairConfig)
    # Name of the lead agent the TUI/CLI bootstraps by default. Other leads
    # (additional ``role: lead`` YAMLs) are switchable in the multi-lead UI.
    default_lead: str = "lead"


@dataclass(slots=True)
class AgentConfig:
    """Config for one agent definition.

    ``provider`` and ``model`` are optional per-agent overrides. ``None``
    means inherit from the app-level active provider and its default
    model. The agent factory resolves these at build time so individual
    agents can target different LLM providers without touching app.yaml.

    ``temperature`` and ``max_output_tokens`` are optional per-agent
    overrides too: when set they flow into ``ProviderRequestConfig`` and
    override the provider's app-level defaults for this agent only.
    ``None`` means inherit. Useful when a single app provider serves
    multiple agents at different sampling profiles (e.g. an exploratory
    agent at temperature 0.8 alongside a deterministic validator at 0.0).

    ``reasoning`` is an optional per-agent override for thinking effort
    and summary verbosity. ``None`` means inherit the provider-level
    default. When set, the agent forwards this on every provider request
    via ``ProviderRequestConfig.reasoning``, so different agents can run
    at different reasoning depths against the same model.
    """

    name: str
    role: str
    personality: str
    prompt_modules: list[str]
    registered_tools: list[str]
    memory_enabled: bool = False
    description: str = ""
    inline_prompt: str = ""
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning: ReasoningConfig | None = None
    mcp_servers: tuple[MCPServerConfig, ...] = ()
    # Lead identity ("soul") + display metadata. All optional so every existing
    # agent YAML keeps loading unchanged. ``soul`` is longer-form persona prose
    # injected into the prompt when present (distinct from the one-line
    # ``personality``); ``color``/``emoji`` are TUI display hints. ``capabilities``
    # carries per-field overrides consumed by ``CapabilityProfile.from_config``.
    soul: str = ""
    color: str | None = None
    emoji: str | None = None
    capabilities: dict[str, bool] = field(default_factory=dict)


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


@dataclass(slots=True, frozen=True)
class TraceContext:
    """Per-turn observability identity threaded through the provider call.

    Carries the small, stable identity bundle (session, agent name, agent
    role) that observability backends like Comet Opik need to group traces
    cleanly. Providers that support trace metadata broadcast (currently
    OpenRouter) read it; providers that don't (OpenAI Responses API)
    ignore it. Frozen so it's safe to share across concurrent turns.
    """

    session_id: str
    agent_name: str
    agent_role: str | None = None


@dataclass(slots=True)
class ProviderRequestConfig:
    """Optional per-request LLM overrides.

    ``response_schema`` + ``response_schema_name`` drive strict-JSON-schema
    structured output on providers that support it (OpenAI Responses API
    translates to ``text.format`` with ``strict=True``). The schema is a
    Pydantic ``BaseModel`` class; the provider converts ``model_json_schema()``
    into the wire format and defensively enforces ``additionalProperties:false``
    on every nested object, which OpenAI strict mode requires.

    ``trace_context`` carries the per-turn identity (session, agent) used
    by providers that broadcast trace metadata (OpenRouter → Opik etc.).
    Always safe to set; providers that don't consume it ignore it.
    """

    model: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    reasoning: ReasoningConfig | None = None
    response_schema: type[Any] | None = None
    response_schema_name: str | None = None
    mcp_servers: tuple[MCPServerConfig, ...] = ()
    trace_context: TraceContext | None = None


@dataclass(slots=True)
class ToolCall:
    """A parsed tool call emitted by a provider."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolExecutionResult:
    """Outcome of one local tool execution."""

    output: str
    await_user_question: str | None = None
    loaded_skill_name: str | None = None


@dataclass(slots=True, frozen=True)
class ToolExecutionContext:
    """Runtime context passed to one tool invocation.

    ``is_lead`` mirrors the calling agent's ``CapabilityProfile.is_lead`` so
    tools can scope behavior to a lead vs a sub-agent without hard-coding the
    literal agent name ``"lead"`` (which breaks once leads carry custom names
    like ``Tim``/``Sophia``).
    """

    session_id: str
    agent_name: str
    is_lead: bool = False


@dataclass(slots=True)
class ToolOutputArtifact:
    """A persisted tool-output artifact written to temporary storage."""

    tool_name: str
    file_ref: str
    text: str
    reference_text: str


@dataclass(slots=True)
class ModelTurn:
    """Normalized model response for one provider turn."""

    response_id: str | None
    output_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] | None = None


@dataclass(slots=True)
class RuntimeEvent:
    """A small runtime event used to drive the CLI display."""

    kind: str
    text: str | None = None
    tool_name: str | None = None
    payload: dict[str, Any] | None = None


@dataclass(slots=True)
class AgentRunResult:
    """Outcome returned to the CLI after one run cycle.

    ``total_tool_calls`` counts tool executions across every loop
    iteration in this run. Used by the sub-agent subprocess entry point
    to flag wasted spawns — a research/explore/validate sub-agent that
    exits with zero tool calls produced only an acknowledgement, not
    the work it was dispatched to do.
    """

    status: AgentOutcome
    session_id: str
    assistant_text: str
    question: str | None = None
    total_tool_calls: int = 0


EventHandler = Callable[[RuntimeEvent], None]
