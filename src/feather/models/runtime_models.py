"""Runtime/transport dataclasses: tool calls, model turns, and events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from feather.models.config_models import MCPServerConfig, ReasoningConfig


class AgentOutcome(str, Enum):
    """Terminal outcomes for a single agent run."""

    COMPLETED = "completed"
    AWAITING_USER = "awaiting_user"


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


__all__ = (
    "AgentOutcome",
    "AgentRunResult",
    "EventHandler",
    "ModelTurn",
    "ProviderRequestConfig",
    "RuntimeEvent",
    "ToolCall",
    "ToolExecutionContext",
    "ToolExecutionResult",
    "ToolOutputArtifact",
    "TraceContext",
)
