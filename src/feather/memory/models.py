"""Data model for the memory subsystem.

Contains three groups of types:

- **Qdrant payload** — ``MemoryPointPayload`` (Pydantic, strict). One instance
  per chunk; enforces the wire-level schema on every upsert and parse.
- **Structured-output schemas** — Pydantic response models (``ExtractionResponse``,
  ``ClassificationResponse``, ``QueryBuildResponse``) supplied to the
  provider's strict-JSON-schema mode. Their JSON schemas are what OpenAI
  validates against; ``extra='forbid'`` ensures the model cannot invent fields.
- **Internal dataclasses** — cheap, slots-based transport types used between
  the service layers. No Pydantic overhead.

All public types are deliberately small; they exist so every boundary has a
typed object crossing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from feather.memory.enums import EmbedType, MemoryOp, MemoryOwner

if TYPE_CHECKING:
    from feather.models import SessionMessage


# -----------------------------------------------------------------------------
# Qdrant payload
# -----------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return a timezone-aware UTC ``datetime``."""
    return datetime.now(timezone.utc)


class MemoryPointPayload(BaseModel):
    """Qdrant point payload. One row represents one chunk of one atomic memory.

    All chunks produced by a single atomic-memory extraction share ``group_id``
    and are linked through ``previous_chunk_id`` / ``next_chunk_id``. The
    tenancy key is ``type``; payload indexes on this and other filter keys are
    created at collection-startup time.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid",
    )

    # Identity
    id: UUID = Field(default_factory=uuid4, description="Qdrant point id; also this chunk's id.")
    type: EmbedType = Field(..., description="Tenancy partition key.")
    memory_owner: MemoryOwner = Field(..., description="Semantic owner; only USER populated today.")

    # Content
    content: str = Field(..., description="The embedded text (one chunk).")
    purpose: str = Field(..., description="How a future agent could use this memory.")
    filepath: str | None = Field(None, description="Source file for attachment:* types.")

    # Chunk linkage within a group
    group_id: UUID = Field(..., description="Shared by all chunks from one atomic memory.")
    previous_chunk_id: UUID | None = None
    next_chunk_id: UUID | None = None
    chunk_index: int = Field(default=0, ge=0, description="0-indexed chunk position within group.")

    # Provenance
    session_id: UUID = Field(..., description="Session that produced the memory.")
    start_message_id: UUID = Field(..., description="First message.id of the extraction window.")
    end_message_id: UUID = Field(..., description="Last message.id of the extraction window.")

    # Timestamps (UTC)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("content", "purpose")
    @classmethod
    def _strip_and_require_nonempty(cls, v: str) -> str:
        """Strip surrounding whitespace and reject empty strings."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("must be non-empty after strip")
        return stripped


# -----------------------------------------------------------------------------
# Structured-output response schemas (Pydantic, strict JSON mode)
# -----------------------------------------------------------------------------


class AtomicMemoryOut(BaseModel):
    """One extracted atomic memory as emitted by the extraction LLM."""

    model_config = ConfigDict(extra="forbid")

    who: str = Field(..., description="Subject of the memory.")
    what: str = Field(..., description="The fact in one declarative sentence.")
    when: str = Field(..., description="Temporal anchor, or 'unspecified'.")
    where: str = Field(..., description="Context/location, or 'unspecified'.")
    why: str = Field(..., description="Motivation or rationale.")
    how: str = Field(..., description="Mechanism/means/method.")
    purpose: str = Field(..., description="How a future agent would use this memory.")
    content: str = Field(..., min_length=1, description="Self-contained canonical text to embed.")


class ExtractionResponse(BaseModel):
    """Top-level response from the extraction LLM call."""

    model_config = ConfigDict(extra="forbid")
    memories: list[AtomicMemoryOut] = Field(
        default_factory=list,
        description="Possibly empty — an uneventful window yields no memories.",
    )


class ClassificationResponse(BaseModel):
    """Top-level response from the CRUD classification LLM call."""

    model_config = ConfigDict(extra="forbid")
    op: Literal["CREATE", "UPDATE", "DELETE", "NO_OP"] = Field(
        ..., description="The CRUD operation to apply."
    )
    target_group_id: str | None = Field(
        default=None,
        description="REQUIRED for UPDATE and DELETE; must equal one candidate group_id.",
    )
    reasoning: str = Field(..., description="One-sentence justification.")


class QueryBuildResponse(BaseModel):
    """Top-level response from the memory query-builder LLM call."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(
        ...,
        description=(
            "A self-contained natural-language query about the user. Empty string "
            "is allowed and pairs with should_skip=True."
        ),
    )
    should_skip: bool = Field(
        default=False,
        description="True if the recent conversation has no memory relevance.",
    )
    reasoning: str = Field(..., description="One-sentence justification.")


# -----------------------------------------------------------------------------
# Internal dataclasses (cheap transport types)
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class AtomicMemory:
    """Pre-chunking representation of one extracted memory."""

    who: str
    what: str
    when: str
    where: str
    why: str
    how: str
    purpose: str
    content: str


@dataclass(slots=True)
class MemoryWindow:
    """A 10-turn extraction window, compact rows excluded."""

    session_id: str
    start_message_id: str
    end_message_id: str
    messages: list["SessionMessage"]


@dataclass(slots=True)
class MemorySearchResult:
    """A single hit from the vector store."""

    payload: MemoryPointPayload
    score: float


@dataclass(slots=True)
class ClassifiedOp:
    """Output of the CRUD classifier for one atomic memory."""

    op: MemoryOp
    target_group_id: str | None
    candidates: list[MemorySearchResult]


@dataclass(slots=True)
class AppliedOp:
    """The result of applying one classified op to Qdrant."""

    op: MemoryOp
    group_id: UUID | None
    chunk_count: int
    error: str | None = None

    @classmethod
    def failed(cls, op: MemoryOp, error: str) -> "AppliedOp":
        """Construct an AppliedOp representing a per-atom failure."""
        return cls(op=op, group_id=None, chunk_count=0, error=error)


@dataclass(slots=True, frozen=True)
class QueryDecision:
    """Output of ``MemoryQueryBuilder.build`` consumed by ``MemoryReader``."""

    query: str
    should_skip: bool
    reasoning: str


@dataclass(slots=True)
class MemoryExtractionReport:
    """Structured summary of one ``extract_and_store`` invocation."""

    session_id: str
    correlation_id: str | None
    window: MemoryWindow | None
    applied_ops: list[AppliedOp] = field(default_factory=list)
    reason: str | None = None

    @classmethod
    def empty(
        cls,
        session_id: str,
        *,
        reason: str,
        correlation_id: str | None = None,
    ) -> "MemoryExtractionReport":
        """Construct a 'nothing happened' report with a structured ``reason``."""
        return cls(
            session_id=session_id,
            correlation_id=correlation_id,
            window=None,
            applied_ops=[],
            reason=reason,
        )

    def to_log_fields(self) -> dict[str, object]:
        """Flatten into a dict suitable for structured logging."""
        counts: dict[str, int] = {}
        errors = 0
        for applied in self.applied_ops:
            counts[applied.op.value] = counts.get(applied.op.value, 0) + 1
            if applied.error is not None:
                errors += 1
        return {
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "ops": counts,
            "errors": errors,
            "reason": self.reason,
        }
