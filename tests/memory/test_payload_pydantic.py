"""Tests for MemoryPointPayload — the Qdrant payload Pydantic model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from feather.memory.enums import EmbedType, MemoryOwner
from feather.memory.models import MemoryPointPayload


def _minimal_kwargs(**overrides: object) -> dict[str, object]:
    """Return a valid payload kwargs dict with one row's worth of fields."""
    base: dict[str, object] = dict(
        type=EmbedType.MEMORY,
        memory_owner=MemoryOwner.USER,
        content="The user prefers Python for async code.",
        purpose="tailor library suggestions to async Python",
        group_id=uuid4(),
        session_id=uuid4(),
        start_message_id=uuid4(),
        end_message_id=uuid4(),
    )
    base.update(overrides)
    return base


def test_minimal_required_fields_produce_valid_payload() -> None:
    """All required fields supplied → model instantiates, defaults populated."""
    p = MemoryPointPayload(**_minimal_kwargs())

    assert p.type == EmbedType.MEMORY.value
    assert p.memory_owner == MemoryOwner.USER.value
    assert p.chunk_index == 0
    assert p.previous_chunk_id is None
    assert p.next_chunk_id is None
    assert p.filepath is None
    assert isinstance(p.id, UUID)
    assert isinstance(p.created_at, datetime)
    assert isinstance(p.updated_at, datetime)


def test_missing_required_field_raises_validation_error() -> None:
    """Dropping a required field must raise ValidationError, not silently default."""
    kwargs = _minimal_kwargs()
    del kwargs["content"]
    with pytest.raises(ValidationError):
        MemoryPointPayload(**kwargs)


def test_extra_fields_are_forbidden() -> None:
    """extra='forbid' must reject unknown keys to prevent schema drift."""
    with pytest.raises(ValidationError):
        MemoryPointPayload(**_minimal_kwargs(), not_a_real_field="oops")


def test_empty_content_is_rejected() -> None:
    """Content must be non-empty after strip — the Gemini API rejects empty strings."""
    with pytest.raises(ValidationError):
        MemoryPointPayload(**_minimal_kwargs(content=""))
    with pytest.raises(ValidationError):
        MemoryPointPayload(**_minimal_kwargs(content="   \n\t  "))


def test_content_is_stripped() -> None:
    """Content is stripped at validation time so downstream embedders see canonical text."""
    p = MemoryPointPayload(**_minimal_kwargs(content="   hello   "))
    assert p.content == "hello"


def test_purpose_cannot_be_empty() -> None:
    """Purpose is user-facing in the prompt block; empty purposes leak noise."""
    with pytest.raises(ValidationError):
        MemoryPointPayload(**_minimal_kwargs(purpose=""))


def test_chunk_index_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        MemoryPointPayload(**_minimal_kwargs(chunk_index=-1))


def test_enum_values_serialized_as_strings() -> None:
    """use_enum_values=True so payload.model_dump stores raw strings, which is what Qdrant sees."""
    p = MemoryPointPayload(**_minimal_kwargs())
    dumped = p.model_dump(mode="json")
    assert dumped["type"] == "memory"
    assert dumped["memory_owner"] == "user"


def test_model_dump_json_is_valid_json_with_uuid_and_datetime_strings() -> None:
    """model_dump(mode='json') must produce a dict we can json.dumps without custom encoders."""
    p = MemoryPointPayload(**_minimal_kwargs())
    dumped = p.model_dump(mode="json")
    raw = json.dumps(dumped)
    reloaded = json.loads(raw)
    assert UUID(reloaded["id"])
    assert datetime.fromisoformat(reloaded["created_at"])


def test_uuid_fields_accept_uuid_instances_and_strings() -> None:
    """Qdrant round-trip: when we read back a scrolled payload, UUID fields come as strings."""
    gid = uuid4()
    p1 = MemoryPointPayload(**_minimal_kwargs(group_id=gid))
    p2 = MemoryPointPayload(**_minimal_kwargs(group_id=str(gid)))
    assert p1.group_id == p2.group_id == gid


def test_default_timestamps_are_utc() -> None:
    """created_at / updated_at must be tz-aware UTC so ordering in Qdrant is consistent."""
    p = MemoryPointPayload(**_minimal_kwargs())
    assert p.created_at.tzinfo is not None
    assert p.created_at.utcoffset() == timezone.utc.utcoffset(None)
    assert p.updated_at.tzinfo is not None


def test_previous_and_next_chunk_ids_are_optional_uuids() -> None:
    """Chunk linkage forms a doubly-linked list within a group_id."""
    prev_id = uuid4()
    nxt_id = uuid4()
    p = MemoryPointPayload(**_minimal_kwargs(previous_chunk_id=prev_id, next_chunk_id=nxt_id))
    assert p.previous_chunk_id == prev_id
    assert p.next_chunk_id == nxt_id


def test_validate_assignment_enforces_types_on_mutation() -> None:
    """Payloads are mutated between build + upsert; bad assignment must fail loudly."""
    p = MemoryPointPayload(**_minimal_kwargs())
    with pytest.raises(ValidationError):
        p.chunk_index = "not an int"  # type: ignore[assignment]


def test_round_trip_through_qdrant_shape() -> None:
    """Simulate reading a payload back from Qdrant: dump + reparse → equal."""
    original = MemoryPointPayload(**_minimal_kwargs())
    reparsed = MemoryPointPayload.model_validate(original.model_dump(mode="json"))
    assert reparsed.id == original.id
    assert reparsed.group_id == original.group_id
    assert reparsed.content == original.content
    assert reparsed.type == original.type
