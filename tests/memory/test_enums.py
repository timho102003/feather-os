"""Tests for memory-subsystem enums."""

from __future__ import annotations

import pytest

from feather.memory.enums import EmbedType, EmbeddingTaskType, MemoryOp, MemoryOwner


def test_embed_type_values_match_spec() -> None:
    """EmbedType string values are the tenancy keys used in Qdrant payload."""
    assert EmbedType.MEMORY.value == "memory"
    assert EmbedType.ATTACHMENT_TEXT.value == "attachment:text"
    assert EmbedType.ATTACHMENT_IMAGE.value == "attachment:image"
    assert EmbedType.ATTACHMENT_PDF.value == "attachment:pdf"


def test_embed_type_is_str_enum_and_round_trips() -> None:
    """EmbedType must be a str-enum so it survives JSON round-trips unchanged."""
    assert isinstance(EmbedType.MEMORY, str)
    assert EmbedType("memory") is EmbedType.MEMORY
    assert str(EmbedType.MEMORY.value) == "memory"


def test_memory_owner_values_match_spec() -> None:
    """MemoryOwner covers user (produced today) and lead (reserved)."""
    assert MemoryOwner.USER.value == "user"
    assert MemoryOwner.LEAD.value == "lead"


def test_memory_owner_is_str_enum() -> None:
    assert isinstance(MemoryOwner.USER, str)
    assert MemoryOwner("user") is MemoryOwner.USER


def test_memory_op_values_match_spec() -> None:
    """MemoryOp is uppercase because it's produced by the classification LLM via Literal[...]."""
    assert MemoryOp.CREATE.value == "CREATE"
    assert MemoryOp.UPDATE.value == "UPDATE"
    assert MemoryOp.DELETE.value == "DELETE"
    assert MemoryOp.NO_OP.value == "NO_OP"


def test_memory_op_is_str_enum() -> None:
    assert isinstance(MemoryOp.CREATE, str)
    assert MemoryOp("NO_OP") is MemoryOp.NO_OP


def test_embedding_task_type_values_match_gemini_skill() -> None:
    """Task types must match the Gemini embeddings API contract."""
    assert EmbeddingTaskType.RETRIEVAL_DOCUMENT.value == "RETRIEVAL_DOCUMENT"
    assert EmbeddingTaskType.RETRIEVAL_QUERY.value == "RETRIEVAL_QUERY"


def test_unknown_value_raises() -> None:
    """Constructors should reject unknown values so typos fail loudly."""
    with pytest.raises(ValueError):
        MemoryOp("create")  # lowercase — not allowed
    with pytest.raises(ValueError):
        MemoryOwner("admin")
    with pytest.raises(ValueError):
        EmbedType("memory:attachment")  # wrong shape
