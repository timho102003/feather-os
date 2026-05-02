"""Tests for structured-output Pydantic response models and internal dataclasses."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from feather.memory.enums import MemoryOp
from feather.memory.models import (
    AppliedOp,
    AtomicMemoryOut,
    ClassificationResponse,
    ExtractionResponse,
    MemoryExtractionReport,
    QueryBuildResponse,
    QueryDecision,
)


# ExtractionResponse ----------------------------------------------------------


def test_extraction_response_accepts_empty_memories() -> None:
    """An uneventful window → zero memories; that must parse cleanly."""
    parsed = ExtractionResponse.model_validate_json('{"memories": []}')
    assert parsed.memories == []


def test_extraction_response_omitting_memories_uses_default_empty_list() -> None:
    """Defaults allow {} to parse; strict schema never emits this, but model is lenient."""
    parsed = ExtractionResponse.model_validate_json("{}")
    assert parsed.memories == []


def test_extraction_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ExtractionResponse.model_validate_json('{"memories": [], "extra": 1}')


def test_atomic_memory_out_requires_all_fields_and_non_empty_content() -> None:
    """Every 5W1H field is required; content cannot be empty."""
    base = dict(who="the user", what="prefers Python", when="ongoing", where="unspecified",
                why="productivity", how="async-first", purpose="suggest Python libs",
                content="the user prefers Python")
    # sanity: valid
    AtomicMemoryOut.model_validate(base)
    # missing a field raises
    for key in list(base):
        dropped = {k: v for k, v in base.items() if k != key}
        with pytest.raises(ValidationError):
            AtomicMemoryOut.model_validate(dropped)
    # empty content raises
    with pytest.raises(ValidationError):
        AtomicMemoryOut.model_validate({**base, "content": ""})


# ClassificationResponse ------------------------------------------------------


def test_classification_response_accepts_all_valid_ops() -> None:
    for op in ("CREATE", "UPDATE", "DELETE", "NO_OP"):
        parsed = ClassificationResponse.model_validate(
            {"op": op, "target_group_id": None, "reasoning": "r"}
        )
        assert parsed.op == op


def test_classification_response_rejects_unknown_op_at_schema_layer() -> None:
    """The Literal[...] enforces casing so strict mode refuses any drift."""
    with pytest.raises(ValidationError):
        ClassificationResponse.model_validate(
            {"op": "create", "target_group_id": None, "reasoning": "r"}
        )


def test_classification_response_target_group_id_is_optional() -> None:
    """target_group_id is only required at the semantic (classifier.py) layer, not here."""
    parsed = ClassificationResponse.model_validate({"op": "NO_OP", "reasoning": "r"})
    assert parsed.target_group_id is None


def test_classification_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ClassificationResponse.model_validate(
            {"op": "NO_OP", "reasoning": "r", "sneaky": 1}
        )


# QueryBuildResponse ----------------------------------------------------------


def test_query_build_response_empty_query_is_valid_when_should_skip_true() -> None:
    parsed = QueryBuildResponse.model_validate(
        {"query": "", "should_skip": True, "reasoning": "greeting"}
    )
    assert parsed.query == ""
    assert parsed.should_skip is True


def test_query_build_response_should_skip_defaults_false() -> None:
    parsed = QueryBuildResponse.model_validate(
        {"query": "the user's python preferences", "reasoning": "rewrite"}
    )
    assert parsed.should_skip is False


def test_query_build_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        QueryBuildResponse.model_validate(
            {"query": "q", "reasoning": "r", "another": "oops"}
        )


# Dataclass sanity ------------------------------------------------------------


def test_query_decision_is_frozen_and_slotted() -> None:
    d = QueryDecision(query="q", should_skip=False, reasoning="r")
    with pytest.raises(Exception):  # FrozenInstanceError subclass of AttributeError
        d.query = "other"  # type: ignore[misc]


def test_applied_op_failed_factory_marks_error_and_zero_chunks() -> None:
    failed = AppliedOp.failed(MemoryOp.CREATE, "embed service timed out")
    assert failed.op is MemoryOp.CREATE
    assert failed.group_id is None
    assert failed.chunk_count == 0
    assert failed.error == "embed service timed out"


def test_memory_extraction_report_empty_factory() -> None:
    report = MemoryExtractionReport.empty("session-1", reason="below_turn_threshold")
    assert report.session_id == "session-1"
    assert report.window is None
    assert report.reason == "below_turn_threshold"
    fields = report.to_log_fields()
    assert fields["ops"] == {}
    assert fields["errors"] == 0
    assert fields["reason"] == "below_turn_threshold"


def test_memory_extraction_report_log_fields_counts_ops_and_errors() -> None:
    """to_log_fields aggregates per-op counts + total error count."""
    report = MemoryExtractionReport(
        session_id="s", correlation_id="cid", window=None,
        applied_ops=[
            AppliedOp(op=MemoryOp.CREATE, group_id=uuid4(), chunk_count=1),
            AppliedOp(op=MemoryOp.CREATE, group_id=uuid4(), chunk_count=2),
            AppliedOp(op=MemoryOp.NO_OP, group_id=None, chunk_count=0),
            AppliedOp.failed(MemoryOp.UPDATE, "bang"),
        ],
    )
    fields = report.to_log_fields()
    assert fields["ops"] == {"CREATE": 2, "NO_OP": 1, "UPDATE": 1}
    assert fields["errors"] == 1
