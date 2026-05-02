"""Sanity tests for the memory-subsystem prompt strings.

Prompts are declarative data — these tests only guard against catastrophic
drift (file empty, key rules missing) so the LLM side never silently loses
an instruction that later code relies on.
"""

from __future__ import annotations

from feather.memory.prompts.classification_prompt import CLASSIFICATION_PROMPT
from feather.memory.prompts.extraction_prompt import EXTRACTION_PROMPT
from feather.memory.prompts.query_prompt import QUERY_PROMPT


def test_extraction_prompt_mentions_the_canonical_rules() -> None:
    assert EXTRACTION_PROMPT
    # Each ground rule and every memory field name must be present — the
    # response schema enforces the field set, the prompt enforces semantics.
    for token in (
        "ATOMIC",
        "DURABLE",
        "ABOUT THE USER",
        "EVIDENCE-BACKED",
        "SKIP IF UNCERTAIN",
        "`content`",
        "`purpose`",
        "memories",
    ):
        assert token in EXTRACTION_PROMPT, f"missing token {token!r}"


def test_classification_prompt_defines_every_op_and_guardrails() -> None:
    assert CLASSIFICATION_PROMPT
    for op in ("CREATE", "UPDATE", "DELETE", "NO_OP"):
        assert op in CLASSIFICATION_PROMPT
    # Guardrail language the service layer relies on:
    assert "target_group_id" in CLASSIFICATION_PROMPT
    assert "NO_OP over CREATE" in CLASSIFICATION_PROMPT


def test_query_prompt_mentions_should_skip_and_rewrite_rules() -> None:
    assert QUERY_PROMPT
    assert "should_skip" in QUERY_PROMPT
    assert "third person" in QUERY_PROMPT
    # Few-shot examples anchor the contract; drop one and memory retrieval
    # quality degrades silently.
    assert QUERY_PROMPT.count("Input:") >= 3
