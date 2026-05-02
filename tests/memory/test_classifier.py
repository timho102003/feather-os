"""Tests for CrudClassifier — the CREATE/UPDATE/DELETE/NO_OP decision layer."""

from __future__ import annotations

import json
from typing import Any, Sequence
from uuid import UUID, uuid4

import pytest

from feather.memory.classifier import CrudClassifier
from feather.memory.config import MemoryOperationModelConfig, MemoryRetrievalConfig
from feather.memory.embedding.base import BaseEmbeddingProvider
from feather.memory.enums import EmbedType, MemoryOp, MemoryOwner
from feather.memory.models import (
    AtomicMemory,
    MemoryPointPayload,
    MemorySearchResult,
)
from feather.memory.prompts.classification_prompt import CLASSIFICATION_PROMPT
from feather.memory.store.base import BaseVectorStore
from feather.models import ModelTurn, ProviderRequestConfig
from feather.providers.base import BaseLLMProvider


# Fakes -----------------------------------------------------------------------


class _FakeEmbedder(BaseEmbeddingProvider):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class _FakeStore(BaseVectorStore):
    def __init__(self, results: list[MemorySearchResult]) -> None:
        self._results = results
        self.search_calls: list[dict[str, Any]] = []

    async def ensure_schema(self) -> None:  # pragma: no cover
        return None

    async def upsert_group(self, *a: Any, **k: Any) -> None:  # pragma: no cover
        return None

    async def delete_group(self, *a: Any, **k: Any) -> None:  # pragma: no cover
        return None

    async def search(
        self,
        *,
        query: Sequence[float],
        top_k: int,
        filters: dict[str, str],
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        self.search_calls.append(
            {"top_k": top_k, "filters": filters, "session_id": session_id}
        )
        return list(self._results)[:top_k]

    async def latest_memory_for_session(
        self, session_id: str
    ) -> MemoryPointPayload | None:  # pragma: no cover
        return None


class _FakeProvider(BaseLLMProvider):
    def __init__(self, output_text: str | None = None) -> None:
        self._output_text = output_text
        self.calls: list[dict[str, Any]] = []

    async def complete(  # type: ignore[override]
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        event_handler: Any = None,
        request_config: ProviderRequestConfig | None = None,
    ) -> ModelTurn:
        self.calls.append(
            {
                "instructions": instructions,
                "input_items": input_items,
                "tools": tools,
                "previous_response_id": previous_response_id,
                "request_config": request_config,
            }
        )
        assert self._output_text is not None, "provider called unexpectedly"
        return ModelTurn(response_id="r", output_text=self._output_text)


# Helpers ---------------------------------------------------------------------


def _atom(content: str = "the user likes Python") -> AtomicMemory:
    return AtomicMemory(
        who="the user",
        what="likes Python",
        when="ongoing",
        where="unspecified",
        why="unspecified",
        how="unspecified",
        purpose="pick default language",
        content=content,
    )


def _result(group_id: UUID, score: float, content: str = "existing") -> MemorySearchResult:
    payload = MemoryPointPayload(
        type=EmbedType.MEMORY,
        memory_owner=MemoryOwner.USER,
        content=content,
        purpose="x",
        group_id=group_id,
        session_id=uuid4(),
        start_message_id=uuid4(),
        end_message_id=uuid4(),
    )
    return MemorySearchResult(payload=payload, score=score)


def _cfg_retrieval(threshold: float = 0.75, top_k: int = 3) -> MemoryRetrievalConfig:
    return MemoryRetrievalConfig(
        classifier_top_k=top_k,
        classifier_score_threshold=threshold,
    )


# Below-threshold: CREATE without LLM ----------------------------------------


async def test_create_when_no_candidates_and_no_llm_call() -> None:
    store = _FakeStore(results=[])
    provider = _FakeProvider(output_text=None)  # must NOT be called
    classifier = CrudClassifier(
        provider=provider,
        prompt=CLASSIFICATION_PROMPT,
        cfg=MemoryOperationModelConfig(),
        store=store,
        embedder=_FakeEmbedder(),
        retrieval_cfg=_cfg_retrieval(),
    )
    out = await classifier.classify(
        _atom(), agent_model="gpt-5-mini", owner=MemoryOwner.USER, session_id="sess"
    )
    assert out.op is MemoryOp.CREATE
    assert out.target_group_id is None
    assert provider.calls == []


async def test_create_when_best_score_below_threshold() -> None:
    low = _result(uuid4(), score=0.5)
    store = _FakeStore(results=[low])
    provider = _FakeProvider(output_text=None)
    classifier = CrudClassifier(
        provider=provider,
        prompt=CLASSIFICATION_PROMPT,
        cfg=MemoryOperationModelConfig(),
        store=store,
        embedder=_FakeEmbedder(),
        retrieval_cfg=_cfg_retrieval(threshold=0.75),
    )
    out = await classifier.classify(
        _atom(), agent_model="gpt-5-mini", owner=MemoryOwner.USER, session_id="sess"
    )
    assert out.op is MemoryOp.CREATE
    assert out.candidates and out.candidates[0].score == 0.5
    assert provider.calls == []


# Cross-session search -------------------------------------------------------


async def test_classifier_searches_cross_session_by_default() -> None:
    """Memory is per-user: the retrieval step for classification is cross-session."""
    store = _FakeStore(results=[])
    classifier = CrudClassifier(
        provider=_FakeProvider(output_text=None),
        prompt=CLASSIFICATION_PROMPT,
        cfg=MemoryOperationModelConfig(),
        store=store,
        embedder=_FakeEmbedder(),
        retrieval_cfg=_cfg_retrieval(),
    )
    await classifier.classify(
        _atom(),
        agent_model="gpt-5-mini",
        owner=MemoryOwner.USER,
        session_id="sess-a",
    )
    assert store.search_calls[0]["session_id"] is None
    assert store.search_calls[0]["filters"] == {
        "type": EmbedType.MEMORY.value,
        "memory_owner": MemoryOwner.USER.value,
    }


# Above-threshold: LLM path --------------------------------------------------


async def test_llm_returns_update_with_valid_target_group_id() -> None:
    gid = uuid4()
    high = _result(gid, score=0.9, content="existing")
    store = _FakeStore(results=[high])
    provider = _FakeProvider(
        output_text=json.dumps(
            {"op": "UPDATE", "target_group_id": str(gid), "reasoning": "refine"}
        )
    )
    classifier = CrudClassifier(
        provider=provider,
        prompt=CLASSIFICATION_PROMPT,
        cfg=MemoryOperationModelConfig(),
        store=store,
        embedder=_FakeEmbedder(),
        retrieval_cfg=_cfg_retrieval(),
    )
    out = await classifier.classify(
        _atom(), agent_model="gpt-5-mini", owner=MemoryOwner.USER, session_id="sess"
    )
    assert out.op is MemoryOp.UPDATE
    assert out.target_group_id == str(gid)
    assert provider.calls, "provider MUST be called when score >= threshold"


async def test_llm_returns_delete_with_valid_target_group_id() -> None:
    gid = uuid4()
    high = _result(gid, score=0.9)
    store = _FakeStore(results=[high])
    provider = _FakeProvider(
        output_text=json.dumps(
            {"op": "DELETE", "target_group_id": str(gid), "reasoning": "retracted"}
        )
    )
    classifier = CrudClassifier(
        provider=provider,
        prompt=CLASSIFICATION_PROMPT,
        cfg=MemoryOperationModelConfig(),
        store=store,
        embedder=_FakeEmbedder(),
        retrieval_cfg=_cfg_retrieval(),
    )
    out = await classifier.classify(
        _atom(), agent_model="gpt-5-mini", owner=MemoryOwner.USER, session_id="sess"
    )
    assert out.op is MemoryOp.DELETE
    assert out.target_group_id == str(gid)


async def test_llm_returns_no_op_passes_through() -> None:
    high = _result(uuid4(), score=0.85)
    store = _FakeStore(results=[high])
    provider = _FakeProvider(
        output_text=json.dumps({"op": "NO_OP", "reasoning": "duplicate"})
    )
    classifier = CrudClassifier(
        provider=provider,
        prompt=CLASSIFICATION_PROMPT,
        cfg=MemoryOperationModelConfig(),
        store=store,
        embedder=_FakeEmbedder(),
        retrieval_cfg=_cfg_retrieval(),
    )
    out = await classifier.classify(
        _atom(), agent_model="gpt-5-mini", owner=MemoryOwner.USER, session_id="sess"
    )
    assert out.op is MemoryOp.NO_OP
    assert out.target_group_id is None


async def test_llm_returns_create_when_similarity_is_false_positive() -> None:
    high = _result(uuid4(), score=0.8)
    store = _FakeStore(results=[high])
    provider = _FakeProvider(
        output_text=json.dumps({"op": "CREATE", "reasoning": "different fact"})
    )
    classifier = CrudClassifier(
        provider=provider,
        prompt=CLASSIFICATION_PROMPT,
        cfg=MemoryOperationModelConfig(),
        store=store,
        embedder=_FakeEmbedder(),
        retrieval_cfg=_cfg_retrieval(),
    )
    out = await classifier.classify(
        _atom(), agent_model="gpt-5-mini", owner=MemoryOwner.USER, session_id="sess"
    )
    assert out.op is MemoryOp.CREATE
    assert out.target_group_id is None


# Guardrails -----------------------------------------------------------------


async def test_invalid_target_group_id_is_coerced_to_no_op(caplog: pytest.LogCaptureFixture) -> None:
    """Hallucinated group_ids must collapse to NO_OP with a warning log."""
    gid = uuid4()
    high = _result(gid, score=0.85)
    store = _FakeStore(results=[high])
    bogus = str(uuid4())
    provider = _FakeProvider(
        output_text=json.dumps(
            {"op": "UPDATE", "target_group_id": bogus, "reasoning": "hallucinated"}
        )
    )
    classifier = CrudClassifier(
        provider=provider,
        prompt=CLASSIFICATION_PROMPT,
        cfg=MemoryOperationModelConfig(),
        store=store,
        embedder=_FakeEmbedder(),
        retrieval_cfg=_cfg_retrieval(),
    )
    with caplog.at_level("WARNING"):
        out = await classifier.classify(
            _atom(),
            agent_model="gpt-5-mini",
            owner=MemoryOwner.USER,
            session_id="sess",
        )
    assert out.op is MemoryOp.NO_OP
    assert out.target_group_id is None
    assert any("classifier" in rec.message.lower() for rec in caplog.records)


async def test_update_without_target_group_id_is_coerced_to_no_op() -> None:
    """UPDATE with null target_group_id is invalid — coerce."""
    gid = uuid4()
    high = _result(gid, score=0.85)
    store = _FakeStore(results=[high])
    provider = _FakeProvider(
        output_text=json.dumps({"op": "UPDATE", "reasoning": "forgot id"})
    )
    classifier = CrudClassifier(
        provider=provider,
        prompt=CLASSIFICATION_PROMPT,
        cfg=MemoryOperationModelConfig(),
        store=store,
        embedder=_FakeEmbedder(),
        retrieval_cfg=_cfg_retrieval(),
    )
    out = await classifier.classify(
        _atom(), agent_model="gpt-5-mini", owner=MemoryOwner.USER, session_id="sess"
    )
    assert out.op is MemoryOp.NO_OP


# Structured-output wiring ---------------------------------------------------


async def test_classifier_uses_classification_prompt_and_response_schema() -> None:
    gid = uuid4()
    high = _result(gid, score=0.85)
    store = _FakeStore(results=[high])
    provider = _FakeProvider(
        output_text=json.dumps({"op": "NO_OP", "reasoning": "dup"})
    )
    classifier = CrudClassifier(
        provider=provider,
        prompt=CLASSIFICATION_PROMPT,
        cfg=MemoryOperationModelConfig(),
        store=store,
        embedder=_FakeEmbedder(),
        retrieval_cfg=_cfg_retrieval(),
    )
    await classifier.classify(
        _atom(), agent_model="gpt-5-mini", owner=MemoryOwner.USER, session_id="sess"
    )
    call = provider.calls[0]
    assert call["instructions"] == CLASSIFICATION_PROMPT
    assert call["tools"] == []
    assert call["previous_response_id"] is None
    assert call["request_config"].response_schema.__name__ == "ClassificationResponse"
