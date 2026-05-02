"""Tests for the proactive CRUD methods on :class:`MemoryService`.

These methods bypass the extractor/classifier pipeline. They exist so the
lead can act on direct user instructions like "remember X" / "forget Y" /
"update what you know about Z" without waiting for the auto-extractor's
turn-windowed sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
from uuid import UUID, uuid4

import pytest

from feather.memory.config import (
    MemoryChunkingConfig,
    MemoryConfig,
    MemoryOperationModelConfig,
    MemoryRetrievalConfig,
    MemoryTriggerConfig,
)
from feather.memory.chunker import Chunker
from feather.memory.classifier import CrudClassifier
from feather.memory.embedding.base import BaseEmbeddingProvider
from feather.memory.enums import EmbedType, MemoryOp, MemoryOwner
from feather.memory.extractor import MemoryExtractor
from feather.memory.models import (
    MemoryPointPayload,
    MemorySearchResult,
)
from feather.memory.prompts.classification_prompt import CLASSIFICATION_PROMPT
from feather.memory.prompts.extraction_prompt import EXTRACTION_PROMPT
from feather.memory.service import MemoryService
from feather.memory.store.base import BaseVectorStore
from feather.memory.tokenizer import CharApproxEstimator
from feather.models import MessageRole
from feather.providers.base import BaseLLMProvider
from feather.storage.session_store import SessionStore


# -----------------------------------------------------------------------------
# Fakes (parallel structure to test_service.py — kept self-contained on purpose)
# -----------------------------------------------------------------------------


class _FakeEmbedder(BaseEmbeddingProvider):
    def __init__(self, dims: int = 4) -> None:
        self.dims = dims
        self.doc_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.doc_calls.append(list(texts))
        return [[1.0] * self.dims for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0] * self.dims


class _FakeStore(BaseVectorStore):
    def __init__(self) -> None:
        self.points: list[tuple[MemoryPointPayload, list[float]]] = []
        self.deleted_groups: list[UUID] = []
        self.upsert_calls: list[list[MemoryPointPayload]] = []
        self.search_results: list[MemorySearchResult] = []
        self.search_calls: list[dict[str, object]] = []

    async def ensure_schema(self) -> None:
        return None

    async def upsert_group(
        self,
        payloads: Sequence[MemoryPointPayload],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        self.upsert_calls.append(list(payloads))
        for p, v in zip(payloads, vectors):
            self.points.append((p, list(v)))

    async def delete_group(self, group_id: UUID) -> None:
        self.deleted_groups.append(group_id)
        self.points = [
            (p, v) for p, v in self.points if str(p.group_id) != str(group_id)
        ]

    async def search(
        self,
        *,
        query: Sequence[float],
        top_k: int,
        filters: dict[str, str],
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        self.search_calls.append(
            {
                "top_k": top_k,
                "filters": dict(filters),
                "session_id": session_id,
            }
        )
        return list(self.search_results)

    async def latest_memory_for_session(
        self, session_id: str
    ) -> MemoryPointPayload | None:
        return None


class _UnusedProvider(BaseLLMProvider):
    """Proactive methods must NOT call the LLM — fail loudly if they do."""

    async def complete(self, **_kw: object) -> object:  # type: ignore[override]
        raise AssertionError("proactive ops must not invoke the LLM provider")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _cfg() -> MemoryConfig:
    cfg = MemoryConfig()
    cfg.chunking = MemoryChunkingConfig(
        chunk_size_tokens=10000,  # never chunk in unit tests
        chunk_overlap_tokens=10,
        tokenizer="char4",
    )
    cfg.retrieval = MemoryRetrievalConfig(
        classifier_top_k=3, classifier_score_threshold=0.75
    )
    cfg.trigger = MemoryTriggerConfig(trigger_turns=2, background=False)
    cfg.extraction = MemoryOperationModelConfig()
    cfg.classification = MemoryOperationModelConfig()
    return cfg


def _service(
    *,
    store: _FakeStore,
    embedder: _FakeEmbedder,
    session_store: SessionStore,
) -> MemoryService:
    cfg = _cfg()
    provider = _UnusedProvider()
    chunker = Chunker(
        CharApproxEstimator(),
        size_tokens=cfg.chunking.chunk_size_tokens,
        overlap_tokens=cfg.chunking.chunk_overlap_tokens,
    )
    extractor = MemoryExtractor(
        provider=provider, prompt=EXTRACTION_PROMPT, cfg=cfg.extraction
    )
    classifier = CrudClassifier(
        provider=provider,
        prompt=CLASSIFICATION_PROMPT,
        cfg=cfg.classification,
        store=store,
        embedder=embedder,
        retrieval_cfg=cfg.retrieval,
    )
    return MemoryService(
        cfg=cfg,
        store=store,
        embedder=embedder,
        chunker=chunker,
        extractor=extractor,
        classifier=classifier,
        session_store=session_store,
    )


def _existing_payload(content: str = "old fact about the user") -> MemoryPointPayload:
    """Build a payload with random ids that the FakeStore can return as a search hit."""
    return MemoryPointPayload(
        type=EmbedType.MEMORY,
        memory_owner=MemoryOwner.USER,
        content=content,
        purpose="for tests",
        filepath=None,
        group_id=uuid4(),
        session_id=uuid4(),
        start_message_id=uuid4(),
        end_message_id=uuid4(),
    )


async def _seed_user_message(sess_store: SessionStore) -> tuple[str, str]:
    """Create a session + one user message; return (session_id, message_id)."""
    session = await sess_store.create_session("Lead")
    msg = await sess_store.add_message(session.id, MessageRole.USER, "remember X")
    return session.id, msg.id


# -----------------------------------------------------------------------------
# proactive_create
# -----------------------------------------------------------------------------


async def test_proactive_create_chunks_embeds_and_upserts(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session_id, msg_id = await _seed_user_message(sess_store)

        svc = _service(store=store, embedder=embedder, session_store=sess_store)
        applied = await svc.proactive_create(
            content="The user prefers Python.",
            purpose="route language-specific suggestions to Python.",
            session_id=session_id,
        )

        assert applied.op is MemoryOp.CREATE
        assert applied.error is None
        assert applied.group_id is not None
        assert applied.chunk_count >= 1
        # One upsert with N>=1 chunks; embedder called with the same N strings
        assert len(store.upsert_calls) == 1
        assert len(embedder.doc_calls) == 1
        assert len(embedder.doc_calls[0]) == applied.chunk_count
        # All payloads share the same group_id and reference a real user message
        payloads = store.upsert_calls[0]
        assert {p.group_id for p in payloads} == {applied.group_id}
        assert str(payloads[0].session_id) == session_id
        assert str(payloads[0].start_message_id) == msg_id
        assert str(payloads[0].end_message_id) == msg_id
        assert payloads[0].content == "The user prefers Python."
        assert payloads[0].purpose == "route language-specific suggestions to Python."
        assert payloads[0].memory_owner == MemoryOwner.USER.value
    finally:
        await sess_store.close()


async def test_proactive_create_rejects_empty_content(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session_id, _ = await _seed_user_message(sess_store)

        svc = _service(store=store, embedder=embedder, session_store=sess_store)
        with pytest.raises(ValueError, match="content"):
            await svc.proactive_create(
                content="   ",
                purpose="ok",
                session_id=session_id,
            )
        assert store.upsert_calls == []
        assert embedder.doc_calls == []
    finally:
        await sess_store.close()


async def test_proactive_create_rejects_empty_purpose(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session_id, _ = await _seed_user_message(sess_store)

        svc = _service(store=store, embedder=embedder, session_store=sess_store)
        with pytest.raises(ValueError, match="purpose"):
            await svc.proactive_create(
                content="non-empty",
                purpose="",
                session_id=session_id,
            )
    finally:
        await sess_store.close()


async def test_proactive_create_raises_when_session_has_no_user_message(
    tmp_path: Path,
) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session = await sess_store.create_session("Lead")
        # No user messages — only an assistant turn (corner case).
        await sess_store.add_message(session.id, MessageRole.ASSISTANT, "hi")

        svc = _service(store=store, embedder=embedder, session_store=sess_store)
        with pytest.raises(ValueError, match="no user message"):
            await svc.proactive_create(
                content="x", purpose="y", session_id=session.id
            )
    finally:
        await sess_store.close()


# -----------------------------------------------------------------------------
# proactive_update
# -----------------------------------------------------------------------------


async def test_proactive_update_finds_match_and_replaces_group(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    existing = _existing_payload("user likes Python")
    store.search_results = [MemorySearchResult(payload=existing, score=0.91)]
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session_id, _ = await _seed_user_message(sess_store)
        svc = _service(store=store, embedder=embedder, session_store=sess_store)

        applied = await svc.proactive_update(
            target_query="programming language preference",
            content="The user now prefers Rust over Python.",
            purpose="route language-specific suggestions to Rust.",
            session_id=session_id,
        )

        assert applied.op is MemoryOp.UPDATE
        assert applied.error is None
        assert applied.group_id == existing.group_id
        # The matched group must be deleted before re-upsert.
        assert existing.group_id in store.deleted_groups
        # New payloads were upserted with the SAME group_id as the matched one.
        assert len(store.upsert_calls) == 1
        new_payloads = store.upsert_calls[0]
        assert {p.group_id for p in new_payloads} == {existing.group_id}
        assert new_payloads[0].content == "The user now prefers Rust over Python."
        # Search ran on the embedded query.
        assert embedder.query_calls == ["programming language preference"]
    finally:
        await sess_store.close()


async def test_proactive_update_returns_failed_when_no_match(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    store.search_results = []  # nothing to update
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session_id, _ = await _seed_user_message(sess_store)
        svc = _service(store=store, embedder=embedder, session_store=sess_store)

        applied = await svc.proactive_update(
            target_query="nothing matches",
            content="new content",
            purpose="new purpose",
            session_id=session_id,
        )
        assert applied.op is MemoryOp.UPDATE
        assert applied.error is not None
        assert "no match" in applied.error.lower()
        assert store.upsert_calls == []
        assert store.deleted_groups == []
    finally:
        await sess_store.close()


async def test_proactive_update_rejects_below_threshold(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    weak = _existing_payload("vaguely related")
    store.search_results = [MemorySearchResult(payload=weak, score=0.10)]
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session_id, _ = await _seed_user_message(sess_store)
        svc = _service(store=store, embedder=embedder, session_store=sess_store)

        applied = await svc.proactive_update(
            target_query="something the user said",
            content="new content",
            purpose="new purpose",
            session_id=session_id,
            score_threshold=0.5,
        )
        assert applied.op is MemoryOp.UPDATE
        assert applied.error is not None
        assert store.upsert_calls == []
        assert store.deleted_groups == []
    finally:
        await sess_store.close()


# -----------------------------------------------------------------------------
# proactive_delete
# -----------------------------------------------------------------------------


async def test_proactive_delete_finds_match_and_deletes_group(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    existing = _existing_payload("user dislikes loud cafes")
    store.search_results = [MemorySearchResult(payload=existing, score=0.85)]
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session_id, _ = await _seed_user_message(sess_store)
        svc = _service(store=store, embedder=embedder, session_store=sess_store)

        applied = await svc.proactive_delete(
            target_query="cafe preferences",
            session_id=session_id,
        )

        assert applied.op is MemoryOp.DELETE
        assert applied.error is None
        assert applied.group_id == existing.group_id
        assert existing.group_id in store.deleted_groups
        assert store.upsert_calls == []
        assert embedder.query_calls == ["cafe preferences"]
    finally:
        await sess_store.close()


async def test_proactive_delete_returns_failed_when_no_match(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    store.search_results = []
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session_id, _ = await _seed_user_message(sess_store)
        svc = _service(store=store, embedder=embedder, session_store=sess_store)

        applied = await svc.proactive_delete(
            target_query="nope",
            session_id=session_id,
        )
        assert applied.op is MemoryOp.DELETE
        assert applied.error is not None
        assert "no match" in applied.error.lower()
        assert store.deleted_groups == []
    finally:
        await sess_store.close()


async def test_proactive_delete_rejects_empty_target_query(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session_id, _ = await _seed_user_message(sess_store)
        svc = _service(store=store, embedder=embedder, session_store=sess_store)

        with pytest.raises(ValueError, match="target_query"):
            await svc.proactive_delete(target_query="   ", session_id=session_id)
    finally:
        await sess_store.close()


# -----------------------------------------------------------------------------
# Owner / search filter discipline
# -----------------------------------------------------------------------------


async def test_proactive_update_search_filters_on_owner_and_type(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    existing = _existing_payload("x")
    store.search_results = [MemorySearchResult(payload=existing, score=0.99)]
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session_id, _ = await _seed_user_message(sess_store)
        svc = _service(store=store, embedder=embedder, session_store=sess_store)
        await svc.proactive_update(
            target_query="q",
            content="c",
            purpose="p",
            session_id=session_id,
        )
        assert len(store.search_calls) == 1
        call = store.search_calls[0]
        # Cross-session search by default — UUID filter omitted.
        assert call["session_id"] is None
        assert call["filters"] == {
            "type": EmbedType.MEMORY.value,
            "memory_owner": MemoryOwner.USER.value,
        }
    finally:
        await sess_store.close()
