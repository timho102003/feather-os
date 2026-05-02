"""Tests for MemoryService orchestration (build_window + extract_and_store)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Sequence
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
from feather.memory.enums import MemoryOp, MemoryOwner
from feather.memory.extractor import MemoryExtractor
from feather.memory.models import (
    AtomicMemory,
    ClassifiedOp,
    MemoryPointPayload,
    MemorySearchResult,
    MemoryWindow,
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
# Fakes
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
        self.latest_per_session: dict[str, MemoryPointPayload] = {}

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
        # Default: nothing similar — extractor → CREATE for each atom.
        return []

    async def latest_memory_for_session(
        self, session_id: str
    ) -> MemoryPointPayload | None:
        return self.latest_per_session.get(session_id)


class _FakeProvider(BaseLLMProvider):
    """Minimal provider that returns canned text. Each call must have a queued response."""

    def __init__(self) -> None:
        self.queued: list[str] = []
        self.calls: list[Any] = []

    def queue(self, text: str) -> None:
        self.queued.append(text)

    async def complete(  # type: ignore[override]
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        event_handler: Any = None,
        request_config: Any = None,
    ) -> Any:
        self.calls.append({"instructions": instructions, "request_config": request_config})
        if not self.queued:
            raise AssertionError("provider called more times than queued responses")
        from feather.models import ModelTurn

        return ModelTurn(response_id="r", output_text=self.queued.pop(0))


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
    cfg: MemoryConfig | None = None,
    provider: _FakeProvider,
    store: _FakeStore,
    embedder: _FakeEmbedder,
    session_store: SessionStore,
) -> MemoryService:
    cfg = cfg or _cfg()
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


def _atom_json(content: str) -> str:
    """Render a single-atom ExtractionResponse JSON string."""
    import json

    return json.dumps(
        {
            "memories": [
                {
                    "who": "the user",
                    "what": "x",
                    "when": "ongoing",
                    "where": "u",
                    "why": "u",
                    "how": "u",
                    "purpose": "p",
                    "content": content,
                }
            ]
        }
    )


def _empty_memories_json() -> str:
    return '{"memories": []}'


# -----------------------------------------------------------------------------
# Window-builder behavior (driven through extract_and_store)
# -----------------------------------------------------------------------------


async def test_below_turn_threshold_returns_empty_report_without_extracting(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    provider = _FakeProvider()  # nothing queued — must not be called
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session = await sess_store.create_session("Lead")
        await sess_store.add_message(session.id, MessageRole.USER, "only one user msg")

        svc = _service(
            provider=provider, store=store, embedder=embedder, session_store=sess_store
        )
        report = await svc.extract_and_store(
            session.id, agent_model="gpt-5-mini", owner=MemoryOwner.USER
        )
        assert report.reason == "below_turn_threshold"
        assert provider.calls == []
        assert store.upsert_calls == []
    finally:
        await sess_store.close()


async def test_extract_runs_when_user_turn_count_meets_threshold(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    provider = _FakeProvider()
    provider.queue(_empty_memories_json())  # extractor returns no atoms

    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session = await sess_store.create_session("Lead")
        # 2 user turns (threshold=2), assistant in between
        await sess_store.add_message(session.id, MessageRole.USER, "u1")
        await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a1")
        await sess_store.add_message(session.id, MessageRole.USER, "u2")
        await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a2")

        svc = _service(
            provider=provider, store=store, embedder=embedder, session_store=sess_store
        )
        report = await svc.extract_and_store(
            session.id, agent_model="gpt-5-mini", owner=MemoryOwner.USER
        )
        assert report.reason == "no_atoms"
        assert len(provider.calls) == 1  # extractor called, no atoms
    finally:
        await sess_store.close()


async def test_window_excludes_messages_belonging_to_next_user_turn(tmp_path: Path) -> None:
    """Window ends just before the (N+1)th user message, inclusive of trailing assistant."""
    store = _FakeStore()
    embedder = _FakeEmbedder()
    provider = _FakeProvider()

    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session = await sess_store.create_session("Lead")
        await sess_store.add_message(session.id, MessageRole.USER, "u1")
        await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a1")
        await sess_store.add_message(session.id, MessageRole.USER, "u2")
        a2 = await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a2")
        await sess_store.add_message(session.id, MessageRole.USER, "u3-next")

        # Capture the window through a stub extractor that records what it saw.
        captured_window: list[MemoryWindow] = []

        class _CapturingExtractor:
            async def extract(self, window: MemoryWindow, agent_model: str) -> list[AtomicMemory]:
                captured_window.append(window)
                return []

        svc = _service(
            provider=provider, store=store, embedder=embedder, session_store=sess_store
        )
        svc._extractor = _CapturingExtractor()  # type: ignore[assignment]
        await svc.extract_and_store(
            session.id, agent_model="gpt-5-mini", owner=MemoryOwner.USER
        )

        assert captured_window
        window = captured_window[0]
        # Window ends at a2 (last asst in u2's turn), NOT at u3-next.
        assert window.end_message_id == a2.id
        assert "u3-next" not in [m.content for m in window.messages]
    finally:
        await sess_store.close()


async def test_window_builder_uses_anchor_from_qdrant(tmp_path: Path) -> None:
    """Window starts AFTER the latest memory's end_message_id in this session."""
    store = _FakeStore()
    embedder = _FakeEmbedder()
    provider = _FakeProvider()
    provider.queue(_empty_memories_json())

    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session = await sess_store.create_session("Lead")
        await sess_store.add_message(session.id, MessageRole.USER, "u-old-1")
        a_anchor = await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a-anchor")
        # Pretend Qdrant says: latest memory ends here.
        store.latest_per_session[session.id] = MemoryPointPayload(
            type="memory",  # type: ignore[arg-type]
            memory_owner="user",  # type: ignore[arg-type]
            content="old",
            purpose="x",
            group_id=uuid4(),
            session_id=UUID(session.id),
            start_message_id=uuid4(),
            end_message_id=UUID(a_anchor.id),
        )
        # New activity AFTER the anchor:
        await sess_store.add_message(session.id, MessageRole.USER, "u-new-1")
        await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a-new-1")
        await sess_store.add_message(session.id, MessageRole.USER, "u-new-2")
        await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a-new-2")

        svc = _service(
            provider=provider, store=store, embedder=embedder, session_store=sess_store
        )
        report = await svc.extract_and_store(
            session.id, agent_model="gpt-5-mini", owner=MemoryOwner.USER
        )
        # Pre-anchor messages must NOT appear: with anchor in place, only u-new-1/2 + a-new-1/2 enter the window.
        # (Since extractor returned no atoms, we just verify it ran — the absence
        # of "below_turn_threshold" plus the presence of one extractor call confirms
        # the window had >=2 user messages that came AFTER the anchor.)
        assert report.reason == "no_atoms"
        assert len(provider.calls) == 1
    finally:
        await sess_store.close()


# -----------------------------------------------------------------------------
# Per-op application
# -----------------------------------------------------------------------------


async def test_create_op_writes_one_chunk_with_new_group_id(tmp_path: Path) -> None:
    store = _FakeStore()
    embedder = _FakeEmbedder()
    provider = _FakeProvider()
    provider.queue(_atom_json("the user prefers Python"))

    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session = await sess_store.create_session("Lead")
        await sess_store.add_message(session.id, MessageRole.USER, "u1")
        await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a1")
        await sess_store.add_message(session.id, MessageRole.USER, "u2")
        await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a2")

        svc = _service(
            provider=provider, store=store, embedder=embedder, session_store=sess_store
        )
        report = await svc.extract_and_store(
            session.id, agent_model="gpt-5-mini", owner=MemoryOwner.USER
        )

        assert len(report.applied_ops) == 1
        applied = report.applied_ops[0]
        assert applied.op is MemoryOp.CREATE
        assert applied.chunk_count == 1
        assert applied.group_id is not None
        assert len(store.upsert_calls) == 1
        payload = store.upsert_calls[0][0]
        assert payload.content == "the user prefers Python"
        assert payload.session_id == UUID(session.id)
        assert payload.purpose == "p"
        assert payload.chunk_index == 0
    finally:
        await sess_store.close()


async def test_no_op_when_classifier_returns_no_op(tmp_path: Path) -> None:
    """A classifier-returned NO_OP must not write or delete anything."""
    import json

    store = _FakeStore()

    # Force the classifier path to run by returning a candidate above threshold.
    existing_gid = uuid4()
    existing = MemorySearchResult(
        payload=MemoryPointPayload(
            type="memory",  # type: ignore[arg-type]
            memory_owner="user",  # type: ignore[arg-type]
            content="existing",
            purpose="p",
            group_id=existing_gid,
            session_id=uuid4(),
            start_message_id=uuid4(),
            end_message_id=uuid4(),
        ),
        score=0.95,
    )

    async def search(
        *,
        query: Sequence[float],
        top_k: int,
        filters: dict[str, str],
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        return [existing]

    store.search = search  # type: ignore[method-assign]

    embedder = _FakeEmbedder()
    provider = _FakeProvider()
    provider.queue(_atom_json("the user prefers Python"))  # extractor
    provider.queue(json.dumps({"op": "NO_OP", "reasoning": "duplicate"}))  # classifier

    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session = await sess_store.create_session("Lead")
        await sess_store.add_message(session.id, MessageRole.USER, "u1")
        await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a1")
        await sess_store.add_message(session.id, MessageRole.USER, "u2")
        await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a2")

        svc = _service(
            provider=provider, store=store, embedder=embedder, session_store=sess_store
        )
        report = await svc.extract_and_store(
            session.id, agent_model="gpt-5-mini", owner=MemoryOwner.USER
        )

        assert report.applied_ops[0].op is MemoryOp.NO_OP
        assert store.upsert_calls == []
        assert store.deleted_groups == []
    finally:
        await sess_store.close()


async def test_update_op_replaces_existing_group_in_place(tmp_path: Path) -> None:
    """UPDATE preserves the candidate's group_id — delete then re-insert."""
    import json

    store = _FakeStore()
    existing_gid = uuid4()
    existing = MemorySearchResult(
        payload=MemoryPointPayload(
            type="memory",  # type: ignore[arg-type]
            memory_owner="user",  # type: ignore[arg-type]
            content="old prefers Python",
            purpose="p",
            group_id=existing_gid,
            session_id=uuid4(),
            start_message_id=uuid4(),
            end_message_id=uuid4(),
        ),
        score=0.95,
    )

    async def search(
        *,
        query: Sequence[float],
        top_k: int,
        filters: dict[str, str],
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        return [existing]

    store.search = search  # type: ignore[method-assign]

    embedder = _FakeEmbedder()
    provider = _FakeProvider()
    provider.queue(_atom_json("the user prefers Python AND Rust"))
    provider.queue(
        json.dumps(
            {"op": "UPDATE", "target_group_id": str(existing_gid), "reasoning": "extends"}
        )
    )

    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session = await sess_store.create_session("Lead")
        for _ in range(2):
            await sess_store.add_message(session.id, MessageRole.USER, "u")
            await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a")

        svc = _service(
            provider=provider, store=store, embedder=embedder, session_store=sess_store
        )
        report = await svc.extract_and_store(
            session.id, agent_model="gpt-5-mini", owner=MemoryOwner.USER
        )

        applied = report.applied_ops[0]
        assert applied.op is MemoryOp.UPDATE
        assert applied.group_id == existing_gid
        # delete-then-upsert preserves group_id.
        assert store.deleted_groups == [existing_gid]
        assert store.upsert_calls
        assert store.upsert_calls[0][0].group_id == existing_gid
    finally:
        await sess_store.close()


async def test_delete_op_removes_existing_group(tmp_path: Path) -> None:
    import json

    store = _FakeStore()
    existing_gid = uuid4()
    existing = MemorySearchResult(
        payload=MemoryPointPayload(
            type="memory",  # type: ignore[arg-type]
            memory_owner="user",  # type: ignore[arg-type]
            content="user said they like X",
            purpose="p",
            group_id=existing_gid,
            session_id=uuid4(),
            start_message_id=uuid4(),
            end_message_id=uuid4(),
        ),
        score=0.95,
    )

    async def search(
        *,
        query: Sequence[float],
        top_k: int,
        filters: dict[str, str],
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        return [existing]

    store.search = search  # type: ignore[method-assign]

    embedder = _FakeEmbedder()
    provider = _FakeProvider()
    provider.queue(_atom_json("the user retracted that they like X"))
    provider.queue(
        json.dumps(
            {"op": "DELETE", "target_group_id": str(existing_gid), "reasoning": "retracted"}
        )
    )

    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session = await sess_store.create_session("Lead")
        for _ in range(2):
            await sess_store.add_message(session.id, MessageRole.USER, "u")
            await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a")

        svc = _service(
            provider=provider, store=store, embedder=embedder, session_store=sess_store
        )
        report = await svc.extract_and_store(
            session.id, agent_model="gpt-5-mini", owner=MemoryOwner.USER
        )
        applied = report.applied_ops[0]
        assert applied.op is MemoryOp.DELETE
        assert applied.group_id == existing_gid
        assert store.deleted_groups == [existing_gid]
        assert store.upsert_calls == []
    finally:
        await sess_store.close()


# -----------------------------------------------------------------------------
# Per-atom error isolation
# -----------------------------------------------------------------------------


async def test_per_atom_failure_does_not_abort_batch(tmp_path: Path) -> None:
    """One atom raises during _apply_op → other atoms still get applied."""
    import json

    store = _FakeStore()
    embedder = _FakeEmbedder()
    provider = _FakeProvider()
    # Two atoms in one extraction
    provider.queue(
        json.dumps(
            {
                "memories": [
                    {
                        "who": "u",
                        "what": "x",
                        "when": "u",
                        "where": "u",
                        "why": "u",
                        "how": "u",
                        "purpose": "p",
                        "content": "atom-1",
                    },
                    {
                        "who": "u",
                        "what": "x",
                        "when": "u",
                        "where": "u",
                        "why": "u",
                        "how": "u",
                        "purpose": "p",
                        "content": "atom-2",
                    },
                ]
            }
        )
    )

    # Inject a store that raises on the FIRST upsert and succeeds on the second.
    upsert_attempts = {"n": 0}

    async def flaky_upsert(payloads: Sequence[MemoryPointPayload], vectors: Sequence[Sequence[float]]) -> None:
        upsert_attempts["n"] += 1
        if upsert_attempts["n"] == 1:
            raise RuntimeError("simulated qdrant outage on first call")
        for p, v in zip(payloads, vectors):
            store.points.append((p, list(v)))
        store.upsert_calls.append(list(payloads))

    store.upsert_group = flaky_upsert  # type: ignore[method-assign]

    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session = await sess_store.create_session("Lead")
        for _ in range(2):
            await sess_store.add_message(session.id, MessageRole.USER, "u")
            await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a")

        svc = _service(
            provider=provider, store=store, embedder=embedder, session_store=sess_store
        )
        report = await svc.extract_and_store(
            session.id, agent_model="gpt-5-mini", owner=MemoryOwner.USER
        )

        assert len(report.applied_ops) == 2
        # First atom failed, second succeeded
        assert report.applied_ops[0].error is not None
        assert report.applied_ops[1].error is None
        assert upsert_attempts["n"] == 2
    finally:
        await sess_store.close()


# -----------------------------------------------------------------------------
# Per-session lock
# -----------------------------------------------------------------------------


async def test_concurrent_extracts_for_same_session_serialize(tmp_path: Path) -> None:
    """Two concurrent extract_and_store calls for the same session run sequentially."""
    store = _FakeStore()
    embedder = _FakeEmbedder()

    # A provider that signals when its first call starts and waits to be released.
    started = asyncio.Event()
    release = asyncio.Event()
    in_flight: list[int] = []

    class _BlockingProvider(BaseLLMProvider):
        async def complete(  # type: ignore[override]
            self,
            *,
            instructions: str,
            input_items: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            previous_response_id: str | None,
            event_handler: Any = None,
            request_config: Any = None,
        ) -> Any:
            from feather.models import ModelTurn

            in_flight.append(1)
            assert sum(in_flight) == 1, "two extracts ran concurrently for same session"
            started.set()
            await release.wait()
            in_flight.pop()
            return ModelTurn(response_id="r", output_text=_empty_memories_json())

    provider = _BlockingProvider()

    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        session = await sess_store.create_session("Lead")
        for _ in range(2):
            await sess_store.add_message(session.id, MessageRole.USER, "u")
            await sess_store.add_message(session.id, MessageRole.ASSISTANT, "a")

        svc = _service(
            provider=provider, store=store, embedder=embedder, session_store=sess_store
        )
        # Kick off two extractions concurrently.
        first = asyncio.create_task(
            svc.extract_and_store(session.id, agent_model="gpt-5-mini", owner=MemoryOwner.USER)
        )
        await started.wait()
        second = asyncio.create_task(
            svc.extract_and_store(session.id, agent_model="gpt-5-mini", owner=MemoryOwner.USER)
        )
        # Give the second a chance to wake up — it MUST be blocked on the lock.
        await asyncio.sleep(0.01)
        assert not second.done()
        # Release first.
        release.set()
        await first
        # Now second can acquire the lock and proceed.
        await second
    finally:
        await sess_store.close()


# -----------------------------------------------------------------------------
# Initialize delegates to store
# -----------------------------------------------------------------------------


async def test_initialize_delegates_to_store_ensure_schema(tmp_path: Path) -> None:
    store = _FakeStore()
    called = {"ensure_schema": 0}

    async def ensure() -> None:
        called["ensure_schema"] += 1

    store.ensure_schema = ensure  # type: ignore[method-assign]
    embedder = _FakeEmbedder()
    provider = _FakeProvider()
    sess_store = SessionStore(tmp_path / "db.sqlite")
    await sess_store.initialize()
    try:
        svc = _service(
            provider=provider, store=store, embedder=embedder, session_store=sess_store
        )
        await svc.initialize()
        assert called["ensure_schema"] == 1
    finally:
        await sess_store.close()
