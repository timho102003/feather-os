"""Tests for LiveMemoryReader and NoOpMemoryReader."""

from __future__ import annotations

import asyncio
from typing import Any, Sequence
from uuid import UUID, uuid4

from feather.memory.config import (
    MemoryOperationModelConfig,
    MemoryRetrievalConfig,
)
from feather.memory.embedding.base import BaseEmbeddingProvider
from feather.memory.enums import EmbedType, MemoryOwner
from feather.memory.models import (
    MemoryPointPayload,
    MemorySearchResult,
    QueryDecision,
)
from feather.memory.prompts.query_prompt import QUERY_PROMPT
from feather.memory.query_builder import MemoryQueryBuilder
from feather.memory.reader import (
    LiveMemoryReader,
    NoOpMemoryReader,
    format_memory_block,
)
from feather.memory.store.base import BaseVectorStore
from feather.models import (
    MessageRole,
    ModelTurn,
    ProviderRequestConfig,
    SessionMessage,
)
from feather.providers.base import BaseLLMProvider


class _FakeEmbedder(BaseEmbeddingProvider):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        return [[1.0] * 4 for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class _FakeStore(BaseVectorStore):
    def __init__(self, results: list[MemorySearchResult]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def ensure_schema(self) -> None: ...
    async def upsert_group(self, *a: Any, **k: Any) -> None: ...
    async def delete_group(self, *a: Any, **k: Any) -> None: ...

    async def search(
        self,
        *,
        query: Sequence[float],
        top_k: int,
        filters: dict[str, str],
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        self.calls.append(
            {"top_k": top_k, "filters": filters, "session_id": session_id}
        )
        return list(self._results)[:top_k]

    async def latest_memory_for_session(
        self, session_id: str
    ) -> MemoryPointPayload | None:  # pragma: no cover
        return None


class _FakeProvider(BaseLLMProvider):
    """Always returns a query-builder JSON that builds a non-empty query."""

    def __init__(self, *, output_text: str | None = None, exc: BaseException | None = None) -> None:
        self._output_text = output_text or '{"query":"the user","should_skip":false,"reasoning":"r"}'
        self._exc = exc
        self.calls: list[Any] = []

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
        self.calls.append({"instructions": instructions, "request_config": request_config})
        if self._exc is not None:
            raise self._exc
        return ModelTurn(response_id="r", output_text=self._output_text)


def _payload(
    content: str,
    group_id: UUID | None = None,
    *,
    embed_type: EmbedType = EmbedType.MEMORY,
    filepath: str | None = None,
) -> MemoryPointPayload:
    from datetime import datetime, timezone

    return MemoryPointPayload(
        type=embed_type,
        memory_owner=MemoryOwner.USER,
        content=content,
        purpose="for tests",
        filepath=filepath,
        group_id=group_id or uuid4(),
        session_id=uuid4(),
        start_message_id=uuid4(),
        end_message_id=uuid4(),
        created_at=datetime(2026, 4, 16, tzinfo=timezone.utc),
    )


def _result(
    content: str,
    score: float,
    group_id: UUID | None = None,
    *,
    embed_type: EmbedType = EmbedType.MEMORY,
    filepath: str | None = None,
) -> MemorySearchResult:
    return MemorySearchResult(
        payload=_payload(
            content,
            group_id,
            embed_type=embed_type,
            filepath=filepath,
        ),
        score=score,
    )


def _msg(role: MessageRole, content: str, seq: int) -> SessionMessage:
    return SessionMessage(
        id=str(uuid4()),
        session_id="sess",
        role=role,
        content=content,
        file_ref=None,
        is_compact=False,
        sequence=seq,
        created_at="2026-04-16T00:00:00Z",
    )


def _reader(
    *,
    cfg: MemoryRetrievalConfig | None = None,
    store_results: list[MemorySearchResult],
    provider_text: str | None = None,
    provider_exc: BaseException | None = None,
) -> tuple[LiveMemoryReader, _FakeStore, _FakeProvider]:
    cfg = cfg or MemoryRetrievalConfig()
    provider = _FakeProvider(output_text=provider_text, exc=provider_exc)
    qb = MemoryQueryBuilder(
        provider=provider, prompt=QUERY_PROMPT, cfg=MemoryOperationModelConfig()
    )
    store = _FakeStore(store_results)
    embedder = _FakeEmbedder()
    return LiveMemoryReader(
        embedder=embedder, store=store, query_builder=qb, cfg=cfg
    ), store, provider


# augment_instructions -------------------------------------------------------


async def test_augment_returns_empty_when_disabled() -> None:
    cfg = MemoryRetrievalConfig(enabled=False)
    reader, store, _ = _reader(cfg=cfg, store_results=[_result("x", 0.9)])
    out = await reader.augment_instructions(
        session_id="sess",
        recent_messages=[_msg(MessageRole.USER, "hi", 1)],
        latest_user_text="hi",
        agent_model="gpt-5-mini",
    )
    assert out == ""
    assert store.calls == []


async def test_augment_returns_empty_when_latest_text_empty() -> None:
    reader, store, _ = _reader(store_results=[_result("x", 0.9)])
    out = await reader.augment_instructions(
        session_id="sess",
        recent_messages=[],
        latest_user_text="",
        agent_model="gpt-5-mini",
    )
    assert out == ""
    assert store.calls == []


async def test_augment_skips_search_when_query_builder_says_skip() -> None:
    skip_text = '{"query":"","should_skip":true,"reasoning":"greeting"}'
    reader, store, _ = _reader(store_results=[_result("x", 0.9)], provider_text=skip_text)
    out = await reader.augment_instructions(
        session_id="sess",
        recent_messages=[_msg(MessageRole.USER, "hi", 1)],
        latest_user_text="hi",
        agent_model="gpt-5-mini",
    )
    assert out == ""
    assert store.calls == []


async def test_augment_returns_empty_when_search_yields_nothing() -> None:
    reader, store, _ = _reader(store_results=[])
    out = await reader.augment_instructions(
        session_id="sess",
        recent_messages=[_msg(MessageRole.USER, "hi", 1)],
        latest_user_text="hi",
        agent_model="gpt-5-mini",
    )
    assert out == ""
    # search WAS called (we wanted to retrieve)
    assert store.calls and store.calls[0]["session_id"] is None


async def test_augment_returns_block_with_top_results() -> None:
    reader, store, _ = _reader(
        store_results=[
            _result("the user prefers Python", 0.9),
            _result("the user is a data scientist", 0.85),
        ]
    )
    out = await reader.augment_instructions(
        session_id="sess",
        recent_messages=[_msg(MessageRole.USER, "code review please", 1)],
        latest_user_text="code review please",
        agent_model="gpt-5-mini",
    )
    assert "Relevant memory" in out
    assert "the user prefers Python" in out
    assert "the user is a data scientist" in out


async def test_augment_filters_by_threshold_and_dedupes_by_group_id() -> None:
    """Multi-chunk results from the same group_id collapse to one entry."""
    gid = uuid4()
    reader, store, _ = _reader(
        cfg=MemoryRetrievalConfig(score_threshold=0.6, top_k_prompt_injection=5),
        store_results=[
            _result("chunk-a", 0.9, group_id=gid),
            _result("chunk-b", 0.85, group_id=gid),
            _result("other", 0.4),  # below threshold
        ],
    )
    out = await reader.augment_instructions(
        session_id="sess",
        recent_messages=[_msg(MessageRole.USER, "q", 1)],
        latest_user_text="q",
        agent_model="gpt-5-mini",
    )
    assert "chunk-a" in out  # higher-scored chunk wins
    assert "chunk-b" not in out
    assert "other" not in out


async def test_augment_honors_retrieval_timeout(monkeypatch) -> None:
    """If the read path exceeds retrieval_timeout_s, return empty (fail-open)."""

    class _SlowStore(BaseVectorStore):
        async def ensure_schema(self) -> None: ...
        async def upsert_group(self, *a, **k) -> None: ...
        async def delete_group(self, *a, **k) -> None: ...

        async def search(self, **k) -> list[MemorySearchResult]:
            await asyncio.sleep(1.0)
            return []

        async def latest_memory_for_session(self, sid):  # pragma: no cover
            return None

    cfg = MemoryRetrievalConfig(retrieval_timeout_s=0.05)
    provider = _FakeProvider()
    qb = MemoryQueryBuilder(
        provider=provider, prompt=QUERY_PROMPT, cfg=MemoryOperationModelConfig()
    )
    reader = LiveMemoryReader(
        embedder=_FakeEmbedder(), store=_SlowStore(), query_builder=qb, cfg=cfg
    )
    out = await reader.augment_instructions(
        session_id="sess",
        recent_messages=[_msg(MessageRole.USER, "q", 1)],
        latest_user_text="q",
        agent_model="gpt-5-mini",
    )
    assert out == ""


# recall ----------------------------------------------------------------------


async def test_recall_with_session_scoped_filters_by_session() -> None:
    reader, store, _ = _reader(store_results=[_result("x", 0.9)])
    out = await reader.recall(
        query="anything",
        top_k=3,
        score_threshold=0.5,
        session_id="sess-123",
        owner=MemoryOwner.USER,
    )
    assert len(out) == 1
    assert store.calls[0]["session_id"] == "sess-123"


async def test_recall_default_is_cross_session() -> None:
    reader, store, _ = _reader(store_results=[_result("x", 0.9)])
    await reader.recall(
        query="anything",
        top_k=3,
        score_threshold=0.5,
        session_id=None,
        owner=MemoryOwner.USER,
    )
    assert store.calls[0]["session_id"] is None


async def test_recall_searches_attachment_embedding_types() -> None:
    reader, store, _ = _reader(
        store_results=[
            _result(
                "attachment text",
                0.9,
                embed_type=EmbedType.ATTACHMENT_TEXT,
                filepath=".feather/attachments/s/report.txt",
            )
        ]
    )

    out = await reader.recall(
        query="report",
        top_k=3,
        score_threshold=0.5,
        session_id="sess-123",
        owner=MemoryOwner.USER,
    )

    searched_types = {call["filters"]["type"] for call in store.calls}
    assert EmbedType.MEMORY.value in searched_types
    assert EmbedType.ATTACHMENT_TEXT.value in searched_types
    assert EmbedType.ATTACHMENT_PDF.value in searched_types
    assert EmbedType.ATTACHMENT_IMAGE.value in searched_types
    assert out[0].payload.filepath == ".feather/attachments/s/report.txt"


# format_memory_block --------------------------------------------------------


def test_format_memory_block_includes_query_and_purposes() -> None:
    block = format_memory_block(
        results=[_result("the user prefers Python", 0.92)],
        cfg=MemoryRetrievalConfig(),
        query="the user's coding preferences",
    )
    assert "the user's coding preferences" in block
    assert "Relevant memory" in block
    assert "0.92" in block
    assert "the user prefers Python" in block
    assert "for tests" in block  # purpose field


def test_format_memory_block_includes_attachment_source_path() -> None:
    block = format_memory_block(
        results=[
            _result(
                "attachment text",
                0.92,
                embed_type=EmbedType.ATTACHMENT_TEXT,
                filepath=".feather/attachments/s/report.txt",
            )
        ],
        cfg=MemoryRetrievalConfig(),
        query="report",
    )

    assert "source file: .feather/attachments/s/report.txt" in block


def test_format_memory_block_returns_clear_markers() -> None:
    block = format_memory_block(
        results=[_result("x", 0.5)], cfg=MemoryRetrievalConfig(), query="q"
    )
    assert block.startswith("## Relevant memory")
    assert "End of memory block" in block


# NoOpMemoryReader -----------------------------------------------------------


async def test_noop_reader_returns_empty_string_for_augment() -> None:
    reader = NoOpMemoryReader()
    out = await reader.augment_instructions(
        session_id="sess",
        recent_messages=[],
        latest_user_text="anything",
        agent_model="gpt-5-mini",
    )
    assert out == ""


async def test_noop_reader_returns_empty_list_for_recall() -> None:
    reader = NoOpMemoryReader()
    out = await reader.recall(
        query="x",
        top_k=10,
        score_threshold=0.5,
        session_id=None,
        owner=MemoryOwner.USER,
    )
    assert out == []
