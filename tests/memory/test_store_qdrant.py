"""Tests for QdrantVectorStore against an in-memory Qdrant instance."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from feather.memory.config import MemoryQdrantConfig
from feather.memory.enums import EmbedType, MemoryOwner
from feather.memory.models import MemoryPointPayload
from feather.memory.store.base import BaseVectorStore
from feather.memory.store.qdrant import MemorySchemaError, QdrantVectorStore


def _cfg(**overrides: object) -> MemoryQdrantConfig:
    base = MemoryQdrantConfig(
        collection_name=f"test_{uuid4().hex[:8]}",
        embedding_dims=8,  # tiny for tests
        hnsw_m=4,
        hnsw_ef_construct=16,
        hnsw_ef_search=16,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _payload(
    *,
    content: str = "something",
    group_id: UUID | None = None,
    session_id: UUID | None = None,
    owner: MemoryOwner = MemoryOwner.USER,
    etype: EmbedType = EmbedType.MEMORY,
    created_at: datetime | None = None,
) -> MemoryPointPayload:
    return MemoryPointPayload(
        type=etype,
        memory_owner=owner,
        content=content,
        purpose="test purpose",
        group_id=group_id or uuid4(),
        session_id=session_id or uuid4(),
        start_message_id=uuid4(),
        end_message_id=uuid4(),
        created_at=created_at or datetime.now(timezone.utc),
    )


def _vec(seed: float = 1.0, dims: int = 8) -> list[float]:
    """Return a deterministic unit-ish vector — distinct seeds give distinct angles."""
    import math

    raw = [seed * (i + 1) for i in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


@pytest.fixture
async def client() -> AsyncQdrantClient:
    c = AsyncQdrantClient(location=":memory:")
    try:
        yield c
    finally:
        await c.close()


# ensure_schema --------------------------------------------------------------


async def test_ensure_schema_creates_collection_and_is_idempotent(client: AsyncQdrantClient) -> None:
    cfg = _cfg()
    store = QdrantVectorStore(client=client, cfg=cfg)

    await store.ensure_schema()
    assert await client.collection_exists(cfg.collection_name)

    # Second call is a no-op, must not raise.
    await store.ensure_schema()
    assert await client.collection_exists(cfg.collection_name)


async def test_ensure_schema_raises_memory_schema_error_on_dim_mismatch(
    client: AsyncQdrantClient,
) -> None:
    cfg = _cfg(embedding_dims=8)
    store = QdrantVectorStore(client=client, cfg=cfg)
    await store.ensure_schema()

    # Reconfigure with different dims; re-running ensure_schema must refuse.
    cfg_bad = _cfg(embedding_dims=16, collection_name=cfg.collection_name)
    store_bad = QdrantVectorStore(client=client, cfg=cfg_bad)
    with pytest.raises(MemorySchemaError):
        await store_bad.ensure_schema()


# upsert / search / delete ---------------------------------------------------


async def test_upsert_and_search_round_trip(client: AsyncQdrantClient) -> None:
    cfg = _cfg()
    store = QdrantVectorStore(client=client, cfg=cfg)
    await store.ensure_schema()

    target_payload = _payload(content="the user prefers Python")
    other_payload = _payload(content="the user likes coffee")
    await store.upsert_group([target_payload], [_vec(1.0)])
    await store.upsert_group([other_payload], [_vec(-1.0)])

    hits = await store.search(
        query=_vec(1.0),
        top_k=5,
        filters={"type": "memory", "memory_owner": "user"},
    )
    assert len(hits) == 2
    # Highest similarity first; the content match should be our target.
    assert hits[0].payload.content == "the user prefers Python"
    assert hits[0].score > hits[1].score


async def test_search_filters_by_session_id(client: AsyncQdrantClient) -> None:
    cfg = _cfg()
    store = QdrantVectorStore(client=client, cfg=cfg)
    await store.ensure_schema()

    session_a = uuid4()
    session_b = uuid4()
    await store.upsert_group([_payload(session_id=session_a, content="A")], [_vec(1.0)])
    await store.upsert_group([_payload(session_id=session_b, content="B")], [_vec(1.0)])

    hits_a = await store.search(
        query=_vec(1.0),
        top_k=5,
        filters={"type": "memory", "memory_owner": "user"},
        session_id=str(session_a),
    )
    assert len(hits_a) == 1
    assert hits_a[0].payload.content == "A"


async def test_delete_group_removes_all_chunks_of_that_group(client: AsyncQdrantClient) -> None:
    cfg = _cfg()
    store = QdrantVectorStore(client=client, cfg=cfg)
    await store.ensure_schema()

    target_gid = uuid4()
    other_gid = uuid4()
    await store.upsert_group(
        [
            _payload(group_id=target_gid, content="chunk-1"),
            _payload(group_id=target_gid, content="chunk-2"),
        ],
        [_vec(1.0), _vec(1.2)],
    )
    await store.upsert_group(
        [_payload(group_id=other_gid, content="survivor")],
        [_vec(-1.0)],
    )

    await store.delete_group(target_gid)

    hits = await store.search(
        query=_vec(1.0),
        top_k=10,
        filters={"type": "memory", "memory_owner": "user"},
    )
    contents = sorted(h.payload.content for h in hits)
    assert contents == ["survivor"]


# latest_memory_for_session --------------------------------------------------


async def test_latest_memory_for_session_returns_none_when_empty(client: AsyncQdrantClient) -> None:
    cfg = _cfg()
    store = QdrantVectorStore(client=client, cfg=cfg)
    await store.ensure_schema()
    assert await store.latest_memory_for_session(str(uuid4())) is None


async def test_latest_memory_for_session_returns_most_recent_by_created_at(
    client: AsyncQdrantClient,
) -> None:
    cfg = _cfg()
    store = QdrantVectorStore(client=client, cfg=cfg)
    await store.ensure_schema()

    session = uuid4()
    now = datetime.now(timezone.utc)
    older = _payload(session_id=session, content="older", created_at=now - timedelta(minutes=10))
    newer = _payload(session_id=session, content="newer", created_at=now)
    await store.upsert_group([older], [_vec(1.0)])
    await store.upsert_group([newer], [_vec(1.0)])

    latest = await store.latest_memory_for_session(str(session))
    assert latest is not None
    assert latest.content == "newer"


async def test_latest_memory_for_session_ignores_other_sessions(client: AsyncQdrantClient) -> None:
    cfg = _cfg()
    store = QdrantVectorStore(client=client, cfg=cfg)
    await store.ensure_schema()

    session_a = uuid4()
    session_b = uuid4()
    await store.upsert_group(
        [_payload(session_id=session_a, content="A")], [_vec(1.0)]
    )
    await store.upsert_group(
        [_payload(session_id=session_b, content="B")], [_vec(1.0)]
    )

    out = await store.latest_memory_for_session(str(session_a))
    assert out is not None
    assert out.content == "A"


# Protocol conformance -------------------------------------------------------


def test_qdrant_store_is_base_vector_store() -> None:
    class _StubClient:
        pass

    store = QdrantVectorStore(client=_StubClient(), cfg=_cfg())
    assert isinstance(store, BaseVectorStore)
