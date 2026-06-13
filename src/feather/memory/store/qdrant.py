"""Qdrant-backed vector store for memory points.

Wraps :class:`qdrant_client.AsyncQdrantClient`. Every public method enforces
the :class:`MemoryPointPayload` schema on the way in and out — callers never
see raw Qdrant model types.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from feather.memory.config import MemoryQdrantConfig
from feather.memory.enums import EmbedType
from feather.memory.models import MemoryPointPayload, MemorySearchResult
from feather.memory.store.base import BaseVectorStore

logger = logging.getLogger(__name__)


class MemorySchemaError(RuntimeError):
    """Raised when an existing Qdrant collection doesn't match the configured schema.

    This is fail-closed by design: silent coercion on dim mismatch would hide a
    model change that invalidated the entire vector space.
    """


class QdrantVectorStore(BaseVectorStore):
    """Async Qdrant adapter."""

    def __init__(self, *, client: AsyncQdrantClient, cfg: MemoryQdrantConfig) -> None:
        self._client = client
        self._cfg = cfg

    @property
    def client(self) -> AsyncQdrantClient:
        return self._client

    async def aclose(self) -> None:
        """Close the underlying Qdrant client if it supports ``close``."""
        closer = getattr(self._client, "close", None)
        if closer is None:
            return
        try:
            await closer()
        except Exception:  # noqa: BLE001
            logger.exception("memory.store.close_error")

    # -- ensure_schema --------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Idempotently create or validate the memory collection + payload indexes."""
        collection = self._cfg.collection_name
        exists = await self._client.collection_exists(collection)
        if not exists:
            await self._create_collection()
            await self._create_payload_indexes()
            logger.info(
                "memory.store.schema.created",
                extra={"collection": collection, "dims": self._cfg.embedding_dims},
            )
            return

        info = await self._client.get_collection(collection)
        existing_dims = _extract_vector_size(info)
        if existing_dims != self._cfg.embedding_dims:
            raise MemorySchemaError(
                f"collection {collection!r} has vector size {existing_dims} but config is "
                f"{self._cfg.embedding_dims}. Re-embedding required. See qdrant-model-migration."
            )
        # Idempotently ensure indexes exist — Qdrant treats re-create as a no-op.
        await self._create_payload_indexes(ignore_errors=True)

    async def _create_collection(self) -> None:
        await self._client.create_collection(
            collection_name=self._cfg.collection_name,
            vectors_config=models.VectorParams(
                size=self._cfg.embedding_dims,
                distance=models.Distance.COSINE,
                on_disk=self._cfg.on_disk_vectors,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=self._cfg.hnsw_m,
                ef_construct=self._cfg.hnsw_ef_construct,
                full_scan_threshold=self._cfg.hnsw_full_scan_threshold,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=self._cfg.default_segment_number,
                indexing_threshold=self._cfg.indexing_threshold,
            ),
            on_disk_payload=self._cfg.on_disk_payload,
        )

    async def _create_payload_indexes(self, *, ignore_errors: bool = False) -> None:
        collection = self._cfg.collection_name

        async def safe(
            field_name: str, field_schema: Any
        ) -> None:
            try:
                await self._client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception:
                if not ignore_errors:
                    raise
                logger.debug(
                    "memory.store.index.exists", extra={"field": field_name}
                )

        # Tenancy key — tenant-aware cluster layout.
        await safe(
            "type",
            models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD, is_tenant=True
            ),
        )
        await safe(
            "memory_owner",
            models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
        )
        for uid_field in (
            "session_id",
            "group_id",
            "start_message_id",
            "end_message_id",
        ):
            await safe(
                uid_field,
                models.UuidIndexParams(type=models.UuidIndexType.UUID),
            )
        await safe(
            "created_at",
            models.DatetimeIndexParams(type=models.DatetimeIndexType.DATETIME),
        )
        await safe(
            "chunk_index",
            models.IntegerIndexParams(
                type=models.IntegerIndexType.INTEGER, range=True
            ),
        )

    # -- upsert / delete ------------------------------------------------------

    async def upsert_group(
        self,
        payloads: Sequence[MemoryPointPayload],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(payloads) != len(vectors):
            raise ValueError("payloads and vectors must have the same length")
        if not payloads:
            return
        points = [
            models.PointStruct(
                id=str(payload.id),
                vector=list(vector),
                payload=payload.model_dump(mode="json"),
            )
            for payload, vector in zip(payloads, vectors, strict=True)
        ]
        await self._client.upsert(
            collection_name=self._cfg.collection_name,
            points=points,
            wait=True,
        )

    async def delete_group(self, group_id: UUID) -> None:
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="group_id",
                        match=models.MatchValue(value=str(group_id)),
                    )
                ]
            )
        )
        await self._client.delete(
            collection_name=self._cfg.collection_name,
            points_selector=selector,
            wait=True,
        )

    # -- search ---------------------------------------------------------------

    async def search(
        self,
        *,
        query: Sequence[float],
        top_k: int,
        filters: dict[str, str],
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        must: list[Any] = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filters.items()
        ]
        if session_id is not None:
            must.append(
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                )
            )
        response = await self._client.query_points(
            collection_name=self._cfg.collection_name,
            query=list(query),
            limit=top_k,
            query_filter=models.Filter(must=must),
            search_params=models.SearchParams(
                hnsw_ef=self._cfg.hnsw_ef_search, exact=False
            ),
            with_payload=True,
        )
        return [
            MemorySearchResult(
                payload=MemoryPointPayload.model_validate(pt.payload),
                score=float(pt.score),
            )
            for pt in response.points
        ]

    # -- latest_memory_for_session -------------------------------------------

    async def latest_memory_for_session(
        self, session_id: str
    ) -> MemoryPointPayload | None:
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                ),
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=EmbedType.MEMORY.value),
                ),
            ]
        )
        points, _ = await self._client.scroll(
            collection_name=self._cfg.collection_name,
            scroll_filter=scroll_filter,
            limit=1,
            order_by=models.OrderBy(
                key="created_at", direction=models.Direction.DESC
            ),
            with_payload=True,
        )
        if not points:
            return None
        return MemoryPointPayload.model_validate(points[0].payload)


def _extract_vector_size(info: Any) -> int:
    """Read the vector dimensionality from a Qdrant ``CollectionInfo``.

    Qdrant's ``vectors_config`` can be a single ``VectorParams`` or a named
    mapping. For memory we always use the unnamed single-vector form, but we
    handle both defensively.
    """
    vectors = info.config.params.vectors
    if hasattr(vectors, "size"):
        return int(vectors.size)
    if isinstance(vectors, dict):
        first_key = next(iter(vectors))
        return int(vectors[first_key].size)
    raise MemorySchemaError("unrecognized vectors_config layout on existing collection")
