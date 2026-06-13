"""Vector-store protocol.

All memory components depend only on this interface so the concrete backend
(Qdrant today) can be swapped or faked. Tests use a ``FakeVectorStore`` built
per-test; the service layer never touches ``AsyncQdrantClient`` directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
from uuid import UUID

from feather.memory.models import MemoryPointPayload, MemorySearchResult


class BaseVectorStore(ABC):
    """Abstract Qdrant-like vector store.

    ``ensure_schema`` must be idempotent and fail-closed on schema drift
    (e.g. vector dim mismatch) — silent coercion would corrupt retrieval
    without warning.
    """

    @abstractmethod
    async def ensure_schema(self) -> None:
        """Create the collection + payload indexes if absent, or validate
        that an existing collection matches the configured dims / distance."""

    @abstractmethod
    async def upsert_group(
        self,
        payloads: Sequence[MemoryPointPayload],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Insert-or-replace a batch of points. Must write synchronously
        (``wait=True`` or equivalent) so subsequent reads see the change."""

    @abstractmethod
    async def delete_group(self, group_id: UUID) -> None:
        """Delete every point whose payload ``group_id`` matches."""

    @abstractmethod
    async def search(
        self,
        *,
        query: Sequence[float],
        top_k: int,
        filters: dict[str, str],
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        """Run a nearest-neighbor search with optional exact-match filters.

        Args:
            query: The query vector.
            top_k: Maximum hits to return.
            filters: Exact-match constraints (``field → value``) combined
                as an AND ``must`` clause.
            session_id: Optional filter on ``payload.session_id``; ``None``
                means cross-session search (the default for user memory).
        """

    @abstractmethod
    async def latest_memory_for_session(
        self, session_id: str
    ) -> MemoryPointPayload | None:
        """Return the most recently ``created_at`` memory for ``session_id``."""

    async def aclose(self) -> None:
        """Release resources held by this store.

        Concrete no-op — subclasses override when they own a closable client.
        Defined here (non-abstract) so existing fakes and third-party
        implementations don't break.
        """
