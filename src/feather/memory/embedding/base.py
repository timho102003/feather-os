"""Embedding-provider protocol.

All memory components depend only on this interface so the concrete provider
(Gemini today) can be swapped without touching the write/read path. Tests
inject a :class:`FakeEmbeddingProvider` (defined per-test) that implements
the same two methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class BaseEmbeddingProvider(ABC):
    """Abstract embedding provider.

    Implementations must distinguish documents (indexed content) from queries
    (search inputs) so asymmetric task-type routing is preserved — Gemini
    retrieval quality depends on this split per the ``gemini-embedding`` skill.
    """

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many document texts.

        Args:
            texts: Strings to embed. Implementations filter out empty/blank
                strings; a fully empty input must raise.

        Returns:
            One vector per non-empty input, in input order.
        """

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed one query text.

        Args:
            text: The natural-language query to embed.

        Returns:
            A single embedding vector.
        """
