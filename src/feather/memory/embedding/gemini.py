"""Gemini embedding provider.

Wraps the synchronous ``google-genai`` client inside an async facade so the
agent's event loop never blocks on an embedding call. Supports batching to
Gemini's 100-items-per-call limit, retry with exponential backoff on
``ResourceExhausted`` and timeouts, and L2 normalization for Matryoshka
reduced-dim outputs (the API normalizes the full 3072-d output but not
smaller ones — missing this is the most common integration bug per the
``gemini-embedding`` skill).
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Iterator, Sequence

try:
    from google import genai  # type: ignore[import-not-found]
    from google.genai import types as _genai_types  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — the package is a declared dep
    genai = None  # type: ignore[assignment]
    _genai_types = None  # type: ignore[assignment]

try:
    from google.api_core.exceptions import ResourceExhausted  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — fallback stub for stripped envs

    class ResourceExhausted(Exception):  # type: ignore[no-redef]
        """Fallback when google.api_core isn't importable."""

from feather.memory.config import MemoryEmbeddingConfig
from feather.memory.embedding.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

_MAX_BATCH = 100


def _l2_normalize(vector: list[float]) -> list[float]:
    """Return the unit-norm version of ``vector`` (no-op if norm is zero)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    inv = 1.0 / norm
    return [x * inv for x in vector]


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield consecutive batches of at most ``size`` items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Async Gemini embedding adapter.

    Construction:
    - ``client``: inject a pre-built ``genai.Client`` (used by tests).
    - ``api_key``: build a ``genai.Client(api_key=...)`` internally.
    Exactly one must be provided.
    """

    def __init__(
        self,
        *,
        cfg: MemoryEmbeddingConfig,
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError(
                    "GeminiEmbeddingProvider requires either a client or an api_key"
                )
            if genai is None:  # pragma: no cover — declared dep
                raise RuntimeError("google-genai is not installed")
            client = genai.Client(api_key=api_key)
        self._cfg = cfg
        self._client = client

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(list(texts), self._cfg.task_type_document)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text], self._cfg.task_type_query)
        return vectors[0]

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Filter empty strings, batch to ``_MAX_BATCH``, and call the API.

        Args:
            texts: Input strings (may contain empty/whitespace entries).
            task_type: Gemini task-type literal.

        Returns:
            Vectors in the order of non-empty inputs.

        Raises:
            ValueError: If the input contains only empty/whitespace strings.
            ResourceExhausted: On retry budget exhaustion.
            asyncio.TimeoutError: On retry budget exhaustion after timeouts.
        """
        cleaned = [t.strip() for t in texts if t is not None and t.strip()]
        if not cleaned:
            raise ValueError("embed called with no non-empty strings")
        vectors: list[list[float]] = []
        for batch in _batched(cleaned, _MAX_BATCH):
            vectors.extend(await self._call_with_retry(batch, task_type))
        return vectors

    async def _call_with_retry(self, batch: list[str], task_type: str) -> list[list[float]]:
        """Invoke the API with exponential backoff on known-retryable errors."""
        config = _build_embed_config(task_type, self._cfg.output_dimensionality)
        last_exc: BaseException | None = None
        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._client.models.embed_content,
                        model=self._cfg.model,
                        contents=batch,
                        config=config,
                    ),
                    timeout=self._cfg.request_timeout_s,
                )
            except ResourceExhausted as exc:
                last_exc = exc
                if attempt >= self._cfg.max_retries:
                    raise
                await asyncio.sleep(self._cfg.retry_backoff_s ** attempt)
                continue
            except asyncio.TimeoutError as exc:
                last_exc = exc
                if attempt >= self._cfg.max_retries:
                    raise
                await asyncio.sleep(self._cfg.retry_backoff_s ** attempt)
                continue
            vectors = [list(emb.values) for emb in response.embeddings]
            if (
                self._cfg.output_dimensionality < 3072
                and self._cfg.normalize_reduced_dims
            ):
                vectors = [_l2_normalize(v) for v in vectors]
            return vectors
        # Unreachable: the loop either returns or raises.
        if last_exc is not None:  # pragma: no cover
            raise last_exc
        raise RuntimeError("unreachable")  # pragma: no cover


def _build_embed_config(task_type: str, output_dimensionality: int) -> Any:
    """Construct the ``EmbedContentConfig`` — shimmed so tests can stub genai."""
    if _genai_types is None:  # pragma: no cover
        # Minimal duck-typed stand-in for tests that don't install google-genai.
        from types import SimpleNamespace

        return SimpleNamespace(task_type=task_type, output_dimensionality=output_dimensionality)
    return _genai_types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=output_dimensionality,
    )
