"""Tests for GeminiEmbeddingProvider using a fake google-genai client.

The tests do not import ``google-genai`` — we stub its surface exactly where
``GeminiEmbeddingProvider`` touches it. ``ResourceExhausted`` and
``InvalidArgument`` are real exception classes from the ``google-api-core``
package but the provider catches them by type; for tests we substitute
lookalike classes that inherit from the real ones when available, else
from a local base so the retry path is exercised deterministically.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

import pytest

from feather.memory.config import MemoryEmbeddingConfig
from feather.memory.embedding.base import BaseEmbeddingProvider
from feather.memory.embedding.gemini import GeminiEmbeddingProvider


# Fake google-genai shape ----------------------------------------------------


@dataclass
class _FakeEmbedding:
    values: list[float]


@dataclass
class _FakeEmbedResponse:
    embeddings: list[_FakeEmbedding]


class _FakeGeminiModels:
    """Matches the surface of ``genai.Client().models``."""

    def __init__(self, outer: "_FakeGeminiClient") -> None:
        self._outer = outer

    def embed_content(
        self,
        *,
        model: str,
        contents: Any,
        config: Any,
    ) -> _FakeEmbedResponse:
        self._outer.calls.append({"model": model, "contents": contents, "config": config})
        if self._outer.raise_times > 0:
            self._outer.raise_times -= 1
            raise self._outer.raise_exc or RuntimeError("simulated failure")
        texts: list[str] = contents if isinstance(contents, list) else [contents]
        dims = getattr(config, "output_dimensionality", 0) or self._outer.fixed_dims
        # Produce deterministic non-unit vectors so we can verify normalization.
        vectors: list[_FakeEmbedding] = []
        for i, _ in enumerate(texts):
            v = [float(i + 1)] * dims
            vectors.append(_FakeEmbedding(values=v))
        return _FakeEmbedResponse(embeddings=vectors)


class _FakeGeminiClient:
    """Matches the surface of ``genai.Client``."""

    def __init__(self, fixed_dims: int = 3072) -> None:
        self.calls: list[dict[str, Any]] = []
        self.raise_times: int = 0
        self.raise_exc: BaseException | None = None
        self.fixed_dims = fixed_dims
        self.models = _FakeGeminiModels(self)


# Helpers --------------------------------------------------------------------


def _cfg(**overrides: Any) -> MemoryEmbeddingConfig:
    base = MemoryEmbeddingConfig(
        provider="gemini",
        model="gemini-embedding-2-preview",
        output_dimensionality=3072,
        request_timeout_s=0.5,
        max_retries=3,
        retry_backoff_s=1.001,  # keep sleep tiny in tests
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _make(client: _FakeGeminiClient, cfg: MemoryEmbeddingConfig | None = None) -> GeminiEmbeddingProvider:
    """Construct the provider with an injected fake client."""
    return GeminiEmbeddingProvider(cfg=cfg or _cfg(), client=client)


# Protocol conformance -------------------------------------------------------


def test_gemini_provider_is_base_embedding_provider() -> None:
    client = _FakeGeminiClient()
    provider = _make(client)
    assert isinstance(provider, BaseEmbeddingProvider)


# embed_documents / embed_query ----------------------------------------------


async def test_embed_documents_uses_retrieval_document_task_type() -> None:
    client = _FakeGeminiClient()
    provider = _make(client)
    out = await provider.embed_documents(["a", "b"])
    assert len(out) == 2
    call = client.calls[0]
    assert call["contents"] == ["a", "b"]
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert call["config"].output_dimensionality == 3072


async def test_embed_query_uses_retrieval_query_task_type_and_returns_flat_list() -> None:
    client = _FakeGeminiClient()
    provider = _make(client)
    vec = await provider.embed_query("hello")
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)
    call = client.calls[0]
    assert call["contents"] == ["hello"]
    assert call["config"].task_type == "RETRIEVAL_QUERY"


async def test_empty_and_whitespace_strings_are_filtered_before_call() -> None:
    """Gemini API rejects empty strings; the provider filters them out."""
    client = _FakeGeminiClient()
    provider = _make(client)
    out = await provider.embed_documents(["hello", "", "   ", "world"])
    # Only two non-empty strings reach the API.
    assert client.calls[0]["contents"] == ["hello", "world"]
    assert len(out) == 2


async def test_all_empty_input_raises_before_api_call() -> None:
    client = _FakeGeminiClient()
    provider = _make(client)
    with pytest.raises(ValueError):
        await provider.embed_documents(["", "   "])
    assert client.calls == []


# Batching --------------------------------------------------------------------


async def test_batches_larger_than_one_hundred_are_split() -> None:
    client = _FakeGeminiClient()
    provider = _make(client)
    texts = [f"t{i}" for i in range(250)]
    out = await provider.embed_documents(texts)
    assert len(out) == 250
    # 250 / 100 → 3 API calls: 100 + 100 + 50
    assert [len(call["contents"]) for call in client.calls] == [100, 100, 50]


# Retry / timeout -------------------------------------------------------------


class _FakeResourceExhausted(Exception):
    """Stand-in for google.api_core.exceptions.ResourceExhausted."""


async def test_retries_on_resource_exhausted_and_eventually_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two failures → third try succeeds."""
    # Monkey-patch the ResourceExhausted class the provider catches.
    from feather.memory.embedding import gemini as gem

    monkeypatch.setattr(gem, "ResourceExhausted", _FakeResourceExhausted)
    # Speed up backoff.
    _real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_: _real_sleep(0))

    client = _FakeGeminiClient()
    client.raise_times = 2
    client.raise_exc = _FakeResourceExhausted("rate")
    provider = _make(client)
    out = await provider.embed_documents(["one"])
    assert len(out) == 1
    assert len(client.calls) == 3  # 2 failures + 1 success


async def test_retry_budget_exhausted_re_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from feather.memory.embedding import gemini as gem

    monkeypatch.setattr(gem, "ResourceExhausted", _FakeResourceExhausted)
    _real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_: _real_sleep(0))

    client = _FakeGeminiClient()
    client.raise_times = 100
    client.raise_exc = _FakeResourceExhausted("rate")
    provider = _make(client, _cfg(max_retries=2))
    with pytest.raises(_FakeResourceExhausted):
        await provider.embed_documents(["one"])
    assert len(client.calls) == 2


# L2 normalization -----------------------------------------------------------


async def test_embeddings_are_l2_normalized_when_dims_below_three_thousand_seventy_two() -> None:
    """For reduced-dim Matryoshka outputs, the Gemini API does NOT normalize — we must."""
    client = _FakeGeminiClient(fixed_dims=768)
    provider = _make(client, _cfg(output_dimensionality=768, normalize_reduced_dims=True))
    out = await provider.embed_documents(["a", "b"])
    for v in out:
        norm = math.sqrt(sum(x * x for x in v))
        assert math.isclose(norm, 1.0, abs_tol=1e-6)


async def test_embeddings_pass_through_untouched_at_three_thousand_seventy_two() -> None:
    """At 3072 the API already normalizes; the provider must NOT re-normalize/mutate."""
    client = _FakeGeminiClient(fixed_dims=3072)
    provider = _make(client, _cfg(output_dimensionality=3072, normalize_reduced_dims=True))
    out = await provider.embed_documents(["a"])
    # Our fake returns non-unit vectors; the provider should NOT touch them at 3072.
    assert out[0][0] == 1.0
    assert len(out[0]) == 3072


async def test_embeddings_not_normalized_when_flag_off() -> None:
    client = _FakeGeminiClient(fixed_dims=768)
    provider = _make(client, _cfg(output_dimensionality=768, normalize_reduced_dims=False))
    out = await provider.embed_documents(["a"])
    # Un-normalized vector of [1]*768 has norm sqrt(768) ≈ 27.7
    norm = math.sqrt(sum(x * x for x in out[0]))
    assert norm > 2.0


# Timeout --------------------------------------------------------------------


async def test_timeout_is_enforced_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow client call must be timed out at ``request_timeout_s``."""
    from feather.memory.embedding import gemini as gem

    # Make sleep a no-op inside retry backoff
    _real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_: _real_sleep(0))

    slow_event = asyncio.Event()

    def slow_embed(**kwargs: Any) -> Any:  # runs in a thread
        # Block until slow_event is set; tests will never set it.
        import time

        time.sleep(2)
        return _FakeEmbedResponse(embeddings=[])

    client = _FakeGeminiClient()
    client.models.embed_content = slow_embed  # type: ignore[method-assign]

    # Low timeout so the test doesn't hang long.
    provider = _make(client, _cfg(request_timeout_s=0.05, max_retries=2))

    # After retries exhaust, the last TimeoutError is re-raised.
    with pytest.raises(asyncio.TimeoutError):
        await provider.embed_documents(["slow"])
    # Don't leave anything blocked.
    slow_event.set()
