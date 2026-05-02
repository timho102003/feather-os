"""Tests for token estimators."""

from __future__ import annotations

from typing import Any

import pytest

from feather.memory.tokenizer import (
    CharApproxEstimator,
    GeminiEstimator,
    TiktokenEstimator,
    build_estimator,
)


# TiktokenEstimator -----------------------------------------------------------


def test_tiktoken_estimator_counts_tokens_deterministically() -> None:
    """count must be deterministic across calls for the same text."""
    est = TiktokenEstimator("o200k_base")
    text = "the user prefers Python for async code"
    first = est.count(text)
    second = est.count(text)
    assert first == second
    assert first > 0


def test_tiktoken_estimator_longer_text_has_more_tokens() -> None:
    """Longer strings must have strictly more tokens (monotonic)."""
    est = TiktokenEstimator("o200k_base")
    short = est.count("hi")
    long = est.count("the quick brown fox jumped over the lazy dog " * 10)
    assert long > short


def test_tiktoken_estimator_handles_empty_string() -> None:
    """Empty string → 0 tokens (not a crash). Chunker's fast path relies on this."""
    est = TiktokenEstimator("o200k_base")
    assert est.count("") == 0


def test_tiktoken_estimator_exposes_encoding_for_chunker() -> None:
    """Chunker uses TiktokenEstimator.encoding for token-accurate slicing."""
    est = TiktokenEstimator("o200k_base")
    ids = est.encoding.encode("hello world")
    assert ids
    assert isinstance(ids, list)


# CharApproxEstimator ---------------------------------------------------------


def test_char_approx_estimator_returns_len_over_four_with_min_one() -> None:
    est = CharApproxEstimator()
    assert est.count("") == 1
    assert est.count("abcd") == 1
    assert est.count("a" * 16) == 4
    assert est.count("a" * 17) == 4  # integer floor


# GeminiEstimator -------------------------------------------------------------


class _FakeGeminiClient:
    """Stub with the minimum surface GeminiEstimator touches."""

    def __init__(self, total_tokens: int) -> None:
        self._total = total_tokens
        self.calls: list[dict[str, Any]] = []

        class _Models:
            def __init__(inner) -> None:  # noqa: N805 — pytest-style inner class
                inner.outer = self

            def count_tokens(inner, *, model: str, contents: str):  # noqa: N805
                inner.outer.calls.append({"model": model, "contents": contents})
                return type("Resp", (), {"total_tokens": inner.outer._total})()

        self.models = _Models()


def test_gemini_estimator_delegates_to_count_tokens_with_model_and_content() -> None:
    fake = _FakeGeminiClient(total_tokens=42)
    est = GeminiEstimator(fake, "gemini-embedding-2-preview")
    assert est.count("any text") == 42
    assert fake.calls == [
        {"model": "gemini-embedding-2-preview", "contents": "any text"}
    ]


# build_estimator -------------------------------------------------------------


def test_build_estimator_dispatches_by_key() -> None:
    from feather.memory.config import MemoryChunkingConfig

    cfg_tik = MemoryChunkingConfig(tokenizer="tiktoken", tokenizer_encoding="o200k_base")
    cfg_char = MemoryChunkingConfig(tokenizer="char4")
    cfg_gem = MemoryChunkingConfig(tokenizer="gemini")

    assert isinstance(build_estimator(cfg_tik), TiktokenEstimator)
    assert isinstance(build_estimator(cfg_char), CharApproxEstimator)
    assert isinstance(
        build_estimator(cfg_gem, gemini_client=_FakeGeminiClient(1), embed_model="m"),
        GeminiEstimator,
    )


def test_build_estimator_raises_on_unknown_key() -> None:
    from feather.memory.config import MemoryChunkingConfig

    with pytest.raises(ValueError):
        build_estimator(MemoryChunkingConfig(tokenizer="whatever"))


def test_build_estimator_gemini_without_client_raises() -> None:
    """Gemini estimator needs a client+model; absent either → clear error."""
    from feather.memory.config import MemoryChunkingConfig

    with pytest.raises(ValueError):
        build_estimator(MemoryChunkingConfig(tokenizer="gemini"))
