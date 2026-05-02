"""Tests for the token-aware Chunker."""

from __future__ import annotations

import pytest

from feather.memory.chunker import Chunker, TextChunk
from feather.memory.tokenizer import CharApproxEstimator, TiktokenEstimator


# Construction invariants -----------------------------------------------------


def test_chunker_raises_when_overlap_not_less_than_size() -> None:
    est = TiktokenEstimator("o200k_base")
    with pytest.raises(ValueError):
        Chunker(est, size_tokens=100, overlap_tokens=100)
    with pytest.raises(ValueError):
        Chunker(est, size_tokens=100, overlap_tokens=150)


def test_chunker_raises_on_zero_or_negative_size() -> None:
    est = TiktokenEstimator("o200k_base")
    with pytest.raises(ValueError):
        Chunker(est, size_tokens=0, overlap_tokens=0)


# Single-chunk fast path ------------------------------------------------------


def test_short_text_returns_single_chunk() -> None:
    est = TiktokenEstimator("o200k_base")
    c = Chunker(est, size_tokens=100, overlap_tokens=10)
    out = c.chunk("the user prefers Python for async code")
    assert len(out) == 1
    assert out[0].index == 0
    assert out[0].text == "the user prefers Python for async code"
    assert out[0].token_count > 0


def test_empty_text_returns_single_chunk_with_zero_tokens() -> None:
    est = TiktokenEstimator("o200k_base")
    c = Chunker(est, size_tokens=100, overlap_tokens=10)
    out = c.chunk("")
    assert len(out) == 1
    assert out[0].text == ""
    assert out[0].token_count == 0


# Token-accurate multi-chunk path (tiktoken) ---------------------------------


def test_oversized_text_splits_into_multiple_chunks_with_overlap() -> None:
    est = TiktokenEstimator("o200k_base")
    size, overlap = 50, 10
    c = Chunker(est, size_tokens=size, overlap_tokens=overlap)

    # Build text with ~200 tokens so we get ~5 chunks with overlap=10
    text = " ".join(f"word{i:04d}" for i in range(250))
    out = c.chunk(text)

    assert len(out) >= 2
    for i, chunk in enumerate(out):
        assert chunk.index == i
        assert 0 < chunk.token_count <= size


def test_chunks_overlap_in_token_space() -> None:
    """Consecutive chunks share `overlap` tokens at the join so retrieval doesn't miss boundaries."""
    est = TiktokenEstimator("o200k_base")
    size, overlap = 40, 10
    c = Chunker(est, size_tokens=size, overlap_tokens=overlap)
    text = " ".join(f"word{i:04d}" for i in range(300))
    out = c.chunk(text)

    # For each pair, last `overlap` token ids of chunk i should equal first
    # `overlap` token ids of chunk i+1.
    enc = est.encoding
    for i in range(len(out) - 1):
        a = enc.encode(out[i].text, disallowed_special=())
        b = enc.encode(out[i + 1].text, disallowed_special=())
        assert a[-overlap:] == b[:overlap], (
            f"overlap not preserved between chunk {i} and {i + 1}"
        )


def test_chunks_cover_the_text_contiguously_in_token_space() -> None:
    """Re-joining chunks (minus overlap) must reproduce the original token sequence."""
    est = TiktokenEstimator("o200k_base")
    size, overlap = 40, 10
    c = Chunker(est, size_tokens=size, overlap_tokens=overlap)
    text = " ".join(f"word{i:04d}" for i in range(300))
    out = c.chunk(text)

    enc = est.encoding
    original_ids = enc.encode(text, disallowed_special=())

    # Reconstruct by concatenating the first chunk's ids, plus every next
    # chunk's ids with the overlap prefix dropped.
    reconstructed: list[int] = list(enc.encode(out[0].text, disallowed_special=()))
    for later in out[1:]:
        ids = enc.encode(later.text, disallowed_special=())
        reconstructed.extend(ids[overlap:])

    assert reconstructed == original_ids


# Word-packer fallback path (non-tiktoken estimator) -------------------------


def test_word_packer_keeps_counts_at_or_below_size() -> None:
    est = CharApproxEstimator()
    size, overlap = 10, 2
    c = Chunker(est, size_tokens=size, overlap_tokens=overlap)
    # CharApprox: 1 token per 4 chars. Build text around ~200 chars.
    text = ("lorem ipsum dolor sit amet " * 10).strip()
    out = c.chunk(text)
    assert len(out) >= 2
    for chunk in out:
        assert chunk.token_count <= size


def test_word_packer_indices_are_monotonic_and_start_at_zero() -> None:
    est = CharApproxEstimator()
    c = Chunker(est, size_tokens=10, overlap_tokens=2)
    text = ("word " * 80).strip()
    out = c.chunk(text)
    assert [chunk.index for chunk in out] == list(range(len(out)))


def test_text_chunk_is_a_frozen_dataclass() -> None:
    """TextChunk is used as a value type; mutation must be disallowed."""
    tc = TextChunk(index=0, text="hi", token_count=1)
    with pytest.raises(Exception):
        tc.index = 5  # type: ignore[misc]
