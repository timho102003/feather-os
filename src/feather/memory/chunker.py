"""Token-aware chunker for atomic memory content.

Most atomic memories are well under the 1000-token default and pass through
the fast single-chunk path. Only truly long extractions trigger real chunking,
in which case consecutive chunks share ``overlap_tokens`` at their boundaries
so retrieval doesn't miss matches that fall across a cut.

Two code paths:

- :meth:`Chunker._chunk_on_tokens` (preferred): used when the estimator is a
  :class:`~feather.memory.tokenizer.TiktokenEstimator`. Slices directly on
  token ids and decodes — token-accurate and cheap.
- :meth:`Chunker._chunk_on_words`: used with any count-only estimator. Greedy
  word packer with an estimator-guided overlap. Correct but slower and not
  token-exact.
"""

from __future__ import annotations

from dataclasses import dataclass

from feather.memory.tokenizer import TiktokenEstimator, TokenEstimator


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One chunk of text with its 0-indexed position and token count."""

    index: int
    text: str
    token_count: int


class Chunker:
    """Split text into token-sized chunks with configurable overlap.

    The chunker owns no I/O — it's a pure, deterministic transform driven by
    the injected ``TokenEstimator``. Overlap must be strictly less than size
    (otherwise chunks would regress indefinitely).
    """

    def __init__(
        self,
        estimator: TokenEstimator,
        *,
        size_tokens: int = 1000,
        overlap_tokens: int = 100,
    ) -> None:
        if size_tokens <= 0:
            raise ValueError("size_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if overlap_tokens >= size_tokens:
            raise ValueError("overlap_tokens must be < size_tokens")
        self._est = estimator
        self._size = size_tokens
        self._overlap = overlap_tokens

    def chunk(self, text: str) -> list[TextChunk]:
        """Return ``text`` split into overlapping chunks.

        Empty text produces a single empty chunk (``token_count=0``) so the
        caller can iterate uniformly.
        """
        total = self._est.count(text)
        if total <= self._size:
            return [TextChunk(index=0, text=text, token_count=total)]
        if isinstance(self._est, TiktokenEstimator):
            return self._chunk_on_tokens(text)
        return self._chunk_on_words(text)

    # -- tiktoken path --------------------------------------------------------

    def _chunk_on_tokens(self, text: str) -> list[TextChunk]:
        assert isinstance(self._est, TiktokenEstimator)
        encoding = self._est.encoding
        ids = encoding.encode(text, disallowed_special=())
        step = self._size - self._overlap

        chunks: list[TextChunk] = []
        start = 0
        idx = 0
        while start < len(ids):
            end = min(start + self._size, len(ids))
            piece_ids = ids[start:end]
            chunk_text = encoding.decode(piece_ids)
            chunks.append(TextChunk(index=idx, text=chunk_text, token_count=len(piece_ids)))
            if end == len(ids):
                break
            start += step
            idx += 1
        return chunks

    # -- word-packer path (generic estimator) ---------------------------------

    def _chunk_on_words(self, text: str) -> list[TextChunk]:
        words = text.split()
        if not words:
            return [TextChunk(index=0, text="", token_count=0)]

        chunks: list[TextChunk] = []
        cursor = 0
        idx = 0
        while cursor < len(words):
            end = self._greedy_fit(words, cursor)
            # Ensure we always consume at least one word so the loop terminates
            # even when a single "word" is itself larger than size_tokens.
            if end == cursor:
                end = cursor + 1
            chunk_text = " ".join(words[cursor:end])
            chunks.append(
                TextChunk(
                    index=idx,
                    text=chunk_text,
                    token_count=self._est.count(chunk_text),
                )
            )
            idx += 1
            if end >= len(words):
                break
            cursor = self._rewind_for_overlap(words, cursor, end)
        return chunks

    def _greedy_fit(self, words: list[str], start: int) -> int:
        """Return the exclusive end index that fits within ``size_tokens``."""
        end = start
        while end < len(words):
            candidate = " ".join(words[start : end + 1])
            if self._est.count(candidate) > self._size:
                break
            end += 1
        return end

    def _rewind_for_overlap(self, words: list[str], chunk_start: int, chunk_end: int) -> int:
        """Find the next chunk's start index by walking backwards from ``chunk_end``.

        Returns the smallest index ``k`` in ``(chunk_start, chunk_end]`` such
        that ``words[k:chunk_end]`` counts for at least ``overlap_tokens``.
        Falls back to ``chunk_end`` (no overlap) if even the full chunk has
        fewer than ``overlap_tokens`` — this can happen with a single
        oversized word that was emitted as its own chunk above.
        """
        k = chunk_end
        while k > chunk_start + 1:
            tail = " ".join(words[k - 1 : chunk_end])
            if self._est.count(tail) >= self._overlap:
                return k - 1
            k -= 1
        # Default: step forward by one word to guarantee progress.
        return chunk_end
