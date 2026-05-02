"""Token estimators used by ``Chunker`` to size memory chunks.

Three implementations behind a common ``TokenEstimator`` protocol:

- :class:`TiktokenEstimator` — fast, local, token-id-accurate. Preferred
  because it also exposes the underlying encoder so the chunker can slice
  on token boundaries (not character boundaries). Not Gemini-exact, but
  stable and deterministic; at 1000-token chunk size that drift is well
  under Gemini's 8192-token input ceiling, so it is harmless.
- :class:`GeminiEstimator` — exact against the Gemini tokenizer via
  ``client.models.count_tokens``, at the cost of one API round-trip per
  count. Useful when exact sizing matters (rare).
- :class:`CharApproxEstimator` — fallback when neither of the above is
  available. Uses ``len(text)//4`` (matches the existing compaction
  heuristic). No slicing support; chunker falls back to its word packer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import tiktoken

if TYPE_CHECKING:
    from feather.memory.config import MemoryChunkingConfig


class TokenEstimator(Protocol):
    """Protocol every token estimator satisfies."""

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...


class TiktokenEstimator:
    """Local, deterministic estimator backed by :mod:`tiktoken`.

    Exposes :attr:`encoding` so the chunker can produce token-id-accurate
    slices via ``encoding.encode`` / ``encoding.decode``.
    """

    def __init__(self, encoding_name: str = "o200k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def encoding(self) -> "tiktoken.Encoding":
        """The underlying ``tiktoken`` encoding object."""
        return self._encoding

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))


class GeminiEstimator:
    """Calls ``google.genai`` Client.models.count_tokens for exact counts."""

    def __init__(self, client: object, model: str) -> None:
        self._client = client
        self._model = model

    def count(self, text: str) -> int:
        response = self._client.models.count_tokens(  # type: ignore[attr-defined]
            model=self._model,
            contents=text,
        )
        return int(response.total_tokens)


class CharApproxEstimator:
    """Fallback estimator: ``max(1, len(text) // 4)``."""

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


def build_estimator(
    cfg: "MemoryChunkingConfig",
    *,
    gemini_client: object | None = None,
    embed_model: str = "",
) -> TokenEstimator:
    """Construct the configured estimator.

    Args:
        cfg: The chunking config carrying the ``tokenizer`` selector.
        gemini_client: Required when ``cfg.tokenizer == 'gemini'``.
        embed_model: Model name passed to the Gemini token counter.

    Returns:
        The instantiated estimator.

    Raises:
        ValueError: When ``cfg.tokenizer`` is unknown, or when selecting
            ``'gemini'`` without a ``gemini_client``.
    """
    match cfg.tokenizer:
        case "tiktoken":
            return TiktokenEstimator(cfg.tokenizer_encoding)
        case "char4":
            return CharApproxEstimator()
        case "gemini":
            if gemini_client is None:
                raise ValueError("tokenizer='gemini' requires a gemini_client")
            return GeminiEstimator(gemini_client, embed_model)
        case other:
            raise ValueError(f"unknown tokenizer {other!r}")
