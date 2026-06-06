"""Prompt-cache hit accounting from normalized provider usage dicts.

Surfaces whether prompt caching is actually working at runtime. The three
providers report cache usage under different keys; :func:`read_cache_usage`
normalizes them and :class:`CacheStats` accumulates a hit rate, so an
accidental prefix change — which silently drops caching to zero — shows up as
a collapsing gauge rather than a quiet cost regression.

Provider shapes handled:

- **Anthropic** (and Claude-via-OpenRouter): ``cache_read_input_tokens`` /
  ``cache_creation_input_tokens``.
- **OpenAI / OpenRouter**: ``prompt_tokens_details.cached_tokens`` (read only;
  no separate cache-write field is reported).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ("CacheUsage", "read_cache_usage", "CacheStats")


def _coerce_int(value: Any) -> int:
    """Return a non-negative int, treating anything else as 0."""

    return value if isinstance(value, int) and value >= 0 else 0


@dataclass(slots=True, frozen=True)
class CacheUsage:
    """Per-turn prompt-cache token counts, provider-normalized."""

    read: int
    write: int

    @property
    def total(self) -> int:
        """Cacheable tokens this turn (served from cache + written to cache)."""

        return self.read + self.write

    @property
    def hit_rate(self) -> float:
        """Fraction of cacheable tokens served from cache (0.0 when none)."""

        return self.read / self.total if self.total else 0.0


def read_cache_usage(usage: dict[str, Any] | None) -> CacheUsage:
    """Normalize a provider ``usage`` dict into a :class:`CacheUsage`.

    Falls back to the OpenAI/OpenRouter ``prompt_tokens_details.cached_tokens``
    only when no Anthropic-style read field is present, so a dict carrying both
    shapes prefers the explicit cache fields. Unknown / ``None`` usage → zeros.
    """

    if not isinstance(usage, dict):
        return CacheUsage(read=0, write=0)

    read = _coerce_int(usage.get("cache_read_input_tokens"))
    write = _coerce_int(usage.get("cache_creation_input_tokens"))
    if not read:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            read = _coerce_int(details.get("cached_tokens"))
    return CacheUsage(read=read, write=write)


@dataclass(slots=True)
class CacheStats:
    """Running prompt-cache accumulator across many turns.

    Feed each turn's ``usage`` dict to :meth:`record`; read :attr:`hit_rate`
    for a steady-state gauge. A hit rate that collapses while traffic is normal
    is the production signature of a prefix that stopped being byte-stable.
    """

    cache_reads: int = 0
    cache_writes: int = 0

    def record(self, usage: dict[str, Any] | None) -> None:
        """Fold one turn's cache usage into the running totals."""

        seen = read_cache_usage(usage)
        self.cache_reads += seen.read
        self.cache_writes += seen.write

    @property
    def hit_rate(self) -> float:
        """Cumulative ``reads / (reads + writes)`` (0.0 before any cacheable tokens)."""

        denom = self.cache_reads + self.cache_writes
        return self.cache_reads / denom if denom else 0.0
