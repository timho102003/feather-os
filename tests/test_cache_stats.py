"""Tests for prompt-cache hit accounting."""

from __future__ import annotations

from feather.observability.cache_stats import CacheStats, CacheUsage, read_cache_usage


# ---------------------------------------------------------------------------
# read_cache_usage — provider shape normalization
# ---------------------------------------------------------------------------


def test_read_cache_usage_anthropic_shape() -> None:
    usage = {
        "input_tokens": 12,
        "cache_read_input_tokens": 980,
        "cache_creation_input_tokens": 40,
    }
    seen = read_cache_usage(usage)
    assert seen == CacheUsage(read=980, write=40)


def test_read_cache_usage_openai_openrouter_shape() -> None:
    usage = {"prompt_tokens": 2006, "prompt_tokens_details": {"cached_tokens": 1920}}
    seen = read_cache_usage(usage)
    assert seen.read == 1920
    assert seen.write == 0


def test_read_cache_usage_prefers_explicit_cache_fields() -> None:
    """When both shapes are present, the explicit Anthropic read field wins."""

    usage = {
        "cache_read_input_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": 999},
    }
    assert read_cache_usage(usage).read == 500


def test_read_cache_usage_handles_none_and_missing() -> None:
    assert read_cache_usage(None) == CacheUsage(read=0, write=0)
    assert read_cache_usage({}) == CacheUsage(read=0, write=0)
    # Defensive against non-int / negative junk.
    assert read_cache_usage({"cache_read_input_tokens": None}).read == 0
    assert read_cache_usage({"cache_read_input_tokens": -5}).read == 0
    assert read_cache_usage({"prompt_tokens_details": "nope"}).read == 0


def test_cache_usage_hit_rate() -> None:
    assert CacheUsage(read=900, write=100).hit_rate == 0.9
    assert CacheUsage(read=0, write=0).hit_rate == 0.0  # no division by zero
    assert CacheUsage(read=0, write=50).hit_rate == 0.0


# ---------------------------------------------------------------------------
# CacheStats — running accumulator
# ---------------------------------------------------------------------------


def test_cache_stats_accumulates_hit_rate() -> None:
    stats = CacheStats()
    assert stats.hit_rate == 0.0  # empty → no division by zero

    stats.record({"cache_creation_input_tokens": 1000})  # cold write
    assert stats.hit_rate == 0.0

    stats.record({"cache_read_input_tokens": 1000})  # warm read
    assert stats.hit_rate == 0.5

    stats.record({"cache_read_input_tokens": 2000})
    assert stats.cache_reads == 3000
    assert stats.cache_writes == 1000
    assert stats.hit_rate == 0.75


def test_cache_stats_ignores_uncacheable_turns() -> None:
    stats = CacheStats()
    stats.record(None)
    stats.record({"input_tokens": 50})  # below cache floor — nothing cached
    assert stats.cache_reads == 0
    assert stats.cache_writes == 0
    assert stats.hit_rate == 0.0
