"""Tests for feather.providers.retry_utils."""

from __future__ import annotations

import time
from unittest.mock import patch

from feather.providers.retry_utils import (
    backoff_delay,
    seconds_from_retry_after,
    seconds_until_unix_timestamp,
)


# ------------------------------------------------------------------ backoff_delay


def test_backoff_delay_lower_bound() -> None:
    """Result must be >= base * 2^attempt (the exponential component)."""
    for attempt in range(5):
        base = 0.5
        result = backoff_delay(attempt, base)
        assert result >= base * (2**attempt)


def test_backoff_delay_upper_bound() -> None:
    """Result must be < base * 2^attempt + base (jitter at most one base)."""
    for attempt in range(5):
        base = 0.5
        result = backoff_delay(attempt, base)
        assert result < base * (2**attempt) + base + 1e-9


def test_backoff_delay_attempt_zero_bounds() -> None:
    base = 0.5
    result = backoff_delay(0, base)
    # base * 2^0 = 0.5; jitter in [0, 0.5) → result in [0.5, 1.0)
    assert 0.5 <= result < 1.0 + 1e-9


def test_backoff_delay_attempt_two_bounds() -> None:
    base = 0.5
    result = backoff_delay(2, base)
    # base * 4 = 2.0; jitter in [0, 0.5) → result in [2.0, 2.5)
    assert 2.0 <= result < 2.5 + 1e-9


# ------------------------------------------------------------------ seconds_until_unix_timestamp


def test_seconds_until_unix_timestamp_future_ts_returns_positive() -> None:
    future = str(int(time.time()) + 30)
    result = seconds_until_unix_timestamp(future, max_wait=60.0)
    assert result is not None
    assert 0.0 < result <= 30.0 + 1.0  # +1s for clock drift


def test_seconds_until_unix_timestamp_past_ts_returns_zero() -> None:
    past = str(int(time.time()) - 10)
    result = seconds_until_unix_timestamp(past, max_wait=60.0)
    assert result == 0.0


def test_seconds_until_unix_timestamp_clamps_to_max_wait() -> None:
    far_future = str(int(time.time()) + 9999)
    result = seconds_until_unix_timestamp(far_future, max_wait=60.0)
    assert result == 60.0


def test_seconds_until_unix_timestamp_none_returns_none() -> None:
    assert seconds_until_unix_timestamp(None, max_wait=60.0) is None


def test_seconds_until_unix_timestamp_non_digit_returns_none() -> None:
    assert seconds_until_unix_timestamp("not-a-number", max_wait=60.0) is None


def test_seconds_until_unix_timestamp_empty_string_returns_none() -> None:
    assert seconds_until_unix_timestamp("", max_wait=60.0) is None


# ------------------------------------------------------------------ seconds_from_retry_after


def test_seconds_from_retry_after_integer_honored() -> None:
    """Pure digit header must return exactly that many seconds (within clamp)."""
    result = seconds_from_retry_after("3", max_wait=60.0)
    assert result == 3.0


def test_seconds_from_retry_after_integer_clamped() -> None:
    """Values above max_wait must be clamped."""
    result = seconds_from_retry_after("9999", max_wait=60.0)
    assert result == 60.0


def test_seconds_from_retry_after_zero_honored() -> None:
    result = seconds_from_retry_after("0", max_wait=60.0)
    assert result == 0.0


def test_seconds_from_retry_after_valid_http_date_returns_bounded_wait() -> None:
    """A valid HTTP-date in the future must return a positive bounded wait.

    The implementation uses naive strptime (matching the original claude_provider
    code) so the reference datetime must also be naive-local to produce a
    consistent delta.
    """
    from datetime import datetime

    # Build a naive datetime 30 seconds in the future using local time, then
    # format it — the parser will interpret it as local too, so delta ≈ 30s.
    future_ts = time.time() + 30
    dt = datetime.fromtimestamp(future_ts)  # naive local
    http_date = dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
    result = seconds_from_retry_after(http_date, max_wait=60.0)
    assert result is not None
    assert 0.0 < result <= 30.0 + 2.0  # allow 2s for clock drift


def test_seconds_from_retry_after_past_http_date_returns_zero() -> None:
    """An HTTP-date already in the past must return 0.0."""
    from datetime import datetime

    past_ts = time.time() - 10
    dt = datetime.fromtimestamp(past_ts)  # naive local
    http_date = dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
    result = seconds_from_retry_after(http_date, max_wait=60.0)
    assert result == 0.0


def test_seconds_from_retry_after_none_returns_none() -> None:
    assert seconds_from_retry_after(None, max_wait=60.0) is None


def test_seconds_from_retry_after_malformed_returns_none() -> None:
    """Malformed string (not digits, not HTTP-date) must return None."""
    assert seconds_from_retry_after("not-a-date", max_wait=60.0) is None


def test_seconds_from_retry_after_empty_returns_none() -> None:
    assert seconds_from_retry_after("", max_wait=60.0) is None


def test_seconds_from_retry_after_http_date_clamped_to_max_wait() -> None:
    """An HTTP-date far in the future must be clamped to max_wait."""
    from datetime import datetime

    far_future_ts = time.time() + 9999
    dt = datetime.fromtimestamp(far_future_ts)  # naive local
    http_date = dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
    result = seconds_from_retry_after(http_date, max_wait=60.0)
    assert result == 60.0
