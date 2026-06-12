"""Shared retry/backoff math for streaming providers."""

from __future__ import annotations

import random
import time
from datetime import datetime

__all__ = (
    "backoff_delay",
    "seconds_from_retry_after",
    "seconds_until_unix_timestamp",
)


def backoff_delay(attempt: int, base_delay: float) -> float:
    """Exponential backoff with full jitter: base * 2^attempt + U(0, base).

    Args:
        attempt: Zero-based retry attempt index.
        base_delay: Base delay in seconds.

    Returns:
        Computed sleep duration in seconds.
    """
    return base_delay * (2**attempt) + random.uniform(0.0, base_delay)


def seconds_until_unix_timestamp(header: str | None, *, max_wait: float) -> float | None:
    """X-RateLimit-Reset-style unix-seconds header -> bounded wait, else None.

    Parses a header whose value is a unix epoch timestamp (integer digits only).
    Returns the number of seconds until that timestamp, clamped to ``max_wait``
    and floored at 0.0. Returns ``None`` if the header is absent or non-numeric.

    Args:
        header: Header value string, or None.
        max_wait: Maximum wait to return; larger hints are clamped.

    Returns:
        Seconds to wait, or None if header is unusable.
    """
    if header and header.isdigit():
        delta = max(0.0, int(header) - time.time())
        return min(delta, max_wait)
    return None


def seconds_from_retry_after(header: str | None, *, max_wait: float) -> float | None:
    """RFC 7231 Retry-After (delta-seconds or HTTP-date) -> bounded wait, else None.

    Handles both forms that RFC 7231 allows:
    - Integer delta-seconds (the form Anthropic actually emits).
    - HTTP-date in the format ``%a, %d %b %Y %H:%M:%S GMT`` (naive UTC).

    Malformed values (non-digit, unparseable date) return ``None`` so the
    caller can fall back to exponential backoff.

    Args:
        header: Retry-After header value, or None.
        max_wait: Maximum wait to return; larger hints are clamped.

    Returns:
        Seconds to wait, or None if header is absent or malformed.
    """
    if not header:
        return None
    hint = header.strip()
    if hint.isdigit():
        return min(float(hint), max_wait)
    try:
        target = datetime.strptime(hint, "%a, %d %b %Y %H:%M:%S GMT")
        delta = max(0.0, target.timestamp() - time.time())
        return min(delta, max_wait)
    except ValueError:
        return None
