"""Lead-worker subprocess decision + hang-watcher state helpers (pure)."""

from __future__ import annotations

import os


_LEAD_WORKER_ENV = "FEATHER_USE_LEAD_WORKER"

# Hang-watcher cadence. The supervisor's default staleness threshold is
# 5 s and the worker's heartbeat cadence is 1 s, so polling every 2 s
# means a real hang surfaces in at most one threshold window plus one
# poll interval (~7 s) — fast enough to be useful, slow enough that
# noise from a single missed beat doesn't fire.
_HANG_WATCHER_POLL_SECONDS = 2.0


def decide_hang_alert(prev_stale: bool, current_stale: bool) -> str | None:
    """Pure state-machine helper for the hang watcher.

    Returns ``"alert"`` on a not-stale → stale transition,
    ``"recover"`` on the inverse, and ``None`` for no-change ticks.
    Extracted so the TUI's polling loop is a thin wrapper that's easy
    to reason about and the actual state logic is unit-testable.
    """

    if current_stale and not prev_stale:
        return "alert"
    if prev_stale and not current_stale:
        return "recover"
    return None


def _env_says_use_lead_worker() -> bool | None:
    """Return the env-var override or ``None`` if not set.

    Recognised truthy values: ``1`` / ``true`` / ``yes`` / ``on``.
    Recognised falsy override: ``0`` / ``false`` / ``no`` / ``off``.
    Anything else (including unset) returns ``None`` so the YAML wins.
    """

    raw = os.environ.get(_LEAD_WORKER_ENV)
    if raw is None:
        return None
    cleaned = raw.strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return None


def _should_use_lead_worker(yaml_enabled: bool = False) -> bool:
    """Decide whether to run the lead in a separate worker subprocess.

    Resolution order:

    1. ``FEATHER_USE_LEAD_WORKER`` env var — power-user override that
       wins over the persistent setting (handy for one-off testing
       without flipping the YAML).
    2. ``self_repair.enabled`` from ``app.yaml`` — the persistent
       answer the onboarding wizard writes.

    Default is False so users who never opted in get the long-standing
    in-process behavior, byte-identical to before this feature shipped.
    """

    env_choice = _env_says_use_lead_worker()
    if env_choice is not None:
        return env_choice
    return bool(yaml_enabled)
