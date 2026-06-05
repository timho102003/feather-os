"""Shim: the lead-worker subprocess entrypoint body moved to
:mod:`feather.core.leads.worker_entry`.

Kept at the top level — and load-bearing as ``__main__`` — because the
supervisor spawns the worker as ``python -m feather.lead_worker_entry``; that
dotted path must keep resolving and ``-m`` execution must run ``main()``.
"""

from __future__ import annotations

import os

from feather.core.leads.worker_entry import main

__all__ = ("main",)

if __name__ == "__main__":
    # Inherit unbuffered stdio so the supervisor never blocks on a half-buffered
    # pipe (the worker also per-line-flushes its event sink).
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
