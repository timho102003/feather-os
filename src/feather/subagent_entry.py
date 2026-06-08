"""Shim: the sub-agent subprocess entrypoint body moved to
:mod:`feather.core.subagents.entry`.

Kept at the top level — and load-bearing as ``__main__`` — because sub-agents
are spawned as ``python -m feather.subagent_entry`` (argv in
``spawn_agent_tool``); that dotted path must keep resolving and ``-m``
execution must run ``main()``.
"""

from __future__ import annotations

import os

from feather.core.subagents.entry import main, run_subagent_async

__all__ = ("main", "run_subagent_async")

if __name__ == "__main__":
    # Best-effort: keep stdout clean even if a nested library tries to chatter.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
