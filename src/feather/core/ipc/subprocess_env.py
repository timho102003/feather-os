"""Environment construction for feather-spawned subprocesses."""

from __future__ import annotations

import os

__all__ = ("subprocess_env_with_home",)


def subprocess_env_with_home() -> dict[str, str]:
    """Snapshot ``os.environ`` with a guaranteed non-empty ``HOME``.

    ``Path.expanduser()`` raises ``RuntimeError`` when ``HOME`` is
    missing, and worker/sub-agent code calls it on hot paths (seen
    crashing in the field). When ``HOME`` is absent or empty, re-derive
    it from ``pwd`` — the same source ``os.path.expanduser`` consults.
    """

    env = os.environ.copy()
    if not env.get("HOME"):
        try:
            import pwd

            env["HOME"] = pwd.getpwuid(os.getuid()).pw_dir
        except (ImportError, KeyError, OSError):
            pass
    return env
