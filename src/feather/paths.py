"""Backward-compatible shim: ``FeatherPaths`` now lives in
:mod:`feather.config.app_paths`.

Kept as a permanent top-level re-export because ``feather.paths`` is imported by
~27 modules (including ``conftest.py``'s autouse fixture) and is reached
cross-process via the runtime, so relocating it would churn every call site for
no behavioral gain.
"""

from __future__ import annotations

from feather.config.app_paths import FeatherPaths

__all__ = ("FeatherPaths",)
