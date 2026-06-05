"""Feather runtime composition root.

``runtime.py`` became this package; the body lives in :mod:`feather.runtime.root`
and provider construction in :mod:`feather.runtime.provider_factory`. This
re-export keeps ``from feather.runtime import FeatherRuntime`` resolving
unchanged for every caller (cli, api.hub, both subprocess entries, worker_core).
"""

from __future__ import annotations

from feather.runtime.root import ConfigApplyResult, FeatherRuntime

__all__ = ("ConfigApplyResult", "FeatherRuntime")
