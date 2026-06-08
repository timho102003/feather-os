"""Backward-compatible shim: the user-profile store moved to
:mod:`feather.storage.user_profile`.

Kept as a permanent top-level re-export because ``feather.profile`` is imported
by the agent loop, the agent factory, and the user-info tool (cross-process), so
relocating every call site would churn them for no behavioral gain. The
implementation lives in :mod:`feather.storage.user_profile`.
"""

from __future__ import annotations

from feather.storage.user_profile import UserProfile, UserProfileStore

__all__ = ("UserProfile", "UserProfileStore")
