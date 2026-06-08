"""Backward-compatible alias: the onboarding wizard moved to
:mod:`feather.setup.wizard`.

This module is aliased to :mod:`feather.setup.wizard` via ``sys.modules`` so
``feather.onboarding`` and ``feather.setup.wizard`` are the *same* module
object. That preserves not just imports (including the semi-private
``_QDRANT_*`` constants and ``_mask_secret``) but also monkeypatching of
internal helpers — tests still patch e.g. ``feather.onboarding._probe_qdrant_url``
and the wizard's own code, running under the new path, observes the patch.
"""

from __future__ import annotations

import sys

from feather.setup import wizard as _wizard

sys.modules[__name__] = _wizard
