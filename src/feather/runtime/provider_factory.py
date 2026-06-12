"""Provider construction helpers for the Feather runtime.

Building the session-wide LLM provider and the optional Parallel web-tools
client lives here so :mod:`feather.runtime.root` doesn't carry the concrete
provider imports — it depends on these two factories, not the provider classes.
"""

from __future__ import annotations

import logging
from typing import Any

from feather.providers.base import BaseLLMProvider
from feather.providers.catalog import provider_spec
from feather.providers.parallel_client import ParallelClient

logger = logging.getLogger(__name__)

__all__ = ("_build_default_provider", "_try_build_parallel_client")


def _build_default_provider(app_config: Any) -> BaseLLMProvider:
    """Pick the provider implementation for the session-wide active provider.

    ``app_config.active_provider`` defaults to ``"openai"`` and preserves
    existing sessions exactly. Setting it to ``"openrouter"`` or
    ``"claude"`` flips every agent to that provider's path; missing the
    matching config block raises so operator misconfiguration fails
    loudly rather than silently falling back.
    """

    active = (app_config.active_provider or "openai").strip().lower()
    spec = provider_spec(active)
    if spec is None:
        raise ValueError(
            f"unsupported active_provider={active!r} "
            "(expected 'openai', 'openrouter', or 'claude')"
        )
    if spec.config_block(app_config) is None:
        raise ValueError(
            f"active_provider={active} but no `{active}:` block in app.yaml"
        )
    return spec.build(app_config)


def _try_build_parallel_client(app_config: Any) -> ParallelClient | None:
    """Instantiate the Parallel AI client when config and API key are present."""

    parallel_config = getattr(app_config, "parallel", None)
    if parallel_config is None:
        return None
    try:
        return ParallelClient(parallel_config)
    except ValueError as exc:
        logger.warning("parallel web tools disabled: %s", exc)
        return None
