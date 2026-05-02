"""Runtime construction for the memory subsystem.

Builds either a live :class:`MemoryStack` (Qdrant + Gemini wired together)
or a zero-cost no-op stack based on three signals:

1. ``app_config.memory.enabled`` — top-level YAML switch.
2. ``QDRANT_URL`` and a Gemini API key in env (or YAML for the URL).
3. ``FEATHER_MEMORY_DISABLED=1`` env override (kill switch).

When any gate fails the stack is fully no-op and the rest of the runtime
behaves identically to a memory-less build.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from feather.memory.config import (
    MemoryConfig,
    MemoryOperationModelConfig,
)
from feather.memory.reader import LiveMemoryReader, MemoryReader, NoOpMemoryReader
from feather.memory.trigger import LiveMemoryTrigger, MemoryTrigger, NoOpMemoryTrigger

if TYPE_CHECKING:
    from feather.memory.service import MemoryService
    from feather.models import AppConfig
    from feather.providers.base import BaseLLMProvider
    from feather.storage.session_store import SessionStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryStack:
    """Composed reader + trigger + (optional) live service.

    ``owned_providers`` lists alternate provider instances built for memory
    ops that overrode ``provider`` to something other than the app default.
    The runtime closes them via :meth:`aclose` during shutdown so the
    sockets they own don't leak.
    """

    reader: MemoryReader
    trigger: MemoryTrigger
    service: "MemoryService | None"
    enabled: bool
    owned_providers: list["BaseLLMProvider"] = field(default_factory=list)

    async def aclose(self) -> None:
        """Close every alternate provider this stack owns."""

        for provider in self.owned_providers:
            closer = getattr(provider, "aclose", None)
            if closer is None:
                continue
            try:
                await closer()
            except Exception:  # noqa: BLE001
                logger.exception("memory.shutdown.alternate_provider_close_error")


def _resolve_qdrant_url(cfg: MemoryConfig) -> str | None:
    """Return the effective Qdrant URL.

    Precedence (per spec §6.4):

    1. ``QDRANT_URL`` env var — set by Compose / k8s / explicit shell.
    2. ``~/.feather/state/memory.json`` marker — written by
       ``feather init-memory``. Source of truth for "is local-mode
       memory currently set up?".
    3. ``cfg.qdrant.url`` from ``app.yaml``.
    """
    env = os.environ.get("QDRANT_URL")
    if env and env.strip():
        return env
    try:
        from feather.cli_commands import memory_url_from_marker
        from feather.paths import FeatherPaths

        marker_url = memory_url_from_marker(FeatherPaths.global_only())
    except Exception:  # noqa: BLE001 — marker reads must never crash startup
        marker_url = None
    if marker_url:
        return marker_url
    return cfg.qdrant.url


def _resolve_gemini_key() -> str | None:
    """Return a Gemini API key from any of the standard env-var names."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        value = os.environ.get(name)
        if value and value.strip():
            return value
    return None


def _kill_switch_active() -> bool:
    return os.environ.get("FEATHER_MEMORY_DISABLED") == "1"


def _active_provider_name(app_config: "AppConfig") -> str:
    """Return the normalised name of the app-default provider."""

    return (app_config.active_provider or "openai").strip().lower()


def _build_alternate_provider(
    name: str, app_config: "AppConfig", op_label: str
) -> "BaseLLMProvider":
    """Build a fresh provider instance for a cross-provider memory op.

    Errors fail fast at startup so a misconfigured op_label surfaces
    before the first extraction tick rather than mid-session.
    """

    from feather.providers.openai_provider import OpenAIResponsesProvider
    from feather.providers.openrouter_provider import OpenRouterChatProvider

    if name == "openai":
        return OpenAIResponsesProvider(app_config.openai)
    if name == "openrouter":
        if app_config.openrouter is None:
            raise ValueError(
                f"memory.{op_label}.provider=openrouter but no `openrouter:` "
                "block in app.yaml"
            )
        return OpenRouterChatProvider(app_config.openrouter)
    raise ValueError(
        f"memory.{op_label}.provider={name!r} is not a known provider "
        "(expected 'openai' or 'openrouter')"
    )


def _resolve_op_provider(
    op_cfg: MemoryOperationModelConfig,
    *,
    op_label: str,
    app_config: "AppConfig",
    default_provider: "BaseLLMProvider",
    cache: dict[str, "BaseLLMProvider"],
    owned: list["BaseLLMProvider"],
) -> tuple["BaseLLMProvider", str | None]:
    """Resolve a memory op's provider and the model fallback to inherit.

    Returns ``(provider, default_model)``:
    - ``provider`` is the LLM client the op should call.
    - ``default_model`` is the model name to use when ``op_cfg.model`` is
      ``None`` AND the op runs on an alternate provider. If the op uses
      the app-default provider, ``default_model`` is ``None`` so the op
      falls through to the calling agent's conversation model at call
      time (preserves existing behaviour).
    """

    requested = op_cfg.provider
    active = _active_provider_name(app_config)

    if requested is None or requested == active:
        return default_provider, None

    cached = cache.get(requested)
    if cached is not None:
        return cached, _alternate_default_model(requested, app_config)

    instance = _build_alternate_provider(requested, app_config, op_label)
    cache[requested] = instance
    owned.append(instance)
    return instance, _alternate_default_model(requested, app_config)


def _alternate_default_model(name: str, app_config: "AppConfig") -> str:
    """Return the configured default model for an alternate provider."""

    if name == "openai":
        return app_config.openai.model
    if name == "openrouter":
        assert app_config.openrouter is not None
        return app_config.openrouter.model
    raise ValueError(f"unknown provider name: {name!r}")


def build_memory_stack(
    *,
    cfg: MemoryConfig,
    default_provider: "BaseLLMProvider",
    app_config: "AppConfig",
    session_store: "SessionStore",
) -> MemoryStack:
    """Construct the memory stack based on config + env gating.

    Returns a :class:`MemoryStack` whose ``enabled`` flag is True only when
    every gate passes; in all other cases ``reader`` and ``trigger`` are
    no-ops and ``service`` is ``None``.

    ``default_provider`` is the LLM client the agent loop is using —
    memory ops with ``provider=None`` reuse this same instance to share
    its connection pool. Memory ops that override ``provider`` get a
    freshly-built (or cached) alternate, tracked on ``stack.owned_providers``
    so the runtime can close them during shutdown.
    """
    if not cfg.enabled:
        logger.info("memory.runtime.disabled", extra={"reason": "config_disabled"})
        return MemoryStack(
            reader=NoOpMemoryReader(),
            trigger=NoOpMemoryTrigger(),
            service=None,
            enabled=False,
        )
    if _kill_switch_active():
        logger.info(
            "memory.runtime.disabled", extra={"reason": "kill_switch"}
        )
        return MemoryStack(
            reader=NoOpMemoryReader(),
            trigger=NoOpMemoryTrigger(),
            service=None,
            enabled=False,
        )
    qdrant_url = _resolve_qdrant_url(cfg)
    gemini_key = _resolve_gemini_key()
    if not qdrant_url or not gemini_key:
        logger.warning(
            "memory.runtime.disabled",
            extra={
                "reason": "missing_credentials",
                "has_qdrant_url": bool(qdrant_url),
                "has_gemini_key": bool(gemini_key),
            },
        )
        return MemoryStack(
            reader=NoOpMemoryReader(),
            trigger=NoOpMemoryTrigger(),
            service=None,
            enabled=False,
        )

    # Imports kept local so the no-op path doesn't pull in qdrant-client.
    from qdrant_client import AsyncQdrantClient

    from feather.memory.chunker import Chunker
    from feather.memory.classifier import CrudClassifier
    from feather.memory.embedding.gemini import GeminiEmbeddingProvider
    from feather.memory.extractor import MemoryExtractor
    from feather.memory.prompts.classification_prompt import CLASSIFICATION_PROMPT
    from feather.memory.prompts.extraction_prompt import EXTRACTION_PROMPT
    from feather.memory.prompts.query_prompt import QUERY_PROMPT
    from feather.memory.query_builder import MemoryQueryBuilder
    from feather.memory.service import MemoryService
    from feather.memory.store.qdrant import QdrantVectorStore
    from feather.memory.tokenizer import build_estimator

    qdrant_api_key = os.environ.get(cfg.qdrant.api_key_env)
    qdrant_client = AsyncQdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=cfg.qdrant.prefer_grpc,
        timeout=int(cfg.qdrant.request_timeout_s),
    )
    store = QdrantVectorStore(client=qdrant_client, cfg=cfg.qdrant)
    embedder = GeminiEmbeddingProvider(cfg=cfg.embedding, api_key=gemini_key)
    estimator = build_estimator(cfg.chunking)
    chunker = Chunker(
        estimator,
        size_tokens=cfg.chunking.chunk_size_tokens,
        overlap_tokens=cfg.chunking.chunk_overlap_tokens,
    )
    op_provider_cache: dict[str, "BaseLLMProvider"] = {}
    owned_providers: list["BaseLLMProvider"] = []
    extractor_provider, extractor_default_model = _resolve_op_provider(
        cfg.extraction,
        op_label="extraction",
        app_config=app_config,
        default_provider=default_provider,
        cache=op_provider_cache,
        owned=owned_providers,
    )
    classifier_provider, classifier_default_model = _resolve_op_provider(
        cfg.classification,
        op_label="classification",
        app_config=app_config,
        default_provider=default_provider,
        cache=op_provider_cache,
        owned=owned_providers,
    )
    extractor = MemoryExtractor(
        provider=extractor_provider,
        prompt=EXTRACTION_PROMPT,
        cfg=cfg.extraction,
        default_model=extractor_default_model,
    )
    classifier = CrudClassifier(
        provider=classifier_provider,
        prompt=CLASSIFICATION_PROMPT,
        cfg=cfg.classification,
        store=store,
        embedder=embedder,
        retrieval_cfg=cfg.retrieval,
        default_model=classifier_default_model,
    )
    service = MemoryService(
        cfg=cfg,
        store=store,
        embedder=embedder,
        chunker=chunker,
        extractor=extractor,
        classifier=classifier,
        session_store=session_store,
    )
    qb_provider, qb_default_model = _resolve_op_provider(
        cfg.query_builder,
        op_label="query_builder",
        app_config=app_config,
        default_provider=default_provider,
        cache=op_provider_cache,
        owned=owned_providers,
    )
    query_builder = MemoryQueryBuilder(
        provider=qb_provider,
        prompt=QUERY_PROMPT,
        cfg=cfg.query_builder,
        default_model=qb_default_model,
    )
    reader = LiveMemoryReader(
        embedder=embedder, store=store, query_builder=query_builder, cfg=cfg.retrieval
    )
    trigger = LiveMemoryTrigger(service=service, cfg=cfg.trigger)
    logger.info(
        "memory.runtime.enabled",
        extra={
            "qdrant_url": qdrant_url,
            "embedding_model": cfg.embedding.model,
            "embedding_dims": cfg.embedding.output_dimensionality,
        },
    )
    return MemoryStack(
        reader=reader,
        trigger=trigger,
        service=service,
        enabled=True,
        owned_providers=owned_providers,
    )


__all__ = ["MemoryStack", "build_memory_stack"]
