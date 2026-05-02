"""Thin async wrapper over the Parallel AI SDK for Feather web tools."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from parallel import Parallel

from feather.models import ParallelConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ParallelSearchHit:
    """Normalized per-result record returned by `ParallelClient.search`."""

    title: str | None
    url: str
    excerpts: list[str]
    publish_date: str | None


@dataclass(slots=True)
class ParallelSearchResponse:
    """Normalized response for a single Parallel search call."""

    mode: str
    results: list[ParallelSearchHit]


@dataclass(slots=True)
class ParallelExtractHit:
    """Normalized per-URL record returned by `ParallelClient.extract`."""

    title: str | None
    url: str
    excerpts: list[str]
    full_content: str | None
    publish_date: str | None


@dataclass(slots=True)
class ParallelExtractResponse:
    """Normalized response for a single Parallel extract call."""

    results: list[ParallelExtractHit]
    errors: list[str]


class ParallelClient:
    """Async-friendly wrapper around the sync Parallel SDK."""

    def __init__(self, config: ParallelConfig, *, client: Any = None) -> None:
        self._config = config
        if client is not None:
            self._client = client
            return
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing required environment variable: {config.api_key_env}"
            )
        self._client = Parallel(api_key=api_key)

    @property
    def config(self) -> ParallelConfig:
        """Return the config backing this client."""

        return self._config

    async def search(
        self,
        *,
        objective: str,
        search_queries: list[str] | None,
        mode: str,
        max_results: int,
        include_domains: list[str] | None,
    ) -> ParallelSearchResponse:
        """Run one Parallel search request and normalize the response."""

        kwargs: dict[str, Any] = {
            "objective": objective,
            "mode": mode,
            "max_results": max_results,
        }
        if search_queries:
            kwargs["search_queries"] = search_queries
        if include_domains:
            kwargs["source_policy"] = {"include_domains": include_domains}

        logger.info(
            "parallel search objective_chars=%s mode=%s max_results=%s include_domains=%s",
            len(objective),
            mode,
            max_results,
            include_domains or [],
        )
        raw = await asyncio.to_thread(self._client.beta.search, **kwargs)
        return ParallelSearchResponse(
            mode=mode,
            results=[
                ParallelSearchHit(
                    title=_safe_attr(item, "title"),
                    url=_safe_attr(item, "url") or "",
                    excerpts=list(_safe_attr(item, "excerpts") or []),
                    publish_date=_safe_attr(item, "publish_date"),
                )
                for item in (raw.results or [])
            ],
        )

    async def extract(
        self,
        *,
        url: str,
        objective: str | None,
        include_full_content: bool,
    ) -> ParallelExtractResponse:
        """Run one Parallel extract request for a single URL."""

        kwargs: dict[str, Any] = {
            "urls": [url],
            "excerpts": True,
        }
        if include_full_content:
            kwargs["full_content"] = True
        if objective:
            kwargs["objective"] = objective

        logger.info(
            "parallel extract url=%s include_full_content=%s has_objective=%s",
            url,
            include_full_content,
            bool(objective),
        )
        raw = await asyncio.to_thread(self._client.beta.extract, **kwargs)
        results = [
            ParallelExtractHit(
                title=_safe_attr(item, "title"),
                url=_safe_attr(item, "url") or url,
                excerpts=list(_safe_attr(item, "excerpts") or []),
                full_content=_safe_attr(item, "full_content"),
                publish_date=_safe_attr(item, "publish_date"),
            )
            for item in (raw.results or [])
        ]
        errors = [
            _safe_attr(err, "message") or str(err)
            for err in (raw.errors or [])
        ]
        return ParallelExtractResponse(results=results, errors=errors)


def _safe_attr(obj: Any, name: str) -> Any:
    """Return an attribute or mapping key from heterogeneous SDK return values."""

    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
