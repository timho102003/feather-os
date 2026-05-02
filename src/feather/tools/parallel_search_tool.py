"""Web-search tool powered by the Parallel AI Search API."""

from __future__ import annotations

import logging
import re
from typing import Any

from feather.models import ParallelConfig, ToolExecutionContext, ToolExecutionResult
from feather.providers.parallel_client import ParallelClient, ParallelSearchHit
from feather.tools.base import BaseTool

logger = logging.getLogger(__name__)

_SEARCH_MODES: tuple[str, ...] = ("fast", "one-shot", "agentic")

# Parallel's source_policy accepts:
#   - plain domains: example.com, subdomain.example.gov
#   - bare extensions: .gov, .edu, .co.uk
# It rejects schemes, paths, ports, or non-domain keywords ("news", "tech").
# We validate locally because LLMs frequently confuse these with categories
# and a downstream HTTP 422 wastes a turn.
_DOMAIN_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9-]*(\.[A-Za-z0-9][A-Za-z0-9-]*)+$"
)
_EXTENSION_PATTERN = re.compile(
    r"^\.[A-Za-z][A-Za-z0-9]*(\.[A-Za-z][A-Za-z0-9]*)*$"
)


def _is_valid_domain(value: object) -> bool:
    """Return True iff ``value`` is a domain string Parallel will accept.

    Accepts plain domains (``example.com``, ``sub.example.co.uk``) and bare
    extensions (``.gov``, ``.edu``, ``.co.uk``). Rejects empty strings,
    URLs with schemes/paths/ports, and category keywords like ``news``.
    """
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate:
        return False
    return bool(
        _DOMAIN_PATTERN.fullmatch(candidate)
        or _EXTENSION_PATTERN.fullmatch(candidate)
    )


def _render_invalid_domains_error(invalid: list[object]) -> str:
    """Render a self-correcting error for the agent.

    Surfacing the rule + showing which entries violated it lets the model
    fix the next call without an HTTP round-trip to Parallel.
    """
    quoted = ", ".join(repr(d) for d in invalid)
    return (
        f"web_search: invalid include_domains entries: [{quoted}]. "
        "Each entry must be a plain domain (e.g. 'example.com', "
        "'subdomain.example.gov') or a bare extension starting with a "
        "period ('.gov', '.edu', '.co.uk'). Schemes, paths, ports, and "
        "category keywords like 'news' or 'tech' are not allowed. "
        "Either fix the entries or omit include_domains entirely to "
        "search the whole web."
    )


class ParallelSearchTool(BaseTool):
    """Expose Parallel AI Search as a Feather tool named `web_search`."""

    name = "web_search"
    description = (
        "Search the web via Parallel AI. Returns ranked, LLM-ready excerpts with source URLs. "
        "Start with `mode='fast'` for quick lookups. If the returned excerpts are not rich "
        "enough to answer the objective, retry with `mode='one-shot'` (more comprehensive) or "
        "`mode='agentic'` (multi-step, token-efficient for deeper research). Prefer a concise "
        "natural-language `objective`; only supply `search_queries` when specific keywords matter."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "objective": {
                "type": "string",
                "description": "Natural-language description of what you want to learn.",
            },
            "search_queries": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Optional explicit keyword queries to augment the objective.",
            },
            "mode": {
                "type": ["string", "null"],
                "enum": ["fast", "one-shot", "agentic", None],
                "description": (
                    "Search mode. 'fast' (~1s, shallow). 'one-shot' (comprehensive, slower). "
                    "'agentic' (token-efficient for multi-step research). Omit for the configured default."
                ),
            },
            "max_results": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": 10,
                "description": "Upper bound on results. Omit for the configured default.",
            },
            "include_domains": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": (
                    "Optional allowlist of DOMAINS to restrict the search to. "
                    "Each entry must be a plain domain (e.g. 'example.com', "
                    "'subdomain.example.gov') or a bare extension starting "
                    "with a period ('.gov', '.edu', '.co.uk'). Schemes, "
                    "paths, ports, and category keywords ('news', 'tech', "
                    "'science') are NOT valid — Parallel will reject them. "
                    "Omit or pass null to search the whole web."
                ),
            },
        },
        "required": ["objective", "search_queries", "mode", "max_results", "include_domains"],
        "additionalProperties": False,
    }

    def __init__(self, client: ParallelClient, config: ParallelConfig) -> None:
        self._client = client
        self._config = config

    def get_prompt(self) -> str:
        """Describe the tool for prompt assembly."""

        return (
            f"- `{self.name}`: web search via Parallel AI. Returns ranked excerpts for an "
            "`objective`. If the excerpts cannot answer the question, retry with "
            "`mode='one-shot'` or `mode='agentic'` instead of the default `mode='fast'`."
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        """Run one Parallel search and format the results for the model."""

        del context  # unused

        objective = (arguments.get("objective") or "").strip()
        if not objective:
            raise ValueError("`objective` must not be empty.")

        raw_mode = arguments.get("mode")
        mode = raw_mode if raw_mode is not None else self._config.default_search_mode
        if mode not in _SEARCH_MODES:
            raise ValueError(
                f"Invalid mode `{mode}`. Expected one of: {', '.join(_SEARCH_MODES)}."
            )

        raw_max = arguments.get("max_results")
        max_results = int(raw_max) if raw_max is not None else self._config.max_results

        search_queries = arguments.get("search_queries")
        raw_include_domains = arguments.get("include_domains")
        if raw_include_domains:
            invalid = [
                d for d in raw_include_domains if not _is_valid_domain(d)
            ]
            if invalid:
                return ToolExecutionResult(
                    output=_render_invalid_domains_error(invalid)
                )
            include_domains = list(raw_include_domains)
        else:
            include_domains = None

        try:
            response = await self._client.search(
                objective=objective,
                search_queries=list(search_queries) if search_queries else None,
                mode=mode,
                max_results=max_results,
                include_domains=include_domains,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("parallel web_search failed")
            return ToolExecutionResult(
                output=f"web_search failed: {exc}"
            )

        if not response.results:
            return ToolExecutionResult(
                output=(
                    f"No results for objective: {objective!r} (mode={mode}).\n"
                    "Consider retrying with mode='one-shot' or mode='agentic' for deeper search."
                )
            )

        rendered = _render_results(objective, response.mode, response.results)
        return ToolExecutionResult(output=rendered)


def _render_results(
    objective: str, mode: str, hits: list[ParallelSearchHit]
) -> str:
    """Render Parallel search hits into a compact text block for the model."""

    lines: list[str] = [f"objective: {objective}", f"mode={mode} results={len(hits)}", ""]
    for index, hit in enumerate(hits, start=1):
        title = hit.title or "(untitled)"
        lines.append(f"{index}. {title}")
        lines.append(f"   url: {hit.url}")
        if hit.publish_date:
            lines.append(f"   published: {hit.publish_date}")
        for excerpt in hit.excerpts:
            lines.append(f"   excerpt: {excerpt.strip()}")
        lines.append("")
    return "\n".join(lines).rstrip()
