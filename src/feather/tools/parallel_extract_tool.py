"""Web-fetch tool powered by the Parallel AI Extract API."""

from __future__ import annotations

import logging
from typing import Any

from feather.models import ParallelConfig, ToolExecutionContext, ToolExecutionResult
from feather.providers.parallel_client import (
    ParallelClient,
    ParallelExtractHit,
)
from feather.storage.tool_output_store import ToolOutputStore
from feather.tools.base import BaseTool

logger = logging.getLogger(__name__)

_EXTRACT_MODES: tuple[str, ...] = ("excerpts", "full")
_DEFAULT_MODE = "excerpts"


class ParallelExtractTool(BaseTool):
    """Expose Parallel AI Extract as a Feather tool named `web_fetch`."""

    name = "web_fetch"
    description = (
        "Fetch clean content from a specific URL via Parallel AI. Use `mode='excerpts'` "
        "(default) for short, objective-focused sections of the page. Escalate to "
        "`mode='full'` only when excerpts are insufficient — full pages can be large "
        "and are written to disk, referenced by path so you can follow up with `read_file`."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute URL of the page to fetch.",
            },
            "objective": {
                "type": ["string", "null"],
                "description": (
                    "Optional natural-language goal that focuses the returned excerpts."
                ),
            },
            "mode": {
                "type": ["string", "null"],
                "enum": ["excerpts", "full", None],
                "description": (
                    "Extraction mode. 'excerpts' (default) returns short focused sections. "
                    "'full' returns the complete page markdown. Large full pages are saved "
                    "to .feather/tmp and referenced by path."
                ),
            },
        },
        "required": ["url", "objective", "mode"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        client: ParallelClient,
        config: ParallelConfig,
        tool_output_store: ToolOutputStore,
    ) -> None:
        self._client = client
        self._config = config
        self._tool_output_store = tool_output_store

    def get_prompt(self) -> str:
        """Describe the tool for prompt assembly."""

        return (
            f"- `{self.name}`: fetch a URL via Parallel AI. Start with `mode='excerpts'`; "
            "only escalate to `mode='full'` when the excerpts are insufficient, since full "
            "pages can be large and will be saved to disk for follow-up reads with `read_file`."
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        """Fetch one URL via Parallel AI and format the output for the model."""

        del context  # unused

        url = (arguments.get("url") or "").strip()
        if not url:
            raise ValueError("`url` must not be empty.")

        raw_mode = arguments.get("mode")
        mode = raw_mode if raw_mode is not None else _DEFAULT_MODE
        if mode not in _EXTRACT_MODES:
            raise ValueError(
                f"Invalid mode `{mode}`. Expected one of: {', '.join(_EXTRACT_MODES)}."
            )

        objective_raw = arguments.get("objective")
        objective = objective_raw.strip() if isinstance(objective_raw, str) else None
        objective = objective or None

        try:
            response = await self._client.extract(
                url=url,
                objective=objective,
                include_full_content=(mode == "full"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("parallel web_fetch failed")
            return ToolExecutionResult(output=f"web_fetch failed: {exc}")

        if not response.results:
            error_suffix = f" errors: {'; '.join(response.errors)}" if response.errors else ""
            return ToolExecutionResult(
                output=f"No content returned for {url}.{error_suffix}"
            )

        hit = response.results[0]
        if mode == "full":
            return await self._render_full(url, hit)
        return ToolExecutionResult(output=_render_excerpts(url, hit))


    async def _render_full(
        self, url: str, hit: ParallelExtractHit
    ) -> ToolExecutionResult:
        """Format a `mode='full'` result, offloading oversized bodies to disk."""

        full_content = hit.full_content or ""
        threshold = self._config.inline_full_content_threshold
        header = _format_header(url, hit, mode="full")

        if len(full_content) <= threshold:
            return ToolExecutionResult(
                output=f"{header}\n\n{full_content}".rstrip()
            )

        artifact = await self._tool_output_store.write(self.name, full_content)
        body = (
            f"Full content saved to `{artifact.file_ref}` ({len(full_content)} chars). "
            "Use read_file to view slices."
        )
        return ToolExecutionResult(output=f"{header}\n\n{body}")


def _render_excerpts(url: str, hit: ParallelExtractHit) -> str:
    """Render excerpts-mode output for one extract hit."""

    header = _format_header(url, hit, mode="excerpts")
    if not hit.excerpts:
        return f"{header}\n\nNo excerpts returned."
    lines = [header, ""]
    for excerpt in hit.excerpts:
        lines.append(f"- {excerpt.strip()}")
    return "\n".join(lines)


def _format_header(url: str, hit: ParallelExtractHit, *, mode: str) -> str:
    """Build the compact header shown on every extract response."""

    title = hit.title or "(untitled)"
    published = f" published={hit.publish_date}" if hit.publish_date else ""
    return f"url: {hit.url or url}\ntitle: {title}\nmode={mode}{published}"
