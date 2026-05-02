"""Repository grep tool."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.tools.base import BaseTool


class GrepTool(BaseTool):
    """Search text files under the workspace."""

    name = "grep"
    description = "Search repository files by regex pattern. Use this to find code, config, or docs."
    parameters_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {
                "type": ["string", "null"],
                "description": "Optional relative path to limit the search. Defaults to the workspace root.",
            },
            "case_sensitive": {
                "type": ["boolean", "null"],
                "description": "Whether the regex should be case sensitive.",
                "default": False,
            },
            "max_results": {
                "type": ["integer", "null"],
                "description": "Maximum number of matching lines to return.",
                "default": 20,
                "minimum": 1,
                "maximum": 200,
            },
        },
        "required": ["pattern", "path", "case_sensitive", "max_results"],
        "additionalProperties": False,
    }

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        """Search the repository.

        Args:
            arguments: Tool arguments from the model.
            context: Runtime context for the current tool invocation.

        Returns:
            Matching lines and file locations.
        """

        pattern = arguments["pattern"]
        case_sensitive = bool(arguments.get("case_sensitive") or False)
        max_results = int(arguments.get("max_results") or 20)
        search_root = self._resolve_search_root(arguments.get("path") or ".")
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)

        matches: list[str] = []
        for path in sorted(search_root.rglob("*")):
            if len(matches) >= max_results:
                break
            if not path.is_file() or self._should_skip(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            for line_number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    relative = path.relative_to(self._workspace_root)
                    matches.append(f"{relative}:{line_number}: {line.strip()}")
                    if len(matches) >= max_results:
                        break

        if not matches:
            return ToolExecutionResult(output="No matches found.")
        return ToolExecutionResult(output="\n".join(matches))

    def _resolve_search_root(self, raw_path: str) -> Path:
        path = (self._workspace_root / raw_path).resolve()
        if self._workspace_root not in path.parents and path != self._workspace_root:
            raise ValueError("Search path must stay inside the workspace.")
        if not path.exists():
            raise ValueError(f"Search path does not exist: {raw_path}")
        return path

    def _should_skip(self, path: Path) -> bool:
        parts = set(path.relative_to(self._workspace_root).parts)
        return bool(parts & {".git", ".venv", "__pycache__", ".pytest_cache", ".feather", "build", "dist"})
