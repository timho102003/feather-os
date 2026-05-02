"""Persist tool outputs to `.feather/tmp` and expose stable file references."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from feather.models import ToolOutputArtifact


class ToolOutputStore:
    """Write tool outputs to per-tool files under the workspace temp directory."""

    def __init__(self, workspace_root: Path, temp_directory: str) -> None:
        self._workspace_root = workspace_root.resolve()
        self._temp_root = (self._workspace_root / temp_directory).resolve()

    async def write(self, tool_name: str, text: str) -> ToolOutputArtifact:
        """Persist one tool output and return its reference details.

        Args:
            tool_name: Tool that produced the output.
            text: Full tool output.

        Returns:
            Persisted output artifact.
        """

        safe_tool_name = self._sanitize_tool_name(tool_name)
        directory = self._temp_root / safe_tool_name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{uuid4()}.output"
        path.write_text(text, encoding="utf-8")
        file_ref = str(path.relative_to(self._workspace_root))
        return ToolOutputArtifact(
            tool_name=tool_name,
            file_ref=file_ref,
            text=text,
            reference_text=f"{tool_name} tool call output content file: {file_ref}",
        )

    def _sanitize_tool_name(self, tool_name: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", tool_name.strip()).strip("-")
        return sanitized or "tool"
