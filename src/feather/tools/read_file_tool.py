"""Read file contents from the workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.tools.base import BaseTool


class ReadFileTool(BaseTool):
    """Read text files from the workspace, including saved tool-output files."""

    name = "read_file"
    description = (
        "Read a text file from the workspace. Use this when chat history references a stored "
        "tool-output file or when exact file contents matter."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file to read.",
            },
            "start_line": {
                "type": ["integer", "null"],
                "description": "Optional 1-based start line. Defaults to 1.",
                "minimum": 1,
            },
            "end_line": {
                "type": ["integer", "null"],
                "description": "Optional 1-based end line, inclusive. Defaults to the end of file.",
                "minimum": 1,
            },
            "max_chars": {
                "type": ["integer", "null"],
                "description": "Maximum characters to return. Defaults to 12000.",
                "minimum": 200,
                "maximum": 100000,
            },
        },
        "required": ["path", "start_line", "end_line", "max_chars"],
        "additionalProperties": False,
    }

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()

    def get_prompt(self) -> str:
        """Describe how the read-file tool should be used.

        Returns:
            Prompt text for this tool.
        """

        return (
            "- `read_file`: read a text file by relative path. Use this when history references "
            "a stored tool-output file under `.feather/tmp/...` and you need the full contents."
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        """Read file contents with optional line slicing and truncation.

        Args:
            arguments: Tool arguments from the model.
            context: Runtime context for the current tool invocation.

        Returns:
            Tool result containing the requested file slice.
        """

        path = self._resolve_path(arguments["path"])
        start_line = int(arguments.get("start_line") or 1)
        max_chars = int(arguments.get("max_chars") or 12000)

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File is not valid UTF-8 text: {arguments['path']}") from exc

        lines = text.splitlines()
        if not lines:
            rendered = ""
            end_line = start_line
        else:
            default_end_line = len(lines)
            end_line = int(arguments.get("end_line") or default_end_line)
            if end_line < start_line:
                raise ValueError("`end_line` must be greater than or equal to `start_line`.")
            selected = lines[start_line - 1 : end_line]
            rendered = "\n".join(
                f"{line_number}: {line}"
                for line_number, line in enumerate(selected, start=start_line)
            )

        relative = path.relative_to(self._workspace_root)
        output = f"file: {relative}\nlines: {start_line}-{end_line}\n{rendered}".rstrip()
        if len(output) > max_chars:
            output = f"{output[:max_chars]}\n... [truncated]"
        return ToolExecutionResult(output=output)

    def _resolve_path(self, raw_path: str) -> Path:
        path = (self._workspace_root / raw_path).resolve()
        if path != self._workspace_root and self._workspace_root not in path.parents:
            raise ValueError("File path must stay inside the workspace.")
        if not path.exists():
            raise ValueError(f"File does not exist: {raw_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {raw_path}")
        return path
