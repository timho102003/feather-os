"""Tool for extracting readable text from PDF attachments."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.integrations.attachments.pdf import extract_pdf_text
from feather.tools.base import BaseTool


class ReadPdfTool(BaseTool):
    """Extract text from a PDF, optionally using OpenDataLoader hybrid mode."""

    name = "read_pdf"
    description = (
        "Extract readable text from a PDF file. Use `opendataloader_hybrid` "
        "for complex scanned/layout-heavy PDFs when the optional command is configured."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path to a PDF file.",
            },
            "mode": {
                "type": ["string", "null"],
                "enum": ["auto", "text", "opendataloader_hybrid", None],
                "description": "Extraction backend. Defaults to auto.",
            },
            "max_chars": {
                "type": ["integer", "null"],
                "description": "Maximum characters to return. Defaults to 12000.",
                "minimum": 1000,
                "maximum": 100000,
            },
        },
        "required": ["path", "mode", "max_chars"],
        "additionalProperties": False,
    }

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()

    def get_prompt(self) -> str:
        """Describe the PDF extraction workflow."""

        return (
            "- `read_pdf`: extract text from PDFs saved in the workspace. "
            "Use `mode=opendataloader_hybrid` for complex PDFs when configured; "
            "otherwise use `auto`."
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        """Extract text from one PDF file."""

        path = self._resolve_path(arguments["path"])
        mode = str(arguments.get("mode") or "auto")
        max_chars = int(arguments.get("max_chars") or 12000)
        text = await asyncio.to_thread(
            extract_pdf_text,
            path,
            mode=mode,
            max_chars=max_chars,
        )
        relative = path.relative_to(self._workspace_root)
        return ToolExecutionResult(
            output=f"file: {relative}\nmode: {mode}\n{text}".rstrip()
        )

    def _resolve_path(self, raw_path: str) -> Path:
        path = (self._workspace_root / raw_path).resolve()
        if path != self._workspace_root and self._workspace_root not in path.parents:
            raise ValueError("File path must stay inside the workspace.")
        if not path.exists():
            raise ValueError(f"File does not exist: {raw_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {raw_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("read_pdf only supports .pdf files")
        return path
