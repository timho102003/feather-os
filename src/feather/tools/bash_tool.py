"""Workspace-scoped bash tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.tools.base import BaseTool


class BashTool(BaseTool):
    """Execute short bash commands inside the workspace."""

    name = "bash"
    description = (
        "Run a bash command inside the workspace when inspection or simple automation is faster "
        "than other tools. Keep commands short, targeted, and non-interactive."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute.",
            },
            "cwd": {
                "type": ["string", "null"],
                "description": "Optional relative working directory inside the workspace.",
            },
            "timeout_ms": {
                "type": ["integer", "null"],
                "description": "Optional timeout in milliseconds. Defaults to 10000.",
                "minimum": 1,
                "maximum": 120000,
            },
            "max_output_chars": {
                "type": ["integer", "null"],
                "description": "Maximum output characters to return. Defaults to 4000.",
                "minimum": 200,
                "maximum": 20000,
            },
        },
        "required": ["command", "cwd", "timeout_ms", "max_output_chars"],
        "additionalProperties": False,
    }

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        """Execute a non-interactive bash command.

        Args:
            arguments: Tool arguments from the model.
            context: Runtime context for the current tool invocation.

        Returns:
            Command output including exit status.
        """

        command = arguments["command"].strip()
        if not command:
            raise ValueError("`command` must not be empty.")

        cwd = self._resolve_cwd(arguments.get("cwd") or ".")
        timeout_seconds = (int(arguments.get("timeout_ms") or 10000)) / 1000
        max_output_chars = int(arguments.get("max_output_chars") or 4000)

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            executable="/bin/bash",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            process.kill()
            await process.communicate()
            return ToolExecutionResult(
                output=(
                    f"Command timed out after {timeout_seconds:.1f}s.\n"
                    f"cwd: {cwd.relative_to(self._workspace_root) or Path('.')}\n"
                    f"command: {command}"
                )
            )

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        sections = [
            f"exit_code: {process.returncode}",
            f"cwd: {self._display_cwd(cwd)}",
            f"command: {command}",
        ]
        if stdout_text:
            sections.append(f"stdout:\n{stdout_text}")
        if stderr_text:
            sections.append(f"stderr:\n{stderr_text}")

        output = "\n".join(sections)
        if len(output) > max_output_chars:
            output = f"{output[:max_output_chars]}\n... [truncated]"
        return ToolExecutionResult(output=output)

    def _resolve_cwd(self, raw_cwd: str) -> Path:
        cwd = (self._workspace_root / raw_cwd).resolve()
        if cwd != self._workspace_root and self._workspace_root not in cwd.parents:
            raise ValueError("`cwd` must stay inside the workspace.")
        if not cwd.exists():
            raise ValueError(f"Working directory does not exist: {raw_cwd}")
        if not cwd.is_dir():
            raise ValueError(f"Working directory is not a directory: {raw_cwd}")
        return cwd

    def _display_cwd(self, cwd: Path) -> str:
        relative = cwd.relative_to(self._workspace_root)
        return "." if str(relative) == "." else str(relative)
