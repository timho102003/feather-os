"""Read file contents from the workspace and (opt-in) the global ``~/.feather`` tree."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.tools.base import BaseTool


class ReadFileTool(BaseTool):
    """Read text files from the workspace, including saved tool-output files.

    The default sandbox is the workspace root — typically the discovered
    project root, or the directory the user launched ``feather`` from
    when no project is detected.

    When constructed with a :class:`feather.paths.FeatherPaths`, files
    anywhere under the global root (``paths.global_root``, typically
    ``~/.feather``) are also readable so agents can inspect global
    config, agent YAMLs, skill bodies, the persona file, and runtime
    state markers without shelling out. Paths that escape every allowed
    root via ``..`` or symlinks are rejected.

    Defensive deny-list: any file whose name (after symlink resolution)
    starts with ``.env`` — including ``.env``, ``.env.local``,
    ``.env.production``, ``.env_backup``, ``.envrc`` (direnv), etc. —
    is refused, because those files commonly hold API keys. The check
    is name-based: a *hardlink* with a non-``.env`` name pointing at
    the same inode is NOT caught (``Path.resolve()`` walks symlinks but
    not hardlinks). ``read_file`` is best-effort defense in depth, not
    a sandbox; the ``bash`` tool remains the only trusted path for
    files the agent suspects contain secrets.
    """

    name = "read_file"
    description = (
        "Read a UTF-8 text file. Reads anywhere in the workspace and "
        "anywhere under `~/.feather/` (global config, agent YAMLs, "
        "skills, user.md, state markers). Files whose name starts with "
        "`.env` (e.g. `.env`, `.env.local`, `.envrc`) are refused — "
        "use `bash` if you genuinely need them. Paths outside every "
        "allowed root are rejected."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to read. Workspace-relative (e.g. `src/main.py`, "
                    "`.feather/tmp/bash/x.output`) or absolute (including "
                    "`~/.feather/...` which is expanded via $HOME)."
                ),
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

    def __init__(self, workspace_root: Path, paths: object = None) -> None:
        self._workspace_root = workspace_root.resolve()
        roots: list[Path] = [self._workspace_root]
        # When a FeatherPaths is provided, also whitelist the entire
        # ``~/.feather`` tree so agents can inspect global config, agent
        # YAMLs, skill bodies, user.md, and state markers without
        # shelling out.
        if paths is not None:
            try:
                roots.append(paths.global_root.resolve())  # type: ignore[attr-defined]
            except AttributeError:
                pass  # silently ignore non-FeatherPaths objects
        self._readable_roots: tuple[Path, ...] = tuple(roots)

    def get_prompt(self) -> str:
        """Describe how the read-file tool should be used.

        Returns:
            Prompt text for this tool.
        """

        return (
            "- `read_file`: read a text file by relative or absolute "
            "path. Reads anywhere in the workspace and under "
            "`~/.feather/`. Names starting with `.env` (e.g. `.env`, "
            "`.envrc`) are refused — use `bash` if you must. Best for "
            "inspecting tool-output files under `.feather/tmp/...` or "
            "globally-installed agent YAMLs and skills under "
            "`~/.feather/`."
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
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
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

        display = self._display_path(path)
        output = f"file: {display}\nlines: {start_line}-{end_line}\n{rendered}".rstrip()
        if len(output) > max_chars:
            output = f"{output[:max_chars]}\n... [truncated]"
        return ToolExecutionResult(output=output)

    def _display_path(self, path: Path) -> str:
        """Render ``path`` for the user.

        Workspace-relative paths render as ``relative/to/workspace`` to
        keep familiar output. Files under the user's home directory
        (e.g. global reads from ``~/.feather/...``) render with a ``~``
        prefix so the user's real ``$HOME`` is not echoed into chat or
        the prompt cache. Anything else falls back to the absolute path.
        """

        try:
            return str(path.relative_to(self._workspace_root))
        except ValueError:
            pass
        try:
            return f"~/{path.relative_to(Path.home())}"
        except ValueError:
            return str(path)

    @staticmethod
    def _is_dotenv_name(name: str) -> bool:
        """True for any filename starting with ``.env``.

        Catches ``.env``, ``.env.local``, ``.env.production``,
        ``.env_backup``, ``.envrc`` (direnv), and similar. Intentionally
        broad — read_file is best-effort defense in depth, and edge
        files like ``.envoy.yaml`` (rare) can still be reached via
        ``bash`` when needed.
        """

        return name.startswith(".env")

    def _resolve_path(self, raw_path: str) -> Path:
        """Resolve ``raw_path`` to an absolute file inside an allowed root.

        Order of operations matters: ``expanduser`` → ``resolve`` (which
        follows symlinks in every component) → membership check against
        ``_readable_roots`` → ``.env`` deny → existence + filetype
        checks. The membership check must precede ``exists()`` so a
        symlink in the workspace pointing at ``/etc/passwd`` is rejected
        even when the target exists.
        """

        try:
            candidate = Path(raw_path).expanduser()
        except RuntimeError as exc:
            raise ValueError(f"Could not expand path: {raw_path}") from exc
        if candidate.is_absolute():
            path = candidate.resolve()
        else:
            path = (self._workspace_root / candidate).resolve()
        if not any(path == root or root in path.parents for root in self._readable_roots):
            raise ValueError("File path must stay inside the workspace.")
        if self._is_dotenv_name(path.name):
            raise ValueError(
                f"`.env`-style files are refused by `read_file` to avoid "
                f"leaking secrets into chat. Use `bash` (e.g. `cat`) if "
                f"you really need to inspect: {raw_path}"
            )
        if not path.exists():
            raise ValueError(f"File does not exist: {raw_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {raw_path}")
        return path
