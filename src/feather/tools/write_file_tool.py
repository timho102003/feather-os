"""Write text files into the workspace's whitelisted directories."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.tools.base import BaseTool

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 1_048_576
# Project-relative whitelist still covers ``.feather`` (runtime state)
# and ``config`` for back-compat with tests that stage a project-local
# config tree. Production agents that want to edit YAML now point at
# the global override paths exposed via the optional FeatherPaths arg.
_PROJECT_WRITABLE_SUBDIRS: tuple[str, ...] = (".feather", "config")


class WriteFileTool(BaseTool):
    """Write text files atomically inside the writable sandbox.

    The sandbox covers ``.feather/`` (runtime state, scratch artifacts,
    skill files) and ``config/`` (agent + app YAML so agents can edit
    their own configuration). When constructed with a
    :class:`feather.paths.FeatherPaths`, the global config and skills
    directories (``~/.feather/config``, ``~/.feather/skills``) are added
    to the whitelist so the agent-creator skill can persist new agents
    and skills user-globally instead of polluting the project tree.
    Paths outside every allowed root are rejected.
    """

    name = "write_file"
    description = (
        "Write a UTF-8 text file. Writes are restricted to `.feather/` "
        "(runtime state, artifacts, skills) and `config/` (app + agent "
        "YAML). Paths outside both roots are rejected. Creates parent "
        "directories by default. Refuses to overwrite an existing file "
        "unless `overwrite=true`."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Workspace-relative path inside `.feather/` or `config/` "
                    "(e.g. `.feather/artifacts/notes.md`, "
                    "`config/agents/explore.yaml`). Absolute paths are "
                    "accepted only if they fall inside an allowed root."
                ),
            },
            "content": {
                "type": "string",
                "description": "Full UTF-8 file content to write.",
            },
            "overwrite": {
                "type": ["boolean", "null"],
                "description": "If true, replace an existing file. Defaults to false.",
            },
            "create_parents": {
                "type": ["boolean", "null"],
                "description": "If true (default), create missing parent directories.",
            },
        },
        "required": ["path", "content", "overwrite", "create_parents"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace_root: Path,
        paths: object = None,
    ) -> None:
        self._workspace_root = workspace_root.resolve()
        roots: list[Path] = [
            (self._workspace_root / sub).resolve()
            for sub in _PROJECT_WRITABLE_SUBDIRS
        ]
        # When a FeatherPaths is provided, also whitelist the global
        # config + skills dirs so agents can persist new YAMLs and
        # skills user-globally.
        if paths is not None:
            try:
                roots.append(paths.global_config_dir.resolve())  # type: ignore[attr-defined]
                roots.append(paths.global_skills_dir.resolve())  # type: ignore[attr-defined]
            except AttributeError:
                pass  # silently ignore non-FeatherPaths objects
        self._writable_roots: tuple[Path, ...] = tuple(roots)

    def get_prompt(self) -> str:
        """Describe how the write-file tool should be used."""

        return (
            "- `write_file`: write a UTF-8 text file under `.feather/` "
            "(scratch / artifacts / skills) or `config/` (app + agent YAML). "
            "Any other path is rejected. Set `overwrite=true` to replace an "
            "existing file."
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        """Write `content` to `path` atomically inside the `.feather/` sandbox.

        Args:
            arguments: Tool arguments from the model.
            context: Runtime context for the current tool invocation.

        Returns:
            Tool result describing the write.
        """

        raw_path = arguments["path"]
        content = arguments["content"]
        overwrite = bool(arguments.get("overwrite") or False)
        create_parents = arguments.get("create_parents")
        if create_parents is None:
            create_parents = True
        else:
            create_parents = bool(create_parents)

        if not isinstance(content, str):
            raise ValueError("`content` must be a string.")
        if len(content) > _MAX_CONTENT_CHARS:
            raise ValueError(
                f"Content too large: {len(content)} chars exceeds the "
                f"{_MAX_CONTENT_CHARS}-char cap."
            )

        path = self._resolve_path(raw_path)
        if path.is_dir():
            raise ValueError(f"Path is a directory, not a file: {raw_path}")

        existed = path.exists()
        if existed and not overwrite:
            raise ValueError(
                f"File already exists at {raw_path}. Set `overwrite=true` to replace."
            )

        parent = path.parent
        if not parent.exists():
            if not create_parents:
                raise ValueError(
                    f"Parent directory does not exist: {self._display_path(parent)}"
                )
            parent.mkdir(parents=True, exist_ok=True)

        encoded = content.encode("utf-8")
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=parent
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    logger.exception(
                        "write_file.tmp_cleanup_failed path=%s tmp=%s",
                        path,
                        tmp_path,
                    )
            raise

        display = self._display_path(path)
        logger.info(
            "tool.write_file agent=%s session_id=%s path=%s bytes=%d created=%s",
            context.agent_name,
            context.session_id,
            display,
            len(encoded),
            not existed,
        )
        return ToolExecutionResult(
            output=f"wrote {len(encoded)} bytes to {display} (created={not existed})"
        )

    def _display_path(self, path: Path) -> str:
        """Return ``path`` rendered for the user in the most readable form.

        Prefers ``relative_to(workspace_root)`` so project-relative
        writes keep their familiar ``.feather/foo`` rendering. Falls
        back to the matched whitelist root for paths that live outside
        the workspace (e.g. global ``~/.feather/skills`` writes), and
        finally to the absolute path. Only the human-facing message is
        affected; the actual write uses the absolute resolved path.
        """
        try:
            return str(path.relative_to(self._workspace_root))
        except ValueError:
            pass
        for root in self._writable_roots:
            try:
                return str(path.relative_to(root))
            except ValueError:
                continue
        return str(path)

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            path = candidate.resolve()
        else:
            path = (self._workspace_root / candidate).resolve()
        for allowed in self._writable_roots:
            if path == allowed or allowed in path.parents:
                return path
        allowed_summary = ", ".join(str(p) for p in self._writable_roots)
        raise ValueError(
            f"write_file may only write inside one of: {allowed_summary} "
            f"(got: {raw_path})."
        )
