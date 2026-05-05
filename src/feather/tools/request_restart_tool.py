"""Self-repair tool: queue a clean lead-worker restart.

The lead calls this *after* it has patched Feather's own code (via
``write_file``, ``bash``, etc.) and verified the change with tests.
The tool itself does not kill any process — it just sets a flag on
the session row. The supervisor's restart watcher polls that flag,
performs a graceful worker shutdown + respawn on the same
``session_id``, and posts a "restart succeeded" inbox message so the
lead resumes naturally.

Tool surface is intentionally minimal:

* ``reason`` — short human-readable string the supervisor logs for
  diagnostics. Required so the lead has to articulate *why* it's
  asking for a restart, which makes accidental calls less likely.

The tool surfaces install-mode context in its response so the model
can warn the user when the patched files won't survive a package
upgrade (wheel / read-only installs). It never refuses the call —
even an ephemeral patch is useful inside the current session.
"""

from __future__ import annotations

from typing import Any

from feather.core.install_mode import InstallInfo, InstallMode
from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.storage.session_store import SessionStore
from feather.tools.base import BaseTool


class RequestRestartTool(BaseTool):
    """Queue a clean restart of the lead worker subprocess."""

    name = "request_restart"
    description = (
        "After you have patched Feather's own code and verified the change "
        "(eg `bash uv run pytest`), call this to ask the supervisor to "
        "respawn the lead worker so the patched modules are reloaded. The "
        "current conversation continues automatically on the new worker."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": (
                    "One sentence explaining what you patched and why a "
                    "restart is needed. Logged by the supervisor."
                ),
            }
        },
        "required": ["reason"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        session_store: SessionStore,
        install_info: InstallInfo,
    ) -> None:
        self._session_store = session_store
        self._install_info = install_info

    def get_prompt(self) -> str:
        return (
            "- `request_restart`: queue a graceful restart of the lead "
            "worker subprocess so patched feather/* modules reload. The "
            "current session continues on the new worker; conversation "
            "history is preserved. Only call after the patch is committed "
            "to disk and tests pass — do not use as an interrupt mechanism."
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        reason = str(arguments.get("reason", "")).strip()
        if not reason:
            return ToolExecutionResult(
                output=(
                    "request_restart requires a non-empty `reason` so the "
                    "supervisor can log what was patched."
                )
            )

        await self._session_store.mark_restart_requested(
            context.session_id, reason
        )
        notice = _install_mode_notice(self._install_info)
        body = (
            "Restart queued. The supervisor will respawn the lead worker "
            "on its next poll and resume this session.\n"
            f"{notice}"
        )
        return ToolExecutionResult(output=body)


def _install_mode_notice(info: InstallInfo) -> str:
    # Surface relative directory names rather than absolute paths so the
    # notice (which may be quoted in user-shared bug reports later) does
    # not leak the operator's filesystem layout.
    match info.mode:
        case InstallMode.EDITABLE:
            checkout_label = (
                info.repo_root.name if info.repo_root is not None else "<unknown>"
            )
            return (
                "Install mode: editable. Your patch lives in the source "
                f"checkout '{checkout_label}/' and survives package upgrades."
            )
        case InstallMode.WHEEL:
            return (
                "Install mode: wheel. Your patch lives in site-packages "
                "and will be silently overwritten on the next "
                "`pip install --upgrade`. Consider opening an issue or "
                "PR upstream via `submit_github_report` so the fix is "
                "preserved across releases."
            )
        case InstallMode.READ_ONLY:
            return (
                "Install mode: read-only (the package directory is not "
                "writable). Self-repair is unlikely to work — the file "
                "edit will fail. Reinstall feather into a writable venv "
                "before retrying."
            )
    # Defensive default for unforeseen future InstallMode members. Keeps
    # the function's declared return type honest under static analysis
    # without papering over real coverage gaps.
    return f"Install mode: {info.mode.value} (no specific guidance)."
