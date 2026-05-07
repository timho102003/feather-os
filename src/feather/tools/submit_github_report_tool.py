"""Submit an issue (and eventually a PR) to a GitHub repo via the ``gh`` CLI.

Designed to ship the loop "lead noticed a bug → user agreed to file it
→ issue lands upstream so the fix persists across package releases."
The tool itself is intentionally narrow: validate inputs, shell out to
``gh issue create``, return the URL or a structured error. Quality
control (template, "never auto-submit", green-tests precondition) lives
in the ``submit-github-report`` skill, not here.

For v1 only ``kind="issue"`` is supported. ``kind="pr"`` returns a clear
"not yet" so the model surfaces the limit instead of silently failing.

The tool requires the ``gh`` CLI to be installed and authenticated. It
detects this lazily inside ``execute`` so the agent_factory doesn't pay
the subprocess cost at build time, and so the model gets a clear
actionable message at the moment of failure rather than at startup.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any

from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.tools.base import BaseTool

logger = logging.getLogger(__name__)


_GH_TIMEOUT_SECONDS = 30.0
_MAX_TITLE_LENGTH = 200
# GitHub's issue body limit is ~65 535 BYTES (UTF-8), not chars. A body
# of CJK or emoji-heavy text encodes to ~3x its char count, so a pure
# char-length check would let the model send a body that GitHub then
# rejects. We check encoded byte length and leave a small headroom.
_MAX_BODY_BYTES = 60_000


class SubmitGithubReportTool(BaseTool):
    """File a GitHub issue (or, in a future revision, a PR) via the ``gh`` CLI."""

    name = "submit_github_report"
    description = (
        "File a GitHub issue (or, eventually, a PR) on the upstream repo. "
        "Always read the submit-github-report skill first; it carries the "
        "issue template, hard rules ('never auto-submit', 'one bug per "
        "report', 'tests must pass'), and confirmation flow. Requires the "
        "`gh` CLI installed and `gh auth login` completed."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["issue", "pr"],
                "description": (
                    "Always pass 'issue' for now. 'pr' is reserved and will "
                    "return a 'not supported' notice — surface that to the "
                    "user verbatim."
                ),
            },
            "title": {
                "type": "string",
                "description": (
                    "Short title (under 70 chars recommended). Reuse the "
                    "skill's template phrasing."
                ),
            },
            "body": {
                "type": "string",
                "description": (
                    "Markdown body. Use the issue template from the skill "
                    "with the Summary / Expected / Actual / Reproduction / "
                    "Environment / Diagnostic context sections."
                ),
            },
            "repo": {
                "type": "string",
                "description": (
                    "Optional 'owner/repo' (e.g. 'timho102003/feather-os'). "
                    "Defaults to the upstream repo when not supplied."
                ),
            },
        },
        "required": ["kind", "title", "body"],
        "additionalProperties": False,
    }

    def __init__(self, *, default_repo: str = "timho102003/feather-os") -> None:
        self._default_repo = default_repo

    def get_prompt(self) -> str:
        return (
            "- `submit_github_report`: file an issue upstream. ALWAYS load "
            "the submit-github-report skill first via `load_skill`, "
            "confirm the title/body with the user, then call. Never "
            "auto-submit. PR support is not enabled — `kind='pr'` is "
            "rejected with a clear notice."
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        del context  # not used; kind/title/body/repo are sufficient
        kind = str(arguments.get("kind", "")).strip().lower()
        title = str(arguments.get("title", "")).strip()
        body = str(arguments.get("body", "")).strip()
        repo = str(arguments.get("repo") or self._default_repo).strip()

        if kind == "pr":
            return ToolExecutionResult(
                output=(
                    "kind='pr' is not supported in this release of the tool. "
                    "Tell the user: 'I can file a detailed issue describing "
                    "the bug and the patch I'd propose, and you can open "
                    "the PR yourself.' Then retry with kind='issue'."
                )
            )
        if kind != "issue":
            return ToolExecutionResult(
                output=(
                    f"kind must be 'issue' or 'pr' (got {kind!r})."
                )
            )
        if not title:
            return ToolExecutionResult(output="title must be non-empty.")
        if not body:
            return ToolExecutionResult(output="body must be non-empty.")
        if len(title) > _MAX_TITLE_LENGTH:
            return ToolExecutionResult(
                output=(
                    f"title exceeds {_MAX_TITLE_LENGTH} chars — please "
                    "shorten before retrying."
                )
            )
        body_bytes = len(body.encode("utf-8"))
        if body_bytes > _MAX_BODY_BYTES:
            return ToolExecutionResult(
                output=(
                    f"body exceeds {_MAX_BODY_BYTES} UTF-8 bytes "
                    f"(currently {body_bytes}) — GitHub's API caps the "
                    "issue body at ~65 KB. Trim diagnostic context "
                    "before retrying. Note: non-ASCII characters take "
                    "more than 1 byte each."
                )
            )
        if "/" not in repo:
            return ToolExecutionResult(
                output=(
                    f"repo must be in 'owner/repo' form (got {repo!r})."
                )
            )

        if shutil.which("gh") is None:
            return ToolExecutionResult(
                output=(
                    "The `gh` CLI is not installed (or not on PATH). "
                    "Tell the user to install it — `brew install gh` on "
                    "macOS or the distro package on linux — then run "
                    "`gh auth login` before retrying."
                )
            )

        argv = [
            "gh",
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            title,
            "--body",
            body,
        ]
        try:
            stdout, stderr, returncode = await _run_gh(argv)
        except asyncio.TimeoutError:
            return ToolExecutionResult(
                output=(
                    f"gh issue create timed out after {_GH_TIMEOUT_SECONDS}s. "
                    "GitHub may be slow or `gh` may be hung — retry or have "
                    "the user run the equivalent command manually."
                )
            )
        if returncode != 0:
            return ToolExecutionResult(
                output=(
                    f"gh issue create failed (exit={returncode}). "
                    f"stderr (verbatim): {stderr.strip() or '<empty>'}\n"
                    "If this says 'not authenticated', tell the user to "
                    "run `gh auth login`. If it says the repo is missing, "
                    "double-check the `repo` argument."
                )
            )
        # gh emits the issue URL on its own line, but may also print
        # release-update notices or "Creating issue in owner/repo" on
        # stdout depending on version + terminal detection. Scan for the
        # first https://github.com/* line instead of "last line wins".
        url = ""
        for line in stdout.splitlines():
            candidate = line.strip()
            if candidate.startswith("https://github.com/"):
                url = candidate
                break
        if not url:
            return ToolExecutionResult(
                output=(
                    "gh issue create returned exit=0 but no GitHub URL on "
                    f"stdout. Raw stdout: {stdout!r}"
                )
            )
        logger.info(
            "submit_github_report.issue_created repo=%s url=%s", repo, url
        )
        return ToolExecutionResult(
            output=(
                f"Issue filed: {url}\n"
                "Share this URL with the user so they can subscribe to "
                "updates and (if relevant) link a PR back to it."
            )
        )


async def _run_gh(argv: list[str]) -> tuple[str, str, int]:
    """Run a ``gh`` subprocess with a hard timeout; return (stdout, stderr, rc)."""

    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=_GH_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise
    return (
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
        process.returncode if process.returncode is not None else -1,
    )
