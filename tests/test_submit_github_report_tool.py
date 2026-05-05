"""Tests for the submit_github_report tool.

Validation / refusal paths run without ``gh`` on PATH. The success and
gh-failure paths use a fixture script written to ``tmp_path`` and
prepended to PATH so the tool calls *that* script as if it were ``gh``,
without needing real GitHub credentials in CI.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from feather.models import ToolExecutionContext
from feather.tools.submit_github_report_tool import SubmitGithubReportTool


def _ctx() -> ToolExecutionContext:
    return ToolExecutionContext(session_id="s1", agent_name="lead")


def _stage_fake_gh(
    tmp_path: Path,
    monkeypatch,
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
) -> Path:
    """Write a fake ``gh`` to tmp_path/bin and prepend it to PATH."""

    binp = tmp_path / "bin"
    binp.mkdir(parents=True, exist_ok=True)
    fake = binp / "gh"
    # Use a Python shim — works on every platform with python3 on PATH,
    # avoids shell-quoting hazards inside the heredoc.
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{binp}{os.pathsep}{os.environ.get('PATH', '')}")
    return fake


def _hide_gh(monkeypatch, tmp_path: Path) -> None:
    """Force ``shutil.which('gh')`` to return None by isolating PATH."""

    # An empty bin directory is enough — there's no real gh inside.
    empty = tmp_path / "empty_bin"
    empty.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PATH", str(empty))


# --------------------------------------------------------------------- #
# Validation / refusal paths
# --------------------------------------------------------------------- #


async def test_pr_kind_returns_not_supported_notice(tmp_path: Path, monkeypatch) -> None:
    _hide_gh(monkeypatch, tmp_path)
    tool = SubmitGithubReportTool()
    result = await tool.execute(
        {"kind": "pr", "title": "x", "body": "y"}, _ctx()
    )
    assert "not supported" in result.output.lower()
    assert "kind='issue'" in result.output


async def test_unknown_kind_rejected(tmp_path: Path, monkeypatch) -> None:
    _hide_gh(monkeypatch, tmp_path)
    tool = SubmitGithubReportTool()
    result = await tool.execute(
        {"kind": "discussion", "title": "x", "body": "y"}, _ctx()
    )
    assert "must be 'issue' or 'pr'" in result.output


async def test_empty_title_rejected(tmp_path: Path, monkeypatch) -> None:
    _hide_gh(monkeypatch, tmp_path)
    tool = SubmitGithubReportTool()
    result = await tool.execute(
        {"kind": "issue", "title": "   ", "body": "y"}, _ctx()
    )
    assert "title must be non-empty" in result.output


async def test_empty_body_rejected(tmp_path: Path, monkeypatch) -> None:
    _hide_gh(monkeypatch, tmp_path)
    tool = SubmitGithubReportTool()
    result = await tool.execute(
        {"kind": "issue", "title": "x", "body": ""}, _ctx()
    )
    assert "body must be non-empty" in result.output


async def test_oversize_title_rejected(tmp_path: Path, monkeypatch) -> None:
    _hide_gh(monkeypatch, tmp_path)
    tool = SubmitGithubReportTool()
    result = await tool.execute(
        {"kind": "issue", "title": "x" * 500, "body": "y"}, _ctx()
    )
    assert "title exceeds" in result.output.lower()


async def test_invalid_repo_format_rejected(tmp_path: Path, monkeypatch) -> None:
    _hide_gh(monkeypatch, tmp_path)
    tool = SubmitGithubReportTool()
    result = await tool.execute(
        {"kind": "issue", "title": "x", "body": "y", "repo": "not-a-slash"},
        _ctx(),
    )
    assert "owner/repo" in result.output


async def test_missing_gh_cli_returns_actionable_message(
    tmp_path: Path, monkeypatch
) -> None:
    _hide_gh(monkeypatch, tmp_path)
    tool = SubmitGithubReportTool()
    result = await tool.execute(
        {"kind": "issue", "title": "x", "body": "y"}, _ctx()
    )
    assert "`gh` CLI is not installed" in result.output
    assert "gh auth login" in result.output


# --------------------------------------------------------------------- #
# gh subprocess paths (with fixture script on PATH)
# --------------------------------------------------------------------- #


async def test_success_returns_issue_url(tmp_path: Path, monkeypatch) -> None:
    _stage_fake_gh(
        tmp_path,
        monkeypatch,
        stdout="https://github.com/timho102003/feather-os/issues/42\n",
        exit_code=0,
    )
    tool = SubmitGithubReportTool()
    result = await tool.execute(
        {
            "kind": "issue",
            "title": "Compaction loses last_response_id",
            "body": "Repro steps...",
        },
        _ctx(),
    )
    assert "Issue filed" in result.output
    assert "https://github.com/timho102003/feather-os/issues/42" in result.output


async def test_url_extraction_skips_warning_lines_before_url(
    tmp_path: Path, monkeypatch
) -> None:
    """gh sometimes emits a release-update notice on stdout before the URL.
    The tool must scan for the first https://github.com/* line, not blindly
    take the last line of stdout."""

    _stage_fake_gh(
        tmp_path,
        monkeypatch,
        stdout=(
            "! A new release of gh is available: 2.40.0 -> 2.50.0\n"
            "! https://github.com/cli/cli/releases/tag/v2.50.0\n"
            "https://github.com/timho102003/feather-os/issues/99\n"
            "Some trailing chatter\n"
        ),
        exit_code=0,
    )
    tool = SubmitGithubReportTool()
    result = await tool.execute(
        {"kind": "issue", "title": "x", "body": "y"}, _ctx()
    )
    assert "Issue filed" in result.output
    assert "https://github.com/timho102003/feather-os/issues/99" in result.output
    # The cli release-update URL must NOT be picked as the issue URL.
    assert "cli/releases" not in result.output


async def test_oversize_body_rejected_by_utf8_byte_count(
    tmp_path: Path, monkeypatch
) -> None:
    """A body that's well under 60k chars but over 60k UTF-8 bytes
    (e.g. CJK or emoji-heavy) must still be rejected so GitHub doesn't
    do it for us with a less helpful error."""

    _hide_gh(monkeypatch, tmp_path)
    tool = SubmitGithubReportTool()
    # 25k * 3 bytes (CJK) = 75k bytes — over the 60k cap.
    body = "中" * 25_000
    assert len(body) < 60_000
    assert len(body.encode("utf-8")) > 60_000
    result = await tool.execute(
        {"kind": "issue", "title": "x", "body": body}, _ctx()
    )
    assert "exceeds" in result.output.lower()
    assert "utf-8 bytes" in result.output.lower()


async def test_gh_nonzero_exit_surfaces_stderr_verbatim(
    tmp_path: Path, monkeypatch
) -> None:
    _stage_fake_gh(
        tmp_path,
        monkeypatch,
        stderr="HTTP 401: Bad credentials. Run gh auth login.",
        exit_code=4,
    )
    tool = SubmitGithubReportTool()
    result = await tool.execute(
        {"kind": "issue", "title": "x", "body": "y"}, _ctx()
    )
    assert "exit=4" in result.output
    assert "HTTP 401: Bad credentials" in result.output
    assert "gh auth login" in result.output  # remediation hint
