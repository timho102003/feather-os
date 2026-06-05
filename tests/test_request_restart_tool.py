"""Tests for the request_restart tool and its SessionStore plumbing."""

from __future__ import annotations

from pathlib import Path

from feather.core.install_mode import InstallInfo, InstallMode
from feather.models import ToolExecutionContext
from feather.storage.session_store import SessionStore
from feather.tools.request_restart_tool import RequestRestartTool


async def _open_session_store(tmp_path: Path) -> SessionStore:
    store = SessionStore(tmp_path / "feather.db")
    await store.initialize()
    return store


def _ctx(session_id: str) -> ToolExecutionContext:
    return ToolExecutionContext(session_id=session_id, agent_name="lead")


def _editable_info() -> InstallInfo:
    return InstallInfo(
        mode=InstallMode.EDITABLE,
        package_path=Path("/repo/src/feather"),
        repo_root=Path("/repo"),
    )


def _wheel_info() -> InstallInfo:
    return InstallInfo(
        mode=InstallMode.WHEEL,
        package_path=Path("/site-packages/feather"),
        repo_root=None,
    )


def _readonly_info() -> InstallInfo:
    return InstallInfo(
        mode=InstallMode.READ_ONLY,
        package_path=Path("/usr/lib/feather"),
        repo_root=None,
    )


# --------------------------------------------------------------------- #
# SessionStore restart-flag plumbing
# --------------------------------------------------------------------- #


async def test_mark_and_get_restart_request(tmp_path: Path) -> None:
    store = await _open_session_store(tmp_path)
    try:
        session = await store.create_session("lead")
        assert await store.get_restart_request(session.id) is None

        await store.mark_restart_requested(session.id, "patched compaction.py")
        flag = await store.get_restart_request(session.id)
        assert flag is not None
        ts, reason = flag
        assert reason == "patched compaction.py"
        assert "T" in ts  # ISO-8601 timestamp
    finally:
        await store.close()


async def test_clear_restart_request_removes_flag(tmp_path: Path) -> None:
    store = await _open_session_store(tmp_path)
    try:
        session = await store.create_session("lead")
        await store.mark_restart_requested(session.id, "fix")
        await store.clear_restart_request(session.id)
        assert await store.get_restart_request(session.id) is None
    finally:
        await store.close()


async def test_mark_restart_overwrites_previous_request(tmp_path: Path) -> None:
    """A second call replaces the first reason — last writer wins."""

    store = await _open_session_store(tmp_path)
    try:
        session = await store.create_session("lead")
        await store.mark_restart_requested(session.id, "first")
        await store.mark_restart_requested(session.id, "second")
        flag = await store.get_restart_request(session.id)
        assert flag is not None
        assert flag[1] == "second"
    finally:
        await store.close()


async def test_mark_restart_normalises_blank_reason(tmp_path: Path) -> None:
    """A whitespace-only reason gets a placeholder, never empty string."""

    store = await _open_session_store(tmp_path)
    try:
        session = await store.create_session("lead")
        await store.mark_restart_requested(session.id, "   ")
        flag = await store.get_restart_request(session.id)
        assert flag is not None
        assert flag[1] != ""
    finally:
        await store.close()


# --------------------------------------------------------------------- #
# RequestRestartTool behaviour
# --------------------------------------------------------------------- #


async def test_tool_writes_flag_and_returns_editable_notice(tmp_path: Path) -> None:
    store = await _open_session_store(tmp_path)
    try:
        session = await store.create_session("lead")
        tool = RequestRestartTool(store, _editable_info())
        result = await tool.execute(
            {"reason": "patched feather.core.agent.compaction"},
            _ctx(session.id),
        )
        # Flag is set on the session row.
        flag = await store.get_restart_request(session.id)
        assert flag is not None
        assert flag[1] == "patched feather.core.agent.compaction"
        # Response surfaces install mode so the model can warn the user.
        assert "Restart queued" in result.output
        assert "editable" in result.output.lower()
    finally:
        await store.close()


async def test_tool_warns_user_in_wheel_mode(tmp_path: Path) -> None:
    store = await _open_session_store(tmp_path)
    try:
        session = await store.create_session("lead")
        tool = RequestRestartTool(store, _wheel_info())
        result = await tool.execute(
            {"reason": "patched something"}, _ctx(session.id)
        )
        # Even in wheel mode the tool succeeds (does not refuse) — the
        # model needs the option for session-scoped fixes — but the
        # warning must mention the upgrade-overwrite risk.
        assert "wheel" in result.output.lower()
        assert "submit_github_report" in result.output.lower()
        # Flag is still set.
        assert await store.get_restart_request(session.id) is not None
    finally:
        await store.close()


async def test_tool_warns_about_read_only_install(tmp_path: Path) -> None:
    store = await _open_session_store(tmp_path)
    try:
        session = await store.create_session("lead")
        tool = RequestRestartTool(store, _readonly_info())
        result = await tool.execute(
            {"reason": "patched x"}, _ctx(session.id)
        )
        assert "read-only" in result.output.lower()
    finally:
        await store.close()


async def test_tool_rejects_empty_reason(tmp_path: Path) -> None:
    """Force the model to articulate why — refuses blank without setting flag."""

    store = await _open_session_store(tmp_path)
    try:
        session = await store.create_session("lead")
        tool = RequestRestartTool(store, _editable_info())
        result = await tool.execute({"reason": "   "}, _ctx(session.id))
        # No restart queued.
        assert await store.get_restart_request(session.id) is None
        assert "non-empty `reason`" in result.output
    finally:
        await store.close()
