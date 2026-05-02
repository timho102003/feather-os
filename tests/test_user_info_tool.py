"""Tests for the lead-only ``user_info`` tool."""

from __future__ import annotations

from pathlib import Path

from feather.models import ToolExecutionContext
from feather.profile import UserProfileStore
from feather.tools.user_info_tool import UserInfoTool


def _ctx() -> ToolExecutionContext:
    return ToolExecutionContext(session_id="sess-1", agent_name="Lead")


async def test_user_info_create_round_trip(tmp_path: Path) -> None:
    """A CREATE call writes the field and returns a confirmation string."""

    store = UserProfileStore(tmp_path / "user.md")
    tool = UserInfoTool(store)
    result = await tool.execute(
        {"operation": "CREATE", "field": "name", "value": "Tim", "note": None},
        _ctx(),
    )

    assert "created" in result.output.lower()
    assert store.load().fields["name"] == "Tim"


async def test_user_info_update_round_trip(tmp_path: Path) -> None:
    """UPDATE replaces an existing field's value."""

    store = UserProfileStore(tmp_path / "user.md")
    await store.create("role", "Engineer")
    tool = UserInfoTool(store)
    await tool.execute(
        {"operation": "UPDATE", "field": "role", "value": "Senior", "note": None},
        _ctx(),
    )
    assert store.load().fields["role"] == "Senior"


async def test_user_info_delete_round_trip(tmp_path: Path) -> None:
    """DELETE removes the named field."""

    store = UserProfileStore(tmp_path / "user.md")
    await store.create("name", "Tim")
    tool = UserInfoTool(store)
    await tool.execute(
        {"operation": "DELETE", "field": "name", "value": None, "note": None},
        _ctx(),
    )
    assert "name" not in store.load().fields


async def test_user_info_append_note_round_trip(tmp_path: Path) -> None:
    """APPEND_NOTE adds a bullet under ``## Notes``."""

    store = UserProfileStore(tmp_path / "user.md")
    await store.create("name", "Tim")
    tool = UserInfoTool(store)
    result = await tool.execute(
        {"operation": "APPEND_NOTE", "field": None, "value": None, "note": "loves rust"},
        _ctx(),
    )
    assert "appended" in result.output.lower()
    assert "loves rust" in store.render()


async def test_user_info_create_missing_field_returns_error(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "user.md")
    tool = UserInfoTool(store)
    result = await tool.execute(
        {"operation": "CREATE", "field": None, "value": "Tim", "note": None},
        _ctx(),
    )
    assert "requires" in result.output.lower()


async def test_user_info_create_reserved_returns_error(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "user.md")
    tool = UserInfoTool(store)
    result = await tool.execute(
        {"operation": "CREATE", "field": "created_at", "value": "now", "note": None},
        _ctx(),
    )
    assert "reserved" in result.output.lower()


async def test_user_info_update_unknown_returns_error(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "user.md")
    tool = UserInfoTool(store)
    result = await tool.execute(
        {"operation": "UPDATE", "field": "name", "value": "Tim", "note": None},
        _ctx(),
    )
    assert "does not exist" in result.output.lower()


async def test_user_info_invalid_operation_returns_error(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "user.md")
    tool = UserInfoTool(store)
    result = await tool.execute(
        {"operation": "TRUNCATE", "field": "name", "value": None, "note": None},
        _ctx(),
    )
    assert "invalid operation" in result.output.lower()


async def test_user_info_append_note_missing_text_returns_error(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "user.md")
    tool = UserInfoTool(store)
    result = await tool.execute(
        {"operation": "APPEND_NOTE", "field": None, "value": None, "note": None},
        _ctx(),
    )
    assert "requires" in result.output.lower()
