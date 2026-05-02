"""Tests for the write-file tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from feather.models import ToolExecutionContext
from feather.tools.write_file_tool import WriteFileTool


def _ctx() -> ToolExecutionContext:
    return ToolExecutionContext(session_id="session-1", agent_name="Lead")


async def test_write_file_creates_new_file_inside_feather(tmp_path: Path) -> None:
    """Happy path: writes a new file under .feather/ with content intact."""

    tool = WriteFileTool(tmp_path)
    result = await tool.execute(
        {
            "path": ".feather/artifacts/note.txt",
            "content": "hello world",
            "overwrite": None,
            "create_parents": None,
        },
        _ctx(),
    )

    written = tmp_path / ".feather" / "artifacts" / "note.txt"
    assert written.read_text(encoding="utf-8") == "hello world"
    assert "wrote" in result.output
    assert ".feather/artifacts/note.txt" in result.output


async def test_write_file_refuses_overwrite_by_default(tmp_path: Path) -> None:
    """If the file exists and overwrite is false (default), raise."""

    target = tmp_path / ".feather" / "x.txt"
    target.parent.mkdir(parents=True)
    target.write_text("original", encoding="utf-8")

    tool = WriteFileTool(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        await tool.execute(
            {
                "path": ".feather/x.txt",
                "content": "new",
                "overwrite": None,
                "create_parents": None,
            },
            _ctx(),
        )

    assert target.read_text(encoding="utf-8") == "original"


async def test_write_file_overwrites_when_flag_set(tmp_path: Path) -> None:
    """With overwrite=true, an existing file is replaced."""

    target = tmp_path / ".feather" / "x.txt"
    target.parent.mkdir(parents=True)
    target.write_text("original", encoding="utf-8")

    tool = WriteFileTool(tmp_path)
    await tool.execute(
        {
            "path": ".feather/x.txt",
            "content": "replacement",
            "overwrite": True,
            "create_parents": None,
        },
        _ctx(),
    )

    assert target.read_text(encoding="utf-8") == "replacement"


async def test_write_file_auto_creates_parent_dirs(tmp_path: Path) -> None:
    """Default create_parents=true means deep paths just work."""

    tool = WriteFileTool(tmp_path)
    await tool.execute(
        {
            "path": ".feather/artifacts/nested/deep/file.txt",
            "content": "x",
            "overwrite": None,
            "create_parents": None,
        },
        _ctx(),
    )

    assert (tmp_path / ".feather" / "artifacts" / "nested" / "deep" / "file.txt").is_file()


async def test_write_file_refuses_when_create_parents_false(tmp_path: Path) -> None:
    """create_parents=false: raise if the parent does not already exist."""

    tool = WriteFileTool(tmp_path)
    with pytest.raises(ValueError, match="[Pp]arent directory"):
        await tool.execute(
            {
                "path": ".feather/missing/file.txt",
                "content": "x",
                "overwrite": None,
                "create_parents": False,
            },
            _ctx(),
        )


async def test_write_file_rejects_paths_outside_writable_roots(tmp_path: Path) -> None:
    """Paths outside both `.feather/` and `config/` must be rejected."""

    tool = WriteFileTool(tmp_path)
    for forbidden in ["pyproject.toml", "src/feather/x.py", "tests/foo.py"]:
        with pytest.raises(ValueError, match=r"may only write inside"):
            await tool.execute(
                {
                    "path": forbidden,
                    "content": "x",
                    "overwrite": None,
                    "create_parents": None,
                },
                _ctx(),
            )


async def test_write_file_allows_config_directory(tmp_path: Path) -> None:
    """config/ is a whitelisted writable root for editing agent + app YAML."""

    tool = WriteFileTool(tmp_path)
    await tool.execute(
        {
            "path": "config/agents/explore.yaml",
            "content": "name: Explore\n",
            "overwrite": None,
            "create_parents": None,
        },
        _ctx(),
    )
    assert (
        tmp_path / "config" / "agents" / "explore.yaml"
    ).read_text(encoding="utf-8") == "name: Explore\n"


async def test_write_file_allows_top_level_config_file(tmp_path: Path) -> None:
    """Top-level config/app.yaml lives directly under config/."""

    tool = WriteFileTool(tmp_path)
    await tool.execute(
        {
            "path": "config/app.yaml",
            "content": "active_provider: openai\n",
            "overwrite": None,
            "create_parents": None,
        },
        _ctx(),
    )
    assert (tmp_path / "config" / "app.yaml").is_file()


async def test_write_file_rejects_workspace_escape(tmp_path: Path) -> None:
    """`..`-escapes that resolve outside the workspace are rejected."""

    tool = WriteFileTool(tmp_path)
    with pytest.raises(ValueError):
        await tool.execute(
            {
                "path": "../etc/passwd",
                "content": "x",
                "overwrite": None,
                "create_parents": None,
            },
            _ctx(),
        )


async def test_write_file_rejects_overwrite_of_directory(tmp_path: Path) -> None:
    """The path must not refer to an existing directory."""

    target = tmp_path / ".feather" / "adir"
    target.mkdir(parents=True)

    tool = WriteFileTool(tmp_path)
    with pytest.raises(ValueError, match="directory"):
        await tool.execute(
            {
                "path": ".feather/adir",
                "content": "x",
                "overwrite": True,
                "create_parents": None,
            },
            _ctx(),
        )


async def test_write_file_enforces_size_cap(tmp_path: Path) -> None:
    """Content above the configured cap is rejected without writing."""

    huge = "x" * (1_048_577)  # one byte past the 1 MB cap

    tool = WriteFileTool(tmp_path)
    with pytest.raises(ValueError, match="too large"):
        await tool.execute(
            {
                "path": ".feather/big.txt",
                "content": huge,
                "overwrite": None,
                "create_parents": None,
            },
            _ctx(),
        )


async def test_write_file_is_atomic_on_failure(tmp_path: Path) -> None:
    """If the rename step fails, the existing file content must be preserved."""

    target = tmp_path / ".feather" / "atomic.txt"
    target.parent.mkdir(parents=True)
    target.write_text("original", encoding="utf-8")

    tool = WriteFileTool(tmp_path)
    with patch("feather.tools.write_file_tool.os.replace", side_effect=OSError("fs full")):
        with pytest.raises(OSError):
            await tool.execute(
                {
                    "path": ".feather/atomic.txt",
                    "content": "replacement",
                    "overwrite": True,
                    "create_parents": None,
                },
                _ctx(),
            )

    assert target.read_text(encoding="utf-8") == "original"
    # No half-written temp files left behind in the directory.
    leftovers = [p for p in target.parent.iterdir() if p.name != "atomic.txt"]
    assert leftovers == []


def test_write_file_schema_is_openai_strict_compatible(tmp_path: Path) -> None:
    """All properties must be in `required` for strict tool calling."""

    tool = WriteFileTool(tmp_path)
    properties = tool.parameters_schema["properties"]

    assert sorted(tool.parameters_schema["required"]) == sorted(properties.keys())
    assert tool.parameters_schema["additionalProperties"] is False
    assert properties["overwrite"]["type"] == ["boolean", "null"]
    assert properties["create_parents"]["type"] == ["boolean", "null"]


def test_write_file_prompt_describes_writable_roots(tmp_path: Path) -> None:
    """The tool prompt must surface both whitelisted roots so the model uses valid paths."""

    tool = WriteFileTool(tmp_path)
    prompt = tool.get_prompt()
    assert ".feather" in prompt
    assert "config" in prompt
    assert ".feather" in tool.description
    assert "config" in tool.description
