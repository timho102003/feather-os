"""Global config / skills should be writable when FeatherPaths is provided."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.models import ToolExecutionContext
from feather.paths import FeatherPaths
from feather.tools.write_file_tool import WriteFileTool


def _ctx() -> ToolExecutionContext:
    return ToolExecutionContext(agent_name="lead", session_id="s1")


async def test_write_file_accepts_global_config_dir_when_paths_provided(tmp_path):
    home = tmp_path / "home"
    paths = FeatherPaths(project_root=tmp_path, home=home)
    paths.ensure_global_dirs()
    tool = WriteFileTool(tmp_path, paths=paths)

    target = paths.global_agents_dir / "demo-custom.yaml"
    result = await tool.execute(
        {
            "path": str(target),
            "content": "name: demo\n",
            "overwrite": None,
            "create_parents": True,
        },
        _ctx(),
    )
    assert "wrote" in result.output
    assert target.read_text(encoding="utf-8").startswith("name: demo")


async def test_write_file_accepts_global_skills_dir_when_paths_provided(tmp_path):
    home = tmp_path / "home"
    paths = FeatherPaths(project_root=tmp_path, home=home)
    paths.ensure_global_dirs()
    tool = WriteFileTool(tmp_path, paths=paths)

    target = paths.global_skills_dir / "demo" / "SKILL.md"
    result = await tool.execute(
        {
            "path": str(target),
            "content": "---\nname: demo\ndescription: x\n---\n\nbody",
            "overwrite": None,
            "create_parents": True,
        },
        _ctx(),
    )
    assert "wrote" in result.output


async def test_write_file_succeeds_when_global_root_lives_outside_workspace(tmp_path):
    """Regression: red-team finding C1 — when ~/.feather lives outside the
    workspace_root, writing to it must not crash on relative_to()."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "homedir"  # NOT a child of workspace
    paths = FeatherPaths(project_root=workspace, home=home)
    paths.ensure_global_dirs()
    tool = WriteFileTool(workspace, paths=paths)

    target = paths.global_skills_dir / "demo" / "SKILL.md"
    result = await tool.execute(
        {
            "path": str(target),
            "content": "---\nname: demo\ndescription: x\n---\n\nbody",
            "overwrite": None,
            "create_parents": True,
        },
        _ctx(),
    )
    assert "wrote" in result.output
    assert target.is_file()


async def test_write_file_still_rejects_outside_all_roots_with_paths(tmp_path):
    """A path outside the workspace AND outside the global config/skills
    dirs must still be rejected. ``home`` is placed *as a sibling* of the
    workspace so it is genuinely outside every allowed root (otherwise it
    would fall inside the workspace by accident)."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "homedir"  # sibling, not a child of workspace
    paths = FeatherPaths(project_root=workspace, home=home)
    paths.ensure_global_dirs()
    tool = WriteFileTool(workspace, paths=paths)

    # ~/.feather/skills is allowed, but ~/.feather (the parent) is not
    forbidden = home / "secrets.txt"
    with pytest.raises(ValueError, match="may only write inside"):
        await tool.execute(
            {
                "path": str(forbidden),
                "content": "x",
                "overwrite": None,
                "create_parents": True,
            },
            _ctx(),
        )
