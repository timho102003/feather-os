"""Tests for the dotted-path resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.config.resolver import (
    ConfigPathResolver,
    PathResolution,
    PathScope,
)
from feather.paths import FeatherPaths


def _paths(tmp_path: Path) -> FeatherPaths:
    return FeatherPaths(project_root=tmp_path / "proj", home=tmp_path / "global")


def test_resolve_app_path_global(tmp_path: Path) -> None:
    resolver = ConfigPathResolver(_paths(tmp_path))
    res = resolver.resolve("app.openai.model", scope=PathScope.GLOBAL)
    assert isinstance(res, PathResolution)
    assert res.file == tmp_path / "global" / "config" / "app.yaml"
    assert res.yaml_path == ["openai", "model"]


def test_resolve_app_path_project(tmp_path: Path) -> None:
    resolver = ConfigPathResolver(_paths(tmp_path))
    res = resolver.resolve("app.openai.model", scope=PathScope.PROJECT)
    assert res.file == tmp_path / "proj" / "config" / "app.yaml"


def test_resolve_agent_path(tmp_path: Path) -> None:
    """Registry uses the agent's capitalized display name (`Lead`) but the
    shipped YAMLs are lowercase (`lead.yaml`). The resolver lowercases on
    the way to disk so writes land in the file the loader actually reads."""

    resolver = ConfigPathResolver(_paths(tmp_path))
    res = resolver.resolve("agents.Lead.model", scope=PathScope.GLOBAL)
    assert res.file == tmp_path / "global" / "config" / "agents" / "lead.yaml"
    assert res.yaml_path == ["model"]


def test_resolve_rejects_unknown_prefix(tmp_path: Path) -> None:
    resolver = ConfigPathResolver(_paths(tmp_path))
    with pytest.raises(ValueError):
        resolver.resolve("session.foo", scope=PathScope.GLOBAL)


def test_resolve_rejects_short_path(tmp_path: Path) -> None:
    resolver = ConfigPathResolver(_paths(tmp_path))
    with pytest.raises(ValueError):
        resolver.resolve("agents.Lead", scope=PathScope.GLOBAL)
