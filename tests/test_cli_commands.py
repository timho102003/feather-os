"""Tests for the new top-level feather subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feather.cli.commands import (
    init_memory,
    init_project,
    memory_enabled_via_marker,
    memory_url_from_marker,
    remove_memory,
    stop_memory,
)
from feather.paths import FeatherPaths


# ---------------------------------------------------------------------------
# init_project
# ---------------------------------------------------------------------------


def test_init_project_creates_feather_dirs_and_registers(tmp_path):
    proj = tmp_path / "code" / "myproj"
    proj.mkdir(parents=True)
    home = tmp_path / "home"
    paths = FeatherPaths.for_project(proj, home=home)

    output: list[str] = []
    rc = init_project(paths, say=output.append)

    assert rc == 0
    assert (proj / ".feather" / "db").is_dir()
    assert (proj / ".feather" / "tmp").is_dir()
    assert (proj / ".feather" / "logs").is_dir()
    assert (proj / ".feather" / "skills").is_dir()
    assert paths.projects_index.is_file()
    data = json.loads(paths.projects_index.read_text(encoding="utf-8"))
    assert str(proj) in data["projects"]
    assert "Initialized" in output[-1]


def test_init_project_is_idempotent(tmp_path):
    proj = tmp_path / "p"
    proj.mkdir()
    home = tmp_path / "home"
    paths = FeatherPaths.for_project(proj, home=home)

    init_project(paths, say=lambda *_: None)
    init_project(paths, say=lambda *_: None)

    data = json.loads(paths.projects_index.read_text(encoding="utf-8"))
    assert data["projects"].count(str(proj)) == 1


def test_init_project_refuses_in_global_only_mode(tmp_path):
    paths = FeatherPaths.global_only(home=tmp_path / "home")
    output: list[str] = []
    rc = init_project(paths, say=output.append)
    assert rc != 0


# ---------------------------------------------------------------------------
# init_memory / stop_memory / remove_memory
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_paths(tmp_path):
    return FeatherPaths.global_only(home=tmp_path / "home")


def test_init_memory_writes_marker_when_container_starts(
    isolated_paths, monkeypatch
):
    """When ensure_local_qdrant_container succeeds, marker is written."""

    monkeypatch.setattr(
        "feather.cli.commands.docker_available", lambda **_: True
    )
    monkeypatch.setattr(
        "feather.cli.commands.qdrant_container_state",
        lambda **_: type("S", (), {"state": "absent"})(),
    )
    monkeypatch.setattr(
        "feather.cli.commands.ensure_local_qdrant_container",
        lambda **_: "http://localhost:6333",
    )
    output: list[str] = []
    rc = init_memory(isolated_paths, say=output.append)
    assert rc == 0
    assert isolated_paths.memory_marker.is_file()
    payload = json.loads(isolated_paths.memory_marker.read_text(encoding="utf-8"))
    assert payload["url"] == "http://localhost:6333"
    assert payload["mode"] == "local-docker"
    assert payload["container_name"]
    assert payload["started_at"]


def test_init_memory_is_no_op_when_marker_exists_and_container_running(
    isolated_paths, monkeypatch
):
    isolated_paths.ensure_global_dirs()
    isolated_paths.memory_marker.write_text(
        json.dumps({"url": "http://existing:6333"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "feather.cli.commands.qdrant_container_state",
        lambda **_: type("S", (), {"state": "running"})(),
    )

    started = []

    def _no_call(**kwargs):  # pragma: no cover — should not be invoked
        started.append("called")
        return "should-not-happen"

    monkeypatch.setattr("feather.cli.commands.ensure_local_qdrant_container", _no_call)
    output: list[str] = []
    rc = init_memory(isolated_paths, say=output.append)
    assert rc == 0
    assert started == []
    assert any("already running" in line for line in output)


def test_init_memory_returns_error_when_docker_missing(isolated_paths, monkeypatch):
    monkeypatch.setattr(
        "feather.cli.commands.docker_available", lambda **_: False
    )
    monkeypatch.setattr(
        "feather.cli.commands.qdrant_container_state",
        lambda **_: type("S", (), {"state": "absent"})(),
    )
    output: list[str] = []
    rc = init_memory(isolated_paths, say=output.append)
    assert rc == 1
    assert not isolated_paths.memory_marker.exists()
    assert any("Docker" in line for line in output)


def test_stop_memory_calls_helper_and_preserves_marker(
    isolated_paths, monkeypatch
):
    isolated_paths.ensure_global_dirs()
    isolated_paths.memory_marker.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "feather.cli.commands.docker_available", lambda **_: True
    )
    monkeypatch.setattr(
        "feather.cli.commands.stop_local_qdrant_container", lambda **_: "stopped"
    )
    output: list[str] = []
    rc = stop_memory(isolated_paths, say=output.append)
    assert rc == 0
    assert isolated_paths.memory_marker.exists()
    assert any("stopped" in line for line in output)


def test_remove_memory_deletes_container_and_marker(
    isolated_paths, monkeypatch
):
    isolated_paths.ensure_global_dirs()
    isolated_paths.memory_marker.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "feather.cli.commands.docker_available", lambda **_: True
    )
    monkeypatch.setattr(
        "feather.cli.commands.remove_local_qdrant_container", lambda **_: "removed"
    )
    output: list[str] = []
    rc = remove_memory(isolated_paths, purge=False, say=output.append)
    assert rc == 0
    assert not isolated_paths.memory_marker.exists()
    assert any("removed" in line for line in output)


# ---------------------------------------------------------------------------
# Marker query helpers
# ---------------------------------------------------------------------------


def test_memory_enabled_via_marker(isolated_paths):
    assert memory_enabled_via_marker(isolated_paths) is False
    isolated_paths.ensure_global_dirs()
    isolated_paths.memory_marker.write_text("{}", encoding="utf-8")
    assert memory_enabled_via_marker(isolated_paths) is True


def test_memory_url_from_marker_returns_recorded_url(isolated_paths):
    isolated_paths.ensure_global_dirs()
    isolated_paths.memory_marker.write_text(
        json.dumps({"url": "http://x:6333"}), encoding="utf-8"
    )
    assert memory_url_from_marker(isolated_paths) == "http://x:6333"


def test_memory_url_from_marker_returns_none_when_absent(isolated_paths):
    assert memory_url_from_marker(isolated_paths) is None


def test_memory_url_from_marker_returns_none_when_corrupted(isolated_paths):
    isolated_paths.ensure_global_dirs()
    isolated_paths.memory_marker.write_text("not json", encoding="utf-8")
    assert memory_url_from_marker(isolated_paths) is None
