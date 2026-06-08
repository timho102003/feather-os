"""SoulLibrary: layered packaged → global → project discovery."""

from __future__ import annotations

from pathlib import Path

from feather.core.leads.soul_library import SoulLibrary
from feather.paths import FeatherPaths

_TEMPLATE = """title: {title}
personality: One liner.
color: "#123456"
emoji: "🧭"
prose: |
  You work in a {title} way, methodically and with care.
tags: [demo]
"""


def _write_soul(directory: Path, soul_id: str, title: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{soul_id}.yaml").write_text(
        _TEMPLATE.format(title=title), encoding="utf-8"
    )


def test_project_soul_is_discovered(tmp_path: Path) -> None:
    _write_soul(tmp_path / "config" / "souls", "custom-one", "Custom")
    lib = SoulLibrary(tmp_path)  # no paths → packaged + project only
    soul = lib.get("custom-one")
    assert soul is not None and soul.title == "Custom"


def test_project_overrides_global(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    _write_soul(home / "config" / "souls", "dup", "FromGlobal")
    _write_soul(project / "config" / "souls", "dup", "FromProject")
    _write_soul(home / "config" / "souls", "global-only", "GlobalOnly")
    paths = FeatherPaths(project_root=project, home=home)
    lib = SoulLibrary(project, paths=paths)
    ids = {s.id for s in lib.list()}
    assert "global-only" in ids
    assert lib.get("dup").title == "FromProject"  # project wins


def test_broken_yaml_is_skipped(tmp_path: Path) -> None:
    souls = tmp_path / "config" / "souls"
    _write_soul(souls, "good", "Good")
    souls.mkdir(parents=True, exist_ok=True)
    (souls / "bad.yaml").write_text("display_name: OnlyName\n", encoding="utf-8")  # missing fields
    (souls / "garbage.yaml").write_text("::: not yaml :::\n", encoding="utf-8")
    lib = SoulLibrary(tmp_path)
    ids = {s.id for s in lib.list()}
    assert "good" in ids
    assert "bad" not in ids and "garbage" not in ids


def test_get_miss_returns_none(tmp_path: Path) -> None:
    assert SoulLibrary(tmp_path).get("does-not-exist") is None


def test_packaged_library_is_available_without_paths(tmp_path: Path) -> None:
    """The 20-soul built-in library must load even with no project/global dirs."""
    lib = SoulLibrary(tmp_path)
    ids = {s.id for s in lib.list()}
    assert "systems-thinker" in ids  # a known packaged soul
    assert len(ids) >= 20
