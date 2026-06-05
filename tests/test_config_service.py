"""Tests for the ConfigService orchestration layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.config import load_app_config
from feather.config.resolver import PathScope
from feather.config.service import ConfigService, ValueSource
from feather.paths import FeatherPaths


def _service(tmp_path: Path) -> ConfigService:
    paths = FeatherPaths(project_root=tmp_path / "proj", home=tmp_path / "global")
    paths.ensure_global_dirs()
    paths.ensure_project_dirs()
    cfg = load_app_config(paths.project_root, paths=paths)
    return ConfigService(paths=paths, app_config=cfg)


# ---------------------------------------------------------------------------
# Task 15: ConfigService.get
# ---------------------------------------------------------------------------


def test_get_returns_default_source_when_no_override(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    value = svc.get("app.active_provider")

    assert value.source == ValueSource.DEFAULT
    assert value.current in {"openai", "openrouter", "claude"}
    assert value.field.path == "app.active_provider"


def test_get_returns_global_source_when_override_present(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    (svc.paths.global_config_dir / "app.yaml").write_text(
        "active_provider: claude\n", encoding="utf-8"
    )

    # Re-create after writing so loader picks the new file
    svc2 = _service(tmp_path)
    value = svc2.get("app.active_provider")

    assert value.current == "claude"
    assert value.source == ValueSource.GLOBAL


def test_get_unknown_path_raises(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    with pytest.raises(KeyError):
        svc.get("app.nope.foo")


# ---------------------------------------------------------------------------
# Task 16: ConfigService.validate + set
# ---------------------------------------------------------------------------


def test_validate_accepts_valid_enum(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.validate("app.active_provider", "claude")

    assert result.ok


def test_validate_rejects_unknown_enum(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.validate("app.active_provider", "anthropic")

    assert not result.ok
    assert "openai" in (result.error or "")


def test_validate_rejects_negative_for_positive_validator(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.validate("app.memory.retrieval.top_k_tool", -1)

    assert not result.ok


def test_validate_coerces_strings_to_int(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.validate("app.memory.retrieval.top_k_tool", "12")

    assert result.ok
    assert result.coerced == 12


def test_set_writes_to_global_by_default(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.set("app.active_provider", "claude")

    assert result.ok
    overlay = svc.paths.global_config_dir / "app.yaml"
    assert overlay.exists()
    assert "claude" in overlay.read_text(encoding="utf-8")


def test_set_writes_to_project_when_flagged(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.set("app.active_provider", "claude", scope=PathScope.PROJECT)

    assert result.ok
    proj = tmp_path / "proj" / "config" / "app.yaml"
    assert proj.exists()


def test_set_rejects_invalid_value(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.set("app.active_provider", "anthropic")

    assert not result.ok
    overlay = svc.paths.global_config_dir / "app.yaml"
    assert not overlay.exists() or "anthropic" not in overlay.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Task 17: ConfigService.list, diff, reset
# ---------------------------------------------------------------------------


def test_list_filters_by_section(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    rows = svc.list(section="app.openai")

    paths = [r.field.path for r in rows]
    assert "app.openai.model" in paths
    assert all(p.startswith("app.openai") for p in paths)


def test_list_returns_all_when_section_blank(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    rows = svc.list(section="")

    assert len(rows) >= 50


def test_diff_shows_global_vs_default(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc.set("app.active_provider", "claude")
    svc.set("app.openai.temperature", 0.3)

    diff = svc.diff()

    assert "app.active_provider" in diff
    old, new = diff["app.active_provider"]
    assert new == "claude"
    assert old != new


def test_reset_removes_overlay(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc.set("app.active_provider", "claude")

    result = svc.reset("app.active_provider")

    assert result.ok
    diff = svc.diff()
    assert "app.active_provider" not in diff


def test_reset_no_op_when_no_overlay(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = svc.reset("app.active_provider")

    assert result.ok


# ---------------------------------------------------------------------------
# Task 7: self_repair.enabled force carve-out
# ---------------------------------------------------------------------------


def test_set_self_repair_without_force_refuses(tmp_path: Path) -> None:
    """ConfigService.set rejects self_repair.enabled unless force=True."""

    svc = _service(tmp_path)

    result = svc.set("app.self_repair.enabled", True)

    assert not result.ok
    assert "force" in (result.error or "").lower()


def test_set_self_repair_with_force_succeeds(tmp_path: Path) -> None:
    """ConfigService.set accepts self_repair.enabled when force=True."""

    svc = _service(tmp_path)

    result = svc.set("app.self_repair.enabled", True, force=True)

    assert result.ok
