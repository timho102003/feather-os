"""Tests for the ConfigService orchestration layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.config import load_app_config
from feather.config_service import ConfigService, ValueSource
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
