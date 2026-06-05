"""Tests for the /config slash dispatcher."""

from __future__ import annotations

from pathlib import Path

from feather.config import load_app_config
from feather.config.service import ConfigService
from feather.config.slash import handle_config_command
from feather.paths import FeatherPaths


def _service(tmp_path: Path) -> ConfigService:
    paths = FeatherPaths(project_root=tmp_path / "proj", home=tmp_path / "global")
    paths.ensure_global_dirs()
    paths.ensure_project_dirs()
    cfg = load_app_config(paths.project_root, paths=paths)
    return ConfigService(paths=paths, app_config=cfg)


def test_get_returns_current_value(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "get app.active_provider")

    assert result.ok
    assert "app.active_provider" in result.body
    assert "[default]" in result.body


def test_get_unknown_path(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "get app.does.not.exist")

    assert not result.ok
    assert "unknown" in result.body.lower()


def test_set_writes_to_global(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "set app.active_provider claude")

    assert result.ok
    assert result.requires_apply == ["app.active_provider"]
    overlay = svc.paths.global_config_dir / "app.yaml"
    assert "claude" in overlay.read_text(encoding="utf-8")


def test_set_with_project_flag(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "set app.active_provider claude --project")

    assert result.ok
    proj = tmp_path / "proj" / "config" / "app.yaml"
    assert proj.exists()


def test_set_rejects_invalid(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "set app.active_provider anthropic")

    assert not result.ok


def test_list_section(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "list app.openai")

    assert result.ok
    assert "app.openai.model" in result.body


def test_diff_after_set(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    handle_config_command(svc, "set app.active_provider claude")

    result = handle_config_command(svc, "diff")

    assert result.ok
    assert "app.active_provider" in result.body


def test_reset_after_set(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    handle_config_command(svc, "set app.active_provider claude")

    result = handle_config_command(svc, "reset app.active_provider")

    assert result.ok
    diff = handle_config_command(svc, "diff").body
    assert "app.active_provider" not in diff


def test_unknown_subcommand(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "xyz")

    assert not result.ok
    assert "unknown" in result.body.lower()


def test_bare_config_returns_usage_help(tmp_path: Path) -> None:
    svc = _service(tmp_path)

    result = handle_config_command(svc, "")

    # Bare /config is help, not an error (the modal is wired in the TUI now).
    assert result.ok
    body = result.body.lower()
    assert "usage" in body
    for sub in ("get", "set", "list", "diff", "reset"):
        assert sub in body


# ---------------------------------------------------------------------------
# Task 7: self_repair.enabled --force flag in slash dispatcher
# ---------------------------------------------------------------------------


def test_set_self_repair_without_force_refuses(tmp_path: Path) -> None:
    """Headless /config set app.self_repair.enabled true is rejected without --force."""

    svc = _service(tmp_path)

    result = handle_config_command(svc, "set app.self_repair.enabled true")

    assert not result.ok
    assert "force" in result.body.lower()


def test_set_self_repair_with_force(tmp_path: Path) -> None:
    """Headless /config set app.self_repair.enabled true --force succeeds."""

    svc = _service(tmp_path)

    result = handle_config_command(svc, "set app.self_repair.enabled true --force")

    assert result.ok
