"""Tests for install-mode detection (editable / wheel / read_only)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from feather.core.install_mode import InstallMode, detect_install_mode


def _stage_editable_layout(tmp_path: Path) -> Path:
    """Create a ``<repo>/src/feather/__init__.py`` skeleton + pyproject.toml."""

    repo = tmp_path / "repo"
    pkg = repo / "src" / "feather"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("# stub", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='feather'\n", encoding="utf-8")
    return pkg / "__init__.py"


def _stage_wheel_layout(tmp_path: Path) -> Path:
    """Mimic a site-packages layout — no pyproject.toml above the pkg."""

    site = tmp_path / "site-packages"
    pkg = site / "feather"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("# stub", encoding="utf-8")
    return pkg / "__init__.py"


def test_detect_editable_when_pyproject_above_package(tmp_path: Path) -> None:
    init_file = _stage_editable_layout(tmp_path)
    info = detect_install_mode(init_file)
    assert info.mode is InstallMode.EDITABLE
    assert info.repo_root == (tmp_path / "repo").resolve()
    assert info.is_durable() is True


def test_detect_wheel_when_no_pyproject_above_package(tmp_path: Path) -> None:
    init_file = _stage_wheel_layout(tmp_path)
    info = detect_install_mode(init_file)
    assert info.mode is InstallMode.WHEEL
    assert info.repo_root is None
    assert info.is_durable() is False


def test_detect_read_only_when_package_dir_not_writable(tmp_path: Path) -> None:
    init_file = _stage_wheel_layout(tmp_path)
    pkg_dir = init_file.parent
    # Strip write bits on the directory itself; the access check is on the
    # directory, not the file.
    pkg_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        # On systems running as root, os.access lies (root bypasses perms).
        if os.geteuid() == 0:
            pytest.skip("read_only detection cannot be tested as root")
        info = detect_install_mode(init_file)
        assert info.mode is InstallMode.READ_ONLY
        assert info.is_durable() is False
    finally:
        # Restore so pytest can clean up tmp_path.
        pkg_dir.chmod(stat.S_IRWXU)


def test_detect_default_uses_real_feather_init() -> None:
    """No-arg call must inspect the actually-imported feather package."""

    info = detect_install_mode()
    # Whichever mode the test runner is in, the call must succeed and the
    # package_path must point at our installed feather.
    assert info.package_path.exists()
    assert info.mode in {
        InstallMode.EDITABLE,
        InstallMode.WHEEL,
        InstallMode.READ_ONLY,
    }
