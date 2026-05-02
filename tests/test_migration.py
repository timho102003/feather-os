"""Tests for the legacy → global migration prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.migration import (
    detect_legacy_artifacts,
    already_handled,
    maybe_migrate,
)
from feather.paths import FeatherPaths


@pytest.fixture
def paths(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".feather").mkdir()
    return FeatherPaths(project_root=proj, home=tmp_path / "home")


# ---------------------------------------------------------------------------
# detect_legacy_artifacts
# ---------------------------------------------------------------------------


def test_detect_returns_all_none_in_global_only_mode(tmp_path):
    paths = FeatherPaths.global_only(home=tmp_path / "home")
    artifacts = detect_legacy_artifacts(paths)
    assert artifacts.env_file is None
    assert artifacts.user_md is None
    assert artifacts.onboarded_marker is None
    assert artifacts.has_anything is False


def test_detect_finds_env_user_md_and_marker(paths):
    proj = paths.project_root
    (proj / ".env").write_text("OPENAI_API_KEY=sk-x\n")
    (proj / ".feather" / "user.md").write_text("# user\n")
    (proj / ".feather" / "onboarded.json").write_text('{"version": 1}\n')

    artifacts = detect_legacy_artifacts(paths)
    assert artifacts.env_file == proj / ".env"
    assert artifacts.user_md == proj / ".feather" / "user.md"
    assert artifacts.onboarded_marker == proj / ".feather" / "onboarded.json"
    assert artifacts.has_anything is True


def test_detect_skips_empty_files(paths):
    proj = paths.project_root
    (proj / ".env").write_text("")
    artifacts = detect_legacy_artifacts(paths)
    assert artifacts.env_file is None


# ---------------------------------------------------------------------------
# maybe_migrate
# ---------------------------------------------------------------------------


def test_maybe_migrate_is_no_op_when_nothing_to_migrate(paths):
    rc = maybe_migrate(paths, ask=lambda *_: "y", say=lambda *_: None)
    assert rc == "not-applicable"


def test_maybe_migrate_skips_when_already_handled(paths):
    proj = paths.project_root
    (proj / ".env").write_text("OPENAI_API_KEY=sk\n")
    (proj / ".feather" / "MIGRATED_TO_GLOBAL.txt").write_text("done")
    rc = maybe_migrate(paths, ask=lambda *_: "y", say=lambda *_: None)
    assert rc == "already-handled"


def test_maybe_migrate_copies_files_when_user_accepts(paths):
    proj = paths.project_root
    (proj / ".env").write_text("OPENAI_API_KEY=sk-from-project\n")
    (proj / ".feather" / "user.md").write_text("# user content\n")

    output: list[str] = []
    rc = maybe_migrate(paths, ask=lambda *_: "y", say=output.append)

    assert rc == "migrated"
    assert paths.env_file.read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-from-project\n"
    assert paths.global_user_md.read_text(encoding="utf-8") == "# user content\n"
    # Originals untouched
    assert (proj / ".env").exists()
    assert (proj / ".feather" / "user.md").exists()
    # Breadcrumb written
    assert (proj / ".feather" / "MIGRATED_TO_GLOBAL.txt").exists()
    assert any("Migration complete" in line for line in output)


def test_maybe_migrate_default_answer_migrates(paths):
    """Empty input should be treated as 'yes'."""
    proj = paths.project_root
    (proj / ".env").write_text("OPENAI_API_KEY=sk\n")
    rc = maybe_migrate(paths, ask=lambda *_: "", say=lambda *_: None)
    assert rc == "migrated"


def test_maybe_migrate_skip_writes_declined_breadcrumb(paths):
    proj = paths.project_root
    (proj / ".env").write_text("OPENAI_API_KEY=sk\n")
    rc = maybe_migrate(paths, ask=lambda *_: "skip", say=lambda *_: None)
    assert rc == "declined"
    assert (proj / ".feather" / "MIGRATION_DECLINED.txt").exists()


def test_maybe_migrate_no_does_not_write_breadcrumb(paths):
    """'no' leaves the door open to be prompted next run."""
    proj = paths.project_root
    (proj / ".env").write_text("OPENAI_API_KEY=sk\n")
    rc = maybe_migrate(paths, ask=lambda *_: "n", say=lambda *_: None)
    assert rc == "declined"
    assert not (proj / ".feather" / "MIGRATION_DECLINED.txt").exists()


def test_maybe_migrate_does_not_clobber_existing_global_state(paths):
    """Globally-staged files survive a re-migration prompt."""
    proj = paths.project_root
    (proj / ".env").write_text("OPENAI_API_KEY=project-key\n")

    paths.ensure_global_dirs()
    paths.env_file.write_text("OPENAI_API_KEY=already-global\n")

    maybe_migrate(paths, ask=lambda *_: "y", say=lambda *_: None)

    assert paths.env_file.read_text(encoding="utf-8") == "OPENAI_API_KEY=already-global\n"


def test_already_handled_global_only_returns_false(tmp_path):
    paths = FeatherPaths.global_only(home=tmp_path / "home")
    assert already_handled(paths) is False
