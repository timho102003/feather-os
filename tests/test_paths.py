"""Tests for FeatherPaths path resolution."""

from pathlib import Path

import pytest

from feather.paths import FeatherPaths


# ---------------------------------------------------------------------------
# Global path getters
# ---------------------------------------------------------------------------


def test_global_root_uses_explicit_home():
    paths = FeatherPaths(project_root=None, home=Path("/fake/home"))
    assert paths.global_root == Path("/fake/home")


def test_global_subdirs_compose_under_global_root():
    paths = FeatherPaths(project_root=None, home=Path("/h"))
    assert paths.global_config_dir == Path("/h/config")
    assert paths.global_agents_dir == Path("/h/config/agents")
    assert paths.global_skills_dir == Path("/h/skills")
    assert paths.global_state_dir == Path("/h/state")
    assert paths.memory_marker == Path("/h/state/memory.json")
    assert paths.onboarded_marker == Path("/h/state/onboarded.json")
    assert paths.projects_index == Path("/h/state/projects.json")
    assert paths.global_sessions_db == Path("/h/state/sessions.db")
    assert paths.env_file == Path("/h/.env")
    assert paths.global_user_md == Path("/h/user.md")


def test_default_home_uses_path_home(monkeypatch, tmp_path):
    monkeypatch.delenv("FEATHER_HOME", raising=False)
    fake_home = tmp_path / "fakeuser"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    paths = FeatherPaths(project_root=None)
    assert paths.global_root == fake_home / ".feather"


def test_feather_home_env_var_overrides_default(monkeypatch, tmp_path):
    custom = tmp_path / "custom-home"
    monkeypatch.setenv("FEATHER_HOME", str(custom))
    paths = FeatherPaths(project_root=None)
    assert paths.global_root == custom


def test_explicit_home_arg_beats_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("FEATHER_HOME", str(tmp_path / "from-env"))
    paths = FeatherPaths(project_root=None, home=tmp_path / "explicit")
    assert paths.global_root == tmp_path / "explicit"


# ---------------------------------------------------------------------------
# Project path getters
# ---------------------------------------------------------------------------


def test_project_subdirs_compose_under_project_root():
    paths = FeatherPaths(project_root=Path("/proj"), home=Path("/h"))
    assert paths.project_feather_dir == Path("/proj/.feather")
    assert paths.db_path == Path("/proj/.feather/db/feather.db")
    assert paths.tmp_dir == Path("/proj/.feather/tmp")
    assert paths.subagent_staging_dir == Path("/proj/.feather/tmp/subagent_tasks")
    assert paths.log_dir == Path("/proj/.feather/logs")
    assert paths.log_file == Path("/proj/.feather/logs/feather.log")
    assert paths.project_skills_dir == Path("/proj/.feather/skills")
    assert paths.project_user_md == Path("/proj/.feather/user.md")
    assert paths.attachments_dir == Path("/proj/.feather/attachments")
    assert paths.artifacts_dir == Path("/proj/.feather/artifacts")
    assert paths.project_env_file == Path("/proj/.env")


@pytest.mark.parametrize(
    "prop",
    [
        "project_feather_dir",
        "db_path",
        "tmp_dir",
        "subagent_staging_dir",
        "log_dir",
        "log_file",
        "project_skills_dir",
        "project_user_md",
        "attachments_dir",
        "artifacts_dir",
        "project_env_file",
    ],
)
def test_project_getters_raise_in_global_only_mode(prop):
    paths = FeatherPaths(project_root=None, home=Path("/h"))
    with pytest.raises(RuntimeError, match="not in project mode"):
        getattr(paths, prop)


def test_is_project_mode_reflects_project_root():
    assert FeatherPaths(project_root=None, home=Path("/h")).is_project_mode is False
    assert FeatherPaths(project_root=Path("/p"), home=Path("/h")).is_project_mode is True


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def test_ensure_global_dirs_creates_subtree(tmp_path):
    paths = FeatherPaths(project_root=None, home=tmp_path / "home")
    first = paths.ensure_global_dirs()
    assert first is True
    assert (tmp_path / "home" / "config" / "agents").is_dir()
    assert (tmp_path / "home" / "skills").is_dir()
    assert (tmp_path / "home" / "state").is_dir()


def test_ensure_global_dirs_is_idempotent(tmp_path):
    paths = FeatherPaths(project_root=None, home=tmp_path / "home")
    paths.ensure_global_dirs()
    assert paths.ensure_global_dirs() is False


def test_ensure_project_dirs_creates_subtree(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    paths = FeatherPaths(project_root=proj, home=tmp_path / "home")
    paths.ensure_project_dirs()
    assert (proj / ".feather" / "db").is_dir()
    assert (proj / ".feather" / "tmp").is_dir()
    assert (proj / ".feather" / "logs").is_dir()
    assert (proj / ".feather" / "skills").is_dir()
    assert (proj / ".feather" / "attachments").is_dir()
    assert (proj / ".feather" / "artifacts").is_dir()


def test_ensure_project_dirs_raises_in_global_only_mode(tmp_path):
    paths = FeatherPaths(project_root=None, home=tmp_path / "home")
    with pytest.raises(RuntimeError, match="not in project mode"):
        paths.ensure_project_dirs()


# ---------------------------------------------------------------------------
# detect() walk-up
# ---------------------------------------------------------------------------


def _setup_home(monkeypatch, home_dir: Path) -> None:
    """Pin Path.home() and clear the project-root override env var."""
    home_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home_dir))
    monkeypatch.delenv("FEATHER_PROJECT_ROOT", raising=False)


def test_detect_finds_project_in_cwd(tmp_path, monkeypatch):
    user_home = tmp_path / "user"
    _setup_home(monkeypatch, user_home)
    proj = user_home / "myproj"
    proj.mkdir()
    (proj / ".feather").mkdir()
    paths = FeatherPaths.detect(cwd=proj, home=tmp_path / "feather-home")
    assert paths.project_root == proj.resolve()


def test_detect_walks_up_to_find_project(tmp_path, monkeypatch):
    user_home = tmp_path / "user"
    _setup_home(monkeypatch, user_home)
    proj = user_home / "myproj"
    proj.mkdir()
    (proj / ".feather").mkdir()
    nested = proj / "src" / "deep" / "leaf"
    nested.mkdir(parents=True)
    paths = FeatherPaths.detect(cwd=nested, home=tmp_path / "feather-home")
    assert paths.project_root == proj.resolve()


def test_detect_returns_global_only_when_no_project_found(tmp_path, monkeypatch):
    user_home = tmp_path / "user"
    _setup_home(monkeypatch, user_home)
    elsewhere = user_home / "random" / "place"
    elsewhere.mkdir(parents=True)
    paths = FeatherPaths.detect(cwd=elsewhere, home=tmp_path / "feather-home")
    assert paths.project_root is None
    assert paths.is_project_mode is False


def test_detect_does_not_treat_user_home_as_project(tmp_path, monkeypatch):
    """~/.feather/ is the global state dir, not a project."""
    user_home = tmp_path / "user"
    _setup_home(monkeypatch, user_home)
    (user_home / ".feather").mkdir()  # this is the global dir, NOT a project
    nested = user_home / "code" / "anywhere"
    nested.mkdir(parents=True)
    paths = FeatherPaths.detect(cwd=nested, home=user_home / ".feather")
    assert paths.project_root is None


def test_detect_does_not_walk_above_user_home(tmp_path, monkeypatch):
    """Walk must not pick up a .feather/ outside the user's home tree."""
    user_home = tmp_path / "user"
    _setup_home(monkeypatch, user_home)
    (tmp_path / ".feather").mkdir()  # decoy outside user home
    nested = user_home / "code" / "deeply" / "nested"
    nested.mkdir(parents=True)
    paths = FeatherPaths.detect(cwd=nested, home=tmp_path / "feather-home")
    assert paths.project_root is None


def test_detect_respects_feather_project_root_env(tmp_path, monkeypatch):
    user_home = tmp_path / "user"
    user_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: user_home))
    explicit = tmp_path / "explicit-proj"
    explicit.mkdir()
    monkeypatch.setenv("FEATHER_PROJECT_ROOT", str(explicit))
    elsewhere = user_home / "elsewhere"
    elsewhere.mkdir()
    paths = FeatherPaths.detect(cwd=elsewhere, home=tmp_path / "feather-home")
    assert paths.project_root == explicit


def test_detect_uses_cwd_by_default(tmp_path, monkeypatch):
    user_home = tmp_path / "user"
    _setup_home(monkeypatch, user_home)
    proj = user_home / "p"
    proj.mkdir()
    (proj / ".feather").mkdir()
    monkeypatch.chdir(proj)
    paths = FeatherPaths.detect(home=tmp_path / "feather-home")
    assert paths.project_root == proj.resolve()


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def test_for_project_pins_root(tmp_path):
    proj = tmp_path / "p"
    paths = FeatherPaths.for_project(proj, home=tmp_path / "h")
    assert paths.project_root == proj
    assert paths.global_root == tmp_path / "h"


def test_global_only_factory_yields_no_project(tmp_path):
    paths = FeatherPaths.global_only(home=tmp_path / "h")
    assert paths.project_root is None
    assert paths.global_root == tmp_path / "h"
