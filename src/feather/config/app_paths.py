"""Path resolution for Feather global and project state.

This module is the single chokepoint for all filesystem layout decisions.
Every store, loader, and CLI command should construct paths through a
:class:`FeatherPaths` instance instead of building :class:`pathlib.Path`
literals directly. Centralizing the layout makes it possible to test the
``~/.feather`` / ``./.feather`` split deterministically and to relocate
either side later without grep-ing the codebase.

Two scopes are modeled:

* **Global** state lives under ``~/.feather`` (overridable via
  ``FEATHER_HOME``) and follows the user across projects: config
  overlays, user-installed skills, the qdrant memory marker, the
  onboarded marker, and the user's API keys.
* **Project** state lives under ``./.feather`` next to the user's code:
  the SQLite session DB, tool-output overflow, attachments, logs, and
  any project-scoped skill or persona overrides.

Project mode is opted into by walking up from the working directory
looking for an existing ``.feather`` directory. ``FeatherPaths.detect``
returns a *global-only* instance when no project is found, so users can
``feather`` from anywhere without the tool silently scattering ``.feather``
directories across their disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


_GLOBAL_DIR_NAME = ".feather"
_PROJECT_DIR_NAME = ".feather"
_FEATHER_HOME_ENV = "FEATHER_HOME"
_FEATHER_PROJECT_ROOT_ENV = "FEATHER_PROJECT_ROOT"


class FeatherPaths:
    """Resolves global and project paths for a single feather invocation.

    Construct via :meth:`detect` to walk up from a working directory
    looking for an existing project, via :meth:`for_project` when the
    project root is already known, or via :meth:`global_only` for
    commands that don't touch project state (e.g. ``feather init-memory``).

    Attributes:
        global_root: Root of user-global state. Defaults to
            ``~/.feather`` and is overridable via the ``FEATHER_HOME``
            environment variable; an explicit ``home`` argument beats both.
        project_root: Root of the current project (the directory that
            *contains* ``.feather/``) or ``None`` for global-only mode.
    """

    def __init__(
        self,
        project_root: Optional[Path],
        *,
        home: Optional[Path] = None,
    ) -> None:
        if home is not None:
            self.global_root = Path(home)
        else:
            env_home = os.environ.get(_FEATHER_HOME_ENV)
            if env_home:
                self.global_root = Path(env_home)
            else:
                self.global_root = Path.home() / _GLOBAL_DIR_NAME
        self.project_root = Path(project_root) if project_root is not None else None

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    @property
    def is_project_mode(self) -> bool:
        """True when this invocation is operating against a discovered project."""
        return self.project_root is not None

    def _require_project(self) -> Path:
        if self.project_root is None:
            raise RuntimeError(
                "FeatherPaths is not in project mode; "
                "this property requires a discovered project root"
            )
        return self.project_root

    # ------------------------------------------------------------------
    # Global path getters
    # ------------------------------------------------------------------

    @property
    def global_config_dir(self) -> Path:
        return self.global_root / "config"

    @property
    def global_agents_dir(self) -> Path:
        return self.global_config_dir / "agents"

    @property
    def global_souls_dir(self) -> Path:
        """User-supplied soul presets, layered over the packaged library."""
        return self.global_config_dir / "souls"

    @property
    def global_skills_dir(self) -> Path:
        return self.global_root / "skills"

    @property
    def global_state_dir(self) -> Path:
        return self.global_root / "state"

    @property
    def memory_marker(self) -> Path:
        """JSON file recording the active Qdrant deployment.

        Source of truth for "is long-term memory enabled". Written by
        ``feather init-memory``, removed by ``feather remove-memory``,
        consulted by the onboarding wizard to decide whether to ask the
        memory questions.
        """
        return self.global_state_dir / "memory.json"

    @property
    def onboarded_marker(self) -> Path:
        """Per-machine marker recording that the wizard ran to completion."""
        return self.global_state_dir / "onboarded.json"

    @property
    def projects_index(self) -> Path:
        """Index of known project roots; populated by ``feather init``."""
        return self.global_state_dir / "projects.json"

    @property
    def env_file(self) -> Path:
        """Global secrets file (API keys, integration tokens)."""
        return self.global_root / ".env"

    @property
    def global_user_md(self) -> Path:
        """Wizard-written persona / role / about, scoped to the user."""
        return self.global_root / "user.md"

    @property
    def global_sessions_db(self) -> Path:
        """SQLite session DB used when running in global-only mode."""
        return self.global_state_dir / "sessions.db"

    # ------------------------------------------------------------------
    # Project path getters
    # ------------------------------------------------------------------

    @property
    def project_feather_dir(self) -> Path:
        return self._require_project() / _PROJECT_DIR_NAME

    @property
    def db_path(self) -> Path:
        return self.project_feather_dir / "db" / "feather.db"

    @property
    def tmp_dir(self) -> Path:
        return self.project_feather_dir / "tmp"

    @property
    def subagent_staging_dir(self) -> Path:
        """Where ``spawn_agent`` stages task prompts before subprocess fork."""
        return self.tmp_dir / "subagent_tasks"

    @property
    def log_dir(self) -> Path:
        return self.project_feather_dir / "logs"

    @property
    def log_file(self) -> Path:
        return self.log_dir / "feather.log"

    @property
    def project_skills_dir(self) -> Path:
        return self.project_feather_dir / "skills"

    @property
    def project_user_md(self) -> Path:
        """Optional per-project persona override; never auto-created."""
        return self.project_feather_dir / "user.md"

    @property
    def attachments_dir(self) -> Path:
        return self.project_feather_dir / "attachments"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_feather_dir / "artifacts"

    @property
    def project_env_file(self) -> Path:
        """Optional per-project ``.env`` that overrides the global one."""
        return self._require_project() / ".env"

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def ensure_global_dirs(self) -> bool:
        """Create the global directory tree.

        Returns:
            ``True`` on first creation (the state dir didn't exist yet),
            ``False`` if the global tree was already initialized.
        """
        first_run = not self.global_state_dir.exists()
        for d in (
            self.global_root,
            self.global_config_dir,
            self.global_agents_dir,
            self.global_skills_dir,
            self.global_state_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return first_run

    def ensure_project_dirs(self) -> None:
        """Create the project ``.feather/`` subtree. Requires project mode."""
        for d in (
            self.project_feather_dir,
            self.db_path.parent,
            self.tmp_dir,
            self.log_dir,
            self.project_skills_dir,
            self.attachments_dir,
            self.artifacts_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @classmethod
    def detect(
        cls,
        cwd: Optional[Path] = None,
        *,
        home: Optional[Path] = None,
    ) -> "FeatherPaths":
        """Build a :class:`FeatherPaths` by walking up from ``cwd``.

        Walk-up stops at the user's home directory (``Path.home()``) or
        at the filesystem root, whichever comes first. The user home
        itself is *visited but not accepted* as a project, since
        ``~/.feather`` is the global state directory rather than a
        per-project state directory.

        ``FEATHER_PROJECT_ROOT``, when set, short-circuits the walk and
        pins the project root explicitly.

        Args:
            cwd: Directory to start the walk from. Defaults to
                :func:`pathlib.Path.cwd`.
            home: Override for the global root. Forwarded to
                :class:`FeatherPaths`.

        Returns:
            A FeatherPaths instance. ``project_root`` is ``None`` when no
            ``.feather`` directory was found in the walk.
        """
        env_root = os.environ.get(_FEATHER_PROJECT_ROOT_ENV)
        if env_root:
            return cls(project_root=Path(env_root), home=home)

        start = (cwd or Path.cwd()).resolve()
        user_home = Path.home().resolve()
        candidate = start
        while True:
            if candidate != user_home and (candidate / _PROJECT_DIR_NAME).is_dir():
                return cls(project_root=candidate, home=home)
            if candidate == user_home or candidate == candidate.parent:
                return cls(project_root=None, home=home)
            candidate = candidate.parent

    @classmethod
    def for_project(
        cls,
        project_root: Path,
        *,
        home: Optional[Path] = None,
    ) -> "FeatherPaths":
        """Construct paths pinned to a specific project root."""
        return cls(project_root=Path(project_root), home=home)

    @classmethod
    def global_only(cls, *, home: Optional[Path] = None) -> "FeatherPaths":
        """Construct paths with no project (global-only mode)."""
        return cls(project_root=None, home=home)
