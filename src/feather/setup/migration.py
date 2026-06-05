"""One-time migration for users upgrading from the pre-pip-install layout.

The old workflow stored secrets and per-machine markers under the
project's working tree (``./.env``, ``./.feather/onboarded.json``,
``./.feather/user.md``). The pip-installable layout puts those under
``~/.feather/`` instead so they follow the user across projects.

When an existing user upgrades and runs ``feather`` for the first time,
this module detects the legacy layout, asks for consent, and (if
granted) copies the artifacts into the global tree without deleting the
originals — so the user can roll back by clearing ``~/.feather`` and
keeping the project-local copies intact.

A breadcrumb file is left in ``./.feather/`` so the prompt only fires
once per project, regardless of the user's choice.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from feather.paths import FeatherPaths


_MIGRATED_BREADCRUMB = "MIGRATED_TO_GLOBAL.txt"
_DECLINED_BREADCRUMB = "MIGRATION_DECLINED.txt"


class MigrationOutcome(str, Enum):
    """Terminal outcomes of :func:`maybe_migrate`.

    A ``str`` Enum so existing ``== "migrated"`` style comparisons (and the
    CLI, which currently ignores the return) keep working unchanged.
    """

    NOT_APPLICABLE = "not-applicable"
    ALREADY_HANDLED = "already-handled"
    DEFERRED = "deferred"
    DECLINED = "declined"
    MIGRATED = "migrated"


@dataclass(slots=True, frozen=True)
class LegacyArtifacts:
    """What was found under ``./.feather/`` and ``./.env``."""

    env_file: Path | None
    user_md: Path | None
    onboarded_marker: Path | None

    @property
    def has_anything(self) -> bool:
        return any(p is not None for p in (self.env_file, self.user_md, self.onboarded_marker))


def detect_legacy_artifacts(paths: FeatherPaths) -> LegacyArtifacts:
    """Inspect ``paths.project_root`` for legacy state worth migrating.

    Returns paths that exist *and* hold meaningful content; empty files
    don't count. Always returns a :class:`LegacyArtifacts` even in
    global-only mode, where every field is ``None``.
    """

    if not paths.is_project_mode:
        return LegacyArtifacts(env_file=None, user_md=None, onboarded_marker=None)

    project = paths.project_root
    assert project is not None  # type-narrowing for mypy

    env = project / ".env"
    user_md = project / ".feather" / "user.md"
    marker = project / ".feather" / "onboarded.json"

    return LegacyArtifacts(
        env_file=env if env.is_file() and env.stat().st_size > 0 else None,
        user_md=user_md if user_md.is_file() and user_md.stat().st_size > 0 else None,
        onboarded_marker=(
            marker if marker.is_file() and marker.stat().st_size > 0 else None
        ),
    )


def already_handled(paths: FeatherPaths) -> bool:
    """True when a previous migration prompt already ran for this project."""
    if not paths.is_project_mode:
        return False
    project = paths.project_root
    assert project is not None
    crumb_dir = project / ".feather"
    return (crumb_dir / _MIGRATED_BREADCRUMB).exists() or (
        crumb_dir / _DECLINED_BREADCRUMB
    ).exists()


def maybe_migrate(
    paths: FeatherPaths,
    *,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
) -> MigrationOutcome:
    """Run the interactive migration if applicable.

    Returns a :class:`MigrationOutcome` — one of ``NOT_APPLICABLE``,
    ``ALREADY_HANDLED``, ``DEFERRED`` (non-tty, can't prompt), ``DECLINED``,
    or ``MIGRATED``. The behavior is intentionally side-effect-free when
    nothing needs migrating, so the CLI can call it on every run without
    surprising users who set up cleanly via ``pip install``.
    """

    if not paths.is_project_mode:
        return MigrationOutcome.NOT_APPLICABLE
    if already_handled(paths):
        return MigrationOutcome.ALREADY_HANDLED

    artifacts = detect_legacy_artifacts(paths)
    if not artifacts.has_anything:
        return MigrationOutcome.NOT_APPLICABLE

    # Bail out cleanly when stdin is closed (CI smoke runs, piped
    # invocations, sandboxed installs). Only applies when we're going to
    # the real terminal — tests inject their own ``ask`` callable and
    # should always exercise the flow regardless of pytest's stdin.
    import sys as _sys

    if ask is input and not _sys.stdin.isatty():
        return MigrationOutcome.DEFERRED

    say("")
    say("Detected legacy Feather state in this directory:")
    if artifacts.env_file is not None:
        say(f"  - {artifacts.env_file.relative_to(paths.project_root)}")
    if artifacts.user_md is not None:
        say(f"  - {artifacts.user_md.relative_to(paths.project_root)}")
    if artifacts.onboarded_marker is not None:
        say(f"  - {artifacts.onboarded_marker.relative_to(paths.project_root)}")
    say(
        "\nThe new layout keeps these under ~/.feather/ so they follow "
        "you across projects. The original files are NOT deleted; we "
        "just copy them into the global tree."
    )
    answer = ask("Migrate now? [Y/n/skip]: ").strip().lower()

    if answer in {"s", "skip"}:
        _write_breadcrumb(paths, _DECLINED_BREADCRUMB, "Migration declined.\n")
        say("Skipped. We won't ask again for this project.")
        return MigrationOutcome.DECLINED

    if answer in {"n", "no"}:
        # User explicitly said no but didn't elect skip — leave the
        # door open for next run by NOT writing a breadcrumb. They might
        # want to migrate later.
        say("Not migrating now. Run again to be prompted next time.")
        return MigrationOutcome.DECLINED

    # Anything else (including empty input) → migrate.
    paths.ensure_global_dirs()
    if artifacts.env_file is not None:
        _copy_if_missing(artifacts.env_file, paths.env_file, say=say)
    if artifacts.user_md is not None:
        _copy_if_missing(artifacts.user_md, paths.global_user_md, say=say)
    if artifacts.onboarded_marker is not None:
        _copy_if_missing(artifacts.onboarded_marker, paths.onboarded_marker, say=say)

    _write_breadcrumb(
        paths,
        _MIGRATED_BREADCRUMB,
        "Legacy state copied to ~/.feather/. Originals were preserved.\n",
    )
    say("Migration complete. Originals are untouched in this directory.")
    return MigrationOutcome.MIGRATED


def _copy_if_missing(
    src: Path, dst: Path, *, say: Callable[[str], None]
) -> None:
    """Copy ``src`` to ``dst`` unless ``dst`` already exists.

    Refuses to overwrite a globally-staged file so users who set up
    cleanly via ``pip install`` and then later upgrade an old checkout
    don't lose their fresh global state.
    """
    if dst.exists():
        say(f"  skipped {dst} (already exists)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    say(f"  copied {src} → {dst}")


def _write_breadcrumb(paths: FeatherPaths, name: str, body: str) -> None:
    project = paths.project_root
    assert project is not None
    crumb_dir = project / ".feather"
    crumb_dir.mkdir(parents=True, exist_ok=True)
    (crumb_dir / name).write_text(body, encoding="utf-8")
