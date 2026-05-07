"""Detect whether Feather is installed editable or from a wheel.

Self-repair (the lead patches its own code via ``write_file`` then
calls ``request_restart``) only produces a *durable* fix in editable
installs — the patched files live in the user's checkout and survive
``pip install --upgrade``. In wheel installs the patch lands in
``site-packages/feather/...``, works for the live process and any
restart, but is silently overwritten on the next reinstall.

We don't refuse self-repair in wheel mode (the user might want a
session-scoped fix, or to subsequently open a PR upstream). We surface
the install mode to the model so it can warn the user and recommend
contributing the fix back via ``submit_github_report`` instead of
relying on the local edit.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class InstallMode(str, Enum):
    """How the running Feather package was installed."""

    EDITABLE = "editable"
    """``pip install -e .`` or ``uv sync`` — patches survive across upgrades."""

    WHEEL = "wheel"
    """Standard ``pip install`` — patches land in site-packages and are
    overwritten on the next reinstall."""

    READ_ONLY = "read_only"
    """Install location is not writable (eg system pip, root-owned venv).
    Self-repair will fail at the file-write stage."""


@dataclass(slots=True, frozen=True)
class InstallInfo:
    """Detected install context for the currently-running Feather package."""

    mode: InstallMode
    package_path: Path
    repo_root: Path | None
    """Editable-mode only: the parent directory containing ``pyproject.toml``."""

    def is_durable(self) -> bool:
        """True when an in-place edit is expected to survive package upgrades."""

        return self.mode is InstallMode.EDITABLE


def detect_install_mode(package_init_file: str | Path | None = None) -> InstallInfo:
    """Inspect the on-disk layout of the ``feather`` package.

    Args:
        package_init_file: Override for ``feather.__file__``. Tests
            inject a path under ``tmp_path`` so detection is verifiable
            without modifying the actual install.

    Returns:
        :class:`InstallInfo` describing how the package was installed.

    Detection rules:

    * ``READ_ONLY`` if the package directory cannot be written by the
      current process. This is the only failure-mode that matters at
      execution time — the others differ only in upgrade durability.
    * ``EDITABLE`` if a ``pyproject.toml`` exists within two parent
      levels of the package directory. Both ``pip install -e .`` and
      ``uv sync`` produce this layout (the package symlinks back to
      its source tree).
    * ``WHEEL`` otherwise — typically when the package lives under a
      ``site-packages`` directory.
    """

    if package_init_file is None:
        from feather import __file__ as feather_init

        package_init_file = feather_init

    package_path = Path(package_init_file).resolve().parent

    if not os.access(package_path, os.W_OK):
        return InstallInfo(
            mode=InstallMode.READ_ONLY,
            package_path=package_path,
            repo_root=None,
        )

    repo_root = _find_repo_root(package_path)
    if repo_root is not None:
        return InstallInfo(
            mode=InstallMode.EDITABLE,
            package_path=package_path,
            repo_root=repo_root,
        )

    return InstallInfo(
        mode=InstallMode.WHEEL,
        package_path=package_path,
        repo_root=None,
    )


def _find_repo_root(package_path: Path) -> Path | None:
    """Walk up at most three levels looking for a ``pyproject.toml``.

    Editable installs typically live at ``<repo>/src/feather`` (one level
    up to ``src/``, two up to ``<repo>/``). Three levels is enough slack
    for unusual layouts without scanning the entire filesystem.
    """

    current = package_path
    for _ in range(3):
        parent = current.parent
        if (parent / "pyproject.toml").is_file():
            return parent
        current = parent
    return None
