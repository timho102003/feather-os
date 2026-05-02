"""Shared pytest fixtures.

The autouse fixture below isolates every test from the developer's real
``~/.feather`` tree. Without it, code that constructs a
:class:`feather.paths.FeatherPaths` (or builds a :class:`FeatherRuntime`)
would silently load the developer's actual ``~/.feather/.env`` into
``os.environ`` mid-test, which (a) leaks API keys into the test process
and (b) makes test outcomes machine-dependent.

Tests that need a populated global home should use the
:func:`feather_paths` fixture below: it returns a fresh
:class:`feather.paths.FeatherPaths` rooted under ``tmp_path`` for both
sides of the split.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_feather_home(monkeypatch, tmp_path_factory) -> Path:
    """Point ``FEATHER_HOME`` at a per-test temp dir for the whole run.

    Also clears ``FEATHER_PROJECT_ROOT`` so any test that mutates the
    working directory or invokes :meth:`FeatherPaths.detect` doesn't
    inherit stale state from the surrounding shell.
    """

    home = tmp_path_factory.mktemp("feather_home")
    monkeypatch.setenv("FEATHER_HOME", str(home))
    monkeypatch.delenv("FEATHER_PROJECT_ROOT", raising=False)
    # Clear secrets that the wizard now reuses from os.environ. Without
    # this, a parallel test that exported OPENAI_API_KEY would prevent
    # later wizard tests from exercising their interactive ask path.
    for var in (
        "OPENAI_API_KEY",
        "OPEN_ROUTER_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "PARALLEL_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture
def feather_paths(tmp_path, _isolated_feather_home):
    """Build a :class:`FeatherPaths` for a per-test project + home.

    Both global and project trees are pre-created so individual tests
    can immediately stage files without worrying about ``mkdir``.
    """

    from feather.paths import FeatherPaths

    project = tmp_path / "project"
    project.mkdir()
    (project / ".feather").mkdir()
    paths = FeatherPaths(project_root=project, home=_isolated_feather_home)
    paths.ensure_global_dirs()
    return paths
