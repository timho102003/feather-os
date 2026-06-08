"""Accessors for assets bundled inside the ``feather`` wheel.

The package ships default config files, default per-agent YAMLs, and the
built-in skill catalog under :mod:`feather._resources`. Loading happens
through :mod:`importlib.resources` so the same code works for both
editable installs and built wheels.

Direct callers should not reach into ``feather._resources`` themselves —
that path is treated as private and may move without notice. Use the
helpers in this module instead.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Iterator

import yaml


_RESOURCES_PACKAGE = "feather._resources"


def packaged_root() -> Traversable:
    """Return the root of the bundled resources tree."""
    return files(_RESOURCES_PACKAGE)


def packaged_app_yaml_text() -> str:
    """Read the bundled default ``app.yaml`` as text."""
    return (packaged_root() / "config" / "app.yaml").read_text(encoding="utf-8")


def packaged_app_yaml_dict() -> dict:
    """Parse the bundled default ``app.yaml`` into a plain dict."""
    return yaml.safe_load(packaged_app_yaml_text()) or {}


def iter_packaged_agent_names() -> Iterator[str]:
    """Yield the bare names of every bundled agent YAML.

    Names are filenames with the ``.yaml`` suffix stripped, so the
    caller can pass them straight to :func:`feather.config.load_agent_config`.
    """
    agents_dir = packaged_root() / "config" / "agents"
    for child in agents_dir.iterdir():
        if child.is_file() and child.name.endswith(".yaml"):
            yield child.name[: -len(".yaml")]


def packaged_agent_yaml_text(name: str) -> str:
    """Read a bundled agent YAML by bare name (e.g. ``"lead"``)."""
    return (packaged_root() / "config" / "agents" / f"{name}.yaml").read_text(
        encoding="utf-8"
    )


def has_packaged_agent(name: str) -> bool:
    """True when the package bundles an agent YAML with this bare name."""
    return (packaged_root() / "config" / "agents" / f"{name}.yaml").is_file()


def iter_packaged_soul_names() -> Iterator[str]:
    """Yield the bare ids of every bundled soul preset YAML.

    Ids are filenames with the ``.yaml`` suffix stripped, matching the
    ``Soul.id`` selection key.
    """
    souls_dir = packaged_root() / "config" / "souls"
    if not souls_dir.is_dir():
        return
    for child in souls_dir.iterdir():
        if child.is_file() and child.name.endswith(".yaml"):
            yield child.name[: -len(".yaml")]


def packaged_soul_yaml_text(soul_id: str) -> str:
    """Read a bundled soul YAML by bare id (e.g. ``"atlas-architect"``)."""
    return (packaged_root() / "config" / "souls" / f"{soul_id}.yaml").read_text(
        encoding="utf-8"
    )


def has_packaged_soul(soul_id: str) -> bool:
    """True when the package bundles a soul YAML with this bare id."""
    return (packaged_root() / "config" / "souls" / f"{soul_id}.yaml").is_file()


def packaged_skills_root() -> Traversable:
    """Return the root of the bundled built-in skills tree."""
    return packaged_root() / "skills" / "built-in"


def materialize_packaged_dir(rel_path: str) -> Path:
    """Materialize a packaged directory tree to a real filesystem path.

    Useful for code that requires a real ``Path`` (subprocess, glob,
    etc.). Wraps :func:`importlib.resources.as_file` for convenience but
    leaves cleanup to the caller — usually you should call the
    underlying ``as_file`` context manager yourself instead.

    Args:
        rel_path: Path relative to the resources root, e.g.
            ``"config/agents"``.

    Returns:
        A real filesystem path. May reside in a temp directory when the
        package was installed as a zipapp.
    """
    target = packaged_root()
    for part in Path(rel_path).parts:
        target = target / part
    with as_file(target) as real_path:
        return real_path
