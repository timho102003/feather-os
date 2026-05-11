"""Drift tripwire: every leaf in AppConfig / AgentConfig must be in
the registry or explicitly ignored.

This test fails the build whenever a new dataclass field is added
without a corresponding registry entry, forcing an explicit decision
(surface it in /config, or add it to IGNORED_PATHS with a reason).
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_origin

import pytest

from feather.config_schema import IGNORED_PATHS, REGISTRY, Scope
from feather.models import AgentConfig, AppConfig


def _leaf_paths(prefix: str, cls: type) -> list[str]:
    """Walk a dataclass; yield dotted leaf paths."""

    if not is_dataclass(cls):
        return [prefix.rstrip(".")]

    out: list[str] = []
    for f in fields(cls):
        sub = f"{prefix}{f.name}"
        ftype = f.type if isinstance(f.type, type) else None

        # Unwrap Optional[T] / T | None into T
        origin = get_origin(f.type)
        if origin is type(None):
            ftype = None
        elif origin is None:
            ftype = f.type if isinstance(f.type, type) else None
        else:
            args = [a for a in get_args(f.type) if a is not type(None)]
            ftype = args[0] if len(args) == 1 and isinstance(args[0], type) else None

        if ftype is not None and is_dataclass(ftype):
            out.extend(_leaf_paths(f"{sub}.", ftype))
        else:
            out.append(sub)
    return out


@pytest.mark.xfail(
    strict=False,
    reason="REGISTRY filled across Phase 1 Tasks 3-7; remove xfail in Task 7",
)
def test_app_config_fields_are_in_registry_or_ignored() -> None:
    leaves = {f"app.{p}" for p in _leaf_paths("", AppConfig)}
    addressed = {f.path for f in REGISTRY if f.scope is Scope.APP}
    missing = leaves - addressed - IGNORED_PATHS
    assert not missing, (
        "AppConfig has fields not covered by registry or IGNORED_PATHS: "
        + ", ".join(sorted(missing))
    )


@pytest.mark.xfail(
    strict=False,
    reason="REGISTRY filled across Phase 1 Tasks 3-7; remove xfail in Task 7",
)
def test_agent_config_fields_are_in_registry_or_ignored() -> None:
    leaves = {f"agents.*.{p}" for p in _leaf_paths("", AgentConfig)}
    addressed = {
        f.path.replace(f.path.split(".")[1], "*", 1)
        for f in REGISTRY
        if f.scope is Scope.AGENT
    }
    missing = leaves - addressed - IGNORED_PATHS
    assert not missing, (
        "AgentConfig has fields not covered: " + ", ".join(sorted(missing))
    )


def test_registry_paths_are_unique() -> None:
    paths = [f.path for f in REGISTRY]
    assert len(paths) == len(set(paths)), "duplicate paths in REGISTRY"
