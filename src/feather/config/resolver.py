"""Resolve dotted config paths to (file, yaml_path, scope) tuples.

Dotted path conventions:

- ``app.<section>.<...>`` → ``<config_dir>/app.yaml``, with the leading
  ``app.`` stripped from the in-file YAML path.
- ``agents.<name>.<...>`` → ``<config_dir>/agents/<name>.yaml``, with
  ``agents.<name>.`` stripped from the in-file YAML path.

The resolver does not read or write the files — that is the writer's
job. Path inputs are validated for shape only (prefix + length).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from feather.config.app_paths import FeatherPaths


class PathScope(str, Enum):
    """Which YAML file the resolver targets."""

    GLOBAL = "global"
    PROJECT = "project"


@dataclass(slots=True, frozen=True)
class PathResolution:
    """Result of resolving a dotted config path."""

    file: Path
    yaml_path: list[str]
    scope: PathScope


class ConfigPathResolver:
    """Map dotted config paths to filesystem + YAML coordinates."""

    def __init__(self, paths: FeatherPaths) -> None:
        self._paths = paths

    def resolve(self, dotted: str, *, scope: PathScope) -> PathResolution:
        """Return the file and in-file YAML path for ``dotted``.

        Args:
            dotted: Path like ``app.openai.model`` or ``agents.Lead.model``.
            scope: Whether to target the global overlay or the project file.

        Returns:
            Resolution with the absolute file path and the residual
            in-file YAML path.

        Raises:
            ValueError: If ``dotted`` is too short or uses an unknown
                top-level prefix.
        """

        parts = dotted.split(".")
        head, *rest = parts

        if head == "app":
            if len(parts) < 2:
                raise ValueError(
                    f"app config path must have at least 2 segments, got {dotted!r}"
                )
            base = self._app_yaml_dir(scope) / "app.yaml"
            return PathResolution(file=base, yaml_path=rest, scope=scope)

        if head == "agents":
            if len(parts) < 3:
                raise ValueError(
                    f"agents config path must have at least 3 segments, got {dotted!r}"
                )
            agent_name, *yaml_path = rest
            # Shipped agent YAMLs use lowercase filenames (lead.yaml,
            # explore.yaml) even though their YAML `name:` field is
            # capitalized ("Lead", "Explore") and the registry paths
            # use that capitalized form. Lowercase here so writes via
            # `/config set agents.Explore.<knob>` land in the file the
            # loader actually reads.
            base = self._app_yaml_dir(scope) / "agents" / f"{agent_name.lower()}.yaml"
            return PathResolution(file=base, yaml_path=yaml_path, scope=scope)

        raise ValueError(f"unknown config path prefix: {head!r}")

    def _app_yaml_dir(self, scope: PathScope) -> Path:
        if scope is PathScope.GLOBAL:
            return Path(self._paths.global_config_dir)
        if not self._paths.is_project_mode:
            raise ValueError("project scope requires a discovered project root")
        return Path(self._paths.project_root) / "config"
