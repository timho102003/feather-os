"""Orchestration layer for editable Feather configuration.

ConfigService is the single entry point the slash-command CLI and
the future Textual modal both call. It owns:

- field lookup (via :mod:`feather.config_schema`)
- value resolution (current value + source badge)
- write dispatch (via :mod:`feather.config_writer`)
- validation (via the registry's per-field validators + enum lists)

The service is intentionally synchronous; reload orchestration lives
on :class:`feather.runtime.FeatherRuntime`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import yaml

from feather.config_paths import ConfigPathResolver, PathScope
from feather.config_schema import ConfigField, FieldType, REGISTRY, lookup
from feather.models import AppConfig
from feather.paths import FeatherPaths


class ValueSource(str, Enum):
    """Where the current value came from."""

    DEFAULT = "default"
    GLOBAL = "global"
    PROJECT = "project"


@dataclass(slots=True, frozen=True)
class ConfigValue:
    """Result of :meth:`ConfigService.get`."""

    field: ConfigField
    current: Any
    source: ValueSource


@dataclass(slots=True, frozen=True)
class WriteResult:
    """Result of :meth:`ConfigService.set` or :meth:`reset`."""

    ok: bool
    path: str
    error: str | None = None


@dataclass(slots=True, frozen=True)
class ConfigRow:
    """One row returned by :meth:`ConfigService.list`."""

    field: ConfigField
    current: Any
    source: ValueSource


@dataclass(slots=True, frozen=True)
class ValidateResult:
    """Result of :meth:`ConfigService.validate`."""

    ok: bool
    coerced: Any = None
    error: str | None = None


def _coerce(value: Any, field_def: ConfigField) -> Any:
    """Coerce ``value`` to the wire type declared in ``field_def``.

    Args:
        value: Raw user-supplied value (possibly a string).
        field_def: Registry entry whose type governs the coercion.

    Returns:
        The coerced value.

    Raises:
        ValueError: If coercion fails (e.g. non-boolean string for a
            BOOLEAN field, or value not in the allowed enum list).
        TypeError: If the value cannot be converted.
    """

    if field_def.type is FieldType.INTEGER:
        return int(value)
    if field_def.type is FieldType.FLOAT:
        return float(value)
    if field_def.type is FieldType.BOOLEAN:
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "yes", "on", "1"):
                return True
            if v in ("false", "no", "off", "0"):
                return False
            raise ValueError(f"not a boolean: {value!r}")
        return bool(value)
    if field_def.type is FieldType.STRING_LIST:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return list(value)
    if field_def.type is FieldType.ENUM:
        if value not in (field_def.enum or ()):
            raise ValueError(
                f"value {value!r} not in allowed: {sorted(field_def.enum or ())}"
            )
        return value
    return str(value)


class ConfigService:
    """Read/write editable config through the registry.

    Args:
        paths: Resolved filesystem paths for this invocation.
        app_config: The loaded application configuration.
    """

    def __init__(
        self,
        *,
        paths: FeatherPaths,
        app_config: AppConfig,
    ) -> None:
        self._paths = paths
        self._app_config = app_config
        self._resolver = ConfigPathResolver(paths)

    @property
    def paths(self) -> FeatherPaths:
        """Resolved filesystem paths."""
        return self._paths

    def get(self, dotted: str) -> ConfigValue:
        """Resolve ``dotted`` to its current value + source badge.

        Args:
            dotted: A path that exists in :data:`REGISTRY`.

        Returns:
            :class:`ConfigValue` with the current value and its origin.

        Raises:
            KeyError: If the path is unknown.
        """

        field_def = lookup(dotted)
        if field_def is None:
            raise KeyError(dotted)
        current, source = self._resolve_value(field_def)
        return ConfigValue(field=field_def, current=current, source=source)

    def validate(self, dotted: str, value: Any) -> ValidateResult:
        """Coerce + validate ``value`` against the registry entry for ``dotted``.

        Args:
            dotted: Registry path to validate against.
            value: Raw value (possibly a string from CLI input).

        Returns:
            :class:`ValidateResult` with ``ok=True`` and the coerced value,
            or ``ok=False`` and a human-readable error.
        """

        field_def = lookup(dotted)
        if field_def is None:
            return ValidateResult(ok=False, error=f"unknown path: {dotted}")

        try:
            coerced = _coerce(value, field_def)
        except (TypeError, ValueError) as exc:
            return ValidateResult(ok=False, error=str(exc))

        if field_def.validator is not None:
            try:
                field_def.validator(coerced)
            except ValueError as exc:
                return ValidateResult(ok=False, error=str(exc))

        return ValidateResult(ok=True, coerced=coerced)

    def set(
        self,
        dotted: str,
        value: Any,
        *,
        scope: PathScope = PathScope.GLOBAL,
    ) -> WriteResult:
        """Validate ``value`` and write it to the resolved YAML file.

        Args:
            dotted: Registry path to write.
            value: New value (will be coerced + validated first).
            scope: Target scope — global overlay or project config.

        Returns:
            :class:`WriteResult` indicating success or the error message.
        """

        from feather.config_writer import write_yaml_value

        validate = self.validate(dotted, value)
        if not validate.ok:
            return WriteResult(ok=False, path=dotted, error=validate.error)

        res = self._resolver.resolve(dotted, scope=scope)
        try:
            write_yaml_value(res.file, res.yaml_path, validate.coerced)
        except OSError as exc:
            return WriteResult(ok=False, path=dotted, error=str(exc))
        return WriteResult(ok=True, path=dotted)

    def list(self, section: str = "") -> list[ConfigRow]:
        """Return registry rows whose path starts with ``section``.

        Args:
            section: Dotted prefix to filter by. Empty string returns
                every entry.

        Returns:
            List of :class:`ConfigRow` in registry order.
        """

        out: list[ConfigRow] = []
        for field_def in REGISTRY:
            if section and not field_def.path.startswith(section):
                continue
            current, source = self._resolve_value(field_def)
            out.append(ConfigRow(field=field_def, current=current, source=source))
        return out

    def diff(self) -> dict[str, tuple[Any, Any]]:
        """Return paths whose effective value differs from the packaged default.

        Returns:
            Mapping of ``path → (default_value, current_value)`` for every
            field that has been overridden and whose override differs from
            the compiled-in default.
        """

        out: dict[str, tuple[Any, Any]] = {}
        for field_def in REGISTRY:
            current, source = self._resolve_value(field_def)
            if source is ValueSource.DEFAULT:
                continue
            default = self._dig_app_config(field_def.path)
            # When the overlay restored the default explicitly, skip.
            if default == current:
                continue
            out[field_def.path] = (default, current)
        return out

    def reset(
        self, dotted: str, *, scope: PathScope = PathScope.GLOBAL
    ) -> WriteResult:
        """Remove the overlay value for ``dotted``.

        Args:
            dotted: Registry path to reset.
            scope: Which overlay file to remove the value from.

        Returns:
            :class:`WriteResult` indicating success or the error message.
        """

        field_def = lookup(dotted)
        if field_def is None:
            return WriteResult(ok=False, path=dotted, error=f"unknown path: {dotted}")
        from feather.config_writer import delete_yaml_value

        res = self._resolver.resolve(dotted, scope=scope)
        delete_yaml_value(res.file, res.yaml_path)
        return WriteResult(ok=True, path=dotted)

    # ----- internal value lookup ---------------------------------

    def _resolve_value(self, field_def: ConfigField) -> tuple[Any, ValueSource]:
        """Walk PROJECT then GLOBAL overlays; fall back to AppConfig default.

        Args:
            field_def: Registry entry to resolve.

        Returns:
            Tuple of (current_value, source_enum).
        """

        for scope, source_enum in (
            (PathScope.PROJECT, ValueSource.PROJECT),
            (PathScope.GLOBAL, ValueSource.GLOBAL),
        ):
            res = self._resolver.resolve(field_def.path, scope=scope)
            if res.file.exists():
                data = yaml.safe_load(res.file.read_text(encoding="utf-8")) or {}
                value = self._dig(data, res.yaml_path)
                if value is not None:
                    return value, source_enum
        # Fallback: read the resolved current value from the live
        # AppConfig (which already reflects packaged defaults).
        return self._dig_app_config(field_def.path), ValueSource.DEFAULT

    @staticmethod
    def _dig(data: dict[str, Any], yaml_path: list[str]) -> Any:
        """Traverse ``data`` following ``yaml_path`` keys.

        Args:
            data: Parsed YAML mapping.
            yaml_path: Sequence of keys to follow.

        Returns:
            The leaf value, or ``None`` if any key is missing.
        """

        cursor: Any = data
        for key in yaml_path:
            if not isinstance(cursor, dict) or key not in cursor:
                return None
            cursor = cursor[key]
        return cursor

    def _dig_app_config(self, dotted: str) -> Any:
        """Return the AppConfig leaf value for ``dotted``.

        Handles both ``app.*`` paths (walk AppConfig) and
        ``agents.<name>.*`` paths (load the resolved AgentConfig and
        walk it). Uses defensive ``getattr`` to handle ``None``
        intermediate dataclasses (e.g. ``app.parallel`` when parallel
        is disabled).

        Args:
            dotted: Dotted registry path.

        Returns:
            The leaf value from AppConfig, or ``None`` if an intermediate
            attribute is ``None`` or the agent config is not found.

        Raises:
            KeyError: If the path prefix is not ``app.`` or ``agents.``.
        """

        if dotted.startswith("app."):
            cursor: Any = self._app_config
            for part in dotted.split(".")[1:]:
                cursor = getattr(cursor, part, None)
                if cursor is None:
                    return None
            return cursor
        if dotted.startswith("agents."):
            parts = dotted.split(".")
            agent_name = parts[1]
            from feather.config import load_agent_config

            try:
                agent_cfg = load_agent_config(
                    self._paths.project_root, agent_name, paths=self._paths
                )
            except FileNotFoundError:
                return None
            cursor = agent_cfg
            for part in parts[2:]:
                cursor = getattr(cursor, part, None)
                if cursor is None:
                    return None
            return cursor
        raise KeyError(dotted)
