"""Headless `/config` subcommand dispatcher.

The TUI's slash handler calls :func:`handle_config_command` with the
raw arg string. This module parses the subcommand and dispatches to
the appropriate :class:`feather.config.service.ConfigService` method,
returning a rendered string for the TUI to display.

Interactive (modal) handling is wired separately in
``feather.tui.app`` (Phase 2). Phase 1 supports headless only.
"""

from __future__ import annotations

from dataclasses import dataclass

from feather.config.resolver import PathScope
from feather.config.service import ConfigService


@dataclass(slots=True, frozen=True)
class ConfigCommandResult:
    """Outcome of one `/config <sub>` invocation."""

    ok: bool
    body: str
    requires_apply: list[str] | None = None  # paths to feed apply_config_change


def handle_config_command(
    service: ConfigService, args: str
) -> ConfigCommandResult:
    """Parse and dispatch one `/config <sub> [args]` invocation.

    Args:
        service: Configured :class:`~feather.config.service.ConfigService`
            instance to delegate reads/writes to.
        args: Raw argument string after the ``/config`` command token.

    Returns:
        A :class:`ConfigCommandResult` describing the outcome.
    """

    tokens = args.strip().split()
    if not tokens:
        return ConfigCommandResult(
            ok=True,
            body=(
                "Usage: /config <subcommand>\n"
                "  get <path>           show a field's value and where it comes from\n"
                "  set <path> <value>   write a field (project scope) and apply it live\n"
                "  list [section]       list fields, optionally filtered by dotted prefix\n"
                "  diff                 show fields overridden from the packaged default\n"
                "  reset <path>         remove an override\n"
                "The Textual TUI also opens an interactive editor for /config."
            ),
        )

    sub, *rest = tokens
    if sub == "get":
        return _cmd_get(service, rest)
    if sub == "set":
        return _cmd_set(service, rest)
    if sub == "list":
        return _cmd_list(service, rest)
    if sub == "diff":
        return _cmd_diff(service, rest)
    if sub == "reset":
        return _cmd_reset(service, rest)
    return ConfigCommandResult(
        ok=False, body=f"unknown subcommand: {sub} (expected get|set|list|diff|reset)"
    )


def _cmd_get(service: ConfigService, rest: list[str]) -> ConfigCommandResult:
    """Handle ``/config get <path>``.

    Args:
        service: Config service instance.
        rest: Remaining tokens after the ``get`` subcommand.

    Returns:
        :class:`ConfigCommandResult` with the resolved value and source.
    """

    if len(rest) != 1:
        return ConfigCommandResult(ok=False, body="usage: /config get <path>")
    path = rest[0]
    try:
        value = service.get(path)
    except KeyError:
        return ConfigCommandResult(ok=False, body=f"unknown path: {path}")
    body = f"{path} = {value.current!r}  [{value.source.value}]"
    return ConfigCommandResult(ok=True, body=body)


def _parse_scope(rest: list[str]) -> tuple[PathScope, bool, list[str]]:
    """Split ``--global`` / ``--project`` / ``--force`` flags out of ``rest``.

    Args:
        rest: Token list that may contain scope and force flags.

    Returns:
        Tuple of (resolved :class:`~feather.config.resolver.PathScope`,
        force flag, remaining positional tokens).
    """

    scope = PathScope.GLOBAL
    force = False
    remaining: list[str] = []
    for token in rest:
        if token == "--global":
            scope = PathScope.GLOBAL
        elif token == "--project":
            scope = PathScope.PROJECT
        elif token == "--force":
            force = True
        else:
            remaining.append(token)
    return scope, force, remaining


def _cmd_set(service: ConfigService, rest: list[str]) -> ConfigCommandResult:
    """Handle ``/config set <path> <value> [--project|--global] [--force]``.

    The ``--force`` flag is required when setting ``app.self_repair.enabled``
    to acknowledge that the change requires a full TUI restart.

    Args:
        service: Config service instance.
        rest: Remaining tokens after the ``set`` subcommand.

    Returns:
        :class:`ConfigCommandResult` indicating success and the paths
        that need to be applied, or an error body.
    """

    scope, force, positional = _parse_scope(rest)
    if len(positional) < 2:
        return ConfigCommandResult(
            ok=False,
            body="usage: /config set <path> <value> [--project|--global] [--force]",
        )
    path, *value_parts = positional
    value = " ".join(value_parts)
    write = service.set(path, value, scope=scope, force=force)
    if not write.ok:
        return ConfigCommandResult(ok=False, body=f"{path}: {write.error}")
    return ConfigCommandResult(
        ok=True,
        body=f"{path} = {value} (saved to {scope.value})",
        requires_apply=[path],
    )


def _cmd_list(service: ConfigService, rest: list[str]) -> ConfigCommandResult:
    """Handle ``/config list [section]``.

    Args:
        service: Config service instance.
        rest: Remaining tokens; the first (if any) is treated as a
            dotted-prefix filter.

    Returns:
        :class:`ConfigCommandResult` with a multi-line table of all
        matching fields.
    """

    section = rest[0] if rest else ""
    rows = service.list(section=section)
    if not rows:
        return ConfigCommandResult(
            ok=True, body=f"no fields under {section!r}"
        )
    lines = [
        f"{row.field.path}  =  {row.current!r}  [{row.source.value}]"
        for row in rows
    ]
    return ConfigCommandResult(ok=True, body="\n".join(lines))


def _cmd_diff(service: ConfigService, rest: list[str]) -> ConfigCommandResult:
    """Handle ``/config diff``.

    Args:
        service: Config service instance.
        rest: Ignored (no arguments for diff).

    Returns:
        :class:`ConfigCommandResult` listing all active overrides, or
        "no overrides" when the live config matches the packaged defaults.
    """

    del rest
    diff = service.diff()
    if not diff:
        return ConfigCommandResult(ok=True, body="no overrides")
    lines = [f"{path}: {old!r} → {new!r}" for path, (old, new) in sorted(diff.items())]
    return ConfigCommandResult(ok=True, body="\n".join(lines))


def _cmd_reset(service: ConfigService, rest: list[str]) -> ConfigCommandResult:
    """Handle ``/config reset <path> [--project|--global]``.

    Args:
        service: Config service instance.
        rest: Remaining tokens after the ``reset`` subcommand.

    Returns:
        :class:`ConfigCommandResult` indicating success and the path
        that needs to be applied, or an error body.
    """

    scope, _force, positional = _parse_scope(rest)
    if len(positional) != 1:
        return ConfigCommandResult(
            ok=False, body="usage: /config reset <path> [--project|--global]"
        )
    path = positional[0]
    write = service.reset(path, scope=scope)
    if not write.ok:
        return ConfigCommandResult(ok=False, body=f"{path}: {write.error}")
    return ConfigCommandResult(
        ok=True,
        body=f"{path}: reset (scope={scope.value})",
        requires_apply=[path],
    )


__all__ = (
    "ConfigCommandResult",
    "handle_config_command",
)
