"""Slash-command registry, parser, and matcher used by the Feather TUI.

The module is deliberately decoupled from any UI framework so the matching
and parsing rules can be exercised under unit tests without spinning up a
Textual app. The Textual layer (``feather.textual_tui``) wires the registry
into its composer/dropdown widgets and dispatch table.

Design highlights:

- ``parse_slash_input`` is the single source of truth for "is this composer
  text a slash command?" — it strips leading whitespace, looks for ``/`` at
  the start, and splits the first whitespace-delimited token off as the
  command name. Anything after the first whitespace (and any trailing
  newlines) is preserved verbatim as ``args`` so command handlers can use
  the rest of the message body.
- ``SlashCommandRegistry`` tiers matches as: exact name/alias → name prefix
  → alias prefix → substring. Within each tier, ordering follows the
  registration order so the dropdown is stable as the user types.
- Aliases are first-class: any registered alias resolves to its canonical
  command via :meth:`SlashCommandRegistry.find`, and aliases participate
  in prefix matching so ``/quit`` shows up under ``/q`` even though the
  canonical name is ``exit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True, frozen=True)
class SlashCommand:
    """One slash command registered with the TUI.

    Attributes:
        name: Canonical command name (without the leading slash). Must be
            non-empty and contain no slashes or whitespace.
        summary: One-line description shown in the dropdown and ``/help``.
        aliases: Optional alternate names that also resolve to this command.
        category: Free-form grouping label used by the help renderer.
        usage: Optional one-line usage hint (e.g. ``"/help [command]"``).
    """

    name: str
    summary: str
    aliases: tuple[str, ...] = ()
    category: str = "general"
    usage: str | None = None

    @property
    def display(self) -> str:
        """Return the command formatted with its leading slash."""

        return f"/{self.name}"

    def matches_token(self, token: str) -> bool:
        """Return True when ``token`` equals the name or any alias (case-insensitive)."""

        normalized = token.lower()
        if normalized == self.name.lower():
            return True
        return any(normalized == alias.lower() for alias in self.aliases)


@dataclass(slots=True, frozen=True)
class SlashInput:
    """Parsed result from composer text that begins with a slash.

    Attributes:
        raw: Original composer text, unchanged.
        name_token: Characters between the leading ``/`` and the first
            whitespace (or end of first line). Empty when the user has
            only typed ``/``.
        args: Everything after the first whitespace, including any
            additional lines. Empty when no arguments were supplied.
    """

    raw: str
    name_token: str
    args: str = ""

    @property
    def has_args(self) -> bool:
        """Return True when the user has typed any non-empty argument text."""

        return bool(self.args)


def parse_slash_input(text: str) -> SlashInput | None:
    """Parse composer text into a :class:`SlashInput`, or return ``None``.

    Args:
        text: Raw composer text.

    Returns:
        A :class:`SlashInput` when ``text`` (after stripping leading
        whitespace) begins with ``/``; otherwise ``None``.
    """

    if not text:
        return None
    stripped = text.lstrip()
    if not stripped.startswith("/"):
        return None

    first_line, _, rest = stripped.partition("\n")
    after_slash = first_line[1:]
    # ``str.split(maxsplit=1)`` splits on *any* whitespace run, so tabs and
    # carriage returns work as separators just like spaces — this prevents
    # ``/\thelp`` from silently dropping the typed name (review fix M2).
    head_split = after_slash.split(maxsplit=1)
    name_token = head_split[0] if head_split else ""
    head_args = head_split[1] if len(head_split) > 1 else ""

    args = head_args
    if rest:
        args = f"{args}\n{rest}" if args else rest

    return SlashInput(raw=text, name_token=name_token, args=args)


@dataclass(slots=True, frozen=True)
class _IndexedCommand:
    """Internal helper pairing a command with its registration index."""

    index: int
    command: SlashCommand


class SlashCommandRegistry:
    """Immutable registry of slash commands with prefix/alias matching."""

    def __init__(self, commands: Sequence[SlashCommand]) -> None:
        """Initialise the registry.

        Args:
            commands: Commands to register, in display order.

        Raises:
            ValueError: If a command name is empty, contains whitespace or
                a slash, or if any name/alias collides with another
                registered token.
        """

        seen_tokens: set[str] = set()
        validated: list[SlashCommand] = []
        for cmd in commands:
            self._validate_token(cmd.name, label="command name")
            for alias in cmd.aliases:
                self._validate_token(alias, label="alias")
            for token in (cmd.name, *cmd.aliases):
                normalized = token.lower()
                if normalized in seen_tokens:
                    raise ValueError(
                        f"duplicate slash command token: {token!r}"
                    )
                seen_tokens.add(normalized)
            validated.append(cmd)
        self._commands: tuple[SlashCommand, ...] = tuple(validated)

    @staticmethod
    def _validate_token(token: str, *, label: str) -> None:
        if not token:
            raise ValueError(f"{label} must be non-empty")
        if "/" in token:
            raise ValueError(f"{label} {token!r} must not contain '/'")
        if any(ch.isspace() for ch in token):
            raise ValueError(f"{label} {token!r} must not contain whitespace")

    def all(self) -> tuple[SlashCommand, ...]:
        """Return every registered command in registration order."""

        return self._commands

    def find(self, token: str) -> SlashCommand | None:
        """Return the command matching ``token`` exactly (case-insensitive)."""

        if not token:
            return None
        for cmd in self._commands:
            if cmd.matches_token(token):
                return cmd
        return None

    def match(self, query: str) -> tuple[SlashCommand, ...]:
        """Return commands ranked by how well they match ``query``.

        Ranking tiers (highest first):

        1. Exact match on name or alias.
        2. Name starts with the query.
        3. Alias starts with the query.
        4. Substring of name or any alias.

        An empty query returns every command in registration order. A
        query that matches nothing returns an empty tuple.
        """

        if not query:
            return self._commands

        normalized = query.lower()
        exact: list[_IndexedCommand] = []
        prefix_name: list[_IndexedCommand] = []
        prefix_alias: list[_IndexedCommand] = []
        substring: list[_IndexedCommand] = []

        for index, cmd in enumerate(self._commands):
            if cmd.matches_token(query):
                exact.append(_IndexedCommand(index, cmd))
                continue
            if cmd.name.lower().startswith(normalized):
                prefix_name.append(_IndexedCommand(index, cmd))
                continue
            if any(alias.lower().startswith(normalized) for alias in cmd.aliases):
                prefix_alias.append(_IndexedCommand(index, cmd))
                continue
            in_name = normalized in cmd.name.lower()
            in_alias = any(normalized in alias.lower() for alias in cmd.aliases)
            if in_name or in_alias:
                substring.append(_IndexedCommand(index, cmd))

        ordered = exact + prefix_name + prefix_alias + substring
        return tuple(item.command for item in ordered)


def default_registry() -> SlashCommandRegistry:
    """Build the registry shipped with the Textual TUI."""

    commands: tuple[SlashCommand, ...] = (
        SlashCommand(
            name="help",
            summary="Show all slash commands",
            aliases=("?",),
            category="info",
        ),
        SlashCommand(
            name="exit",
            summary="Leave the TUI",
            aliases=("quit",),
            category="session",
        ),
        SlashCommand(
            name="onboard",
            summary="Restart the first-run onboarding wizard",
            category="session",
        ),
        SlashCommand(
            name="qdrant",
            summary="Manage the local Qdrant Docker container",
            usage="/qdrant [status|start|stop|remove|help]",
            category="memory",
        ),
        SlashCommand(
            name="clear",
            summary="Clear the on-screen conversation transcript",
            category="view",
        ),
        SlashCommand(
            name="copy",
            summary="Copy the conversation transcript to the clipboard",
            category="view",
        ),
        SlashCommand(
            name="queue",
            summary="Show queued user inputs waiting for the agent",
            category="info",
        ),
        SlashCommand(
            name="agents",
            summary="Show currently live sub-agents",
            aliases=("agent",),
            category="info",
        ),
        SlashCommand(
            name="tasks",
            summary="Show tracked tasks for this lead session",
            aliases=("task",),
            category="info",
        ),
        SlashCommand(
            name="session",
            summary="Show session ID, agent name, and context usage",
            category="info",
        ),
        SlashCommand(
            name="skills",
            summary="List skills available to the lead agent",
            category="info",
        ),
        SlashCommand(
            name="integrations",
            summary="Show messaging-integration status (telegram/line/whatsapp)",
            category="messaging",
        ),
        SlashCommand(
            name="telegram",
            summary="Manage the Telegram bot integration",
            usage="/telegram [status|connect <token>|disconnect|help]",
            category="messaging",
        ),
        SlashCommand(
            name="line",
            summary="Manage the LINE Messaging API integration",
            usage="/line [status|connect <secret> <token>|disconnect|help]",
            category="messaging",
        ),
        SlashCommand(
            name="whatsapp",
            summary="Manage the WhatsApp Cloud API integration",
            usage=(
                "/whatsapp [status|connect <phone_id> <token> "
                "<verify_token> <app_secret>|disconnect|help]"
            ),
            category="messaging",
        ),
    )
    return SlashCommandRegistry(commands)


__all__ = (
    "SlashCommand",
    "SlashCommandRegistry",
    "SlashInput",
    "default_registry",
    "parse_slash_input",
)
