"""Tests for the slash-command registry, parser, and matcher."""

from __future__ import annotations

import pytest

from feather.slash_commands import (
    SlashCommand,
    SlashCommandRegistry,
    default_registry,
    parse_slash_input,
)


def test_parse_returns_none_for_non_slash_text() -> None:
    assert parse_slash_input("hello world") is None
    assert parse_slash_input("") is None
    assert parse_slash_input("    ") is None


def test_parse_recognises_slash_at_start_after_whitespace() -> None:
    parsed = parse_slash_input("  /help")

    assert parsed is not None
    assert parsed.name_token == "help"
    assert parsed.args == ""
    assert parsed.has_args is False


def test_parse_does_not_recognise_inline_slash() -> None:
    assert parse_slash_input("hello /help") is None


def test_parse_extracts_name_token_and_args() -> None:
    parsed = parse_slash_input("/help exit me")

    assert parsed is not None
    assert parsed.name_token == "help"
    assert parsed.args == "exit me"
    assert parsed.has_args is True


def test_parse_handles_empty_command_name() -> None:
    parsed = parse_slash_input("/")

    assert parsed is not None
    assert parsed.name_token == ""
    assert parsed.has_args is False


def test_parse_keeps_multiline_body_in_args() -> None:
    parsed = parse_slash_input("/note hello\nworld")

    assert parsed is not None
    assert parsed.name_token == "note"
    assert "hello" in parsed.args
    assert "world" in parsed.args


def test_parse_treats_only_first_line_for_name() -> None:
    parsed = parse_slash_input("/exit\nstray text")

    assert parsed is not None
    assert parsed.name_token == "exit"
    # Multi-line content after the first line still counts as args, so the
    # dispatcher can decide what to do with it.
    assert parsed.has_args is True


def _registry() -> SlashCommandRegistry:
    return SlashCommandRegistry(
        [
            SlashCommand(name="help", summary="Show help", aliases=("?",)),
            SlashCommand(name="exit", summary="Leave", aliases=("quit",)),
            SlashCommand(name="clear", summary="Clear transcript"),
            SlashCommand(name="copy", summary="Copy transcript"),
            SlashCommand(name="agents", summary="Show agents", aliases=("agent",)),
        ]
    )


def test_registry_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError):
        SlashCommandRegistry(
            [
                SlashCommand(name="help", summary="a"),
                SlashCommand(name="help", summary="b"),
            ]
        )


def test_registry_rejects_alias_collision_with_name() -> None:
    with pytest.raises(ValueError):
        SlashCommandRegistry(
            [
                SlashCommand(name="help", summary="a"),
                SlashCommand(name="other", summary="b", aliases=("help",)),
            ]
        )


def test_registry_rejects_alias_collision_between_commands() -> None:
    with pytest.raises(ValueError):
        SlashCommandRegistry(
            [
                SlashCommand(name="a", summary="a", aliases=("dup",)),
                SlashCommand(name="b", summary="b", aliases=("dup",)),
            ]
        )


def test_registry_rejects_command_with_slash_in_name() -> None:
    with pytest.raises(ValueError):
        SlashCommandRegistry([SlashCommand(name="he/lp", summary="x")])


def test_registry_rejects_command_with_whitespace_in_name() -> None:
    with pytest.raises(ValueError):
        SlashCommandRegistry([SlashCommand(name="he lp", summary="x")])


def test_registry_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        SlashCommandRegistry([SlashCommand(name="", summary="x")])


def test_match_empty_query_returns_all_commands_in_registration_order() -> None:
    registry = _registry()
    matches = registry.match("")
    names = tuple(c.name for c in matches)

    assert names == ("help", "exit", "clear", "copy", "agents")


def test_match_exact_name_ranks_first() -> None:
    registry = _registry()
    matches = registry.match("clear")
    assert matches[0].name == "clear"


def test_match_alias_resolves_to_command() -> None:
    registry = _registry()
    matches = registry.match("quit")
    assert matches and matches[0].name == "exit"


def test_match_prefix_returns_all_commands_starting_with_query() -> None:
    registry = _registry()
    matches = registry.match("c")
    names = {cmd.name for cmd in matches}

    assert {"clear", "copy"}.issubset(names)


def test_match_is_case_insensitive() -> None:
    registry = _registry()
    matches = registry.match("HE")
    assert matches and matches[0].name == "help"


def test_match_substring_falls_through_after_prefix_matches() -> None:
    registry = SlashCommandRegistry(
        [
            SlashCommand(name="research", summary="r"),
            SlashCommand(name="search", summary="s"),
        ]
    )
    matches = registry.match("sea")
    names = tuple(c.name for c in matches)

    # "search" is a prefix match; "research" contains "sea" as substring.
    assert names == ("search", "research")


def test_match_returns_empty_for_unknown_query() -> None:
    registry = _registry()
    assert registry.match("zzzzzz") == ()


def test_match_does_not_duplicate_when_name_and_alias_both_match() -> None:
    registry = SlashCommandRegistry(
        [SlashCommand(name="help", summary="x", aliases=("hel",))]
    )
    matches = registry.match("hel")
    assert len(matches) == 1
    assert matches[0].name == "help"


def test_find_returns_command_by_name() -> None:
    registry = _registry()
    cmd = registry.find("help")
    assert cmd is not None and cmd.name == "help"


def test_find_returns_command_by_alias() -> None:
    registry = _registry()
    cmd = registry.find("agent")
    assert cmd is not None and cmd.name == "agents"


def test_find_returns_none_for_unknown_token() -> None:
    registry = _registry()
    assert registry.find("zzz") is None


def test_find_is_case_insensitive() -> None:
    registry = _registry()
    cmd = registry.find("HELP")
    assert cmd is not None and cmd.name == "help"


def test_default_registry_includes_core_commands() -> None:
    registry = default_registry()
    names = {cmd.name for cmd in registry.all()}

    assert {"help", "exit", "clear", "copy", "queue", "agents", "tasks"}.issubset(names)


def test_default_registry_exposes_quit_alias_for_exit() -> None:
    registry = default_registry()
    cmd = registry.find("quit")
    assert cmd is not None and cmd.name == "exit"


def test_slash_input_round_trips_raw_text() -> None:
    parsed = parse_slash_input("/help me   please  ")
    assert parsed is not None
    assert parsed.raw == "/help me   please  "


def test_parse_does_not_lose_token_after_tab_separator() -> None:
    # Regression: review M2. Prior implementation silently dropped
    # ``\thelp`` when the user typed slash + non-space whitespace + name.
    parsed = parse_slash_input("/\thelp")
    assert parsed is not None
    # Either the parser treats the tab as a separator (name="", args="help")
    # or treats \thelp as the name token. Both preserve the typed content.
    typed_back_into_some_field = (parsed.name_token + " " + parsed.args).strip()
    assert "help" in typed_back_into_some_field


def test_parse_preserves_typed_content_after_carriage_return() -> None:
    parsed = parse_slash_input("/\rhelp")
    assert parsed is not None
    typed = (parsed.name_token + " " + parsed.args).strip()
    assert "help" in typed


def test_parse_double_slash_is_unknown_command_not_exception() -> None:
    parsed = parse_slash_input("//doublelash")
    assert parsed is not None
    # The second slash being part of the token is acceptable; the
    # registry will reject it cleanly. The point is no crash and no
    # silent loss.
    assert "doublelash" in (parsed.name_token + parsed.args)


def test_default_registry_includes_config() -> None:
    registry = default_registry()
    cmd = registry.find("config")
    assert cmd is not None
    assert cmd.summary
    assert cmd.usage and "get" in cmd.usage and "set" in cmd.usage
