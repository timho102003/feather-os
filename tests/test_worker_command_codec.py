"""Round-trip tests for the supervisor→worker stdin command codec."""

from __future__ import annotations

import json

import pytest

from feather.core.ipc.command_codec import (
    CONFIG_RELOAD_ACK_KIND,
    CommandCodecError,
    ConfigReloadCommand,
    EnqueueUserInputCommand,
    ResumeOnInboxCommand,
    RunCommand,
    ShutdownCommand,
    decode_command,
    encode_command,
)


def _roundtrip(command):  # type: ignore[no-untyped-def]
    line = encode_command(command)
    assert "\n" not in line
    return decode_command(line)


def test_run_command_round_trips() -> None:
    cmd = RunCommand(session_id="s1", incoming_text="hello")
    assert _roundtrip(cmd) == cmd


def test_resume_on_inbox_command_round_trips() -> None:
    cmd = ResumeOnInboxCommand(session_id="s1")
    assert _roundtrip(cmd) == cmd


def test_enqueue_user_input_command_round_trips() -> None:
    cmd = EnqueueUserInputCommand(session_id="s1", text="mid-turn nudge")
    assert _roundtrip(cmd) == cmd


def test_shutdown_command_round_trips() -> None:
    cmd = ShutdownCommand()
    assert _roundtrip(cmd) == cmd


def test_run_command_preserves_unicode_in_text() -> None:
    cmd = RunCommand(session_id="s1", incoming_text="🪶 fly away")
    assert _roundtrip(cmd) == cmd


def test_decode_rejects_blank() -> None:
    with pytest.raises(CommandCodecError):
        decode_command("")


def test_decode_rejects_invalid_json() -> None:
    with pytest.raises(CommandCodecError):
        decode_command("nope")


def test_decode_rejects_non_object() -> None:
    with pytest.raises(CommandCodecError):
        decode_command('"shutdown"')


def test_decode_rejects_missing_cmd_field() -> None:
    with pytest.raises(CommandCodecError):
        decode_command('{"session_id": "s1"}')


def test_decode_rejects_unknown_cmd() -> None:
    with pytest.raises(CommandCodecError):
        decode_command('{"cmd": "do_a_barrel_roll"}')


def test_decode_rejects_run_missing_required_fields() -> None:
    with pytest.raises(CommandCodecError):
        decode_command('{"cmd": "run", "session_id": "s1"}')  # no incoming_text
    with pytest.raises(CommandCodecError):
        decode_command('{"cmd": "run", "incoming_text": "hi"}')  # no session_id


def test_decode_rejects_run_with_non_string_text() -> None:
    with pytest.raises(CommandCodecError):
        decode_command('{"cmd": "run", "session_id": "s1", "incoming_text": 42}')


# ---------------------------------------------------------------------------
# Task 21 — ConfigReloadCommand round-trips
# ---------------------------------------------------------------------------


def test_config_reload_command_round_trips_live() -> None:
    """Live-class reload command encodes and decodes correctly."""
    cmd = ConfigReloadCommand(
        correlation_id="abc123",
        changed_paths=["app.compaction.trigger_ratio"],
        reload_class="live",
    )
    assert _roundtrip(cmd) == cmd


def test_config_reload_command_round_trips_next_turn() -> None:
    """Next-turn-class reload command encodes and decodes correctly."""
    cmd = ConfigReloadCommand(
        correlation_id="xyz789",
        changed_paths=["app.active_provider", "app.openai.model"],
        reload_class="next_turn",
    )
    assert _roundtrip(cmd) == cmd


def test_config_reload_command_preserves_empty_paths() -> None:
    """Empty changed_paths list is faithfully preserved."""
    cmd = ConfigReloadCommand(
        correlation_id="empty",
        changed_paths=[],
        reload_class="live",
    )
    assert _roundtrip(cmd) == cmd


def test_config_reload_command_wire_format() -> None:
    """Encoded JSON contains all required fields with correct types."""
    cmd = ConfigReloadCommand(
        correlation_id="c1",
        changed_paths=["app.openai.model"],
        reload_class="next_turn",
    )
    raw = json.loads(encode_command(cmd))
    assert raw["cmd"] == "reload_config"
    assert raw["correlation_id"] == "c1"
    assert raw["changed_paths"] == ["app.openai.model"]
    assert raw["reload_class"] == "next_turn"


def test_decode_rejects_config_reload_missing_correlation_id() -> None:
    with pytest.raises(CommandCodecError, match="correlation_id"):
        decode_command(
            '{"cmd": "reload_config", "changed_paths": [], "reload_class": "live"}'
        )


def test_decode_rejects_config_reload_non_list_changed_paths() -> None:
    with pytest.raises(CommandCodecError, match="changed_paths"):
        decode_command(
            '{"cmd": "reload_config", "correlation_id": "x", '
            '"changed_paths": "oops", "reload_class": "live"}'
        )


def test_decode_rejects_config_reload_missing_reload_class() -> None:
    with pytest.raises(CommandCodecError, match="reload_class"):
        decode_command(
            '{"cmd": "reload_config", "correlation_id": "x", "changed_paths": []}'
        )


def test_config_reload_ack_kind_constant() -> None:
    """CONFIG_RELOAD_ACK_KIND is the string the worker uses in control events."""
    assert CONFIG_RELOAD_ACK_KIND == "_config_reload_ack"
