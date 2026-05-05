"""Round-trip tests for the supervisor→worker stdin command codec."""

from __future__ import annotations

import pytest

from feather.core.worker_command_codec import (
    CommandCodecError,
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
