"""Round-trip tests for the worker→supervisor RuntimeEvent JSONL codec."""

from __future__ import annotations

import pytest

from feather.core.ipc.event_codec import (
    EventCodecError,
    decode_event,
    encode_event,
)
from feather.models import RuntimeEvent


def _roundtrip(event: RuntimeEvent) -> RuntimeEvent:
    line = encode_event(event)
    assert "\n" not in line, "encoded line must not embed a newline"
    return decode_event(line)


def test_text_only_event_round_trips() -> None:
    event = RuntimeEvent(kind="assistant_text_delta", text="hello world")
    decoded = _roundtrip(event)
    assert decoded == event


def test_tool_only_event_round_trips() -> None:
    event = RuntimeEvent(kind="tool_started", tool_name="bash")
    decoded = _roundtrip(event)
    assert decoded == event


def test_payload_event_round_trips() -> None:
    event = RuntimeEvent(
        kind="usage_updated",
        payload={"input_tokens": 1234, "output_tokens": 56, "ratio": 0.42},
    )
    decoded = _roundtrip(event)
    assert decoded == event


def test_all_fields_event_round_trips() -> None:
    event = RuntimeEvent(
        kind="tool_finished",
        text="done",
        tool_name="grep",
        payload={"matches": 3, "ok": True, "items": [1, 2, 3]},
    )
    decoded = _roundtrip(event)
    assert decoded == event


def test_empty_event_round_trips() -> None:
    """The decoder must accept events with kind only — every other field optional."""

    event = RuntimeEvent(kind="agent_idle")
    decoded = _roundtrip(event)
    assert decoded == event
    assert decoded.text is None
    assert decoded.tool_name is None
    assert decoded.payload is None


def test_unicode_text_round_trips() -> None:
    """Non-ASCII text must survive the wire intact."""

    event = RuntimeEvent(kind="assistant_text_delta", text="héllo 🪶 — Feather")
    decoded = _roundtrip(event)
    assert decoded == event


def test_decode_rejects_blank_line() -> None:
    with pytest.raises(EventCodecError):
        decode_event("")


def test_decode_rejects_non_json() -> None:
    with pytest.raises(EventCodecError):
        decode_event("not json at all")


def test_decode_rejects_non_object() -> None:
    """JSON arrays/scalars are not valid events."""

    with pytest.raises(EventCodecError):
        decode_event("[1, 2, 3]")


def test_decode_rejects_missing_kind() -> None:
    with pytest.raises(EventCodecError):
        decode_event('{"text": "hi"}')


def test_decode_rejects_non_string_kind() -> None:
    with pytest.raises(EventCodecError):
        decode_event('{"kind": 42}')


def test_encode_strips_default_none_fields() -> None:
    """``None`` fields are omitted on the wire to keep lines small."""

    import json as _json

    event = RuntimeEvent(kind="assistant_text_delta", text="hi")
    payload = _json.loads(encode_event(event))
    assert payload == {"kind": "assistant_text_delta", "text": "hi"}
    assert "tool_name" not in payload
    assert "payload" not in payload
