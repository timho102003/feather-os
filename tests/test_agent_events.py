"""Tests for the EventKind taxonomy and EventEmitter null-object."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from feather.core.agent.events import EventEmitter
from feather.core.ipc.event_codec import decode_event, encode_event
from feather.models import EventKind, RuntimeEvent


def test_emit_without_handler_is_noop() -> None:
    emitter = EventEmitter(None)
    assert emitter.enabled is False
    emitter.emit(EventKind.TOOL_STARTED, tool_name="grep")
    emitter.forward(RuntimeEvent(kind="anything"))


def test_emit_builds_runtime_event_fields() -> None:
    seen: list[RuntimeEvent] = []
    emitter = EventEmitter(seen.append)
    assert emitter.enabled is True
    emitter.emit(
        EventKind.TOOL_FINISHED,
        text="done",
        tool_name="grep",
        payload={"a": 1},
    )
    assert len(seen) == 1
    event = seen[0]
    assert event.kind == "tool_finished"
    assert event.text == "done"
    assert event.tool_name == "grep"
    assert event.payload == {"a": 1}


def test_forward_passes_prebuilt_event_identity() -> None:
    seen: list[RuntimeEvent] = []
    emitter = EventEmitter(seen.append)
    event = RuntimeEvent(kind="agent_message_received", text="hi")
    emitter.forward(event)
    assert seen[0] is event


def test_handler_exception_propagates() -> None:
    def boom(_: RuntimeEvent) -> None:
        raise ValueError("handler exploded")

    emitter = EventEmitter(boom)
    with pytest.raises(ValueError, match="handler exploded"):
        emitter.emit(EventKind.AWAITING_USER, text="q")


def test_event_kind_values_are_plain_strings() -> None:
    assert EventKind.TOOL_STARTED == "tool_started"
    assert json.loads(json.dumps({"k": EventKind.TOOL_STARTED})) == {
        "k": "tool_started"
    }
    line = encode_event(RuntimeEvent(kind=EventKind.TOOL_STARTED, tool_name="bash"))
    decoded = decode_event(line)
    assert decoded.kind == "tool_started"
    assert type(decoded.kind) is str


def test_event_kind_covers_all_emitted_literals() -> None:
    """Tripwire: every literal RuntimeEvent kind in src/ must be in EventKind."""

    src_root = Path(__file__).resolve().parents[1] / "src" / "feather"
    pattern = re.compile(r"RuntimeEvent\(\s*kind=\"([a-z_]+)\"")
    known = {member.value for member in EventKind}
    unknown: dict[str, set[str]] = {}
    for path in src_root.rglob("*.py"):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            kind = match.group(1)
            if kind not in known:
                unknown.setdefault(kind, set()).add(path.name)
    assert not unknown, f"RuntimeEvent kinds missing from EventKind: {unknown}"
