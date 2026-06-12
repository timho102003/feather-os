"""Tests for the EventKind taxonomy and EventEmitter null-object."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from feather.core.agent.base import _inbox_received_event
from feather.core.agent.events import EventEmitter
from feather.core.ipc.event_codec import decode_event, encode_event
from feather.models import AgentMessage, AgentMessageStatus, EventKind, RuntimeEvent


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


def _make_agent_message(*, body: str, id: str = "msg-1") -> AgentMessage:
    """Build a minimal AgentMessage for tests."""
    return AgentMessage(
        id=id,
        from_session_id="sess-from",
        from_agent_name="worker",
        to_session_id="sess-to",
        to_agent_name="lead",
        body=body,
        correlation_id=None,
        in_reply_to=None,
        expects_response=False,
        status=AgentMessageStatus.PENDING,
        created_at="2026-01-01T00:00:00",
        delivered_at=None,
        responded_at=None,
    )


def test_inbox_received_event_previews_truncation() -> None:
    # Body with trailing space: raw len = 300, stripped len = 299 after .strip().
    # The "[N chars]" prefix and the "+M chars" marker both use the STRIPPED
    # body length (body = (msg.body or "").strip()); only total_chars in the
    # text/payload uses raw msg.body lengths.
    long_body = "word " * 60  # raw 300 chars; stripped = "word word ..." 299 chars
    stripped = long_body.strip()  # 299 chars
    collapsed = " ".join(long_body.split())  # same 299 chars after whitespace collapse
    empty_body = ""
    msgs = [
        _make_agent_message(body=long_body, id="msg-1"),
        _make_agent_message(body=empty_body, id="msg-2"),
    ]
    event = _inbox_received_event(
        sender_agent="worker", sender_session="sess-1", messages=msgs
    )

    assert event.kind == "agent_message_received"

    previews = event.payload["previews"]
    assert len(previews) == 2

    # Long message: the function uses stripped body for len in the prefix
    long_preview = previews[0]
    # The function does: body = (msg.body or "").strip(); prefix uses len(body)
    assert long_preview.startswith(f"[{len(stripped)} chars] ")
    assert f"… (+{len(stripped) - 240} chars)" in long_preview
    prefix_len = len(f"[{len(stripped)} chars] ")
    head = long_preview[prefix_len:]
    assert head.startswith(collapsed[:240])

    # Empty message
    assert previews[1] == "(empty body)"

    # text formatting: total_chars uses raw msg.body lengths (not stripped)
    total = sum(len(msg.body or "") for msg in msgs)
    assert event.text.startswith(f"worker (sess-1): 2 message(s), {total} chars")
    assert " | " in event.text


def test_inbox_received_event_collapses_internal_whitespace() -> None:
    # The preview head must use the whitespace-COLLAPSED form
    # (" ".join(body.split())), not merely the stripped body. Internal
    # multi-space and newline runs make the two differ within the first 240
    # chars, so a regression that replaces collapse with plain strip fails.
    body = "alpha   beta\n\ngamma " * 30
    stripped = body.strip()
    collapsed = " ".join(body.split())
    # Sanity: prove the test has power — collapse must actually differ from
    # strip in the truncation window, and the truncation path must run.
    assert collapsed[:240] != stripped[:240]
    assert len(collapsed) > 240

    event = _inbox_received_event(
        sender_agent="worker",
        sender_session="sess-1",
        messages=[_make_agent_message(body=body)],
    )

    preview = event.payload["previews"][0]
    assert preview == (
        f"[{len(stripped)} chars] "
        + collapsed[:240]
        + f"… (+{len(stripped) - 240} chars)"
    )


def test_inbox_received_event_exact_boundary_not_truncated() -> None:
    # The threshold is strict > on the collapsed head: exactly 240 chars is
    # kept whole with no truncation marker; 241 truncates with "+1 chars".
    at_limit = "x" * 240
    over_limit = "x" * 241
    event = _inbox_received_event(
        sender_agent="worker",
        sender_session="sess-1",
        messages=[
            _make_agent_message(body=at_limit, id="msg-1"),
            _make_agent_message(body=over_limit, id="msg-2"),
        ],
    )

    previews = event.payload["previews"]
    assert previews[0] == f"[240 chars] {at_limit}"
    assert "…" not in previews[0]
    assert previews[1] == f"[241 chars] {'x' * 240}… (+1 chars)"


def test_inbox_received_event_payload_bodies_roundtrip() -> None:
    bodies = ["hello world", "second message body"]
    msgs = [
        _make_agent_message(body=b, id=f"msg-{i}")
        for i, b in enumerate(bodies)
    ]
    event = _inbox_received_event(
        sender_agent="sub", sender_session="s2", messages=msgs
    )

    assert event.payload["bodies"] == bodies
    assert event.payload["count"] == 2
    assert event.payload["total_chars"] == sum(len(b) for b in bodies)
    assert event.payload["from_agent_name"] == "sub"
    assert event.payload["from_session_id"] == "s2"
