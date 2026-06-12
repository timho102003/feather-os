"""Unit tests for the conversation-state strategies (no BaseAgent).

These pin the exact provider-input shapes each strategy produces: stateful
sends only new items plus the server cursor and replays history solely when
that cursor was reset; stateless owns the full structural transcript that it
re-sends every turn. Every test fails if the strategy logic regresses.
"""

from __future__ import annotations

import json
from typing import Any

from feather.core.agent.conversation import (
    StatefulConversation,
    StatelessConversation,
    model_turn_input_items,
)
from feather.models import ModelTurn, SessionRecord, SessionStatus, ToolCall


def _session(
    *,
    pending: list[dict[str, Any]] | None = None,
    cursor: str | None = None,
) -> SessionRecord:
    """Build a minimal SessionRecord for strategy tests."""

    return SessionRecord(
        id="sess-1",
        agent_name="Lead",
        status=SessionStatus.ACTIVE,
        last_response_id=cursor,
        loaded_skills=[],
        active_mcp_servers=[],
        pending_inputs=pending or [],
        created_at="2026-06-12T00:00:00Z",
        updated_at="2026-06-12T00:00:00Z",
    )


def _replay_stub(
    items: list[dict[str, Any]], calls: list[int]
) -> Any:
    """Return an async replay callable that records each invocation in ``calls``."""

    async def _replay() -> list[dict[str, Any]]:
        calls.append(1)
        return [dict(item) for item in items]

    return _replay


def _msg(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


# --------------------------------------------------------------------------- #
# Stateful
# --------------------------------------------------------------------------- #


async def test_stateful_replays_only_when_cursor_none() -> None:
    history = [_msg("history")]

    # Cursor set → replay NOT called.
    calls_set: list[int] = []
    ctx = StatefulConversation(replay=_replay_stub(history, calls_set))
    items = await ctx.initial_input_items(_session(cursor="resp-1"), [_msg("new")])
    assert calls_set == []
    assert items == [_msg("new")]

    # Cursor None → replay called exactly once.
    calls_none: list[int] = []
    ctx = StatefulConversation(replay=_replay_stub(history, calls_none))
    items = await ctx.initial_input_items(_session(cursor=None), [_msg("new")])
    assert calls_none == [1]
    assert items == [_msg("history"), _msg("new")]


async def test_stateful_orders_pending_then_history_then_new() -> None:
    pending = [{"type": "function_call_output", "call_id": "c1", "output": "x"}]
    history = [_msg("history")]
    new = [_msg("new")]
    calls: list[int] = []
    ctx = StatefulConversation(replay=_replay_stub(history, calls))

    items = await ctx.initial_input_items(
        _session(pending=pending, cursor=None), new
    )

    assert calls == [1]
    assert items == [pending[0], history[0], new[0]]


async def test_stateful_request_returns_cursor_and_copy() -> None:
    calls: list[int] = []
    ctx = StatefulConversation(replay=_replay_stub([], calls))
    session = _session(cursor="resp-9")
    source = [_msg("a"), _msg("b")]

    sent, cursor = ctx.provider_request(session, source)

    assert cursor == "resp-9"
    assert sent == source
    # Returned list is a copy: mutating it must not touch the caller's list.
    sent.append(_msg("c"))
    assert source == [_msg("a"), _msg("b")]


async def test_stateful_record_turn_noop_and_pause_is_outputs_only() -> None:
    calls: list[int] = []
    ctx = StatefulConversation(replay=_replay_stub([], calls))
    turn = ModelTurn(response_id="r", output_text="hi", tool_calls=[])

    # record_turn is a no-op (no internal transcript to mutate).
    assert ctx.record_turn([_msg("sent")], turn) is None

    outputs = [{"type": "function_call_output", "call_id": "c1", "output": "ok"}]
    payload = ctx.pause_payload(outputs)
    assert payload == outputs
    # Copy, not alias.
    payload.append({"type": "function_call_output", "call_id": "c2", "output": "x"})
    assert outputs == [{"type": "function_call_output", "call_id": "c1", "output": "ok"}]


# --------------------------------------------------------------------------- #
# Stateless
# --------------------------------------------------------------------------- #


async def test_stateless_skips_replay_when_pending_has_structural_context() -> None:
    history = [_msg("history")]

    # Pending containing a function_call item → has structural context → no replay.
    structural = [{"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"}]
    calls_struct: list[int] = []
    ctx = StatelessConversation(replay=_replay_stub(history, calls_struct))
    items = await ctx.initial_input_items(_session(pending=structural), [_msg("new")])
    assert calls_struct == []
    assert items == [structural[0], _msg("new")]

    # Pending with only function_call_output (no message/function_call) → replay IS called.
    outputs_only = [{"type": "function_call_output", "call_id": "c1", "output": "x"}]
    calls_out: list[int] = []
    ctx = StatelessConversation(replay=_replay_stub(history, calls_out))
    items = await ctx.initial_input_items(_session(pending=outputs_only), [_msg("new")])
    assert calls_out == [1]
    assert items == [outputs_only[0], history[0], _msg("new")]


async def test_stateless_begin_seeds_replay_on_empty_input() -> None:
    history = [_msg("history")]
    calls: list[int] = []
    ctx = StatelessConversation(replay=_replay_stub(history, calls))

    await ctx.begin([])

    assert calls == [1]
    # The seeded transcript is now sent verbatim on the first provider_request,
    # which (transcript already set) does NOT extend with empty input_items.
    sent, cursor = ctx.provider_request(_session(), [])
    assert cursor is None
    assert sent == history


async def test_stateless_begin_skips_seed_with_input() -> None:
    history = [_msg("history")]
    calls: list[int] = []
    ctx = StatelessConversation(replay=_replay_stub(history, calls))

    await ctx.begin([_msg("new")])

    assert calls == []
    # Transcript unseeded → first provider_request seeds from input_items.
    sent, _ = ctx.provider_request(_session(), [_msg("new")])
    assert sent == [_msg("new")]


async def test_stateless_begin_is_entry_hook_only() -> None:
    # Callers (run_loop) invoke begin exactly once at entry, before the first
    # provider turn. This test documents the current overwrite semantics: a
    # second begin([]) re-seeds the transcript from replay, discarding any
    # in-run extensions — so adding a guard later is a conscious decision.
    history = [_msg("history")]
    calls: list[int] = []
    ctx = StatelessConversation(replay=_replay_stub(history, calls))

    await ctx.begin([])
    sent, _ = ctx.provider_request(_session(), [_msg("mid-run")])
    assert sent == [_msg("history"), _msg("mid-run")]

    await ctx.begin([])

    assert calls == [1, 1]
    # Transcript was re-seeded to the replay value; the extension is gone.
    resent, _ = ctx.provider_request(_session(), [])
    assert resent == history


async def test_stateless_request_seeds_then_extends_transcript() -> None:
    calls: list[int] = []
    ctx = StatelessConversation(replay=_replay_stub([], calls))
    session = _session()

    # None transcript → seed from items.
    first = [_msg("a")]
    sent1, cursor1 = ctx.provider_request(session, first)
    assert cursor1 is None
    assert sent1 == [_msg("a")]
    # Returned list is a copy of the internal transcript.
    sent1.append(_msg("zzz"))

    # Second call with items → extend the (unmutated) transcript.
    sent2, _ = ctx.provider_request(session, [_msg("b")])
    assert sent2 == [_msg("a"), _msg("b")]

    # Empty items → no extension.
    sent3, _ = ctx.provider_request(session, [])
    assert sent3 == [_msg("a"), _msg("b")]


async def test_stateless_record_turn_folds_tool_calls() -> None:
    ctx = StatelessConversation(replay=_replay_stub([], []))
    sent = [_msg("user")]
    turn = ModelTurn(
        response_id="r",
        output_text="thinking",
        tool_calls=[
            ToolCall(call_id="c1", name="alpha", arguments={"b": 2, "a": 1}),
            ToolCall(call_id="c2", name="beta", arguments={}),
        ],
    )

    ctx.record_turn(sent, turn)
    folded, cursor = ctx.provider_request(_session(), [])

    assert cursor is None
    assert folded == [
        _msg("user"),
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "alpha",
            # compact + sorted JSON; output_text rides the FIRST item only.
            "arguments": json.dumps({"a": 1, "b": 2}, separators=(",", ":"), sort_keys=True),
            "content": "thinking",
        },
        {
            "type": "function_call",
            "call_id": "c2",
            "name": "beta",
            "arguments": "{}",
        },
    ]


async def test_stateless_record_turn_text_only_appends_assistant_message() -> None:
    ctx = StatelessConversation(replay=_replay_stub([], []))
    sent = [_msg("user")]
    turn = ModelTurn(response_id="r", output_text="final answer", tool_calls=[])

    ctx.record_turn(sent, turn)
    folded, _ = ctx.provider_request(_session(), [])

    assert folded == [
        _msg("user"),
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "final answer"}],
        },
    ]


async def test_stateless_record_turn_empty_turn_keeps_sent_items() -> None:
    ctx = StatelessConversation(replay=_replay_stub([], []))
    sent = [_msg("user")]
    turn = ModelTurn(response_id="r", output_text="", tool_calls=[])

    ctx.record_turn(sent, turn)
    folded, _ = ctx.provider_request(_session(), [])

    assert folded == [_msg("user")]


async def test_stateless_pause_payload_is_transcript_plus_outputs() -> None:
    calls: list[int] = []
    ctx = StatelessConversation(replay=_replay_stub([], calls))
    # Seed a transcript via a turn.
    ctx.record_turn([_msg("user")], ModelTurn(response_id="r", output_text="ok", tool_calls=[]))

    outputs = [{"type": "function_call_output", "call_id": "c1", "output": "done"}]
    payload = ctx.pause_payload(outputs)

    assert payload == [
        _msg("user"),
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok"}],
        },
        outputs[0],
    ]

    # Cursor is always None across every provider_request, including post-pause.
    _, cursor = ctx.provider_request(_session(), [])
    assert cursor is None


async def test_stateless_pause_payload_without_transcript_is_outputs_only() -> None:
    ctx = StatelessConversation(replay=_replay_stub([], []))
    outputs = [{"type": "function_call_output", "call_id": "c1", "output": "x"}]

    assert ctx.pause_payload(outputs) == outputs


# --------------------------------------------------------------------------- #
# model_turn_input_items (the moved helper)
# --------------------------------------------------------------------------- #


def test_model_turn_input_items_tool_calls_content_on_first_only() -> None:
    turn = ModelTurn(
        response_id="r",
        output_text="prose",
        tool_calls=[
            ToolCall(call_id="c1", name="alpha", arguments={"z": 1, "a": 0}),
            ToolCall(call_id="c2", name="beta", arguments={"k": "v"}),
        ],
    )

    items = model_turn_input_items(turn)

    assert items == [
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "alpha",
            "arguments": json.dumps({"a": 0, "z": 1}, separators=(",", ":"), sort_keys=True),
            "content": "prose",
        },
        {
            "type": "function_call",
            "call_id": "c2",
            "name": "beta",
            "arguments": json.dumps({"k": "v"}, separators=(",", ":"), sort_keys=True),
        },
    ]


def test_model_turn_input_items_text_only_and_empty() -> None:
    text_turn = ModelTurn(response_id="r", output_text="hello", tool_calls=[])
    assert model_turn_input_items(text_turn) == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        }
    ]

    empty_turn = ModelTurn(response_id="r", output_text="", tool_calls=[])
    assert model_turn_input_items(empty_turn) == []
