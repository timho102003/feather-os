# Harness Elegance Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Name the harness's hidden concepts (conversation state strategy, event
emission, tool outcomes, subprocess plumbing, provider dispatch, store lifecycle) as
small tested abstractions and delete the cross-cutting duplication — with byte-identical
behavior except one deliberate hardening (supervisor stderr bounds).

**Spec:** `docs/superpowers/specs/2026-06-12-harness-elegance-design.md`

**Architecture:** Strategy objects + null-object emitter inside `BaseAgent`; shared
pure-function utility modules for providers/subprocess/storage; a declarative provider
spec table consulted by all dispatch sites. Every public seam (`BaseAgent.run/run_loop`
signatures, `BaseLLMProvider`, tool builders, wire codecs, store APIs) is preserved.

**Tech stack:** Python 3.12, asyncio, pytest (`asyncio_mode=auto`), no new dependencies.

**Baseline:** `uv run pytest` green at 1638 passed / 6 skipped (63s) on branch
`harness-elegance` before Task 1.

**Per-task workflow (CLAUDE.md stages 3–5):** implement → add tests (failure + happy
path; new tests must fail if the refactor's behavior contract breaks) → run targeted
tests → run `uv run pytest` full → commit. Announce classification per task.

Verified facts the plan relies on:
- Tests import `_harden_strict_schema` from `feather.providers.openai_provider`
  (`tests/test_openai_provider_structured_outputs.py:143,162`) and
  `_retry_sleep_seconds` from `feather.providers.claude_provider`
  (`tests/test_claude_provider.py:21`) → those names stay importable as delegates.
- No test asserts store `_require_connection` messages or provider-factory error strings.
- No test calls `BaseAgent.run_loop` private helpers directly.
- `open_store_connection(db_path, *, foreign_keys=True, busy_timeout_ms=5000)`.
- `LeadSessionStore.initialize` creates only `LEAD_SESSIONS_TABLE` with
  `foreign_keys=False`; all other SQLite stores run `initialize_database_schema`.

---

### Task 1: `EventKind` + `EventEmitter` (WS2 foundation)

**Classification:** adding new logic (no existing path perturbed yet).

**Files:**
- Modify: `src/feather/models/runtime_models.py` (add `EventKind`, export)
- Modify: `src/feather/models/__init__.py` (re-export `EventKind`)
- Create: `src/feather/core/agent/events.py`
- Test: `tests/test_agent_events.py`

- [ ] **Step 1.1** Verify the emitted-kind inventory before freezing the enum:
  `grep -rn 'RuntimeEvent(' src/feather --include='*.py' | grep -o 'kind="[a-z_]*"' | sort -u`
  Expected kinds (13): assistant_text_delta, tool_started, tool_finished, awaiting_user,
  user_message_injected, agent_message_received, usage_updated, compaction_started,
  compaction_finished, compaction_failed, scheduled_task_triggered,
  scheduled_task_failed, completion_guard_injected. If `scheduled_task_completed` is
  emitted anywhere, include it; if it is only *consumed* (suspected dead branch in
  `cli/__init__.py`), exclude it and note in the commit message.

- [ ] **Step 1.2** Add to `runtime_models.py` (above `RuntimeEvent`) and export in
  `__all__` of both the module and `feather/models/__init__.py`:

```python
class EventKind(str, Enum):
    """Public taxonomy of RuntimeEvent.kind values.

    str-valued so emitters can pass members where plain strings flowed before:
    JSON, the IPC event codec, and every `event.kind == "literal"` comparison
    in the frontends see the identical string.
    """

    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    AWAITING_USER = "awaiting_user"
    USER_MESSAGE_INJECTED = "user_message_injected"
    AGENT_MESSAGE_RECEIVED = "agent_message_received"
    USAGE_UPDATED = "usage_updated"
    COMPACTION_STARTED = "compaction_started"
    COMPACTION_FINISHED = "compaction_finished"
    COMPACTION_FAILED = "compaction_failed"
    SCHEDULED_TASK_TRIGGERED = "scheduled_task_triggered"
    SCHEDULED_TASK_FAILED = "scheduled_task_failed"
    COMPLETION_GUARD_INJECTED = "completion_guard_injected"
```

- [ ] **Step 1.3** Create `src/feather/core/agent/events.py`:

```python
"""Null-object event emission for the agent loop."""

from __future__ import annotations

from typing import Any

from feather.models import EventHandler, RuntimeEvent

__all__ = ("EventEmitter",)


class EventEmitter:
    """Wraps an optional EventHandler so emit sites need no None checks.

    Deliberately adds no error handling: a raising handler propagates
    exactly as it did when call sites invoked the handler directly.
    """

    __slots__ = ("_handler",)

    def __init__(self, handler: EventHandler | None) -> None:
        self._handler = handler

    @property
    def handler(self) -> EventHandler | None:
        """The raw handler, for seams that accept ``EventHandler | None``."""
        return self._handler

    @property
    def enabled(self) -> bool:
        return self._handler is not None

    def emit(
        self,
        kind: str,
        *,
        text: str | None = None,
        tool_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._handler is None:
            return
        self._handler(
            RuntimeEvent(kind=kind, text=text, tool_name=tool_name, payload=payload)
        )

    def forward(self, event: RuntimeEvent) -> None:
        """Pass through a pre-built event (e.g. inbox preview)."""
        if self._handler is None:
            return
        self._handler(event)
```

- [ ] **Step 1.4** `tests/test_agent_events.py` — cases:
  - `test_emit_without_handler_is_noop` (EventEmitter(None).emit → no error)
  - `test_emit_builds_runtime_event_fields` (capture list; assert kind/text/tool/payload)
  - `test_forward_passes_prebuilt_event_identity`
  - `test_handler_exception_propagates` (handler raises ValueError → pytest.raises)
  - `test_event_kind_values_are_plain_strings`
    (`EventKind.TOOL_STARTED == "tool_started"`; `json.dumps({"k": EventKind.TOOL_STARTED})`
    round-trips to the bare string; `encode_event(RuntimeEvent(kind=EventKind.TOOL_STARTED))`
    from `feather.core.ipc.event_codec` decodes back to `"tool_started"`)
  - `test_event_kind_covers_all_emitted_literals` — tripwire: walk
    `src/feather` source with `Path.rglob`, regex `kind="([a-z_]+)"`, assert every hit
    is a member value of `EventKind` (guards future drift).

- [ ] **Step 1.5** Run: `uv run pytest tests/test_agent_events.py -v` → PASS;
  `uv run pytest --tb=short -q` → 1638+6 green (plus new).
- [ ] **Step 1.6** Commit: `refactor(events): add EventKind taxonomy + EventEmitter null-object`

---

### Task 2: Adopt the emitter in `base.py` + `compaction.py`; extract inbox preview (WS2)

**Classification:** modifying existing logic — emission mechanics replaced wholesale;
event payloads byte-identical.

**Files:**
- Modify: `src/feather/core/agent/base.py`
- Modify: `src/feather/core/agent/compaction.py`
- Test: extend `tests/test_agent_events.py` (inbox preview); existing
  `tests/test_base_agent*.py`, `tests/test_compaction.py` are the regression net.

- [ ] **Step 2.1** In `base.py`: at the top of `run_loop`, create
  `emitter = EventEmitter(event_handler)`. Replace every
  `if event_handler is not None: event_handler(RuntimeEvent(kind="X", ...))` with
  `emitter.emit(EventKind.X, ...)` (sites: completion_guard_injected ~line 485,
  awaiting_user ~line 529, usage_updated in `_emit_usage_ratio`, tool events in
  `_execute_tool_calls`, user_message_injected in `_drain_user_input_queue`,
  agent_message_received in `_drain_agent_inbox`). Private helpers
  `_execute_tool_calls`, `_drain_user_input_queue`, `_drain_agent_inbox`,
  `_emit_usage_ratio`, `_maybe_auto_compact` change their parameter from
  `event_handler: EventHandler | None` to `emitter: EventEmitter`. Seams that still
  take a raw handler (`provider.complete`, `compactor.maybe_compact`) receive
  `emitter.handler`.
- [ ] **Step 2.2** Extract the inbox preview block (`base.py:855-891`) into a module
  function in `base.py`:

```python
_INBOX_PREVIEW_CHARS = 240


def _inbox_received_event(
    *, sender_agent: str, sender_session: str, messages: list[AgentMessage]
) -> RuntimeEvent:
    """Build the agent_message_received event, including human-scan previews."""
    previews: list[str] = []
    for msg in messages:
        body = (msg.body or "").strip()
        if not body:
            previews.append("(empty body)")
            continue
        head = " ".join(body.split())
        if len(head) > _INBOX_PREVIEW_CHARS:
            head = head[:_INBOX_PREVIEW_CHARS] + f"… (+{len(body) - _INBOX_PREVIEW_CHARS} chars)"
        previews.append(f"[{len(body)} chars] {head}")
    total_chars = sum(len(msg.body or "") for msg in messages)
    return RuntimeEvent(
        kind=EventKind.AGENT_MESSAGE_RECEIVED,
        text=(
            f"{sender_agent} ({sender_session}): "
            f"{len(messages)} message(s), {total_chars} chars\n"
            f"    {' | '.join(previews)}"
        ),
        payload={
            "from_agent_name": sender_agent,
            "from_session_id": sender_session,
            "count": len(messages),
            "total_chars": total_chars,
            "previews": previews,
            "bodies": [msg.body or "" for msg in messages],
        },
    )
```

  `_drain_agent_inbox` calls `emitter.forward(_inbox_received_event(...))`.
  **Fidelity check:** original preview join is `" | ".join`, truncation marker
  `… (+N chars)`, payload key order as above — keep identical.
- [ ] **Step 2.3** `compaction.py`: `maybe_compact` keeps its public
  `event_handler: EventHandler | None` parameter; internally wraps
  `emitter = EventEmitter(event_handler)` and uses `emitter.emit(EventKind...)` for the
  three sites. No signature change.
- [ ] **Step 2.4** Add tests in `tests/test_agent_events.py`:
  `test_inbox_received_event_previews_truncation` (long body → `… (+N chars)` and
  `[N chars]` prefix; empty body → `(empty body)`), and
  `test_inbox_received_event_payload_bodies_roundtrip`.
- [ ] **Step 2.5** `uv run pytest tests/test_agent_events.py tests/test_base_agent.py tests/test_base_agent_inbox.py tests/test_compaction.py -v` → PASS;
  full suite green.
- [ ] **Step 2.6** Commit: `refactor(agent): route loop events through EventEmitter; extract inbox preview builder`

---

### Task 3: Tool outcome normalization (WS3)

**Classification:** modifying existing logic — both halves replaced by one path; no
surviving duplicate branch.

**Files:**
- Modify: `src/feather/core/agent/base.py` (`_execute_tool_calls` result loop, ~lines 655-716)
- Test: extend `tests/test_base_agent.py`

- [ ] **Step 3.1** Replace the post-gather loop with the normalized form:

```python
        outputs: list[dict[str, Any]] = []
        question: str | None = None

        for tool_call, result, exc in results:
            if exc is not None:
                output_text = f"Tool `{tool_call.name}` failed: {exc}"
                await_question: str | None = None
                loaded_skill: str | None = None
            elif result is None:
                continue
            else:
                output_text = result.output
                await_question = result.await_user_question
                loaded_skill = result.loaded_skill_name

            if loaded_skill is not None:
                await self._session_store.append_loaded_skill(session_id, loaded_skill)
            artifact = await self._tool_output_store.write(tool_call.name, output_text)
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": output_text,
                }
            )
            await self._session_store.add_message(
                session_id,
                MessageRole.TOOL,
                artifact.reference_text,
                file_ref=artifact.file_ref,
            )
            if await_question and question is None:
                question = await_question
            emitter.emit(
                EventKind.TOOL_FINISHED, tool_name=tool_call.name, text=output_text
            )

        return outputs, question
```

  **Fidelity notes:** error path previously wrote the artifact too (same call) and
  emitted tool_finished with the failure text — identical here. Skill-append before
  artifact write — preserved. `result is None` short-circuit — preserved.
- [ ] **Step 3.2** Add `tests/test_base_agent.py::test_tool_failure_and_success_share_output_contract`:
  run a turn with two tools (one raises ValueError("boom"), one succeeds); assert both
  produce a `function_call_output` with matching `call_id`s, both persist TOOL-role
  messages, the failure output is `` "Tool `failing` failed: boom" ``, and exactly two
  `tool_finished` events fire. (Follow the existing fake-provider/fake-tool pattern at
  the top of `tests/test_base_agent.py`.)
- [ ] **Step 3.3** Targeted + full suite green.
- [ ] **Step 3.4** Commit: `refactor(agent): single normalized path for tool success/failure outcomes`

---

### Task 4: `ConversationContext` (WS1 — the core)

**Classification:** modifying existing logic — after this task no code inside
`run`/`run_loop` may consult `self._provider.stateful`; the strategies fully own it.

**Files:**
- Create: `src/feather/core/agent/conversation.py`
- Modify: `src/feather/core/agent/base.py` (`run`, `run_loop`, pause path; delete
  `_model_turn_input_items`, `_has_stateless_pending_context` from the class)
- Test: `tests/test_conversation_context.py`; regression net:
  `tests/test_base_agent.py` (stateful), `tests/test_base_agent_openrouter.py`
  (stateless), `tests/test_base_agent_inbox.py`, `tests/test_base_agent_completion_guard.py`.

- [ ] **Step 4.1** Create `conversation.py`:

```python
"""Conversation-state strategies: how prior context reaches the provider.

Stateful providers (OpenAI Responses) keep server-side history keyed by
``previous_response_id`` — each turn sends only new items plus the cursor.
Stateless providers (OpenRouter / Claude) get no cursor — each turn must
carry the full structural transcript (assistant tool_calls followed by
matching tool outputs), which these strategies own end to end.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from feather.models import ModelTurn, SessionRecord

__all__ = (
    "ConversationContext",
    "StatefulConversation",
    "StatelessConversation",
    "model_turn_input_items",
)

HistoryReplayFn = Callable[[], Awaitable[list[dict[str, Any]]]]


def model_turn_input_items(turn: ModelTurn) -> list[dict[str, Any]]:
    """Convert a model turn into replayable provider input items."""
    # body moved verbatim from BaseAgent._model_turn_input_items (base.py:1134-1168)


def _has_structural_context(pending_inputs: list[dict[str, Any]]) -> bool:
    """True when pending inputs already carry replayed transcript context."""
    return any(
        item.get("type") in {"message", "function_call"} for item in pending_inputs
    )


class ConversationContext(ABC):
    """Per-run strategy owning replay, cursor, and pause semantics."""

    @abstractmethod
    async def initial_input_items(
        self, session: SessionRecord, new_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Items for run_loop at run() entry: pending → (history?) → new."""

    @abstractmethod
    async def begin(self, input_items: list[dict[str, Any]]) -> None:
        """Hook at run_loop entry (stateless seeds replay on empty input)."""

    @abstractmethod
    def provider_request(
        self, session: SessionRecord, input_items: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Consume input_items; return (items to send, cursor or None)."""

    @abstractmethod
    def record_turn(
        self, sent_items: list[dict[str, Any]], turn: ModelTurn
    ) -> None:
        """Fold a completed provider turn back into the strategy state."""

    @abstractmethod
    def pause_payload(
        self, tool_outputs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """pending_inputs to persist when pausing for AWAITING_USER."""


class StatefulConversation(ConversationContext):
    def __init__(self, *, replay: HistoryReplayFn) -> None:
        self._replay = replay

    async def initial_input_items(self, session, new_items):
        items = list(session.pending_inputs)
        if session.last_response_id is None:
            items.extend(await self._replay())
        items.extend(new_items)
        return items

    async def begin(self, input_items):
        return None

    def provider_request(self, session, input_items):
        return list(input_items), session.last_response_id

    def record_turn(self, sent_items, turn):
        return None

    def pause_payload(self, tool_outputs):
        return list(tool_outputs)


class StatelessConversation(ConversationContext):
    def __init__(self, *, replay: HistoryReplayFn) -> None:
        self._replay = replay
        self._transcript: list[dict[str, Any]] | None = None

    async def initial_input_items(self, session, new_items):
        items = list(session.pending_inputs)
        if not _has_structural_context(items):
            items.extend(await self._replay())
        items.extend(new_items)
        return items

    async def begin(self, input_items):
        if not input_items:
            self._transcript = await self._replay()

    def provider_request(self, session, input_items):
        if self._transcript is None:
            self._transcript = list(input_items)
        elif input_items:
            self._transcript.extend(input_items)
        return list(self._transcript), None

    def record_turn(self, sent_items, turn):
        transcript = list(sent_items)
        transcript.extend(model_turn_input_items(turn))
        self._transcript = transcript

    def pause_payload(self, tool_outputs):
        items = list(self._transcript or [])
        items.extend(tool_outputs)
        return items
```

  (Full type annotations on the concrete methods in the real file; elided here for
  plan brevity only — the signatures match the ABC exactly.)

- [ ] **Step 4.2** Rewire `base.py`:
  - Add `def _conversation_context(self) -> ConversationContext:` returning
    `StatefulConversation(replay=...)` when `self._provider.stateful` else
    `StatelessConversation(replay=...)`, with
    `replay=lambda: self._build_history_replay_items(session_id)` — **note** replay
    needs the session id; make the factory take it:
    `_conversation_context(session_id: str)` and bind
    `functools.partial(self._build_history_replay_items, session_id)`.
    This is the ONLY remaining `stateful` read in the class.
  - `run()`: replace lines 268-291 with

```python
            session = await self._session_store.get_session(session_id)
            context = self._conversation_context(session_id)
            pending_inputs = await context.initial_input_items(
                session, await_new := await self._persist_incoming_user_message(session_id, incoming_text)
            )
```

    …expressed without the walrus in the real edit: persist first
    (`new_input_items, _ = await self._persist_incoming_user_message(...)`), then
    `pending_inputs = await context.initial_input_items(session, new_input_items)`.
    **Ordering fidelity:** original persisted the user message *after* computing
    history items but history building only reads prior rows via
    `render_history_for_cache`; persisting first would change replay contents
    (the new user row would appear in history AND as a new item → duplication).
    **Therefore keep the original order**: compute
    `pending = await context.initial_input_items(session, [])` BEFORE persisting?
    No — `initial_input_items` appends new_items internally. Resolution: split the
    strategy call:

```python
            session = await self._session_store.get_session(session_id)
            context = self._conversation_context(session_id)
            pending_inputs = await context.initial_input_items(session, [])
            new_input_items, _ = await self._persist_incoming_user_message(
                session_id, incoming_text
            )
            pending_inputs.extend(new_input_items)
```

    This preserves the original sequence exactly (history replay rendered before the
    new row exists) and keeps `initial_input_items(session, new_items)` general for
    tests. Pass `[]` as `new_items` at this call site.
  - `run_loop(...)` gains keyword-only `context: ConversationContext | None = None`;
    first lines: `ctx = context or self._conversation_context(session_id)` then
    `await ctx.begin(input_items)` (replaces the `stateless_context_items` seeding at
    331-334). `run()` passes its `context=` through.
  - Per-iteration block 381-398 becomes:

```python
                effective_input_items, effective_cursor = ctx.provider_request(
                    session, input_items
                )
                input_items = []
```

  - After `update_response_state` (line 430-435), the 436-440 block becomes
    `ctx.record_turn(effective_input_items, turn)` (unconditional; stateful no-op).
  - Pause path 519-528 becomes:

```python
                if question is not None:
                    pending_inputs = ctx.pause_payload(input_items)
                    await self._session_store.update_response_state(
                        session_id,
                        pending_inputs=pending_inputs,
                        status=SessionStatus.AWAITING_USER,
                    )
```

    (`input_items` here holds the tool outputs returned by `_execute_tool_calls`,
    exactly as before.)
  - Delete `_model_turn_input_items` / `_has_stateless_pending_context` methods;
    import `model_turn_input_items` where compaction-free code needs it (it doesn't —
    only the strategy uses it now).
- [ ] **Step 4.3** `tests/test_conversation_context.py` — pure-unit, no agent needed.
  Build a `SessionRecord` factory and a stub replay returning a sentinel item list.
  Cases:
  - stateful: `test_stateful_replays_only_when_cursor_none`,
    `test_stateful_orders_pending_then_history_then_new`,
    `test_stateful_request_returns_cursor_and_copy`,
    `test_stateful_pause_payload_is_outputs_only`, `test_stateful_record_turn_noop`.
  - stateless: `test_stateless_skips_replay_when_pending_has_structural_context`
    (pending containing `{"type": "function_call"}` → replay NOT called),
    `test_stateless_begin_seeds_replay_on_empty_input` (resume_on_inbox case),
    `test_stateless_request_seeds_then_extends_transcript`,
    `test_stateless_record_turn_folds_tool_calls`
    (turn with 2 tool calls + output_text → transcript = sent + function_call items,
    first carries `content`, arguments JSON has `sort_keys` + compact separators),
    `test_stateless_record_turn_text_only_appends_assistant_message`,
    `test_stateless_pause_payload_is_transcript_plus_outputs`,
    `test_stateless_cursor_always_none`.
- [ ] **Step 4.4** Regression: `uv run pytest tests/test_conversation_context.py tests/test_base_agent.py tests/test_base_agent_openrouter.py tests/test_base_agent_inbox.py tests/test_base_agent_completion_guard.py -v` → PASS.
  Grep check: `grep -n "stateful" src/feather/core/agent/base.py` → exactly one hit
  (inside `_conversation_context`).
- [ ] **Step 4.5** Full suite green. Commit:
  `refactor(agent): extract ConversationContext strategies for stateful/stateless turns`

---

### Task 5: Provider shared utilities (WS5)

**Classification:** modifying existing logic — duplicated bodies replaced by one
implementation; public/provider-local names preserved as delegates.

**Files:**
- Create: `src/feather/providers/schema_utils.py`
- Create: `src/feather/providers/retry_utils.py`
- Modify: `src/feather/providers/openai_provider.py` (replace `_harden_strict_schema`
  body with `from feather.providers.schema_utils import harden_strict_schema as _harden_strict_schema`)
- Modify: `src/feather/providers/openrouter_translator.py` (same delegate)
- Modify: `src/feather/providers/claude_translator.py` (same delegate)
- Modify: `src/feather/providers/openrouter_provider.py`
  (`_retry_sleep_seconds_from_headers` + `_retry_sleep_seconds` delegate to retry_utils)
- Modify: `src/feather/providers/claude_provider.py` (`_retry_sleep_seconds` delegates)
- Test: `tests/test_provider_schema_utils.py`, `tests/test_provider_retry_utils.py`

- [ ] **Step 5.1** Before writing the shared module, diff the three copies to prove
  semantic identity:
  `sed -n '52,97p' src/feather/providers/openai_provider.py` vs
  `sed -n '621,658p' src/feather/providers/openrouter_translator.py` vs
  `sed -n '645,682p' src/feather/providers/claude_translator.py`.
  The canonical body is the openai one; `setdefault("additionalProperties", False)` ≡
  the explicit-check variant. If any copy diverges beyond that, STOP and surface the
  divergence rather than silently unifying.
- [ ] **Step 5.2** `schema_utils.py`: module docstring + `harden_strict_schema(schema:
  dict[str, Any]) -> None` containing the canonical body verbatim; `__all__`.
- [ ] **Step 5.3** `retry_utils.py`:

```python
"""Shared retry/backoff math for streaming providers."""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone

__all__ = (
    "backoff_delay",
    "seconds_from_retry_after",
    "seconds_until_unix_timestamp",
)


def backoff_delay(attempt: int, base_delay: float) -> float:
    """Exponential backoff with full jitter: base * 2^attempt + U(0, base)."""
    return base_delay * (2**attempt) + random.uniform(0.0, base_delay)


def seconds_until_unix_timestamp(header: str | None, *, max_wait: float) -> float | None:
    """Parse an X-RateLimit-Reset-style unix-seconds header into a wait."""
    if not header or not header.strip().isdigit():
        return None
    return min(max(0.0, int(header.strip()) - time.time()), max_wait)


def seconds_from_retry_after(header: str | None, *, max_wait: float) -> float | None:
    """Parse RFC 7231 Retry-After (delta-seconds or HTTP-date) into a wait."""
    if not header:
        return None
    hint = header.strip()
    if hint.isdigit():
        return min(float(hint), max_wait)
    try:
        target = datetime.strptime(hint, "%a, %d %b %Y %H:%M:%S GMT").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return min(max(0.0, target.timestamp() - time.time()), max_wait)
```

  **Then reconcile with the actual provider implementations** (openrouter 237-247,
  claude 181-209): the existing functions are the source of truth for max_wait values,
  status gating, and date handling — adjust the helpers to express exactly the current
  behavior and make the provider functions one-line compositions. If the claude parser
  treats naive datetimes as local time today, replicate that (no silent timezone fix);
  if it uses utc, keep utc. Behavior identical is the bar.
- [ ] **Step 5.4** Tests:
  - schema: port the strict-mode cases from
    `tests/test_openai_provider_structured_outputs.py:118-178` against
    `schema_utils.harden_strict_schema` (nested objects, `$defs`, anyOf branches), keep
    the original test file untouched and passing via the delegate import.
  - retry: bounds-test `backoff_delay` (`base*2^a <= d < base*2^a + base`), header
    parsing happy/garbage/huge-clamp cases, and equality checks that
    `claude_provider._retry_sleep_seconds(429, 0, 0.5, "3")` still returns 3.0 and the
    malformed-date fallback still lands in the jitter window (mirrors
    `tests/test_claude_provider.py:104-123`).
- [ ] **Step 5.5** Targeted: `uv run pytest tests/test_provider_schema_utils.py tests/test_provider_retry_utils.py tests/test_openai_provider_structured_outputs.py tests/test_claude_provider.py tests/test_openrouter_provider.py tests/test_claude_tool_schema_sanitizer.py -v` → PASS. Full suite green.
- [ ] **Step 5.6** Commit: `refactor(providers): single harden_strict_schema + shared backoff/header helpers`

---

### Task 6: Provider catalog (WS6)

**Classification:** modifying existing logic — four name-switches replaced by one
table; all error strings preserved.

**Files:**
- Create: `src/feather/providers/catalog.py`
- Modify: `src/feather/core/agent/factory.py`
  (`_resolve_provider`, `_resolve_model_name`, `_supports_multimodal_attachments`)
- Modify: `src/feather/runtime/provider_factory.py` (`_build_default_provider`)
- Test: `tests/test_provider_catalog.py`; regression:
  `tests/test_agent_factory.py`, `tests/test_runtime_reload.py`, `tests/test_lead_agent.py`

- [ ] **Step 6.1** `catalog.py`:

```python
"""Declarative registry of selectable LLM providers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from feather.providers.base import BaseLLMProvider
from feather.providers.claude_provider import ClaudeMessagesProvider
from feather.providers.openai_provider import OpenAIResponsesProvider
from feather.providers.openrouter_provider import OpenRouterChatProvider

if TYPE_CHECKING:
    from feather.models import AppConfig

__all__ = ("PROVIDER_NAMES", "ProviderSpec", "provider_spec")


@dataclass(slots=True, frozen=True)
class ProviderSpec:
    """Everything dispatch sites need to know about one provider, in one row.

    ``config_block`` returns the provider's app.yaml section (None when the
    operator didn't configure it); callers own their missing-block error
    messages so existing wording is preserved. ``build`` assumes the block
    is present (openai's block is always present on AppConfig).
    """

    name: str
    build: Callable[["AppConfig"], BaseLLMProvider]
    config_block: Callable[["AppConfig"], Any | None]
    default_model: Callable[["AppConfig"], str]
    supports_multimodal: Callable[["AppConfig"], bool]


_SPECS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        build=lambda cfg: OpenAIResponsesProvider(cfg.openai),
        config_block=lambda cfg: cfg.openai,
        default_model=lambda cfg: cfg.openai.model,
        supports_multimodal=lambda cfg: True,
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        build=lambda cfg: OpenRouterChatProvider(cfg.openrouter),
        config_block=lambda cfg: cfg.openrouter,
        default_model=lambda cfg: (
            cfg.openrouter.model if cfg.openrouter is not None else cfg.openai.model
        ),
        supports_multimodal=lambda cfg: (
            cfg.openrouter.supports_multimodal if cfg.openrouter is not None else False
        ),
    ),
    "claude": ProviderSpec(
        name="claude",
        build=lambda cfg: ClaudeMessagesProvider(cfg.claude),
        config_block=lambda cfg: cfg.claude,
        default_model=lambda cfg: (
            cfg.claude.model if cfg.claude is not None else cfg.openai.model
        ),
        supports_multimodal=lambda cfg: (
            cfg.claude.supports_multimodal if cfg.claude is not None else False
        ),
    ),
}

PROVIDER_NAMES: tuple[str, ...] = tuple(_SPECS)


def provider_spec(name: str) -> ProviderSpec | None:
    """Look up a spec by normalized provider name."""
    return _SPECS.get(name)
```

- [ ] **Step 6.2** `factory._resolve_provider` keeps its cache + normalization; the
  if/elif chain becomes: `spec = provider_spec(agent_provider)`; unknown → the existing
  ``f"Agent `{...}` requested unknown provider `{...}` (expected 'openai', 'openrouter', or 'claude')"``
  ValueError; known but `spec.config_block(self._app_config) is None` and name != "openai" →
  the existing ``f"Agent `{...}` requested provider={name} but no `{name}:` block in app.yaml"``
  ValueError; else `built = spec.build(self._app_config)`.
  `_resolve_model_name` → `spec.default_model(self._app_config)` after the explicit
  `agent_config.model` check (unknown name falls back to openai default model exactly
  as today — today's code falls through to `self._app_config.openai.model` for unknown
  names; preserve by `spec = provider_spec(name) or provider_spec("openai")`).
  `_supports_multimodal_attachments` → `spec.supports_multimodal(...)` with the same
  unknown→openai fallback (today returns True for unknown names; openai spec gives True ✓).
- [ ] **Step 6.3** `provider_factory._build_default_provider`: keep its distinct
  app-level error strings ("active_provider=X but no `X:` block in app.yaml" /
  `f"unsupported active_provider={active!r} (expected 'openai', 'openrouter', or 'claude')"`),
  table-driven via `provider_spec` + `config_block`.
- [ ] **Step 6.4** `tests/test_provider_catalog.py`:
  - `test_specs_cover_expected_names` (PROVIDER_NAMES == ("openai","openrouter","claude"))
  - `test_default_model_prefers_provider_block` / `..._falls_back_to_openai`
  - `test_supports_multimodal_matrix` (openai True; openrouter/claude follow block, False when absent)
  - `test_build_returns_distinct_instances_per_call`
  - `test_unknown_name_returns_none`
- [ ] **Step 6.5** Targeted + full suite green (factory error-message tests prove string fidelity).
- [ ] **Step 6.6** Commit: `refactor(providers): declarative ProviderSpec catalog replaces four name-switches`

---

### Task 7: Shared subprocess plumbing (WS4)

**Classification:** modifying existing logic + one deliberate hardening (supervisor
stderr cap + drainer await timeout) — call out in commit message.

**Files:**
- Create: `src/feather/core/ipc/process.py`
- Modify: `src/feather/tools/spawn_agent_tool.py`
  (`launch_subagent_process`, `_drain_stream` → delete, `_terminate_launched`)
- Modify: `src/feather/core/leads/supervisor.py`
  (`_default_subprocess_factory`, `_SubprocessWorkerHandle.__init__/_drain_stderr/wait`)
- Modify: `src/feather/runtime/root.py` (`_terminate_live_subagents` escalation body)
- Test: `tests/test_ipc_process.py`; regression: `tests/test_spawn_agent_tool.py`,
  `tests/test_subagent_reaper.py`, `tests/test_subagent_reaper_race.py`,
  `tests/test_lead_supervisor.py`, `tests/test_runtime_shutdown_tasks.py`

- [ ] **Step 7.1** `core/ipc/process.py`:

```python
"""Shared subprocess spawn/drain/terminate plumbing.

One canonical implementation for the three places Feather runs piped child
processes (sub-agent spawn, lead-worker spawn, runtime shutdown sweep), so
pipe-deadlock prevention and the SIGTERM→SIGKILL escalation can't drift.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field

__all__ = (
    "PipedProcess",
    "cancel_drainers",
    "drain_stream",
    "spawn_piped_process",
    "terminate_process",
)


async def drain_stream(
    stream: asyncio.StreamReader | None,
    buffer: bytearray,
    *,
    max_bytes: int | None = None,
) -> None:
    """Read ``stream`` to EOF into ``buffer``; never raises into the loop.

    With ``max_bytes`` the buffer stops growing but the stream keeps being
    consumed, so a chatty child can't deadlock on a full pipe while the
    parent caps its memory.
    """
    if stream is None:
        return
    try:
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                return
            if max_bytes is None or len(buffer) < max_bytes:
                room = None if max_bytes is None else max_bytes - len(buffer)
                buffer.extend(chunk if room is None else chunk[:room])
    except Exception:  # noqa: BLE001 — drainers must never raise
        return


@dataclass(slots=True)
class PipedProcess:
    """A spawned child plus its capture buffers and drainer tasks."""

    process: asyncio.subprocess.Process
    stdout_buffer: bytearray = field(default_factory=bytearray)
    stderr_buffer: bytearray = field(default_factory=bytearray)
    drainers: tuple[asyncio.Task[None], ...] = ()


async def spawn_piped_process(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    stdin: int,
    capture_stdout: bool = True,
    stderr_max_bytes: int | None = None,
    name: str = "process",
) -> PipedProcess:
    """Spawn ``argv`` with piped stdout/stderr and start drainers.

    ``capture_stdout=False`` leaves stdout undrained for line-oriented
    protocol readers (the lead-worker supervisor) — only stderr gets a
    drainer in that mode.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )
    piped = PipedProcess(process=process)
    drainers: list[asyncio.Task[None]] = []
    if capture_stdout:
        drainers.append(
            asyncio.create_task(
                drain_stream(process.stdout, piped.stdout_buffer),
                name=f"{name}-stdout",
            )
        )
    drainers.append(
        asyncio.create_task(
            drain_stream(
                process.stderr, piped.stderr_buffer, max_bytes=stderr_max_bytes
            ),
            name=f"{name}-stderr",
        )
    )
    piped.drainers = tuple(drainers)
    return piped


async def terminate_process(
    process: asyncio.subprocess.Process,
    *,
    term_timeout: float = 2.0,
    kill_timeout: float = 2.0,
) -> None:
    """SIGTERM → wait → SIGKILL → wait escalation; idempotent and quiet."""
    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=term_timeout)
        return
    except asyncio.TimeoutError:
        pass
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=kill_timeout)


async def cancel_drainers(drainers: tuple[asyncio.Task[None], ...]) -> None:
    """Cancel-and-await drainer tasks, swallowing their exit."""
    for drainer in drainers:
        if not drainer.done():
            drainer.cancel()
        with contextlib.suppress(BaseException):
            await drainer
```

  **Fidelity check against originals before landing:** spawn_agent_tool's
  `_terminate_launched` calls `terminate()` then waits even when terminate raised
  ProcessLookupError — replicate semantics (the helper above returns early only on
  returncode; ProcessLookupError still proceeds to wait — verify against
  `spawn_agent_tool.py:492-518` and `root.py:801-847` line by line; root.py logs a
  warning when SIGKILL wait times out — keep that log at the root.py call site by
  checking `process.returncode is None` after `terminate_process` returns).
- [ ] **Step 7.2** `spawn_agent_tool.launch_subagent_process` → spawn via
  `spawn_piped_process(argv, cwd=str(root), env=subprocess_env, stdin=asyncio.subprocess.DEVNULL, name=f"subagent-{session_id}")`,
  build `LaunchedSubagent(process=piped.process, stdout_buffer=piped.stdout_buffer,
  stderr_buffer=piped.stderr_buffer, drainers=piped.drainers)`; keep the
  task-file cleanup `try/except` around the spawn. Delete `_drain_stream`;
  `_terminate_launched` body → `await terminate_process(launched.process)` +
  `await cancel_drainers(launched.drainers)`.
- [ ] **Step 7.3** `supervisor.py`:
  - `_default_subprocess_factory`: spawn via
    `spawn_piped_process(argv, cwd=str(self._project_root), env=subprocess_env_with_home(), stdin=asyncio.subprocess.PIPE, capture_stdout=False, stderr_max_bytes=_STDERR_MAX_BYTES, name="lead-worker")`
    and pass the `PipedProcess` into `_SubprocessWorkerHandle`.
  - `_SubprocessWorkerHandle.__init__(piped: PipedProcess)`: drop its own
    `_drain_stderr` task creation; `stderr_buffer` property reads
    `bytes(self._piped.stderr_buffer)`; `wait()` awaits drainers with
    `asyncio.wait_for(..., timeout=2.0)` under suppress. Add module constant
    `_STDERR_MAX_BYTES = 1024 * 1024`.
  - All other handle methods unchanged (stdin write no-await invariant untouched).
- [ ] **Step 7.4** `runtime/root._terminate_live_subagents`: replace the inline
  escalation (810-838) with `await terminate_process(proc)` while preserving the
  `killed` bookkeeping (`killed = proc.returncode is None` *before* the call when the
  process was live → after call, if still `returncode is None` log the SIGKILL warning
  exactly as today) and replace the drainer-cancel loop (840-846) with
  `await cancel_drainers(tuple(entry.drainers))`.
- [ ] **Step 7.5** `tests/test_ipc_process.py` (use `sys.executable -c` children):
  - `test_spawn_captures_stdout_and_stderr` (child prints to both; buffers match)
  - `test_capture_stdout_false_leaves_stdout_readable`
    (parent reads `process.stdout.readline()` itself; stderr still drained)
  - `test_stderr_cap_bounds_buffer_but_drains_pipe`
    (child writes 4 MiB to stderr then exits 0 with cap 64 KiB → buffer ≤ cap, process
    exits cleanly — proves no pipe deadlock)
  - `test_terminate_process_sigterm_path` (sleeping child exits on TERM)
  - `test_terminate_process_kill_escalation` (child traps SIGTERM
    via `signal.signal(SIGTERM, SIG_IGN)`; helper falls through to kill; returncode set)
  - `test_terminate_process_idempotent_on_exited_child`
  - `test_cancel_drainers_swallows_cancellation`
- [ ] **Step 7.6** Targeted + full suite green. Commit:
  `refactor(ipc): shared subprocess spawn/drain/terminate; bound supervisor stderr capture`

---

### Task 8: SQLite store base (WS7)

**Classification:** modifying existing logic — lifecycle lifted; per-store behavior and
error strings preserved.

**Files:**
- Create: `src/feather/storage/base.py`
- Modify: `src/feather/storage/session_store.py`, `cron_store.py`,
  `agent_message_store.py`, `task_store.py`, `lead_session_store.py`,
  `worker_heartbeat_store.py`
- Test: `tests/test_storage_base.py`; regression: every `tests/test_*_store*.py` +
  `tests/test_storage_connection.py`

- [ ] **Step 8.1** `storage/base.py`:

```python
"""Shared lifecycle for long-lived SQLite-backed stores."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import aiosqlite

from feather.storage.connection import open_store_connection
from feather.storage.schema import initialize_database_schema

__all__ = ("BaseSQLiteStore",)


class BaseSQLiteStore:
    """One long-lived aiosqlite connection with the house open/close protocol.

    Subclasses may override ``_FOREIGN_KEYS`` (LeadSessionStore: the pointer
    table references nothing) and ``_apply_schema`` (stores that own a single
    table instead of the full schema).
    """

    _FOREIGN_KEYS: ClassVar[bool] = True

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the connection and ensure required tables exist."""
        self._connection = await open_store_connection(
            self._db_path, foreign_keys=self._FOREIGN_KEYS
        )
        await self._apply_schema(self._connection)
        await self._connection.commit()

    async def _apply_schema(self, connection: aiosqlite.Connection) -> None:
        await initialize_database_schema(connection)

    async def close(self) -> None:
        """Close the SQLite connection (idempotent)."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError(
                f"{type(self).__name__}.initialize() must be called before use."
            )
        return self._connection
```

- [ ] **Step 8.2** Adopt per store, smallest first: `WorkerHeartbeatStore`,
  `LeadSessionStore` (`_FOREIGN_KEYS = False`, `_apply_schema` executes only
  `LEAD_SESSIONS_TABLE.create_sql`), `CronJobStore`, `TaskStore`, `SessionStore`
  (keeps its extra `_active_mcp_lock` in its own `__init__` calling `super().__init__`),
  `AgentMessageStore` (keeps `inbox_cap` param; **note** its message today says
  "must be awaited before use" — unify to the base's "must be called before use."
  wording; no test asserts it, and `session_store` already uses that phrasing).
  Delete each store's now-redundant `initialize`/`close`/`_require_connection`
  (keep store-specific `initialize` overrides only where schema differs).
  `SessionStore` has two raise sites (655/660 — `_require_connection` and `_execute`);
  point both at the base method.
- [ ] **Step 8.3** `tests/test_storage_base.py`:
  - `test_require_connection_before_initialize_raises_with_class_name`
  - `test_initialize_close_idempotent_cycle` (initialize → close → close again ok)
  - `test_lead_session_store_opens_without_foreign_keys`
    (`PRAGMA foreign_keys` query returns 0 after initialize; 1 for SessionStore)
  - `test_base_applies_full_schema_by_default` (tables from `schema.py` exist)
- [ ] **Step 8.4** Targeted store tests + full suite green. Commit:
  `refactor(storage): BaseSQLiteStore lifecycle shared by the six long-lived stores`

---

### Task 9: Memory shutdown seam (WS9)

**Classification:** modifying existing logic — close path moves behind owners; same
client ends up closed.

**Files:**
- Modify: `src/feather/memory/store/base.py` (add `async def aclose(self) -> None`
  default no-op to `BaseVectorStore` — verify exact ABC name on read)
- Modify: `src/feather/memory/store/qdrant.py` (`aclose` closes `self._client` when it
  has `close`)
- Modify: `src/feather/memory/service.py` (`MemoryService.aclose()` → `await self._store.aclose()`)
- Modify: `src/feather/memory/runtime.py` (`MemoryStack.aclose` also closes
  `self.service` when not None)
- Modify: `src/feather/runtime/root.py` (delete the spelunking block at 775-782;
  ordering: trigger drain → stack.aclose (now closes qdrant too) — preserve relative
  order of remaining shutdown steps)
- Test: extend `tests/memory/test_runtime.py` + `tests/memory/test_store_qdrant.py`

- [ ] **Step 9.1** Read the three memory files first; if `QdrantVectorStore` already
  exposes a close-like method, reuse it instead of adding `aclose`. Implement the chain
  with swallow-and-log per house style (a failing close must not break shutdown).
- [ ] **Step 9.2** Tests: `test_memory_stack_aclose_closes_service_store_client`
  (stub client with `close()` coroutine → called once),
  `test_memory_stack_aclose_survives_client_close_error` (raising close → logged,
  no raise), plus existing shutdown tests stay green
  (`tests/test_runtime_shutdown_tasks.py`).
- [ ] **Step 9.3** Full suite green. Commit:
  `refactor(memory): close vector-store client via MemoryStack.aclose seam`

---

### Task 10: Simplify pass + docs (CLAUDE.md stage 6)

- [ ] **Step 10.1** Re-read every diff hunk (`git diff master...HEAD`) hunting dead
  code, leftover imports, duplicate copies the refactor obsoleted (e.g. provider-local
  retry bodies, the old `_drain_stream`), and accidental complexity in the new modules.
  Run `uv run pytest -q` after each removal.
- [ ] **Step 10.2** Update `CLAUDE.md` architecture-spine bullet list and
  `docs/architecture.md` only where module names changed (new
  `core/agent/conversation.py`, `core/agent/events.py`, `core/ipc/process.py`,
  `providers/catalog.py`, `storage/base.py`) — one-line mentions, no rewrites.
- [ ] **Step 10.3** Commit: `docs: note new harness modules in CLAUDE.md + architecture`

### Task 11: Red-team review (CLAUDE.md stages 7–8)

- [ ] **Step 11.1** Write the review checklist from the spec's invariants section +
  every touched seam; then trace each changed function's upstream callers and
  downstream callees from the entrypoints (cli/tui/api → runtime → agent → provider/
  tools → stores; subprocess entries: `subagent_entry`, `lead_worker_entry`).
- [ ] **Step 11.2** Hold each task to its declared classification (no surviving old
  paths for "modifying"; zero perturbation for "adding").
- [ ] **Step 11.3** Performance checklist over the diff: no new awaits inside
  transactions, no event-loop blocking, no unbounded growth (new transcript list is
  per-run, bounded by compaction as before; PipedProcess buffers bounded where they
  weren't — improvement), no cache-invalidation changes.
- [ ] **Step 11.4** Full suite ×2 (flake check; known flaky:
  `test_request_input_waits_for_correlated_reply` — re-run before judging).
  Fix anything found; re-run; only then declare done.
