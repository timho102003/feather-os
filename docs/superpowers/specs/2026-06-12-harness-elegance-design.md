# Harness Elegance Refactor — Design

**Date:** 2026-06-12
**Branch:** `harness-elegance`
**Status:** Approved for implementation (autonomous session; user delegated design decisions:
"even if you think huge refactor or redesign is required for better elegant more abstract
more cleaner design, then do it")

## Problem

Post-v0.2.0 the harness is functionally healthy (1638 tests green) but several load-bearing
concepts exist only as **smeared conditionals and copy-paste**, not as named abstractions:

1. **Conversation state strategy.** The stateful-cursor (OpenAI Responses) vs
   stateless-replay (OpenRouter/Claude) duality is implemented as six
   `self._provider.stateful` checks and four mutations of a `stateless_context_items`
   local threaded through `BaseAgent.run` / `run_loop` / the pause path
   (`core/agent/base.py:276-284, 331-334, 381-398, 436-440, 519-528`). The single most
   important behavior of the harness has no name and no unit tests of its own.
2. **Event emission.** `if event_handler is not None: event_handler(RuntimeEvent(...))`
   appears ~12× in `base.py` and 3× in `compaction.py`. The 14 event kinds are
   undocumented magic strings spread across emitters and three frontend switch chains.
   Presentation logic (inbox preview rendering, ~35 lines) lives inside the agent loop
   (`base.py:855-891`).
3. **Tool outcome handling.** `_execute_tool_calls` duplicates its success and error
   halves — both write an artifact, append a session message, build a
   `function_call_output`, and emit `tool_finished` (`base.py:658-714`).
4. **Subprocess plumbing.** Spawn + pipe-drainer + terminate→kill escalation logic is
   copied three ways: `tools/spawn_agent_tool.py:348-518`,
   `core/leads/supervisor.py:629-653, 675-758`, and
   `runtime/root.py:801-847`. The supervisor copy has a real robustness gap: its stderr
   drainer has **no byte cap and no await-timeout** (the reaper path bounds both).
5. **Provider utility duplication.** `_harden_strict_schema` is duplicated verbatim 3×
   (~115 lines: `openai_provider.py:52-97`, `openrouter_translator.py:621-658`,
   `claude_translator.py:645-682`). Exponential-backoff + retry-header parsing is
   duplicated 2× (`openrouter_provider.py:237-247`, `claude_provider.py:181-209`).
6. **Provider dispatch.** Four separate `if name == "openai"/"openrouter"/"claude"`
   switches: `core/agent/factory.py:_resolve_provider`, `_resolve_model_name`,
   `_supports_multimodal_attachments`, and `runtime/provider_factory.py:_build_default_provider`.
   Adding a provider today means finding all four.
7. **Store lifecycle boilerplate.** Six SQLite stores repeat the identical
   `initialize()/close()/_require_connection()` lifecycle (session, cron, agent_message,
   task, lead_session, worker_heartbeat).
8. **Shutdown seam violation.** `FeatherRuntime.shutdown` reaches into
   `_memory_stack.service._store._client` private attributes to close the Qdrant client
   (`runtime/root.py:775-782`) because `MemoryStack.aclose()` only closes owned providers.

## Approaches considered

- **A. Conservative in-place cleanup** (extract private methods, add comments). Low risk,
  but leaves every concept unnamed and untestable in isolation; duplication remains.
- **B. Targeted concept-extraction refactor (chosen).** Name the hidden concepts as small
  classes/modules with byte-identical behavior; kill the cross-cutting duplication with
  shared utilities; preserve every public seam. Each workstream is independently
  testable and revertible.
- **C. Full redesign** (typed event union, chat-completions provider base class, pydantic
  tool schemas, tui/app split). Highest abstraction ceiling, but a giant regression
  surface across 1638 tests plus cross-process wire contracts — not *required* for a
  clean design, and it would violate the red-team bar of "no regression" confidence.
  Staged as follow-ups instead (§ Follow-ups).

## Design (Approach B) — nine workstreams

### WS1 — `ConversationContext` (core/agent/conversation.py, new)

The strategy object that owns *how prior conversation reaches the provider*:

```python
class ConversationContext(ABC):
    async def initial_input_items(session, new_items) -> list      # run() entry
    async def begin(input_items) -> None                           # run_loop() entry
    def provider_request(session, input_items) -> (items, cursor)  # per iteration
    def record_turn(sent_items, turn) -> None                      # after provider call
    def pause_payload(tool_outputs) -> list                        # AWAITING_USER persist
```

- `StatefulConversation`: cursor = `session.last_response_id`; replays history in
  `initial_input_items` only when the cursor is None (first turn / post-compaction);
  `record_turn` is a no-op; pause persists only the tool outputs.
- `StatelessConversation`: holds the in-run structural transcript; replays history unless
  the pending inputs already carry structural context (today's
  `_has_stateless_pending_context`); seeds from replay in `begin` when the run starts
  with no items (`resume_on_inbox`); cursor is always None; `record_turn` folds the
  sent items + the model turn back into the transcript; pause persists transcript + outputs.

`_model_turn_input_items` and `_has_stateless_pending_context` move here as module
functions. `BaseAgent.run` builds the context and passes it to `run_loop` via a new
optional keyword (`context: ConversationContext | None = None`, default builds one) so
`resume_on_inbox` and direct `run_loop` callers are unchanged. Item ordering
(`session.pending_inputs` → history replay → new items), the copy semantics of
`effective_input_items`, and the store-update-before-record ordering are preserved exactly.

**Classification:** modifying existing logic — the smeared branches are fully replaced;
no surviving path may still consult `provider.stateful` inside `run`/`run_loop`.

### WS2 — `EventKind` + `EventEmitter` (models/runtime_models.py + core/agent/events.py, new)

- `EventKind(str, Enum)` in `models/runtime_models.py`: the 14 public kinds
  (`assistant_text_delta`, `tool_started`, `tool_finished`, `awaiting_user`,
  `user_message_injected`, `agent_message_received`, `usage_updated`,
  `compaction_started/finished/failed`, `scheduled_task_triggered/failed`,
  `completion_guard_injected`, plus `scheduled_task_completed` if emitted — verify).
  `str`-enum so every existing `event.kind == "literal"` comparison, the IPC codec, and
  JSON serialization keep working unchanged. IPC control kinds (`_run_complete` etc.)
  stay where they are — they are wire-internal, not part of the public taxonomy.
- `EventEmitter` in `core/agent/events.py`: wraps `EventHandler | None`;
  `emit(kind, *, text=None, tool_name=None, payload=None)` no-ops on None handler;
  `.handler` property passes the raw handler to seams that still take it
  (provider.complete, compactor). No try/except added — a raising handler propagates
  exactly as today.
- `base.py` and `compaction.py` adopt the emitter; the inbox-preview block becomes a
  pure module function `_inbox_received_event(sender_agent, sender_session, messages)
  -> RuntimeEvent` unit-tested directly. Frontend switch chains are NOT rewritten
  (string equality is unaffected).

### WS3 — Tool outcome normalization (base.py)

`_execute_tool_calls` normalizes `(tool_call, result, exc)` to
`(output_text, question, loaded_skill)` first, then runs one shared
artifact-write → output-append → message-persist → event-emit path. Skill-append stays
before the artifact write; `result is None` short-circuit stays; error text stays
`f"Tool `{name}` failed: {exc}"`.

### WS4 — Shared subprocess plumbing (core/ipc/process.py, new)

```python
async def drain_stream(stream, buffer, *, max_bytes: int | None = None) -> None
async def spawn_piped_process(argv, *, cwd, env, stdin, capture_stdout=True) -> PipedProcess
async def terminate_process(process, *, term_timeout=2.0, kill_timeout=2.0) -> None
async def cancel_drainers(tasks) -> None
@dataclass(slots=True) class PipedProcess: process, stdout_buffer, stderr_buffer, drainers
```

Consumers:
- `spawn_agent_tool.launch_subagent_process` builds `LaunchedSubagent` from
  `spawn_piped_process(...)`; `_drain_stream`/`_terminate_launched` bodies delegate.
- `supervisor._SubprocessWorkerHandle` uses shared `drain_stream` with a **1 MiB stderr
  cap** and `wait()` awaits the drainer with a **2 s timeout** (intentional hardening —
  mirrors the reaper's bounds; the only behavior change in this refactor, tested
  explicitly). The supervisor spawn goes through `spawn_piped_process(capture_stdout=False)`
  (stdout must stay undrained — it is read line-by-line for events). The handle-level
  SIGTERM→SIGKILL shutdown protocol is untouched (it operates on the `WorkerHandle`
  abstraction, which fakes implement in tests).
- `runtime/root._terminate_live_subagents` delegates its inline escalation to
  `terminate_process` + `cancel_drainers`, keeping its logging and task-finalize behavior.

### WS5 — Provider shared utilities (providers/schema_utils.py + providers/retry_utils.py, new)

- `harden_strict_schema(schema) -> None`: the single implementation; the three copies
  become imports. (`setdefault` vs explicit-check variants are semantically identical.)
- `retry_utils`: `backoff_delay(attempt, base_delay)` (exp + full jitter),
  `seconds_until_unix_timestamp(header, max_wait)`,
  `seconds_from_retry_after(header, max_wait)`. The provider-local
  `_retry_sleep_seconds*` functions keep their names/signatures (tests reference them)
  and become thin compositions of the shared core.
- Explicitly NOT unifying: SSE parsers, the retry *loop* structure, translators, or a
  chat-completions base class (see Follow-ups).

### WS6 — Provider catalog (providers/catalog.py, new)

A frozen `ProviderSpec` per provider (`name`, `build(app_config)`,
`config_block(app_config)`, `default_model(app_config)`,
`supports_multimodal(app_config)`) in one table. `factory._resolve_provider`,
`_resolve_model_name`, `_supports_multimodal_attachments`, and
`provider_factory._build_default_provider` all consult it. Error-message text for
missing config blocks / unknown providers is preserved exactly (messages are asserted
in tests).

### WS7 — SQLite store base (storage/base.py, new)

`BaseSQLiteStore` owning `__init__(db_path)`, `initialize()` (open via
`open_store_connection` → `initialize_database_schema` → commit), `close()`, and
`_require_connection()`. A class-level knob covers the `LeadSessionStore` foreign-keys
exception. Adopted by the six SQLite stores; per-store extra constructor params
(e.g. `AgentMessageStore.inbox_cap`) call `super().__init__`. Per-store
`_require_connection` error strings are preserved via `type(self).__name__`
(verify each current message during implementation; match exactly).
`MessagingStore` (per-call connections, FK-off by design), `ToolOutputStore`,
`AttachmentStore`, `UserProfileStore` are intentionally untouched.

### WS9 — Memory shutdown seam (memory/ + runtime/root.py)

Give the vector store / service a proper async close (`MemoryService.aclose()` closing
its store's client; store gains `aclose()` if absent), call it from
`MemoryStack.aclose()`, and delete the private-attribute spelunking block in
`FeatherRuntime.shutdown`. Net behavior identical: the same Qdrant client gets closed,
through a seam instead of `getattr` chains.

*(WS8 was considered — frontend event-dispatch dedup — and cut; see Follow-ups.)*

## Invariants that must hold (red-team checklist seeds)

- Stateless pause/resume round-trip: pending_inputs persisted on AWAITING_USER must
  contain transcript + outputs; `run()` must not double-replay when resuming
  (`_has_stateless_pending_context` semantics).
- Stateful post-compaction: cursor None → exactly one history replay, ordered
  pending → history → new.
- Inbox semantics unchanged: one sender-group per iteration, oldest-first fairness,
  DELIVERED flip, keep-alive cap accounting (top-of-loop inbox bumps the cap; queue
  drain does not).
- Event wire contract: every emitted kind serializes to the identical string; the
  worker→supervisor codec and the three frontends see byte-identical events.
- Reaper-vs-terminate single-delivery claim, refcounted session locks, additive-only
  schema, sequence allocation in one INSERT…SELECT, busy_timeout via shared openers —
  all untouched by design; verify by reading final diffs against each.
- Tool execute() bodies still never block the loop; `spawn_piped_process` is async-only.
- Supervisor stdout is never drained by the shared spawner (events would be lost).

## Testing

Each workstream lands with unit tests next to the existing suites
(`tests/test_conversation_context.py`, `tests/test_agent_events.py`,
`tests/test_ipc_process.py`, `tests/test_provider_schema_utils.py`,
`tests/test_provider_retry_utils.py`, `tests/test_provider_catalog.py`,
`tests/test_storage_base.py`, memory aclose cases in `tests/memory/test_runtime.py`),
covering happy path + failure/edge cases (empty inputs, pause/resume, cap overflow,
kill escalation, unknown provider, double-initialize). The existing 1638-test suite is
the regression net and must stay green after every workstream.

## Follow-ups (explicitly out of scope, in rough priority order)

1. Split `tui/app.py` (2692 lines, 83 methods) — slash-command handlers → module;
   already documented deferred debt.
2. Shared frontend event dispatcher (three ~100-line switch chains in
   `tui/app.py` / `tui/__init__.py` / `cli/__init__.py`) and a consumer for the
   currently-unconsumed `completion_guard_injected` kind.
3. Pydantic-derived tool parameter schemas (23 tools, mechanical but huge diff).
4. Chat-completions base provider / SSE parser base for OpenRouter+Claude.
5. Split `tools/task_tools.py` (1016 lines).
6. Shared subprocess entry-point bootstrap (`lead_worker_entry` vs `subagent_entry`).
7. Doc drift: `docs/configuration.md` memory `operations.*` grouping;
   prompt-caching strategy section in `docs/providers.md`.
8. `core/ipc/process.py:cancel_drainers` — found in review: `suppress(BaseException)`
   around `await drainer` masks the *caller's own* CancelledError (empirically
   reproduced; byte-equivalent to the pre-refactor per-site code, so matched
   rather than regressed). Now that the pattern is centralized, the helper could
   re-raise when the cancellation belongs to the calling task.
