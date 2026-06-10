# Harness Hardening & Quality Pass — Design

**Date:** 2026-06-09 · **Branch:** `improve`

## Goal

Burn down the documented repo landmines (CLAUDE.md "Repo-Specific Landmines")
and the quality polish deferred from the 2026-06-05 src refactor: concurrency
safety, DB contention, event-loop blocking, task lifecycle, plus docstring and
style debt. No new features; no behavior change visible to users except the
removal of failure modes.

## Scope (8 work items)

### 1. Shared SQLite connection helper + busy_timeout everywhere

**Problem.** Six stores hand-roll the same `initialize()` boilerplate
(mkdir → connect → row_factory → pragmas → schema → commit), and the pragma
set drifted: `SessionStore` (`storage/session_store.py:35-44`) and
`CronJobStore` (`storage/cron_store.py:26-35`) omit `PRAGMA busy_timeout`,
so any cross-process write contention hard-fails with `database is locked`
instead of backing off. The other four stores set `busy_timeout=5000`.

**Approaches considered.**
- **(A — chosen)** Extract `storage/connection.py:open_store_connection(db_path, *,
  foreign_keys=True, busy_timeout_ms=5000) -> aiosqlite.Connection` that does
  mkdir + connect + row_factory + pragmas. Each store's `initialize()` calls it
  then runs its own schema init. Kills the duplication and the gaps in one move;
  per-store flags (`LeadSessionStore` sets no FK today) preserved via kwargs.
- (B) A `BaseSqliteStore` ABC — rejected: stores differ in schema-init and
  lifecycle details; a base class is more coupling than the seam needs.

### 2. MessagingStore per-call connections get pragmas

**Problem.** `messaging/store.py` opens a *fresh* connection inside every
method with **zero pragmas** — `busy_timeout=0`, no `foreign_keys`. Under any
contention it hard-fails.

**Approaches considered.**
- **(A — chosen)** Keep the per-call connection design (it is cross-process
  tolerant and the regression surface of changing it is the whole messaging
  router), but route every open through one private async contextmanager
  `_connect()` that applies `busy_timeout`. `initialize()` additionally sets
  `journal_mode=WAL` once (WAL is a persistent DB-file property).
  **Implementation amendment:** `foreign_keys` stays OFF — the existing test
  helper documents that chat mappings are intentionally written without a
  matching session row, so enabling FK enforcement would be a behavior
  change, not hardening.
- (B) Convert to the house single-long-lived-connection pattern — deferred:
  correct end-state per CLAUDE.md's store rule, but a lifecycle change touching
  every messaging caller; out of scope for a hardening pass.

### 3. Atomic message-sequence allocation

**Problem.** `session_store.add_message` (`session_store.py:141-156`) computes
`COALESCE(MAX(sequence),0)+1` in one round trip and INSERTs in another, with no
write transaction spanning both and no unique index on
`messages(session_id, sequence)`. Two writers (e.g. lead process + subprocess
touching the same session) can mint the same sequence.

**Approaches considered.**
- **(A — chosen)** Collapse to a single statement:
  `INSERT INTO messages (…) SELECT ?, ?, …, COALESCE(MAX(sequence),0)+1, ?
  FROM messages WHERE session_id = ?` — an aggregate with no GROUP BY always
  yields exactly one row, and a single statement executes atomically under
  SQLite's single-writer lock (WAL writers serialize; with busy_timeout from
  item 1 the second writer waits and then reads the committed row). Read the
  allocated sequence back by primary key (`SELECT sequence FROM messages WHERE
  id = ?` — one indexed lookup; avoids a `RETURNING` floor of SQLite ≥3.35).
- (B) `CREATE UNIQUE INDEX` + retry loop — rejected: existing DBs may already
  contain duplicate pairs from past races, so the migration can fail, and the
  schema is additive-only with no backfill mechanism.
- (C) `BEGIN IMMEDIATE` around SELECT+INSERT — rejected: holds the write lock
  across two round trips, against the perf checklist.

### 4. Offload blocking file I/O in tools

**Problem.** Under `asyncio.gather` tool fan-out (`core/agent/base.py:652`),
three tools block the event loop:
- `grep_tool.py:65,71` — sync `rglob` walk + `read_text` per file.
- `read_file_tool.py:125` — sync `read_text`.
- `write_file_tool.py:131-158` — sync stat/mkdir/write/**fsync**/replace
  (fsync is a disk barrier — worst offender).

**Fix.** Wrap each tool's synchronous body in a single `asyncio.to_thread`
call, matching the existing house pattern in `pdf_tool.py:65-70`. One offload
per execute (not per file) — the thread does the loop; results and error
mapping unchanged. `spawn_agent_tool`'s mkdir/mkstemp are micro-ops on a cold
path; out of scope.

### 5. Track inline-mode memory-trigger tasks

**Problem.** `memory/trigger.py:99` — in `background=False` (inline/test) mode
the task is created untracked, so `drain()`/`cancel_all()` can't see it.

**Fix.** Track all tasks identically (add to `self._tasks` +
`add_done_callback`). The `background` branch then disappears entirely —
both modes were already `create_task`; tracking was the only difference.
Existing tests observe via `await asyncio.sleep(0)`, unaffected.

### 6. Evict per-session locks in SessionRunCoordinator

**Problem.** `core/session/coordinator.py:15` — `defaultdict[str,
asyncio.Lock]` grows one entry per session forever.

**Invariant to preserve.** All concurrent users of one session_id must share
the *same* lock object, or mutual exclusion silently breaks.

**Approaches considered.**
- **(A — chosen)** Refcount eviction inside `acquire()`: increment a per-key
  count before awaiting the lock, decrement in `finally`, delete the lock +
  count entries when the count reaches zero. A waiter increments before
  awaiting, so an entry is never deleted while anyone holds *or waits*; a
  later acquire minting a fresh lock object is safe because nobody references
  the old one. Single event loop ⇒ no preemption between get-and-increment.
  `is_busy()` unchanged: missing entry ⇒ False, which is correct.
- (B) Session-close lifecycle hook — rejected: no "session done" event exists;
  would invent new API across runtime/router.
- (C) `WeakValueDictionary` — rejected: callers don't hold lock refs between
  acquires; GC timing nondeterminism.

### 7. Supervisor subprocess HOME parity

**Problem.** `tools/spawn_agent_tool.py:392-399` gives sub-agent subprocesses
a HOME fallback (so `~/.feather` resolves in service contexts);
`core/leads/supervisor.py:620-647` spawns lead workers with **no env handling**
— a latent parity bug.

**Fix.** Extract the env-with-HOME-fallback logic into one shared helper and
use it from both spawn sites. Placement decided at implementation (a small
shared module both `tools/` and `core/leads/` can import without cycles).

### 8. Docstring + typing polish (deferred from the src refactor)

- Minimal *why*-focused docstrings on the worst offenders found by survey:
  `core/leads/supervisor.py` (protocol methods/properties),
  `core/leads/manager.py` (`run`, `resume_on_inbox`, `enqueue_user_input`,
  `shutdown`), `api/routes.py` handlers, `api/hub.py`,
  `tools/task_tools.py` / `tools/cron_tools.py` `execute()`s,
  `core/agent/catalog.py` (`is_lead`, `dispatchable`),
  `core/agent/base.py` (`run_one`), `memory/trigger.py` / `memory/reader.py`
  interface methods.
- Modern-generics fixes: drop `typing.Optional` in `config/app_paths.py`,
  `typing.Sequence`/`Union` in `skills/catalog.py`, `typing.AsyncIterator` in
  `core/session/coordinator.py` (→ `collections.abc` / `X | None`).
- No other style debt found (zero f-string logging, zero mutable defaults).

## Explicitly out of scope

- Converting MessagingStore to a persistent connection (item 2B).
- Subprocess-spawn/stream-drain consolidation beyond the env helper (item 7).
- Event-sink abstraction across worker/supervisor.
- UNIQUE constraint on `messages(session_id, sequence)` (blocked by
  additive-only schema).
- TUI/app split (separately deferred from the src refactor).

## Logic classification (per CLAUDE.md stage 2)

- Items 1, 2, 4: **following existing logic** (the pragma set, the
  `to_thread` pattern, and per-call connections all have in-repo precedents to
  match exactly).
- Items 3, 5, 6: **modifying existing logic** (sequence allocation strategy,
  task tracking, lock lifetime) — old paths must not survive.
- Item 7: **adding new logic** (shared helper) + following (env fallback
  semantics copied from spawn_agent_tool).
- Item 8: no logic change.

## Test plan

All tests go next to existing ones (`tests/test_<module>.py`), hermetic, no
live network. Each must fail before its fix and pass after.

1. `tests/test_session_store.py` + `tests/test_cron_store.py`: assert
   `PRAGMA busy_timeout` > 0 on the initialized connection (fails today).
2. `tests/messaging/test_store.py`: a store method's connection reports
   `busy_timeout` > 0 (fails today).
3. `tests/test_session_store.py`: two `SessionStore` instances on the same
   `db_path` (two connections, as in cross-process use) `gather` N
   `add_message` calls to one session → all sequences unique and 1..N
   (flaky-fails today; deterministic-passes after). Happy path: sequences
   still monotonic via single store.
4. `tests/test_grep_tool.py` / `test_read_file_tool.py` /
   `test_write_file_tool.py`: monkeypatch the underlying sync call to
   `time.sleep(0.2)`; run the tool concurrently with a coroutine that sets an
   event; assert the event fires while the tool is in flight (loop not
   blocked). Fails today, passes with `to_thread`. Happy-path output tests
   unchanged.
5. `tests/memory/test_trigger.py`: inline-mode task appears in the tracked
   set and is awaited by `drain()` (fails today). All 11 existing tests stay
   green.
6. New `tests/test_session_coordinator.py`: (a) entry evicted after release;
   (b) two concurrent acquires serialize (same lock) and entry survives while
   a waiter exists; (c) `is_busy` true while held, false after; (d) re-acquire
   after eviction works.
7. Env-helper test: HOME missing from base env → helper injects fallback;
   HOME present → untouched; supervisor spawn passes an env built by the
   helper (unit-test the helper + a spawn-call assertion with a fake).
8. Full suite `uv run pytest` green (known flaky:
   `test_request_input_waits_for_correlated_reply` — re-run once if it fails).

## Risks

- **Item 3** changes a hot-path SQL shape; the read-back adds one PK lookup
  per message. Mitigation: covered by existing session-store tests plus the
  new concurrency test; per-message cost is microseconds.
- **Item 6** touches the mutual-exclusion primitive under every agent run.
  Mitigation: invariant-focused tests (6b) and the red-team review trace
  through `BaseAgent.run` / `resume_on_inbox` / router `is_busy`.
- **Item 4** makes tool bodies run in threads: the bodies touch only local
  variables and the filesystem (no shared mutable state), and `to_thread`
  results are awaited in place, so cancellation semantics match pdf_tool.
