# Task 7 — Shared subprocess spawn/drain/terminate (`core/ipc/process.py`)

**Date:** 2026-06-12  **Branch:** `harness-elegance`

## Classification

**Modifying existing logic** (consolidating three byte-equivalent copies into
one canonical module) **+ ONE deliberate hardening**:

1. Supervisor stderr drainer gains a **1 MiB cap** (was unbounded).
2. Supervisor `wait()` awaits each drainer with a **2.0s timeout** (was an
   unbounded `await` that could hang `wait()` forever).

Everything else must be byte-equivalent to the originals.

A third, unavoidable cosmetic delta: drainer **task-name format** changes from
`subagent-stdout-{sid}` to `subagent-{sid}-stdout` (and the supervisor's
`supervisor.stderr_drain` → `lead-worker-stderr`). Verified no test or
production code depends on these names (grep clean).

## The three originals (semantic comparison — Step 1)

| Concern | spawn_agent_tool | supervisor | root.py |
| --- | --- | --- | --- |
| spawn stdin | `DEVNULL` | `PIPE` | n/a (re-uses existing proc) |
| stdout drained | yes (into buffer) | **no** (line-read via `read_event_line`) | n/a |
| stderr drained | yes (uncapped) | yes (uncapped, own task) | n/a |
| terminate | term→2s→kill→2s | (via handle, in shutdown()) | term→2s→kill→2s + SIGKILL warn |
| drainer cancel | cancel+await each, swallow | (handle.wait awaits drainer) | cancel+await each, swallow |

Known nuances to preserve (from task brief):
- (a) spawn_agent_tool `_terminate_launched` proceeds to `wait()` even when
  `terminate()` raised `ProcessLookupError`. The helper's
  `suppress(ProcessLookupError)` + `wait_for(wait())` matches: `wait()` returns
  immediately for an already-reaped child. ✔ semantically equivalent.
- (b) root.py distinguishes `killed` (it issued terminate) and logs a SIGKILL
  warning if the post-kill wait also timed out. This bookkeeping STAYS at the
  root.py call site via `returncode` checks around `terminate_process`.
- (c) the 2.0s timeouts are the helper's defaults.

## Plan

### Step 1 — `src/feather/core/ipc/process.py` (NEW)
Exactly the module from the brief: `drain_stream`, `PipedProcess`,
`spawn_piped_process`, `terminate_process`, `cancel_drainers`. `__all__` tuple.

### Step 2 — `spawn_agent_tool.py`
- Import `cancel_drainers`, `spawn_piped_process`, `terminate_process` from
  `feather.core.ipc.process`.
- `launch_subagent_process`: keep `_write_task_file` + the `_remove_task_file`
  on-spawn-failure try/except. Spawn via `spawn_piped_process(argv,
  cwd=str(root), env=subprocess_env, stdin=DEVNULL, name=f"subagent-{sid}")`.
  Build `LaunchedSubagent(process=piped.process, stdout_buffer=…, stderr_buffer=…,
  drainers=piped.drainers)`.
- DELETE `_drain_stream`.
- `_terminate_launched`: body becomes
  `await terminate_process(launched.process)` +
  `await cancel_drainers(launched.drainers)`.

### Step 3 — `supervisor.py`
- Module constant `_STDERR_MAX_BYTES = 1024 * 1024` with a why-comment.
- Import `PipedProcess`, `spawn_piped_process` from ipc.process.
- `_default_subprocess_factory`: spawn via `spawn_piped_process(argv,
  cwd=str(self._project_root), env=subprocess_env_with_home(), stdin=PIPE,
  capture_stdout=False, stderr_max_bytes=_STDERR_MAX_BYTES, name="lead-worker")`;
  pass the `PipedProcess` into `_SubprocessWorkerHandle`.
- `_SubprocessWorkerHandle.__init__(self, piped: PipedProcess)`: store `piped`;
  DELETE its own `_drain_stderr` + task creation. `pid`/`returncode` delegate to
  `piped.process`. `stderr_buffer` returns `bytes(piped.stderr_buffer)`.
  `send_command`/`read_event_line`/`close_stdin`/`terminate`/`kill` operate on
  `piped.process` (PRESERVE the no-await invariant comment verbatim).
  `wait()` awaits `piped.process.wait()` then awaits each `piped.drainers`
  task under `contextlib.suppress(Exception)` with
  `asyncio.wait_for(..., timeout=2.0)` (the hardening).
- `signal` import: still used by `terminate()` (`signal.SIGTERM`). Keep.

### Step 4 — `runtime/root.py` `_terminate_live_subagents`
- Import `cancel_drainers`, `terminate_process` (top-level, alongside other
  `feather.core...` imports).
- Per entry: `was_live = proc.returncode is None`; if `was_live:
  await terminate_process(proc)`. `killed = was_live` (matches HEAD: killed is
  set the moment terminate is attempted). If `proc.returncode is None` after →
  the existing SIGKILL warning log.
- `await self._finalize_live_subagent_on_shutdown(entry, killed=killed)` (unchanged).
- Second loop: replace drainer cancel/await body with
  `await cancel_drainers(tuple(entry.drainers))`; keep `registry.remove`.
- DELETE `import asyncio as _asyncio` (nothing else uses it after the rewrite).

### Step 5 — `tests/test_ipc_process.py` (NEW)
Eight tests per brief, `sys.executable -c '...'` children, each < 5s.

## Contract / behavioral deltas (must end up exactly this list)
1. Supervisor stderr buffer capped at 1 MiB.
2. Supervisor `wait()` drainer await now has a 2.0s timeout.
3. Drainer task-name format: `subagent-{sid}-stdout|-stderr`,
   `lead-worker-stderr` (cosmetic).

## Touchpoints to re-verify in red-team (Step 7)
- `WorkerHandle` Protocol body unchanged.
- Reaper reads `entry.process/.stdout_buffer/.stderr_buffer/.drainers` unchanged.
- `LaunchedSubagent` dataclass shape unchanged (consumed by `SpawnAgentTool`
  + `LiveSubagent`).
- `_FakeProc` in test_runtime_shutdown_tasks: `terminate()` sets returncode
  non-None → `terminate_process` returns after first wait → `killed=True` →
  STOPPED/KILLED preserved.
