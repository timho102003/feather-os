# Harness Hardening & Quality Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the documented repo landmines (busy_timeout gaps, non-atomic sequence allocation, event-loop-blocking tools, untracked trigger tasks, unbounded lock dict, supervisor HOME parity) plus the docstring/typing polish deferred from the src refactor.

**Architecture:** Eight independent, individually-testable changes per the spec at `docs/superpowers/specs/2026-06-09-harness-hardening-design.md`. Items 1–2 consolidate SQLite connection bootstrap; item 3 makes sequence allocation a single atomic statement; item 4 offloads sync I/O with `asyncio.to_thread` (house pattern: `pdf_tool.py:65`); items 5–6 fix task/lock lifecycle; item 7 extracts a shared subprocess-env helper; item 8 is doc/typing polish.

**Tech Stack:** Python 3.12, asyncio, aiosqlite, pytest (`asyncio_mode=auto` — never add `@pytest.mark.asyncio`). Run tests with `uv run pytest …`. No linter/formatter exists — match surrounding style by hand.

**House rules that bind every task:** `from __future__ import annotations` first; modern generics only; full annotations incl. `-> None`; minimal why-docstrings; `%`-style lazy logging; tests are hermetic (autouse conftest fixture sets `FEATHER_HOME`).

---

### Task 1: Shared store-connection helper + busy_timeout for SessionStore/CronJobStore

**Files:**
- Create: `src/feather/storage/connection.py`
- Modify: `src/feather/storage/session_store.py:35-44`, `src/feather/storage/cron_store.py:26-35`, `src/feather/storage/agent_message_store.py:59-71`, `src/feather/storage/task_store.py:42-52`, `src/feather/storage/lead_session_store.py:31-40`, `src/feather/storage/worker_heartbeat_store.py:47-60`
- Test: `tests/test_storage_connection.py` (new), `tests/test_session_store.py`, `tests/test_cron_store.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_session_store.py` (match the file's existing fixture style — stores there are constructed with `tmp_path / "feather.db"`):

```python
async def test_session_store_connection_sets_busy_timeout(tmp_path):
    store = SessionStore(tmp_path / "feather.db")
    await store.initialize()
    try:
        cursor = await store._connection.execute("PRAGMA busy_timeout;")
        row = await cursor.fetchone()
        assert int(row[0]) == 5000
    finally:
        await store.close()
```

Append the same test (s/SessionStore/CronJobStore/, name `test_cron_store_connection_sets_busy_timeout`) to `tests/test_cron_store.py`.

Create `tests/test_storage_connection.py`:

```python
"""Tests for the shared store-connection bootstrap."""

from __future__ import annotations

from feather.storage.connection import open_store_connection


async def test_open_store_connection_applies_house_pragmas(tmp_path):
    connection = await open_store_connection(tmp_path / "sub" / "feather.db")
    try:
        for pragma, expected in (
            ("busy_timeout", 5000),
            ("foreign_keys", 1),
        ):
            cursor = await connection.execute(f"PRAGMA {pragma};")
            row = await cursor.fetchone()
            assert int(row[0]) == expected, pragma
        cursor = await connection.execute("PRAGMA journal_mode;")
        row = await cursor.fetchone()
        assert str(row[0]).lower() == "wal"
    finally:
        await connection.close()


async def test_open_store_connection_can_skip_foreign_keys(tmp_path):
    connection = await open_store_connection(
        tmp_path / "feather.db", foreign_keys=False
    )
    try:
        cursor = await connection.execute("PRAGMA foreign_keys;")
        row = await cursor.fetchone()
        assert int(row[0]) == 0
    finally:
        await connection.close()
```

- [ ] **Step 2: Run tests to verify the right ones fail**

Run: `uv run pytest tests/test_storage_connection.py tests/test_session_store.py::test_session_store_connection_sets_busy_timeout tests/test_cron_store.py::test_cron_store_connection_sets_busy_timeout -v`
Expected: storage_connection tests ERROR (module not found); busy_timeout tests FAIL (`busy_timeout` is 0).

- [ ] **Step 3: Create the helper**

`src/feather/storage/connection.py`:

```python
"""Shared aiosqlite connection bootstrap for feather stores."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

_DEFAULT_BUSY_TIMEOUT_MS = 5000

__all__ = ("open_store_connection",)


async def open_store_connection(
    db_path: Path,
    *,
    foreign_keys: bool = True,
    busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
) -> aiosqlite.Connection:
    """Open one long-lived store connection with the house pragma set.

    Every store opens its own connection to the same ``feather.db``,
    across multiple OS processes. WAL keeps readers from blocking the
    writer; ``busy_timeout`` makes a contended write back off for up to
    5s instead of hard-failing with ``database is locked``.
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(db_path)
    connection.row_factory = aiosqlite.Row
    if foreign_keys:
        await connection.execute("PRAGMA foreign_keys=ON;")
    await connection.execute("PRAGMA journal_mode=WAL;")
    await connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms};")
    return connection
```

- [ ] **Step 4: Adopt it in all six stores**

For each store, replace the body of `initialize()` between the mkdir and `initialize_database_schema` with a helper call, preserving each store's current flags exactly. Example for `session_store.py` (the other five follow the identical shape):

```python
    async def initialize(self) -> None:
        """Create the database and required tables if missing."""

        self._connection = await open_store_connection(self._db_path)
        await initialize_database_schema(self._connection)
        await self._connection.commit()
```

Per-store flags (read each `initialize()` first; preserve current behavior exactly):
- `session_store.py`, `cron_store.py`, `agent_message_store.py`, `task_store.py`, `worker_heartbeat_store.py`: defaults (`foreign_keys=True`).
- `lead_session_store.py`: currently sets **no** `foreign_keys` pragma → call with `foreign_keys=False` and keep its existing comment about why.

In each file: add `from feather.storage.connection import open_store_connection`, delete the now-dead `aiosqlite.connect`/pragma/mkdir lines (keep the `import aiosqlite` only if the file still references `aiosqlite.Row`/`aiosqlite.IntegrityError` elsewhere — check before removing). Keep any store-specific explanatory comments by moving them onto the call site only if they say something the helper docstring doesn't.

- [ ] **Step 5: Run the affected suites**

Run: `uv run pytest tests/test_storage_connection.py tests/test_session_store.py tests/test_cron_store.py tests/test_agent_message_store.py tests/test_task_store.py tests/test_lead_session_store.py tests/test_worker_heartbeat_store.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/feather/storage/ tests/test_storage_connection.py tests/test_session_store.py tests/test_cron_store.py
git commit -m "Consolidate store connection bootstrap; add missing busy_timeout"
```

---

### Task 2: MessagingStore per-call connections get pragmas

**Files:**
- Modify: `src/feather/messaging/store.py`
- Test: `tests/messaging/test_store.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/messaging/test_store.py` (reuse its existing store/tmp_path fixture style):

```python
async def test_initialize_sets_persistent_wal_journal_mode(tmp_path):
    db_path = tmp_path / "feather.db"
    store = MessagingStore(db_path)
    await store.initialize()
    async with aiosqlite.connect(db_path) as connection:
        cursor = await connection.execute("PRAGMA journal_mode;")
        row = await cursor.fetchone()
    assert str(row[0]).lower() == "wal"


async def test_per_call_connections_apply_house_pragmas(tmp_path):
    store = MessagingStore(tmp_path / "feather.db")
    await store.initialize()
    async with store._connect() as connection:
        for pragma, expected in (("busy_timeout", 5000), ("foreign_keys", 1)):
            cursor = await connection.execute(f"PRAGMA {pragma};")
            row = await cursor.fetchone()
            assert int(row[0]) == expected, pragma
```

(Add `import aiosqlite` to the test module if missing.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/messaging/test_store.py -v -k "journal_mode or house_pragmas"`
Expected: FAIL — journal_mode is `delete`; `_connect` does not exist.

- [ ] **Step 3: Implement `_connect` and adopt it**

In `src/feather/messaging/store.py` add imports `from collections.abc import AsyncIterator` and `from contextlib import asynccontextmanager`, a module constant, and the contextmanager on `MessagingStore`:

```python
_BUSY_TIMEOUT_MS = 5000
```

```python
    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Open one short-lived connection with the house pragma set.

        This store deliberately opens a connection per call — webhook
        handlers run in many tasks and there is no long-lived connection
        to contend over — but each connection still needs ``busy_timeout``
        and ``foreign_keys`` or a contended write hard-fails immediately.
        """

        async with aiosqlite.connect(self._db_path) as connection:
            connection.row_factory = aiosqlite.Row
            await connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS};")
            await connection.execute("PRAGMA foreign_keys=ON;")
            yield connection
```

Then mechanically replace **every** `async with aiosqlite.connect(self._db_path) as connection:` in the class (11 sites: initialize, save/load/list/delete_credentials, get/upsert_chat_mapping, count_chats_for_platform, claim_inbound, release_inbound, prune_inbound_older_than) with `async with self._connect() as connection:` and delete the now-redundant per-method `connection.row_factory = aiosqlite.Row` lines. `initialize()` becomes:

```python
    async def initialize(self) -> None:
        """Run schema migrations and set the persistent WAL journal mode."""

        async with self._connect() as connection:
            await connection.execute("PRAGMA journal_mode=WAL;")
            await initialize_database_schema(connection)
            await connection.commit()
```

Do not touch the `aiosqlite.IntegrityError` handling in `claim_inbound` or any SQL.

- [ ] **Step 4: Run the messaging suites**

Run: `uv run pytest tests/messaging/ -v`
Expected: all PASS (the dedup/credentials behavior tests prove no regression).

- [ ] **Step 5: Commit**

```bash
git add src/feather/messaging/store.py tests/messaging/test_store.py
git commit -m "Apply house SQLite pragmas to MessagingStore per-call connections"
```

---

### Task 3: Atomic message-sequence allocation in `add_message`

**Files:**
- Modify: `src/feather/storage/session_store.py:120-166`
- Test: `tests/test_session_store.py`

- [ ] **Step 1: Write the failing concurrency test**

Append to `tests/test_session_store.py`:

```python
async def test_add_message_sequences_unique_across_connections(tmp_path):
    """Two connections to one db (the cross-process shape) must never mint
    the same sequence — the old SELECT-MAX-then-INSERT raced here."""

    db_path = tmp_path / "feather.db"
    store_a = SessionStore(db_path)
    store_b = SessionStore(db_path)
    await store_a.initialize()
    await store_b.initialize()
    try:
        session = await store_a.create_session("lead")

        async def add(store: SessionStore, index: int) -> SessionMessage:
            return await store.add_message(
                session.id, MessageRole.USER, f"message-{index}"
            )

        results = await asyncio.gather(
            *(add(store_a if i % 2 == 0 else store_b, i) for i in range(20))
        )
        sequences = sorted(message.sequence for message in results)
        assert sequences == list(range(1, 21))
    finally:
        await store_a.close()
        await store_b.close()
```

Add any missing imports at the top of the test file (`asyncio`, `SessionMessage` from `feather.models`).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_session_store.py::test_add_message_sequences_unique_across_connections -v`
Expected: FAIL — duplicate sequences (assert mismatch). If it flakily passes, re-run; the interleaving across 20 alternating writes makes a pre-fix pass very unlikely.

- [ ] **Step 3: Collapse allocation+insert into one statement**

In `add_message` (`session_store.py:141-156`), replace the SELECT-MAX block and INSERT with:

```python
        message_id = str(uuid4())
        now = _utc_now()
        # Allocate the sequence inside the INSERT itself: a single
        # statement executes atomically under SQLite's write lock, so two
        # connections (or processes) can never read the same MAX. An
        # aggregate with no GROUP BY always yields exactly one row.
        await self._execute(
            """
            INSERT INTO messages (id, session_id, role, content, file_ref, is_compact, sequence, created_at)
            SELECT ?, ?, ?, ?, ?, ?, COALESCE(MAX(sequence), 0) + 1, ?
            FROM messages WHERE session_id = ?
            """,
            (
                message_id,
                session_id,
                role.value,
                content,
                file_ref,
                int(is_compact),
                now,
                session_id,
            ),
        )
        await self._touch_session(session_id)
        await self._connection.commit()
        sequence_row = await self._fetchone(
            "SELECT sequence FROM messages WHERE id = ?", (message_id,)
        )
        sequence = int(sequence_row["sequence"])
```

Keep the trailing `return SessionMessage(...)` exactly as is (it already uses `sequence`). The read-back by primary key avoids a `RETURNING` clause (SQLite ≥3.35 floor).

- [ ] **Step 4: Run the session-store suite**

Run: `uv run pytest tests/test_session_store.py -v`
Expected: all PASS, including the new concurrency test and every pre-existing ordering test (proves sequences still start at 1 and increment).

- [ ] **Step 5: Commit**

```bash
git add src/feather/storage/session_store.py tests/test_session_store.py
git commit -m "Make message sequence allocation atomic across connections"
```

---

### Task 4: Offload blocking file I/O in grep / read_file / write_file

**Files:**
- Modify: `src/feather/tools/grep_tool.py`, `src/feather/tools/read_file_tool.py:124-127`, `src/feather/tools/write_file_tool.py:100-182`
- Test: `tests/test_grep_tool.py`, `tests/test_read_file_tool.py`, `tests/test_write_file_tool.py`

- [ ] **Step 1: Write the failing loop-responsiveness tests**

Each test uses the same deterministic shape: pre-fix, `execute` contains **no awaits**, so once awaited it runs to completion before any other task; post-fix the `to_thread` await yields. Append to `tests/test_grep_tool.py` (mirror its existing `ToolExecutionContext` construction):

```python
import time
from pathlib import Path


async def test_grep_does_not_block_event_loop(tmp_path, monkeypatch):
    (tmp_path / "file.txt").write_text("needle here", encoding="utf-8")
    tool = GrepTool(tmp_path)
    original_read_text = Path.read_text

    def slow_read_text(self: Path, *args: object, **kwargs: object) -> str:
        time.sleep(0.2)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", slow_read_text)
    order: list[str] = []

    async def run_tool() -> None:
        await tool.execute(
            {"pattern": "needle", "path": None, "case_sensitive": None, "max_results": None},
            _context(),
        )
        order.append("tool")

    async def heartbeat() -> None:
        await asyncio.sleep(0.05)
        order.append("heartbeat")

    await asyncio.gather(run_tool(), heartbeat())
    assert order == ["heartbeat", "tool"]
```

(`_context()` = however the file already builds `ToolExecutionContext`; reuse it. Add `import asyncio` if missing.)

Add the analogous test to `tests/test_read_file_tool.py` (same `slow_read_text` monkeypatch; `execute({"path": "file.txt", "start_line": None, "end_line": None, "max_chars": None}, …)`), and to `tests/test_write_file_tool.py` monkeypatching `os.fsync`:

```python
    def slow_fsync(fd: int) -> None:
        time.sleep(0.2)

    monkeypatch.setattr(os, "fsync", slow_fsync)
```

with `execute({"path": "out.txt", "content": "hello", "overwrite": None, "create_parents": None}, …)`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_grep_tool.py tests/test_read_file_tool.py tests/test_write_file_tool.py -v -k "block"`
Expected: all three FAIL with `order == ["tool", "heartbeat"]`.

- [ ] **Step 3: Offload each tool body with `asyncio.to_thread`**

Match the house pattern (`pdf_tool.py:65-70`) — one offload per execute, the thread runs the loop.

`grep_tool.py`: add `import asyncio`; move the walk into a sync method and await it:

```python
        matches = await asyncio.to_thread(
            self._search_sync, search_root, regex, max_results
        )

        if not matches:
            return ToolExecutionResult(output="No matches found.")
        return ToolExecutionResult(output="\n".join(matches))

    def _search_sync(
        self, search_root: Path, regex: re.Pattern[str], max_results: int
    ) -> list[str]:
        """Walk and read files on a worker thread — rglob/read_text block."""

        matches: list[str] = []
        for path in sorted(search_root.rglob("*")):
            if len(matches) >= max_results:
                break
            if not path.is_file() or self._should_skip(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    relative = path.relative_to(self._workspace_root)
                    matches.append(f"{relative}:{line_number}: {line.strip()}")
                    if len(matches) >= max_results:
                        break
        return matches
```

`read_file_tool.py`: add `import asyncio`; replace line 125:

```python
        try:
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File is not valid UTF-8 text: {arguments['path']}") from exc
```

`write_file_tool.py`: add `import asyncio`; in `execute`, after the content checks and `path = self._resolve_path(raw_path)`, replace everything from `if path.is_dir():` through `os.replace(...)`'s try/except with:

```python
        encoded = content.encode("utf-8")
        existed = await asyncio.to_thread(
            self._write_sync, path, raw_path, encoded, overwrite, create_parents
        )
```

and add the sync helper (verbatim move of the old body — stat/mkdir/mkstemp/fsync/replace all block, so the whole sequence runs on a worker thread):

```python
    def _write_sync(
        self,
        path: Path,
        raw_path: str,
        encoded: bytes,
        overwrite: bool,
        create_parents: bool,
    ) -> bool:
        """Run the stat/mkdir/write/fsync/replace sequence on a worker thread.

        Returns:
            Whether the file existed before the write.
        """

        if path.is_dir():
            raise ValueError(f"Path is a directory, not a file: {raw_path}")

        existed = path.exists()
        if existed and not overwrite:
            raise ValueError(
                f"File already exists at {raw_path}. Set `overwrite=true` to replace."
            )

        parent = path.parent
        if not parent.exists():
            if not create_parents:
                raise ValueError(
                    f"Parent directory does not exist: {self._display_path(parent)}"
                )
            parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=parent
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    logger.exception(
                        "write_file.tmp_cleanup_failed path=%s tmp=%s",
                        path,
                        tmp_path,
                    )
            raise
        return existed
```

The trailing logging/return in `execute` keeps using `existed`/`encoded` unchanged.

- [ ] **Step 4: Run the three tool suites**

Run: `uv run pytest tests/test_grep_tool.py tests/test_read_file_tool.py tests/test_write_file_tool.py -v`
Expected: all PASS (sandbox/error-path tests prove the moved code still raises the same ValueErrors through `await`).

- [ ] **Step 5: Commit**

```bash
git add src/feather/tools/grep_tool.py src/feather/tools/read_file_tool.py src/feather/tools/write_file_tool.py tests/test_grep_tool.py tests/test_read_file_tool.py tests/test_write_file_tool.py
git commit -m "Offload blocking file I/O in grep/read_file/write_file to threads"
```

---

### Task 5: Track inline-mode memory-trigger tasks

**Files:**
- Modify: `src/feather/memory/trigger.py:94-105` (+ module docstring)
- Test: `tests/memory/test_trigger.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/memory/test_trigger.py`, reusing the file's existing fake-service fixture and config builder (inline mode = `background=False`):

```python
async def test_inline_mode_tasks_are_tracked_and_drained(trigger_factory):
    """Inline (background=False) tasks must be visible to drain() —
    previously they were created untracked (the documented landmine)."""

    trigger, service = trigger_factory(background=False)
    trigger.maybe_schedule("session-1", agent_model="model", owner=MemoryOwner.LEAD)
    assert len(trigger._tasks) == 1
    await trigger.drain(timeout_s=1.0)
    assert service.extract_calls == 1
```

Adapt fixture/attribute names to what the file actually defines (it has 11 existing tests with a fake `MemoryService`; reuse, don't reinvent). The load-bearing asserts are `len(trigger._tasks) == 1` before drain and the extraction having completed after drain.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/memory/test_trigger.py -v -k inline`
Expected: new test FAILS at `len(trigger._tasks) == 1` (set is empty); the existing inline test still PASSES.

- [ ] **Step 3: Remove the untracked branch**

In `maybe_schedule` (`trigger.py:94-105`), replace:

```python
        coro = self._run(session_id, agent_model, owner)
        if not self._cfg.background:
            # Inline mode (tests): still schedule on the loop so we don't
            # block the caller, but do not track for drain — tests await
            # asyncio.sleep(0) to observe the result.
            loop.create_task(coro)
            return
        task = loop.create_task(
            coro, name=f"memory-extract:{session_id[:8]}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
```

with:

```python
        # Both modes schedule a detached task; tracking is unconditional so
        # drain()/cancel_all() always see in-flight work. ``background``
        # remains accepted in config but no longer changes behavior.
        task = loop.create_task(
            self._run(session_id, agent_model, owner),
            name=f"memory-extract:{session_id[:8]}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
```

Update the module docstring's sentence about inline mode (lines 5-7) to say both modes fire tracked detached tasks.

- [ ] **Step 4: Run the memory suites**

Run: `uv run pytest tests/memory/ -v`
Expected: all PASS (the 11 existing trigger tests prove inline observation via `asyncio.sleep(0)` still works).

- [ ] **Step 5: Commit**

```bash
git add src/feather/memory/trigger.py tests/memory/test_trigger.py
git commit -m "Track inline-mode memory-trigger tasks for drain/cancel"
```

---

### Task 6: Refcounted lock eviction in SessionRunCoordinator

**Files:**
- Modify: `src/feather/core/session/coordinator.py` (whole file)
- Test: Create `tests/test_session_coordinator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_session_coordinator.py`:

```python
"""Lifecycle tests for the per-session run lock map."""

from __future__ import annotations

import asyncio

from feather.core.session.coordinator import SessionRunCoordinator


async def test_lock_entry_evicted_after_release():
    coordinator = SessionRunCoordinator()
    async with coordinator.acquire("s1"):
        assert coordinator.is_busy("s1")
    assert not coordinator.is_busy("s1")
    assert coordinator._locks == {}
    assert coordinator._refcounts == {}


async def test_concurrent_acquires_serialize_and_share_one_lock():
    coordinator = SessionRunCoordinator()
    events: list[str] = []
    first_inside = asyncio.Event()
    release_first = asyncio.Event()

    async def first() -> None:
        async with coordinator.acquire("s1"):
            events.append("first-in")
            first_inside.set()
            await release_first.wait()
        events.append("first-out")

    async def second() -> None:
        await first_inside.wait()
        async with coordinator.acquire("s1"):
            events.append("second-in")

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())
    await first_inside.wait()
    await asyncio.sleep(0)  # let second() block on the lock
    assert "s1" in coordinator._locks  # waiter keeps the entry alive
    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert events == ["first-in", "first-out", "second-in"]
    assert coordinator._locks == {}
    assert coordinator._refcounts == {}


async def test_cancelled_waiter_does_not_leak_entry():
    coordinator = SessionRunCoordinator()
    holder_inside = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with coordinator.acquire("s1"):
            holder_inside.set()
            await release_holder.wait()

    async def waiter() -> None:
        async with coordinator.acquire("s1"):
            pass

    holder_task = asyncio.create_task(holder())
    await holder_inside.wait()
    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    waiter_task.cancel()
    try:
        await waiter_task
    except asyncio.CancelledError:
        pass
    release_holder.set()
    await holder_task
    assert coordinator._locks == {}
    assert coordinator._refcounts == {}


async def test_reacquire_after_eviction_works():
    coordinator = SessionRunCoordinator()
    async with coordinator.acquire("s1"):
        pass
    async with coordinator.acquire("s1"):
        assert coordinator.is_busy("s1")
    assert not coordinator.is_busy("s1")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_session_coordinator.py -v`
Expected: FAIL — `_refcounts` doesn't exist and `_locks` retains entries.

- [ ] **Step 3: Rewrite the coordinator**

Replace `src/feather/core/session/coordinator.py` with:

```python
"""Per-session coordination for serialized agent execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class SessionRunCoordinator:
    """Provide one shared async lock per session ID.

    Entries are reference-counted and evicted when the last holder or
    waiter releases, so the map is bounded by *concurrently active*
    sessions instead of growing with every session ever seen.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._refcounts: dict[str, int] = {}

    @asynccontextmanager
    async def acquire(self, session_id: str) -> AsyncIterator[None]:
        """Serialize work for one session.

        The refcount is incremented before awaiting the lock, so an
        entry is never evicted while any task holds *or waits on* it;
        a fresh lock object after eviction is safe because nothing
        references the old one. Single event loop ⇒ no race between
        the lookup and the increment.
        """

        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        self._refcounts[session_id] = self._refcounts.get(session_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            remaining = self._refcounts[session_id] - 1
            if remaining:
                self._refcounts[session_id] = remaining
            else:
                del self._refcounts[session_id]
                del self._locks[session_id]

    def is_busy(self, session_id: str) -> bool:
        """Return True when a run is currently in flight for the session.

        Read-only inspection used by the messaging router to decide
        whether to spawn a new run or enqueue via
        :class:`UserInputQueue`. A missing entry means no holder and no
        waiter, hence not busy.
        """

        lock = self._locks.get(session_id)
        if lock is None:
            return False
        return lock.locked()
```

- [ ] **Step 4: Run coordinator + consumers**

Run: `uv run pytest tests/test_session_coordinator.py tests/test_base_agent.py tests/messaging/test_router.py tests/messaging/test_redteam_regressions.py -v`
Expected: all PASS (router `is_busy` semantics unchanged; agent serialization unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/feather/core/session/coordinator.py tests/test_session_coordinator.py
git commit -m "Evict per-session run locks when the last holder releases"
```

---

### Task 7: Shared subprocess-env helper + supervisor HOME parity

**Files:**
- Create: `src/feather/core/ipc/subprocess_env.py`
- Modify: `src/feather/tools/spawn_agent_tool.py:383-399`, `src/feather/core/leads/supervisor.py:624-647`
- Test: Create `tests/test_subprocess_env.py`; extend `tests/test_lead_supervisor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_subprocess_env.py`:

```python
"""Tests for the shared subprocess environment builder."""

from __future__ import annotations

from feather.core.ipc.subprocess_env import subprocess_env_with_home


def test_home_present_is_untouched(monkeypatch):
    monkeypatch.setenv("HOME", "/tmp/somewhere")
    env = subprocess_env_with_home()
    assert env["HOME"] == "/tmp/somewhere"


def test_missing_home_is_rederived(monkeypatch):
    monkeypatch.delenv("HOME", raising=False)
    env = subprocess_env_with_home()
    assert env.get("HOME")  # pwd-derived on POSIX


def test_empty_home_is_rederived(monkeypatch):
    monkeypatch.setenv("HOME", "")
    env = subprocess_env_with_home()
    assert env.get("HOME")
```

Append to `tests/test_lead_supervisor.py` (reuse the file's existing supervisor construction; adapt fixture names to what's there):

```python
async def test_default_subprocess_factory_passes_home_env(monkeypatch, ...):
    captured: dict[str, Any] = {}

    class _FakeProcess:
        pid = 4242
        stdin = None
        stdout = None
        stderr = None

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        captured.update(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await supervisor._default_subprocess_factory("session-1")
    env = captured.get("env")
    assert env is not None and env.get("HOME")
```

(`supervisor` = an instance built the way the file's other tests build one; `stderr = None` keeps `_SubprocessWorkerHandle` from starting a drainer.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_subprocess_env.py tests/test_lead_supervisor.py -v -k "subprocess_env or home_env"`
Expected: subprocess_env tests ERROR (module missing); supervisor test FAILS (`env` kwarg is absent).

- [ ] **Step 3: Create the helper and wire both spawn sites**

`src/feather/core/ipc/subprocess_env.py`:

```python
"""Environment construction for feather-spawned subprocesses."""

from __future__ import annotations

import os

__all__ = ("subprocess_env_with_home",)


def subprocess_env_with_home() -> dict[str, str]:
    """Snapshot ``os.environ`` with a guaranteed non-empty ``HOME``.

    ``Path.expanduser()`` raises ``RuntimeError`` when ``HOME`` is
    missing, and worker/sub-agent code calls it on hot paths (seen
    crashing in the field). When ``HOME`` is absent or empty, re-derive
    it from ``pwd`` — the same source ``os.path.expanduser`` consults.
    """

    env = os.environ.copy()
    if not env.get("HOME"):
        try:
            import pwd

            env["HOME"] = pwd.getpwuid(os.getuid()).pw_dir
        except (ImportError, KeyError, OSError):
            pass
    return env
```

`spawn_agent_tool.py`: replace lines 383-399 (the long comment + `subprocess_env` block) with:

```python
    # HOME propagation rationale lives in subprocess_env_with_home.
    subprocess_env = subprocess_env_with_home()
```

and add `from feather.core.ipc.subprocess_env import subprocess_env_with_home` to its imports.

`supervisor.py` `_default_subprocess_factory`: add the same import and pass the env:

```python
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._project_root),
            env=subprocess_env_with_home(),
        )
```

Check `src/feather/core/ipc/__init__.py` — if it re-exports the codecs, leave it alone (import the helper by full module path); the helper must not create an import cycle (the test imports will catch one immediately).

- [ ] **Step 4: Run the affected suites**

Run: `uv run pytest tests/test_subprocess_env.py tests/test_lead_supervisor.py tests/test_spawn_agent_tool.py -v`
Expected: all PASS. (If `tests/test_spawn_agent_tool.py` doesn't exist under that name, run `uv run pytest tests/ -k spawn -v`.)

- [ ] **Step 5: Commit**

```bash
git add src/feather/core/ipc/subprocess_env.py src/feather/tools/spawn_agent_tool.py src/feather/core/leads/supervisor.py tests/test_subprocess_env.py tests/test_lead_supervisor.py
git commit -m "Share HOME-safe subprocess env between sub-agent and lead-worker spawns"
```

---

### Task 8: Docstring + typing polish

**Files:**
- Modify: `src/feather/config/app_paths.py`, `src/feather/skills/catalog.py`, `src/feather/core/leads/manager.py`, `src/feather/core/leads/supervisor.py`, `src/feather/api/routes.py`, `src/feather/api/hub.py`, `src/feather/tools/task_tools.py`, `src/feather/tools/cron_tools.py`, `src/feather/core/agent/catalog.py`, `src/feather/core/agent/base.py`, `src/feather/memory/reader.py`
- Test: none new — behavior must not change; full suite green is the gate.

- [ ] **Step 1: Typing modernization (3 mechanical edits)**

- `config/app_paths.py`: delete `from typing import Optional`; change the two `Optional[Path]` annotations in `__init__` to `Path | None`.
- `skills/catalog.py`: change `from typing import Sequence, Union` to `from collections.abc import Sequence`; change `SkillSource = Union[Path, Traversable]` to `SkillSource = Path | Traversable`.
- (`core/session/coordinator.py` already fixed in Task 6.)

- [ ] **Step 2: Docstrings — one-liners, why-focused, never restating the code**

Add a one-line docstring to each currently-undocumented public method/property below. Concrete texts for the highest-traffic ones; for the rest follow the same register (what the caller gets / why the seam exists — not how):

- `core/agent/base.py` `run_one` (L609): `"""Execute one tool call under the parallel-tools semaphore."""`
- `core/agent/catalog.py` `is_lead` (L41): `"""True when this definition declares a lead role."""`; `dispatchable` (L45): `"""True when the capability profile allows spawn_agent dispatch."""`
- `core/leads/manager.py` — `run`: `"""Run one turn on this lead's session, serialized per lead."""`; `resume_on_inbox`: `"""Run a turn only if the lead's inbox has pending envelopes."""`; `enqueue_user_input`: `"""Steer an in-flight turn by queueing text for the next iteration."""`; `shutdown`: `"""Stop the worker/agent and release the lead's session."""` — and analogous one-liners for the remaining undocumented public methods on both classes in the file.
- `core/leads/supervisor.py` — the `WorkerHandle` protocol members and `_SubprocessWorkerHandle` properties (`pid`, `returncode`, `session_id`, `is_running`, …): one-liners stating what the value means for supervision (e.g. `pid`: `"""OS pid of the worker, or None before spawn."""`).
- `api/routes.py` — every route handler gets a one-liner naming the REST surface, e.g. `list_leads`: `"""GET /api/leads — all discovered leads with live status."""`, `create_lead`: `"""POST /api/leads — scaffold a new lead YAML (optionally from a soul)."""`, matching each route's decorator.
- `api/hub.py` — `create`: `"""Build the runtime and one LeadChannel per discovered lead."""`; `channel`: `"""Return the LeadChannel for a lead key, raising for unknown leads."""`; one-liners for the other undocumented public methods.
- `tools/task_tools.py` / `tools/cron_tools.py` — each undocumented `execute()`: one-liner naming the operation, e.g. `"""Create a task row and return its id for later dispatch."""` (read each class's `name`/`description` and match it).
- `memory/reader.py` — undocumented interface methods: one-liners (e.g. `augment_instructions`: `"""Inject recalled memories into the system prompt block."""`).
- `memory/trigger.py` — already documented at class level; only add one-liners where a public method has none after Task 5.

Write each docstring against the actual code you see in the file — if a suggested text above mismatches what the method really does, write the truthful one-liner instead.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest --tb=short`
Expected: all PASS (known flaky: `test_request_input_waits_for_correlated_reply` — re-run once if it's the only failure).

- [ ] **Step 4: Commit**

```bash
git add src/feather/
git commit -m "Docstring and typing polish deferred from the src refactor"
```

---

### Task 9: Simplify, full-suite gate, red-team review

**Files:** whole diff (`git diff master...improve`)

- [ ] **Step 1: Simplify pass**

Re-read every file changed in Tasks 1–8. Remove anything dead (e.g. unused `aiosqlite` imports after Task 1, unused `defaultdict` import after Task 6, unused `os`/`pwd` remnants after Task 7). Re-run `uv run pytest --tb=short` after any removal.

- [ ] **Step 2: Code-review plan (checklist for the red-team)**

Verify each invariant:
1. Every store still initializes its schema and commits; `lead_session_store` still skips `foreign_keys`.
2. `add_message` returns the same `SessionMessage` shape; sequences start at 1; `_touch_session` still inside the same commit window.
3. Tool error messages and sandbox checks are byte-identical (moved, not rewritten).
4. Trigger: drain/cancel_all see all tasks; no test relied on `_tasks` being empty in inline mode.
5. Coordinator: same-object lock invariant under waiters; `is_busy` semantics for the router unchanged.
6. Supervisor subprocess still inherits the full parent env (helper copies `os.environ`).
7. Cross-process consumers (`subagent_entry`, `lead_worker_entry`) only touch changed code via `SessionStore.add_message` and store initialization — both covered above.

- [ ] **Step 3: Red-team review**

Walk the CLAUDE.md red-team procedure over the full diff (trace from `cli.py main` → runtime → agent loop → stores/tools; classify each change following/modifying/adding and hold it to that bar). Fix anything found, re-run the suite.

- [ ] **Step 4: Final full suite**

Run: `uv run pytest --tb=short`
Expected: green. Then `git log --oneline master..improve` to confirm one commit per task.
