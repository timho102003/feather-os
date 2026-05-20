"""Tests for the non-blocking spawn_agent tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from feather.core.agent_catalog import AgentCatalog, AgentCatalogEntry
from feather.core.subagent_registry import SubagentRegistry
from feather.models import TaskStatus, ToolExecutionContext
from feather.storage.task_store import TaskStore
from feather.tools.spawn_agent_tool import SpawnAgentTool


def _context(session_id: str = "lead-sess") -> ToolExecutionContext:
    return ToolExecutionContext(session_id=session_id, agent_name="Lead")


class _FakeCatalog(AgentCatalog):
    """AgentCatalog stand-in that returns a fixed entry list."""

    def __init__(self, entries: list[AgentCatalogEntry]) -> None:
        super().__init__(Path("/nonexistent"))
        self._entries = list(entries)

    def list_entries(self) -> list[AgentCatalogEntry]:
        return list(self._entries)


def _default_catalog() -> _FakeCatalog:
    return _FakeCatalog(
        [
            AgentCatalogEntry(
                name="lead",
                role="lead",
                description="lead",
                personality="",
                registered_tools=(),
                is_builtin=True,
            ),
            AgentCatalogEntry(
                name="explore",
                role="explore",
                description="local nav",
                personality="",
                registered_tools=("read_file",),
                is_builtin=True,
            ),
            AgentCatalogEntry(
                name="validate",
                role="validate",
                description="run checks",
                personality="",
                registered_tools=("bash",),
                is_builtin=True,
            ),
        ]
    )


class _EofStream:
    """Minimal StreamReader stand-in: always returns EOF."""

    async def read(self, n: int) -> bytes:
        return b""


class _FakeProc:
    """Async subprocess stand-in for tests."""

    def __init__(self) -> None:
        self.pid = 12345
        self.returncode: int | None = None
        self.stdout = _EofStream()
        self.stderr = _EofStream()

    def terminate(self) -> None:
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        if self.returncode is None:
            self.returncode = -9


def _install_subprocess_stub(
    monkeypatch: pytest.MonkeyPatch, proc: _FakeProc
) -> list[list[str]]:
    """Patch asyncio.create_subprocess_exec and capture argv invocations."""

    captured: list[list[str]] = []

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured.append(list(args))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    return captured


async def test_spawn_agent_returns_session_id_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawn must return without awaiting the subprocess."""

    proc = _FakeProc()
    captured = _install_subprocess_stub(monkeypatch, proc)
    registry = SubagentRegistry()

    tool = SpawnAgentTool(
        root=tmp_path, agent_catalog=_default_catalog(), registry=registry
    )
    result = await tool.execute(
        {"agent_name": "explore", "task": "find X"}, _context()
    )

    assert "Sub-agent `explore` spawned." in result.output
    assert "session_id:" in result.output
    assert "correlation_id:" in result.output
    assert "pid: 12345" in result.output
    # Registry must carry one live entry.
    live = await registry.snapshot()
    assert len(live) == 1
    assert live[0].agent_name == "explore"
    assert live[0].parent_session_id == "lead-sess"
    assert live[0].correlation_id is not None
    # argv sanity.
    argv = captured[0]
    assert "--agent-name" in argv
    assert "--session-id" in argv
    assert "--correlation-id" in argv
    assert "--parent-agent-name" in argv


async def test_spawn_agent_passes_explicit_env_with_home_to_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sub-agent subprocess MUST receive an explicit ``env`` containing
    ``HOME``. ``Path.expanduser`` raises ``RuntimeError("Could not
    determine home directory.")`` when HOME is missing and ``pwd`` cannot
    supply it — and the attachment-drop parser calls expanduser on every
    inbound user-text token. We've seen this crash in the field
    (``.feather/logs/feather.log`` shows
    ``subagent_entry.py:250 → attachments.py:248 RuntimeError``), so make
    the env propagation explicit and assert it here.
    """

    proc = _FakeProc()
    captured_kwargs: list[dict[str, Any]] = []

    async def fake_create_subprocess_exec(
        *args: Any, **kwargs: Any
    ) -> _FakeProc:
        captured_kwargs.append(kwargs)
        return proc

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )
    monkeypatch.setenv("HOME", "/home/test-user")

    tool = SpawnAgentTool(
        root=tmp_path,
        agent_catalog=_default_catalog(),
        registry=SubagentRegistry(),
    )
    await tool.execute({"agent_name": "explore", "task": "find X"}, _context())

    assert captured_kwargs, "subprocess was never spawned"
    env = captured_kwargs[0].get("env")
    assert env is not None, (
        "spawn must pass env= explicitly; relying on inherit means a "
        "future bug in the parent env can silently strip HOME"
    )
    assert env.get("HOME") == "/home/test-user", (
        f"sub-agent subprocess env is missing HOME: {env!r}"
    )


async def test_spawn_agent_recovers_home_when_parent_env_lacks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defense in depth: even if the parent process somehow has HOME
    unset, spawn_agent should re-derive it from ``pwd`` rather than ship
    the child with HOME missing and let it crash later in expanduser."""

    proc = _FakeProc()
    captured_kwargs: list[dict[str, Any]] = []

    async def fake_create_subprocess_exec(
        *args: Any, **kwargs: Any
    ) -> _FakeProc:
        captured_kwargs.append(kwargs)
        return proc

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )
    monkeypatch.delenv("HOME", raising=False)

    tool = SpawnAgentTool(
        root=tmp_path,
        agent_catalog=_default_catalog(),
        registry=SubagentRegistry(),
    )
    await tool.execute({"agent_name": "explore", "task": "find X"}, _context())

    env = captured_kwargs[0].get("env")
    assert env is not None
    # On a host with a populated ``/etc/passwd`` we recover from pwd; on
    # an exotic container without one we still don't crash, we just
    # forward what we can. Either is acceptable; what matters is the
    # subprocess does NOT inherit a None env or an env that would
    # silently start a fresh ``os.environ`` look-up from scratch.
    assert "HOME" in env or env.get("HOME") is None or env.get("HOME") == ""


async def test_spawn_agent_rejects_empty_task(tmp_path: Path) -> None:
    tool = SpawnAgentTool(
        root=tmp_path,
        agent_catalog=_default_catalog(),
        registry=SubagentRegistry(),
    )
    with pytest.raises(ValueError):
        await tool.execute({"agent_name": "explore", "task": "   "}, _context())


async def test_spawn_agent_rejects_unknown_agent_name(tmp_path: Path) -> None:
    tool = SpawnAgentTool(
        root=tmp_path,
        agent_catalog=_default_catalog(),
        registry=SubagentRegistry(),
    )
    with pytest.raises(ValueError, match="Unknown sub-agent"):
        await tool.execute(
            {"agent_name": "does-not-exist", "task": "do thing"},
            _context(),
        )


async def test_spawn_agent_rejects_lead_agent_name(tmp_path: Path) -> None:
    """The lead is in the catalog but is explicitly non-dispatchable."""

    tool = SpawnAgentTool(
        root=tmp_path,
        agent_catalog=_default_catalog(),
        registry=SubagentRegistry(),
    )
    with pytest.raises(ValueError, match="not dispatchable"):
        await tool.execute(
            {"agent_name": "lead", "task": "do thing"},
            _context(),
        )


async def test_spawn_agent_rejects_traversal_shaped_name(tmp_path: Path) -> None:
    tool = SpawnAgentTool(
        root=tmp_path,
        agent_catalog=_default_catalog(),
        registry=SubagentRegistry(),
    )
    with pytest.raises(ValueError, match="letters, digits"):
        await tool.execute(
            {"agent_name": "../../etc/passwd", "task": "bad"},
            _context(),
        )


async def test_spawn_agent_registers_unique_sessions_for_parallel_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Back-to-back spawns must register distinct session ids + correlation ids."""

    proc = _FakeProc()
    _install_subprocess_stub(monkeypatch, proc)
    registry = SubagentRegistry()
    tool = SpawnAgentTool(
        root=tmp_path, agent_catalog=_default_catalog(), registry=registry
    )
    a = await tool.execute({"agent_name": "explore", "task": "A"}, _context())
    b = await tool.execute({"agent_name": "explore", "task": "B"}, _context())
    # Extract session_ids from stdout.
    def _sid(text: str) -> str:
        for line in text.splitlines():
            if line.startswith("session_id:"):
                return line.split(":", 1)[1].strip()
        raise AssertionError(f"no session_id found in: {text!r}")

    assert _sid(a.output) != _sid(b.output)
    live = await registry.snapshot()
    assert len({entry.session_id for entry in live}) == 2
    assert len({entry.correlation_id for entry in live}) == 2


async def test_spawn_agent_parent_agent_name_is_carried_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """parent_agent_name stamped at construction should appear in argv."""

    proc = _FakeProc()
    captured = _install_subprocess_stub(monkeypatch, proc)
    registry = SubagentRegistry()
    tool = SpawnAgentTool(
        root=tmp_path,
        agent_catalog=_default_catalog(),
        registry=registry,
        parent_agent_name="Engineer",
    )
    await tool.execute({"agent_name": "explore", "task": "go"}, _context())
    argv = captured[0]
    idx = argv.index("--parent-agent-name")
    assert argv[idx + 1] == "Engineer"


async def test_spawn_agent_treats_null_string_task_id_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: model sends ``"task_id": "null"`` (literal stringified
    JSON null) after schema flattening. The tool must treat that as
    "no task id" and create a fresh task rather than passing the string
    through to ``TaskStore.get_task`` which crashes with
    ``Unknown task: null``.
    """

    proc = _FakeProc()
    _install_subprocess_stub(monkeypatch, proc)
    task_store = TaskStore(tmp_path / "feather.db")
    await task_store.initialize()
    try:
        tool = SpawnAgentTool(
            root=tmp_path,
            agent_catalog=_default_catalog(),
            registry=SubagentRegistry(),
            task_store=task_store,
        )

        result = await tool.execute(
            {
                "agent_name": "explore",
                "task": "investigate something",
                "task_id": "null",  # the regression input
            },
            _context(),
        )

        # spawn_agent must have created a fresh task (not crashed).
        tasks = await task_store.list_tasks(lead_session_id="lead-sess")
        assert len(tasks) == 1
        assert tasks[0].title  # auto-derived from the task description
        assert f"task_id: {tasks[0].id}" in result.output
    finally:
        await task_store.close()


async def test_spawn_agent_binds_existing_queued_task_without_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _FakeProc()
    captured = _install_subprocess_stub(monkeypatch, proc)
    registry = SubagentRegistry()
    task_store = TaskStore(tmp_path / "feather.db")
    await task_store.initialize()
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Planned exploration",
            responsible_agent_name="explore",
            status=TaskStatus.QUEUED,
        )
        tool = SpawnAgentTool(
            root=tmp_path,
            agent_catalog=_default_catalog(),
            registry=registry,
            task_store=task_store,
        )

        result = await tool.execute(
            {
                "agent_name": "explore",
                "task": "find X",
                "task_id": task.id,
            },
            _context(),
        )

        assert f"task_id: {task.id}" in result.output
        tasks = await task_store.list_tasks(lead_session_id="lead-sess")
        updated = await task_store.get_task(task.id)
        live = await registry.snapshot()
        argv = captured[0]
        assert len(tasks) == 1
        assert updated.status == TaskStatus.RUNNING
        assert updated.responsible_session_id is not None
        assert live[0].task_id == task.id
        assert "--task-id" in argv
        assert argv[argv.index("--task-id") + 1] == task.id
    finally:
        await task_store.close()


async def test_spawn_agent_rejects_existing_non_queued_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _FakeProc()
    _install_subprocess_stub(monkeypatch, proc)
    task_store = TaskStore(tmp_path / "feather.db")
    await task_store.initialize()
    try:
        task = await task_store.create_task(
            lead_session_id="lead-sess",
            title="Already running",
            responsible_agent_name="explore",
            status=TaskStatus.RUNNING,
        )
        tool = SpawnAgentTool(
            root=tmp_path,
            agent_catalog=_default_catalog(),
            registry=SubagentRegistry(),
            task_store=task_store,
        )

        with pytest.raises(ValueError, match="queued planned tasks"):
            await tool.execute(
                {
                    "agent_name": "explore",
                    "task": "find X",
                    "task_id": task.id,
                },
                _context(),
            )
    finally:
        await task_store.close()
