"""Sub-agent transcript drill-down modal (transcript-on-demand)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from feather.textual_tui import SubagentDrillScreen


class _FakeOptionList:
    def __init__(self) -> None:
        self.options: list = []

    def clear_options(self) -> None:
        self.options.clear()

    def add_option(self, option) -> None:
        self.options.append(option)


class _FakeLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def clear(self) -> None:
        self.lines.clear()

    def write(self, renderable, **_kw) -> None:
        self.lines.append(getattr(renderable, "plain", str(renderable)))


def _live(agent_name, session_id, parent, task):
    return SimpleNamespace(
        agent_name=agent_name,
        session_id=session_id,
        parent_session_id=parent,
        task_text=task,
    )


def _msg(role, content):
    return SimpleNamespace(role=SimpleNamespace(value=role), content=content)


class _FakeRuntime:
    def __init__(self, live, messages):
        self._live = live
        self._messages = messages
        self.subagent_registry = SimpleNamespace(snapshot=self._snapshot)
        self.session_store = SimpleNamespace(list_messages=self._list_messages)

    async def _snapshot(self):
        return self._live

    async def _list_messages(self, session_id):
        return self._messages.get(session_id, [])


def _wire(monkeypatch, screen):
    opt = _FakeOptionList()
    log = _FakeLog()

    def query_one(selector, *_a):
        return opt if "list" in selector else log

    monkeypatch.setattr(screen, "query_one", query_one)
    return opt, log


async def test_drill_lists_subagents_and_loads_first_transcript(monkeypatch) -> None:
    runtime = _FakeRuntime(
        live=[
            _live("explore", "sub-1", "lead-sess", "find the bug"),
            _live("research", "sub-2", "lead-sess", "survey docs"),
            _live("explore", "other", "OTHER-sess", "not this lead"),
        ],
        messages={
            "sub-1": [_msg("user", "go"), _msg("assistant", "found it in foo.py")],
        },
    )
    screen = SubagentDrillScreen(
        runtime=runtime, parent_session_id="lead-sess", lead_display_name="Tim"
    )
    opt, log = _wire(monkeypatch, screen)

    await screen._reload()

    # Only this lead's sub-agents are listed.
    ids = [o.id for o in opt.options]
    assert ids == ["sub-1", "sub-2"]
    # First one's transcript loaded.
    joined = "\n".join(log.lines)
    assert "found it in foo.py" in joined
    assert "[assistant]" in joined


async def test_drill_handles_no_subagents(monkeypatch) -> None:
    runtime = _FakeRuntime(live=[], messages={})
    screen = SubagentDrillScreen(
        runtime=runtime, parent_session_id="lead-sess", lead_display_name="Tim"
    )
    opt, log = _wire(monkeypatch, screen)
    await screen._reload()
    assert opt.options == []
    assert any("No live sub-agents" in line for line in log.lines)


async def test_drill_handles_empty_transcript(monkeypatch) -> None:
    runtime = _FakeRuntime(
        live=[_live("explore", "sub-1", "lead-sess", "just started")],
        messages={},  # no messages yet
    )
    screen = SubagentDrillScreen(
        runtime=runtime, parent_session_id="lead-sess", lead_display_name="Tim"
    )
    _opt, log = _wire(monkeypatch, screen)
    await screen._reload()
    assert any("no transcript yet" in line.lower() for line in log.lines)
