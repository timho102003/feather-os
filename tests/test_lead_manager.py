"""LeadHandle + LeadManager: the multi-lead substrate.

These tests exercise the in-process path with a light fake runtime; the
supervised (worker subprocess) path is covered by the supervisor tests.
"""

from __future__ import annotations

import pytest

from feather.core.agent.catalog import AgentCatalogEntry
from feather.core.leads.manager import (
    InProcessLeadHandle,
    LeadHandle,
    LeadManager,
)
from feather.models import AgentConfig, AgentOutcome, AgentRunResult


# --------------------------- fakes ---------------------------


class _FakeInputQueue:
    def __init__(self):
        self.enqueued: list[tuple[str, str]] = []

    async def enqueue(self, session_id: str, text: str) -> bool:
        self.enqueued.append((session_id, text))
        return True


class _FakeAgent:
    def __init__(self, name: str, *, counter):
        self._counter = counter
        self.calls: list[tuple] = []
        self.config = AgentConfig(
            name=name.title(),
            role="lead",
            personality=f"{name} is decisive.",
            prompt_modules=[],
            registered_tools=[],
            soul=f"You are {name}.",
            color="#abcdef",
            emoji="🧭",
        )

    async def create_session(self) -> str:
        self._counter["n"] += 1
        return f"sess-{self._counter['n']}"

    async def ensure_session_with_id(self, session_id: str) -> str:
        self.calls.append(("ensure", session_id))
        return session_id

    async def run(self, session_id, text, on_event):
        self.calls.append(("run", session_id, text))
        return AgentRunResult(AgentOutcome.COMPLETED, session_id, "ok")

    async def resume_on_inbox(self, session_id, on_event):
        self.calls.append(("resume", session_id))
        return None


class _FakeCatalog:
    def __init__(self, lead_names: list[str]):
        self._lead_names = lead_names

    def list_leads(self):
        return [
            AgentCatalogEntry(name=n, role="lead", description="", personality="")
            for n in self._lead_names
        ]


class _FakeRuntime:
    def __init__(self, lead_session_store, lead_names):
        self.lead_session_store = lead_session_store
        self.input_queue = _FakeInputQueue()
        self.agent_catalog = _FakeCatalog(lead_names)
        self.default_lead_name = "lead"
        self._counter = {"n": 0}
        self._agents: dict[str, _FakeAgent] = {}

    def build_agent(self, name: str) -> _FakeAgent:
        agent = self._agents.get(name)
        if agent is None:
            agent = _FakeAgent(name, counter=self._counter)
            self._agents[name] = agent
        return agent


@pytest.fixture
async def lead_session_store(tmp_path):
    from feather.storage.lead_session_store import LeadSessionStore

    s = LeadSessionStore(tmp_path / "feather.db")
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()


# --------------------------- handle ---------------------------


async def test_in_process_handle_binds_session_and_routes():
    counter = {"n": 0}
    agent = _FakeAgent("tim", counter=counter)
    iq = _FakeInputQueue()
    handle = InProcessLeadHandle(name="tim", session_id="sess-1", agent=agent, input_queue=iq)
    assert isinstance(handle, LeadHandle)  # runtime_checkable protocol
    assert handle.name == "tim" and handle.session_id == "sess-1"

    res = await handle.run("hi", None)
    assert res.status == AgentOutcome.COMPLETED
    assert agent.calls[0] == ("run", "sess-1", "hi")

    await handle.resume_on_inbox(None)
    assert ("resume", "sess-1") in agent.calls

    assert await handle.enqueue_user_input("queued") is True
    assert iq.enqueued == [("sess-1", "queued")]
    await handle.shutdown()  # no-op


async def test_in_process_handle_enqueue_without_queue_returns_false():
    agent = _FakeAgent("tim", counter={"n": 0})
    handle = InProcessLeadHandle(name="tim", session_id="s", agent=agent, input_queue=None)
    assert await handle.enqueue_user_input("x") is False


# --------------------------- manager ---------------------------


async def test_manager_creates_then_persists_session(lead_session_store):
    runtime = _FakeRuntime(lead_session_store, ["lead"])
    mgr = LeadManager(runtime, worker_mode=False)
    await mgr.start(["lead"])

    handle = mgr.handle("lead")
    assert handle.session_id == "sess-1"
    # The session was recorded so it can resume next launch.
    assert await lead_session_store.get("lead") == "sess-1"


async def test_manager_resumes_existing_session(lead_session_store):
    await lead_session_store.upsert("lead", "prior-sess")
    runtime = _FakeRuntime(lead_session_store, ["lead"])
    mgr = LeadManager(runtime, worker_mode=False)
    await mgr.start(["lead"])

    handle = mgr.handle("lead")
    assert handle.session_id == "prior-sess"
    # ensure_session_with_id was used (resume), not create_session.
    agent = runtime.build_agent("lead")
    assert ("ensure", "prior-sess") in agent.calls


async def test_manager_add_lead_is_idempotent(lead_session_store):
    runtime = _FakeRuntime(lead_session_store, ["lead"])
    mgr = LeadManager(runtime, worker_mode=False)
    h1 = await mgr.add_lead("lead")
    h2 = await mgr.add_lead("lead")
    assert h1 is h2


async def test_manager_discovers_multiple_leads(lead_session_store):
    runtime = _FakeRuntime(lead_session_store, ["lead", "sophia"])
    mgr = LeadManager(runtime, worker_mode=False)
    await mgr.start()  # discover

    assert mgr.active_names() == ["lead", "sophia"]
    infos = mgr.list_leads()
    assert [i.name for i in infos] == ["lead", "sophia"]
    sophia = mgr.info("sophia")
    assert sophia.soul == "You are sophia."
    assert sophia.emoji == "🧭"
    # Two distinct durable sessions were recorded.
    recorded = dict(await lead_session_store.list())
    assert set(recorded) == {"lead", "sophia"}
    assert recorded["lead"] != recorded["sophia"]


async def test_manager_shutdown_clears_state(lead_session_store):
    runtime = _FakeRuntime(lead_session_store, ["lead"])
    mgr = LeadManager(runtime, worker_mode=False)
    await mgr.start(["lead"])
    await mgr.shutdown()
    assert mgr.active_names() == []
    assert mgr.list_leads() == []
