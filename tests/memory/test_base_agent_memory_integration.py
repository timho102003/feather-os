"""Tests for BaseAgent's memory hooks (read-path injection + write-path trigger)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

from feather.core.agent.base import BaseAgent
from feather.core.agent.prompt_builder import PromptBuilder
from feather.memory.enums import MemoryOwner
from feather.memory.models import MemorySearchResult
from feather.memory.reader import MemoryReader, NoOpMemoryReader
from feather.memory.trigger import MemoryTrigger, NoOpMemoryTrigger
from feather.models import (
    AgentConfig,
    ModelTurn,
    ProviderRequestConfig,
)
from feather.providers.base import BaseLLMProvider
from feather.skills.catalog import SkillCatalog
from feather.storage.session_store import SessionStore
from feather.storage.tool_output_store import ToolOutputStore
from feather.tools.registry import ToolRegistry


class _FakeProvider(BaseLLMProvider):
    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def complete(  # type: ignore[override]
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        event_handler: Any = None,
        request_config: ProviderRequestConfig | None = None,
    ) -> ModelTurn:
        self.calls.append(
            {
                "instructions": instructions,
                "input_items": input_items,
                "tools": tools,
                "previous_response_id": previous_response_id,
                "request_config": request_config,
            }
        )
        return self._turns.pop(0)


class _FakeReader(MemoryReader):
    def __init__(
        self,
        *,
        block: str = "",
        raise_on_augment: BaseException | None = None,
    ) -> None:
        self._block = block
        self._raise = raise_on_augment
        self.augment_calls: list[dict[str, Any]] = []

    async def augment_instructions(  # type: ignore[override]
        self,
        *,
        session_id: str,
        recent_messages: Sequence[Any],
        latest_user_text: str,
        agent_model: str,
        owner: MemoryOwner = MemoryOwner.USER,
    ) -> str:
        self.augment_calls.append(
            {
                "session_id": session_id,
                "recent_messages": list(recent_messages),
                "latest_user_text": latest_user_text,
                "agent_model": agent_model,
                "owner": owner,
            }
        )
        if self._raise is not None:
            raise self._raise
        return self._block

    async def recall(self, **_kwargs: object) -> list[MemorySearchResult]:  # pragma: no cover
        return []


class _FakeTrigger(MemoryTrigger):
    def __init__(self, *, raise_on_schedule: BaseException | None = None) -> None:
        self.schedule_calls: list[dict[str, Any]] = []
        self.drained = False
        self.cancelled = False
        self._raise = raise_on_schedule

    def maybe_schedule(  # type: ignore[override]
        self, session_id: str, *, agent_model: str, owner: MemoryOwner
    ) -> None:
        self.schedule_calls.append(
            {"session_id": session_id, "agent_model": agent_model, "owner": owner}
        )
        if self._raise is not None:
            raise self._raise

    async def drain(self, timeout_s: float) -> None:  # pragma: no cover
        self.drained = True

    def cancel_all(self) -> None:  # pragma: no cover
        self.cancelled = True


async def _make_agent(
    tmp_path: Path,
    *,
    provider: _FakeProvider,
    memory_reader: MemoryReader | None = None,
    memory_trigger: MemoryTrigger | None = None,
    model_name: str = "fake-model",
) -> tuple[BaseAgent, SessionStore]:
    (tmp_path / ".feather" / "skills").mkdir(parents=True, exist_ok=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    prompt_builder = PromptBuilder(
        SkillCatalog(tmp_path / ".feather" / "skills"),
        ToolRegistry([]),
    )

    class _ConcreteAgent(BaseAgent):
        pass

    agent = _ConcreteAgent(
        agent_config=AgentConfig(
            name="Lead",
            role="lead",
            personality="Direct",
            prompt_modules=[
                "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
            ],
            registered_tools=[],
        ),
        prompt_builder=prompt_builder,
        provider=provider,
        session_store=session_store,
        tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
        tool_registry=ToolRegistry([]),
        memory_reader=memory_reader or NoOpMemoryReader(),
        memory_trigger=memory_trigger or NoOpMemoryTrigger(),
        model_name=model_name,
    )
    return agent, session_store


# Default (NoOp) wiring is invisible -----------------------------------------


async def test_noop_wiring_keeps_existing_behavior(tmp_path: Path) -> None:
    provider = _FakeProvider(
        [ModelTurn(response_id="r", output_text="ok", tool_calls=[])]
    )
    agent, session_store = await _make_agent(tmp_path, provider=provider)
    try:
        sid = await agent.create_session()
        result = await agent.run(sid, "hello")
        assert result.assistant_text == "ok"
        # Provider was called exactly once — no extra read-path or write-path LLM calls.
        assert len(provider.calls) == 1
        # Memory block must NOT have been added to instructions.
        assert "Relevant memory" not in provider.calls[0]["instructions"]
    finally:
        await session_store.close()


# Read-path injection --------------------------------------------------------


async def test_memory_block_is_injected_into_instructions(tmp_path: Path) -> None:
    block = "## Relevant memory from past conversations\n1. [0.9] X"
    reader = _FakeReader(block=block)
    provider = _FakeProvider(
        [ModelTurn(response_id="r", output_text="ok", tool_calls=[])]
    )
    agent, session_store = await _make_agent(
        tmp_path, provider=provider, memory_reader=reader
    )
    try:
        sid = await agent.create_session()
        await agent.run(sid, "hello world")
        # Reader was called with the latest user text and agent model.
        assert reader.augment_calls
        first = reader.augment_calls[0]
        assert first["latest_user_text"] == "hello world"
        assert first["agent_model"] == "fake-model"
        # The block landed inside the prompt.
        assert "## Relevant memory" in provider.calls[0]["instructions"]
    finally:
        await session_store.close()


async def test_reader_exception_is_swallowed_and_agent_continues(tmp_path: Path) -> None:
    """A misbehaving reader must NOT break the agent turn."""
    reader = _FakeReader(raise_on_augment=RuntimeError("boom"))
    provider = _FakeProvider(
        [ModelTurn(response_id="r", output_text="ok", tool_calls=[])]
    )
    agent, session_store = await _make_agent(
        tmp_path, provider=provider, memory_reader=reader
    )
    try:
        sid = await agent.create_session()
        result = await agent.run(sid, "hello")
        assert result.assistant_text == "ok"
        # No memory block, but the turn completed normally.
        assert "Relevant memory" not in provider.calls[0]["instructions"]
    finally:
        await session_store.close()


# Write-path trigger ---------------------------------------------------------


async def test_trigger_fires_exactly_once_after_run_loop_completes(tmp_path: Path) -> None:
    trigger = _FakeTrigger()
    provider = _FakeProvider(
        [ModelTurn(response_id="r", output_text="ok", tool_calls=[])]
    )
    agent, session_store = await _make_agent(
        tmp_path, provider=provider, memory_trigger=trigger
    )
    try:
        sid = await agent.create_session()
        await agent.run(sid, "hi")
        assert len(trigger.schedule_calls) == 1
        call = trigger.schedule_calls[0]
        assert call["session_id"] == sid
        assert call["agent_model"] == "fake-model"
        assert call["owner"] is MemoryOwner.USER
    finally:
        await session_store.close()


async def test_trigger_exception_is_swallowed_and_agent_returns_normally(tmp_path: Path) -> None:
    trigger = _FakeTrigger(raise_on_schedule=RuntimeError("schedule blew up"))
    provider = _FakeProvider(
        [ModelTurn(response_id="r", output_text="ok", tool_calls=[])]
    )
    agent, session_store = await _make_agent(
        tmp_path, provider=provider, memory_trigger=trigger
    )
    try:
        sid = await agent.create_session()
        result = await agent.run(sid, "hi")
        assert result.assistant_text == "ok"
    finally:
        await session_store.close()


async def test_trigger_fires_even_when_agent_raises(tmp_path: Path) -> None:
    """The write-path trigger lives in `finally` and must still fire on errors."""
    trigger = _FakeTrigger()

    class _BoomProvider(BaseLLMProvider):
        async def complete(  # type: ignore[override]
            self,
            **_kwargs: Any,
        ) -> ModelTurn:
            raise RuntimeError("provider down")

    agent, session_store = await _make_agent(
        tmp_path,
        provider=_BoomProvider(),  # type: ignore[arg-type]
        memory_trigger=trigger,
    )
    try:
        sid = await agent.create_session()
        with pytest.raises(RuntimeError):
            await agent.run(sid, "hi")
        # Trigger STILL ran exactly once.
        assert len(trigger.schedule_calls) == 1
    finally:
        await session_store.close()
