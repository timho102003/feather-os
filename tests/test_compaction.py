"""Tests for automatic active-context compaction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feather.core.agent.compaction import ContextCompactor
from feather.core.agent.base import BaseAgent
from feather.core.agent.prompt_builder import PromptBuilder
from feather.core.prompts.compaction_prompt import COMPACTION_PROMPT
from feather.models import AgentConfig, CompactionConfig, ModelTurn, ProviderRequestConfig
from feather.providers.base import BaseLLMProvider
from feather.skills.catalog import SkillCatalog
from feather.storage.session_store import SessionStore
from feather.storage.tool_output_store import ToolOutputStore
from feather.tools.registry import ToolRegistry


class FakeProvider(BaseLLMProvider):
    """Fake provider that records every request and returns queued turns."""

    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        event_handler=None,
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


async def test_agent_compaction_clears_remote_cursor_and_replays_latest_compact(tmp_path: Path) -> None:
    """After compaction, the next turn should replay only the latest compact history locally."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()

    try:
        provider = FakeProvider(
            [
                ModelTurn(
                    response_id="resp-1",
                    output_text="First answer.",
                    tool_calls=[],
                    usage={"input_tokens": 90, "output_tokens": 10, "total_tokens": 100},
                ),
                ModelTurn(
                    response_id="resp-compact",
                    output_text="## Objective\nCompacted state.\n\n## Next Step\nContinue.",
                    tool_calls=[],
                ),
                ModelTurn(
                    response_id="resp-2",
                    output_text="Second answer.",
                    tool_calls=[],
                    usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                ),
            ]
        )
        prompt_builder = PromptBuilder(SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([]))
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                    "feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT",
                ],
                registered_tools=[],
            ),
            prompt_builder=prompt_builder,
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=ToolRegistry([]),
            compactor=ContextCompactor(
                config=CompactionConfig(
                    enabled=True,
                    trigger_ratio=0.8,
                    context_window_tokens=100,
                    model="gpt-5-mini-compact",
                    max_output_tokens=333,
                    temperature=0.0,
                ),
                provider=provider,
                session_store=session_store,
            ),
        )

        session_id = await agent.create_session()
        first = await agent.run(session_id, "Initial request")

        session = await session_store.get_session(session_id)
        messages = await session_store.list_messages(session_id)

        assert first.assistant_text == "First answer."
        assert session.last_response_id is None
        assert any(message.is_compact for message in messages)
        assert provider.calls[1]["instructions"] == COMPACTION_PROMPT
        assert provider.calls[1]["request_config"] is not None
        assert provider.calls[1]["request_config"].model == "gpt-5-mini-compact"
        assert provider.calls[1]["request_config"].max_output_tokens == 333

        second = await agent.run(session_id, "Follow-up request")

        assert second.assistant_text == "Second answer."
        assert provider.calls[2]["previous_response_id"] is None
        replay_text = provider.calls[2]["input_items"][0]["content"][0]["text"]
        assert "Session context to continue from" in replay_text
        assert "assistant[compact]: ## Objective\nCompacted state." in replay_text
        assert provider.calls[2]["input_items"][1]["content"][0]["text"] == "Follow-up request"
    finally:
        await session_store.close()
