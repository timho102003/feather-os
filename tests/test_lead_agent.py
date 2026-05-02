"""Tests for the lead-agent loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feather.core.lead_agent import LeadAgent
from feather.core.prompt_builder import PromptBuilder
from feather.models import ModelTurn, ProviderRequestConfig, RuntimeEvent, ToolCall
from feather.providers.base import BaseLLMProvider
from feather.skills.catalog import SkillCatalog
from feather.storage.session_store import SessionStore
from feather.storage.tool_output_store import ToolOutputStore
from feather.tools.ask_user_tool import AskUserTool
from feather.tools.grep_tool import GrepTool
from feather.tools.registry import ToolRegistry
from feather.tools.skill_tool import LoadSkillTool
from feather.config import load_agent_config


class FakeProvider(BaseLLMProvider):
    """Simple fake provider for loop testing."""

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
        turn = self._turns.pop(0)
        if event_handler is not None:
            for character in turn.output_text:
                event_handler(RuntimeEvent(kind="assistant_text_delta", text=character))
        return turn


async def test_lead_agent_auto_continues_after_loading_a_skill(tmp_path: Path) -> None:
    """The agent should load a skill and continue automatically."""

    root = tmp_path
    skill_dir = root / ".feather" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo-skill
description: Demo skill
---

# Demo Skill

Loaded skill body.
""",
        encoding="utf-8",
    )
    (root / "config" / "agents").mkdir(parents=True)
    (root / "config" / "agents" / "lead.yaml").write_text(
        """name: Lead
role: lead
personality: Decisive
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
  - feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT
registered_tools:
  - load_skill
""",
        encoding="utf-8",
    )

    session_store = SessionStore(root / "feather.db")
    await session_store.initialize()

    try:
        provider = FakeProvider(
            [
                ModelTurn(
                    response_id="resp-1",
                    output_text="",
                    tool_calls=[ToolCall(call_id="call-1", name="load_skill", arguments={"skill_name": "demo-skill"})],
                ),
                ModelTurn(response_id="resp-2", output_text="Done.", tool_calls=[]),
            ]
        )
        skill_catalog = SkillCatalog(root / ".feather" / "skills")
        tool_registry = ToolRegistry([LoadSkillTool(skill_catalog)])
        prompt_builder = PromptBuilder(skill_catalog, tool_registry)
        agent = LeadAgent(
            agent_config=load_agent_config(root, "lead"),
            prompt_builder=prompt_builder,
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(root, ".feather/tmp"),
            tool_registry=tool_registry,
        )

        session_id = await agent.create_session()
        result = await agent.run(session_id, "Help me")

        assert result.status.value == "completed"
        assert result.assistant_text == "Done."
        session = await session_store.get_session(session_id)
        messages = await session_store.list_messages(session_id)
        assert session.loaded_skills == ["demo-skill"]
        assert "Loaded skill body." in provider.calls[1]["instructions"]
        assert provider.calls[1]["input_items"][0]["output"].startswith("Loaded skill `demo-skill`")
        assert messages[1].role.value == "tool"
        assert messages[1].file_ref is not None
        assert messages[1].content.startswith("load_skill tool call output content file:")
        assert (root / messages[1].file_ref).exists()
    finally:
        await session_store.close()


async def test_lead_agent_pauses_for_ask_user_and_resumes(tmp_path: Path) -> None:
    """The agent should preserve pending tool outputs when it needs user input."""

    root = tmp_path
    (root / "config" / "agents").mkdir(parents=True)
    (root / "config" / "agents" / "lead.yaml").write_text(
        """name: Lead
role: lead
personality: Decisive
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
  - feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT
registered_tools:
  - grep
  - ask_user
  - load_skill
""",
        encoding="utf-8",
    )
    (root / ".feather" / "skills").mkdir(parents=True)
    (root / "app.py").write_text("print('hello')\n", encoding="utf-8")

    session_store = SessionStore(root / "feather.db")
    await session_store.initialize()

    try:
        provider = FakeProvider(
            [
                ModelTurn(
                    response_id="resp-1",
                    output_text="",
                    tool_calls=[ToolCall(call_id="call-1", name="ask_user", arguments={"question": "Which file should I inspect?"})],
                ),
                ModelTurn(response_id="resp-2", output_text="Thanks.", tool_calls=[]),
            ]
        )
        skill_catalog = SkillCatalog(root / ".feather" / "skills")
        tool_registry = ToolRegistry([GrepTool(root), AskUserTool(), LoadSkillTool(skill_catalog)])
        prompt_builder = PromptBuilder(skill_catalog, tool_registry)
        agent = LeadAgent(
            agent_config=load_agent_config(root, "lead"),
            prompt_builder=prompt_builder,
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(root, ".feather/tmp"),
            tool_registry=tool_registry,
        )

        session_id = await agent.create_session()
        first = await agent.run(session_id, "Investigate the repo")
        stored = await session_store.get_session(session_id)

        assert first.status.value == "awaiting_user"
        assert first.question == "Which file should I inspect?"
        assert stored.pending_inputs[0]["type"] == "function_call_output"
        assert "User input required: Which file should I inspect?" in stored.pending_inputs[0]["output"]
        assert stored.last_response_id == "resp-1"

        second = await agent.run(session_id, "Use app.py")

        messages = await session_store.list_messages(session_id)
        assert second.status.value == "completed"
        assert second.assistant_text == "Thanks."
        assert provider.calls[1]["previous_response_id"] == "resp-1"
        assert provider.calls[1]["input_items"][0]["type"] == "function_call_output"
        assert provider.calls[1]["input_items"][1]["type"] == "message"
        assert messages[1].file_ref is not None
        assert messages[1].content.startswith("ask_user tool call output content file:")
    finally:
        await session_store.close()
