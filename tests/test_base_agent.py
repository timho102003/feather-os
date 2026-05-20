"""Tests for the reusable base-agent loop."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from feather.core.base_agent import BaseAgent
from feather.core.compaction import ContextCompactor
from feather.core.input_queue import UserInputQueue
from feather.core.prompt_builder import PromptBuilder
from feather.models import (
    AgentConfig,
    AgentOutcome,
    CompactionConfig,
    MessageRole,
    MCPServerConfig,
    ModelTurn,
    ProviderRequestConfig,
    ReasoningConfig,
    RuntimeEvent,
    ToolCall,
    ToolExecutionContext,
    ToolExecutionResult,
)
from feather.providers.base import BaseLLMProvider
from feather.skills.catalog import SkillCatalog
from feather.storage.attachment_store import AttachmentStore
from feather.storage.session_store import SessionStore
from feather.storage.tool_output_store import ToolOutputStore
from feather.tools.base import BaseTool
from feather.mcp_client import MCPProxyTool
from feather.tools.registry import ToolRegistry


class FakeProvider(BaseLLMProvider):
    """Simple fake provider for base-agent loop testing."""

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


class PrefixedAgent(BaseAgent):
    """Base-agent subclass that customizes how incoming text becomes provider input."""

    def _build_input_items(self, incoming_text: str) -> list[dict[str, Any]]:
        return [self._message_item(f"prefixed: {incoming_text}")]


class BlockingProvider(BaseLLMProvider):
    """Provider stub that blocks the first call so concurrent runs can be observed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.first_call_started = asyncio.Event()
        self.release_first_call = asyncio.Event()

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
        call_index = len(self.calls) + 1
        self.calls.append(
            {
                "input_items": input_items,
                "previous_response_id": previous_response_id,
            }
        )
        if call_index == 1:
            self.first_call_started.set()
            await self.release_first_call.wait()
        return ModelTurn(response_id=f"resp-{call_index}", output_text=f"done-{call_index}", tool_calls=[])


async def test_base_agent_allows_reusable_input_item_customization(tmp_path: Path) -> None:
    """BaseAgent hooks should let future agents reuse the loop with custom input wiring."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()

    try:
        provider = FakeProvider([ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])])
        prompt_builder = PromptBuilder(SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([]))
        agent = PrefixedAgent(
            agent_config=AgentConfig(
                name="Prefixed",
                role="custom",
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
        )

        session_id = await agent.create_session()
        result = await agent.run(session_id, "hello")

        assert result.assistant_text == "ok"
        assert provider.calls[0]["input_items"][0]["content"][0]["text"] == "prefixed: hello"
    finally:
        await session_store.close()


async def test_base_agent_persists_and_sends_file_attachments(
    tmp_path: Path,
) -> None:
    """Dropped files should be saved, linked to the message, and sent to the LLM."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    source = tmp_path / "notes.txt"
    source.write_text("hello from file", encoding="utf-8")

    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        prompt_builder = PromptBuilder(
            SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
        )
        agent = BaseAgent(
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
            attachment_store=AttachmentStore(root=tmp_path, session_store=session_store),
        )

        session_id = await agent.create_session()
        await agent.run(session_id, f"please inspect {source}")

        messages = await session_store.list_messages(session_id)
        attachments = await session_store.list_message_attachments(messages[0].id)
        content = provider.calls[0]["input_items"][0]["content"]

        assert messages[0].content == "please inspect\n[File #1]"
        assert len(attachments) == 1
        assert (tmp_path / attachments[0].filepath).is_file()
        assert content[0] == {"type": "input_text", "text": "please inspect"}
        assert content[1]["type"] == "input_text"
        assert "notes.txt" in content[1]["text"]
        assert "hello from file" in content[1]["text"]
    finally:
        await session_store.close()


async def test_base_agent_sends_pdf_attachment_with_direct_reading_guidance(
    tmp_path: Path,
) -> None:
    """Current-turn PDFs should be sent as bytes with guidance to read them directly."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    source = tmp_path / "label.pdf"
    source.write_bytes(b"%PDF-1.4")

    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
            ),
            prompt_builder=PromptBuilder(
                SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
            ),
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=ToolRegistry([]),
            attachment_store=AttachmentStore(root=tmp_path, session_store=session_store),
        )

        session_id = await agent.create_session()
        await agent.run(session_id, f"read {source}")

        content = provider.calls[0]["input_items"][0]["content"]
        assert content[0] == {"type": "input_text", "text": "read"}
        assert content[1]["type"] == "input_text"
        assert "Inspect current-turn images/PDFs directly" in content[1]["text"]
        assert content[2]["type"] == "input_file"
        assert content[2]["filename"] == "label.pdf"
        assert content[2]["file_data"].startswith("data:application/pdf;base64,")
    finally:
        await session_store.close()


async def test_base_agent_allows_pdf_when_image_support_is_disabled(
    tmp_path: Path,
) -> None:
    """OpenRouter-style file/PDF input should not be blocked by image gating."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4")

    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
            ),
            prompt_builder=PromptBuilder(
                SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
            ),
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=ToolRegistry([]),
            attachment_store=AttachmentStore(root=tmp_path, session_store=session_store),
            supports_multimodal_attachments=False,
        )

        session_id = await agent.create_session()
        await agent.run(session_id, f"read {source}")

        content = provider.calls[0]["input_items"][0]["content"]
        assert content[-1]["type"] == "input_file"
    finally:
        await session_store.close()


async def test_base_agent_rolls_back_attachment_message_on_cancellation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cancelled attachment preparation should not leave ghost placeholders."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4")

    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        attachment_store = AttachmentStore(root=tmp_path, session_store=session_store)
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
            ),
            prompt_builder=PromptBuilder(
                SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
            ),
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=ToolRegistry([]),
            attachment_store=attachment_store,
        )
        session_id = await agent.create_session()

        async def cancel_save(*args: object, **kwargs: object) -> list[object]:
            raise asyncio.CancelledError

        monkeypatch.setattr(attachment_store, "save_pending", cancel_save)

        try:
            await agent.run(session_id, f"inspect {source}")
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("attachment cancellation was swallowed")

        assert await session_store.list_messages(session_id) == []
        assert await session_store.list_session_attachments(session_id) == []
        assert provider.calls == []
    finally:
        await session_store.close()


async def test_base_agent_rejects_oversized_attachment_before_persistence(
    tmp_path: Path,
) -> None:
    """Attachment validation should fail before copying files or writing messages."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    oversized = tmp_path / "large.pdf"
    with oversized.open("wb") as handle:
        handle.truncate((50 * 1024 * 1024) + 1)

    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
            ),
            prompt_builder=PromptBuilder(
                SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
            ),
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=ToolRegistry([]),
            attachment_store=AttachmentStore(root=tmp_path, session_store=session_store),
        )
        session_id = await agent.create_session()

        try:
            await agent.run(session_id, f"inspect {oversized}")
        except ValueError as exc:
            assert "50 MB" in str(exc)
        else:
            raise AssertionError("oversized attachment was accepted")

        assert await session_store.list_messages(session_id) == []
        assert await session_store.list_session_attachments(session_id) == []
        assert not (tmp_path / ".feather" / "attachments").exists()
        assert provider.calls == []
    finally:
        await session_store.close()


async def test_base_agent_rolls_back_provisional_attachment_message_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Copy/build failures should not leave placeholder-only messages behind."""

    class FakeMemoryService:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def index_attachment(self, record: object, *, content: str) -> None:
            self.calls.append({"record": record, "content": content})

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4")

    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        memory = FakeMemoryService()
        attachment_store = AttachmentStore(
            root=tmp_path,
            session_store=session_store,
            memory_service=memory,
        )
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
            ),
            prompt_builder=PromptBuilder(
                SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
            ),
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=ToolRegistry([]),
            attachment_store=attachment_store,
        )
        session_id = await agent.create_session()

        async def fail_build(*args: object, **kwargs: object) -> list[dict[str, str]]:
            raise ValueError("provider block build failed")

        monkeypatch.setattr(agent, "_build_input_items_with_attachments", fail_build)

        try:
            await agent.run(session_id, f"inspect {source}")
        except ValueError as exc:
            assert "provider block build failed" in str(exc)
        else:
            raise AssertionError("attachment failure was accepted")

        assert await session_store.list_messages(session_id) == []
        assert await session_store.list_session_attachments(session_id) == []
        assert not any((tmp_path / ".feather" / "attachments").rglob("*paper.pdf"))
        assert memory.calls == []
        assert provider.calls == []
    finally:
        await session_store.close()


async def test_base_agent_rejects_multimodal_when_provider_disallows_it(
    tmp_path: Path,
) -> None:
    """Provider capability gates should reject image/PDF before persistence."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
            ),
            prompt_builder=PromptBuilder(
                SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
            ),
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=ToolRegistry([]),
            attachment_store=AttachmentStore(root=tmp_path, session_store=session_store),
            supports_multimodal_attachments=False,
        )
        session_id = await agent.create_session()

        try:
            await agent.run(session_id, f"inspect {image}")
        except ValueError as exc:
            assert "does not support image attachments" in str(exc)
        else:
            raise AssertionError("multimodal attachment was accepted")

        assert await session_store.list_messages(session_id) == []
        assert provider.calls == []
    finally:
        await session_store.close()


async def test_stateless_history_replay_includes_prior_image_attachment(
    tmp_path: Path,
) -> None:
    """OpenRouter-style stateless replay should include bounded prior image bytes."""

    class StatelessProvider(FakeProvider):
        stateful = False

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    try:
        provider = StatelessProvider(
            [
                ModelTurn(response_id="resp-1", output_text="seen", tool_calls=[]),
                ModelTurn(response_id="resp-2", output_text="again", tool_calls=[]),
            ]
        )
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
            ),
            prompt_builder=PromptBuilder(
                SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
            ),
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=ToolRegistry([]),
            attachment_store=AttachmentStore(root=tmp_path, session_store=session_store),
        )

        session_id = await agent.create_session()
        await agent.run(session_id, f"inspect {image}")
        await agent.run(session_id, "what color was it?")

        replay_content = provider.calls[1]["input_items"][0]["content"]
        assert any(block.get("type") == "input_image" for block in replay_content)
        assert provider.calls[1]["input_items"][-1]["content"][0]["text"] == (
            "what color was it?"
        )
    finally:
        await session_store.close()


async def test_stateless_history_replay_skips_images_before_latest_compact(
    tmp_path: Path,
) -> None:
    """Compacted-away image bytes should not re-enter stateless replay context."""

    class StatelessProvider(FakeProvider):
        stateful = False

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    try:
        provider = StatelessProvider(
            [
                ModelTurn(response_id="resp-1", output_text="seen", tool_calls=[]),
                ModelTurn(response_id="resp-2", output_text="again", tool_calls=[]),
            ]
        )
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
            ),
            prompt_builder=PromptBuilder(
                SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
            ),
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=ToolRegistry([]),
            attachment_store=AttachmentStore(root=tmp_path, session_store=session_store),
        )

        session_id = await agent.create_session()
        await agent.run(session_id, f"inspect {image}")
        await session_store.add_message(
            session_id,
            MessageRole.ASSISTANT,
            "## Compacted state",
            is_compact=True,
        )
        await agent.run(session_id, "what color was it?")

        replay_content = provider.calls[1]["input_items"][0]["content"]
        assert not any(block.get("type") == "input_image" for block in replay_content)
    finally:
        await session_store.close()


async def test_base_agent_forwards_per_agent_reasoning_to_provider(tmp_path: Path) -> None:
    """An agent with reasoning set in config must pass it to provider.complete via request_config."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        prompt_builder = PromptBuilder(
            SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
        )
        agent = PrefixedAgent(
            agent_config=AgentConfig(
                name="Thinker",
                role="custom",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
                reasoning=ReasoningConfig(effort="high", summary="auto"),
            ),
            prompt_builder=prompt_builder,
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=ToolRegistry([]),
        )
        session_id = await agent.create_session()
        await agent.run(session_id, "hello")
        forwarded = provider.calls[0]["request_config"]
        assert isinstance(forwarded, ProviderRequestConfig)
        assert forwarded.reasoning is not None
        assert forwarded.reasoning.effort == "high"
        assert forwarded.reasoning.summary == "auto"
    finally:
        await session_store.close()


async def test_base_agent_forwards_per_agent_temperature_and_max_tokens(
    tmp_path: Path,
) -> None:
    """An agent with temperature / max_output_tokens overrides must pass them
    to provider.complete via :class:`ProviderRequestConfig`.

    This is the path that lets an agent override the provider's app-level
    sampling and budget without copying the full provider block; the
    provider then merges agent-level override on top of constructor-level
    defaults inside its translator.
    """

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        prompt_builder = PromptBuilder(
            SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
        )
        agent = PrefixedAgent(
            agent_config=AgentConfig(
                name="Tuned",
                role="custom",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
                temperature=0.2,
                max_output_tokens=4096,
            ),
            prompt_builder=prompt_builder,
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=ToolRegistry([]),
        )
        session_id = await agent.create_session()
        await agent.run(session_id, "hello")
        forwarded = provider.calls[0]["request_config"]
        assert isinstance(forwarded, ProviderRequestConfig)
        assert forwarded.temperature == 0.2
        assert forwarded.max_output_tokens == 4096
    finally:
        await session_store.close()


async def test_base_agent_omits_reasoning_when_agent_has_no_override(
    tmp_path: Path,
) -> None:
    """Agents without a `reasoning:` block must not force a request-level override."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        prompt_builder = PromptBuilder(
            SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
        )
        agent = PrefixedAgent(
            agent_config=AgentConfig(
                name="Plain",
                role="custom",
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
        )
        session_id = await agent.create_session()
        await agent.run(session_id, "hello")
        # No agent-level override → provider call's reasoning stays None,
        # so the provider's own default reasoning continues to apply.
        # request_config itself is always constructed so it can carry the
        # per-turn trace context (consumed by tracing-aware providers).
        request_config = provider.calls[0]["request_config"]
        assert request_config is not None
        assert request_config.reasoning is None
        assert request_config.mcp_servers == ()
        assert request_config.trace_context is not None
        assert request_config.trace_context.session_id == session_id
        assert request_config.trace_context.agent_name == "Plain"
    finally:
        await session_store.close()


async def test_base_agent_exposes_mcp_proxy_only_after_session_activation(
    tmp_path: Path,
) -> None:
    """MCP proxy tools should be session-scoped, not listed on every turn."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        server = MCPServerConfig(
            label="docs",
            server_url="https://example.test/mcp",
            providers=("openrouter",),
        )
        provider = FakeProvider(
            [
                ModelTurn(response_id="resp-1", output_text="one", tool_calls=[]),
                ModelTurn(response_id="resp-2", output_text="two", tool_calls=[]),
            ]
        )
        tool_registry = ToolRegistry([MCPProxyTool(server)])
        prompt_builder = PromptBuilder(
            SkillCatalog(tmp_path / ".feather" / "skills"), tool_registry
        )
        agent = PrefixedAgent(
            agent_config=AgentConfig(
                name="Plain",
                role="custom",
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
            tool_registry=tool_registry,
            provider_name="openrouter",
            mcp_servers=(server,),
        )
        session_id = await agent.create_session()
        await agent.run(session_id, "first")
        await session_store.append_active_mcp_server(session_id, "docs")
        await agent.run(session_id, "second")

        assert [tool["name"] for tool in provider.calls[0]["tools"]] == []
        assert [tool["name"] for tool in provider.calls[1]["tools"]] == [
            "mcp_docs"
        ]
        assert "`mcp_docs`" not in provider.calls[0]["instructions"]
        assert "`mcp_docs`" in provider.calls[1]["instructions"]
    finally:
        await session_store.close()


async def test_base_agent_rejects_inactive_hidden_mcp_proxy_calls(
    tmp_path: Path,
) -> None:
    """Hidden MCP proxies must not execute before session registration."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        server = MCPServerConfig(
            label="docs",
            server_url="https://example.test/mcp",
            providers=("openrouter",),
        )
        provider = FakeProvider(
            [
                ModelTurn(
                    response_id="resp-1",
                    output_text="",
                    tool_calls=[
                        ToolCall(
                            call_id="call-1",
                            name="mcp_docs",
                            arguments={
                                "action": "list_tools",
                                "tool_name": None,
                                "arguments": {},
                            },
                        )
                    ],
                ),
                ModelTurn(response_id="resp-2", output_text="done", tool_calls=[]),
            ]
        )
        tool_registry = ToolRegistry([MCPProxyTool(server)])
        prompt_builder = PromptBuilder(
            SkillCatalog(tmp_path / ".feather" / "skills"), tool_registry
        )
        agent = PrefixedAgent(
            agent_config=AgentConfig(
                name="Plain",
                role="custom",
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
            tool_registry=tool_registry,
            provider_name="openrouter",
            mcp_servers=(server,),
        )
        session_id = await agent.create_session()

        result = await agent.run(session_id, "try proxy")

        assert result.assistant_text == "done"
        assert "not available in this session" in provider.calls[1]["input_items"][0][
            "output"
        ]
    finally:
        await session_store.close()


async def test_base_agent_closes_session_mcp_clients_after_run(
    tmp_path: Path,
) -> None:
    """Session-scoped MCP clients should be cleaned up when a run returns."""

    class TrackingMCPManager:
        def __init__(self) -> None:
            self.closed_sessions: list[str] = []

        async def close_session(self, session_id: str) -> None:
            self.closed_sessions.append(session_id)

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    manager = TrackingMCPManager()
    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        prompt_builder = PromptBuilder(
            SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
        )
        agent = PrefixedAgent(
            agent_config=AgentConfig(
                name="Plain",
                role="custom",
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
            mcp_client_manager=manager,  # type: ignore[arg-type]
        )
        session_id = await agent.create_session()

        await agent.run(session_id, "hello")

        assert manager.closed_sessions == [session_id]
    finally:
        await session_store.close()


async def test_base_agent_serializes_concurrent_runs_per_session(tmp_path: Path) -> None:
    """Concurrent runs against one session should be serialized through the shared lock."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()

    try:
        provider = BlockingProvider()
        prompt_builder = PromptBuilder(SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([]))
        agent = PrefixedAgent(
            agent_config=AgentConfig(
                name="Prefixed",
                role="custom",
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
        )

        session_id = await agent.create_session()
        first_task = asyncio.create_task(agent.run(session_id, "first"))
        await provider.first_call_started.wait()

        second_task = asyncio.create_task(agent.run(session_id, "second"))
        await asyncio.sleep(0.05)

        assert len(provider.calls) == 1

        provider.release_first_call.set()
        first = await first_task
        second = await second_task

        assert first.assistant_text == "done-1"
        assert second.assistant_text == "done-2"
        assert len(provider.calls) == 2
        assert provider.calls[1]["previous_response_id"] == "resp-1"
    finally:
        await session_store.close()


async def test_base_agent_emits_usage_updated_event_with_ratio(tmp_path: Path) -> None:
    """BaseAgent should emit usage_updated (ratio) after each provider turn that reports tokens."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()

    try:
        provider = FakeProvider(
            [
                ModelTurn(
                    response_id="resp-1",
                    output_text="ok",
                    tool_calls=[],
                    usage={"input_tokens": 40_000},
                )
            ]
        )
        prompt_builder = PromptBuilder(
            SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
        )
        compactor = ContextCompactor(
            config=CompactionConfig(
                enabled=True,
                trigger_ratio=0.9,
                context_window_tokens=100_000,
            ),
            provider=provider,
            session_store=session_store,
        )
        agent = BaseAgent(
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
            compactor=compactor,
        )

        events: list[RuntimeEvent] = []
        session_id = await agent.create_session()
        await agent.run(session_id, "hello", events.append)

        usage_events = [e for e in events if e.kind == "usage_updated"]
        assert len(usage_events) == 1
        assert usage_events[0].payload == {"usage_ratio": 0.4}
    finally:
        await session_store.close()


async def test_base_agent_executes_parallel_tool_calls_concurrently(
    tmp_path: Path,
) -> None:
    """Multiple model-emitted tool calls should run concurrently, then persist in order."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    class GateTool(BaseTool):
        name = "gate"
        description = "Test tool that blocks until released."
        parameters_schema = {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
            "additionalProperties": False,
        }

        def __init__(self) -> None:
            self.started: list[str] = []
            self.all_started = asyncio.Event()
            self.release = asyncio.Event()

        async def execute(
            self,
            arguments: dict[str, Any],
            context: ToolExecutionContext,
        ) -> ToolExecutionResult:
            label = str(arguments["label"])
            self.started.append(label)
            if len(self.started) == 2:
                self.all_started.set()
            await self.release.wait()
            return ToolExecutionResult(output=f"done-{label}")

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    gate = GateTool()
    provider = FakeProvider(
        [
            ModelTurn(
                response_id="resp-1",
                output_text="",
                tool_calls=[
                    ToolCall(call_id="call-a", name="gate", arguments={"label": "a"}),
                    ToolCall(call_id="call-b", name="gate", arguments={"label": "b"}),
                ],
            ),
            ModelTurn(response_id="resp-2", output_text="done", tool_calls=[]),
        ]
    )

    try:
        tool_registry = ToolRegistry([gate])
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=["gate"],
            ),
            prompt_builder=PromptBuilder(
                SkillCatalog(tmp_path / ".feather" / "skills"), tool_registry
            ),
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=tool_registry,
        )

        session_id = await agent.create_session()
        run_task = asyncio.create_task(agent.run(session_id, "run gates"))
        await asyncio.wait_for(gate.all_started.wait(), timeout=0.5)

        assert gate.started == ["a", "b"]
        assert not run_task.done()
        gate.release.set()
        result = await run_task

        assert result.status == AgentOutcome.COMPLETED
        assert provider.calls[1]["input_items"] == [
            {"type": "function_call_output", "call_id": "call-a", "output": "done-a"},
            {"type": "function_call_output", "call_id": "call-b", "output": "done-b"},
        ]
    finally:
        gate.release.set()
        await session_store.close()


async def test_stateless_agent_persists_tool_call_context_when_pausing(
    tmp_path: Path,
) -> None:
    """Stateless resumes need the assistant tool call plus the tool output."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    class StatelessProvider(FakeProvider):
        stateful = False

    class QuestionTool(BaseTool):
        name = "question"
        description = "Ask a test question."
        parameters_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

        async def execute(
            self,
            arguments: dict[str, Any],
            context: ToolExecutionContext,
        ) -> ToolExecutionResult:
            return ToolExecutionResult(
                output="User input required: choose",
                await_user_question="choose",
            )

    provider = StatelessProvider(
        [
            ModelTurn(
                response_id="resp-1",
                output_text="",
                tool_calls=[
                    ToolCall(call_id="call-q", name="question", arguments={}),
                ],
            ),
            ModelTurn(response_id="resp-2", output_text="done", tool_calls=[]),
        ]
    )
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()

    try:
        tool_registry = ToolRegistry([QuestionTool()])
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=["question"],
            ),
            prompt_builder=PromptBuilder(
                SkillCatalog(tmp_path / ".feather" / "skills"), tool_registry
            ),
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=tool_registry,
        )

        session_id = await agent.create_session()
        result = await agent.run(session_id, "start")
        session = await session_store.get_session(session_id)

        assert result.status == AgentOutcome.AWAITING_USER
        assert session.pending_inputs == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "start"}],
            },
            {
                "type": "function_call",
                "call_id": "call-q",
                "name": "question",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call-q",
                "output": "User input required: choose",
            },
        ]

        resumed = await agent.run(session_id, "answer")
        assert resumed.status == AgentOutcome.COMPLETED
        assert [item["type"] for item in provider.calls[1]["input_items"]] == [
            "message",
            "function_call",
            "function_call_output",
            "message",
        ]
        assert provider.calls[1]["input_items"][0]["content"][0]["text"] == "start"
        assert provider.calls[1]["input_items"][3]["content"][0]["text"] == "answer"
        assert not any(
            "Session context to continue from" in str(item)
            for item in provider.calls[1]["input_items"]
        )
    finally:
        await session_store.close()


async def test_base_agent_caps_parallel_tool_execution(
    tmp_path: Path,
) -> None:
    """Parallel tool fanout should be bounded to avoid resource spikes."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    class CountingTool(BaseTool):
        name = "counting"
        description = "Count active executions."
        parameters_schema = {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
            "additionalProperties": False,
        }

        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.started = 0

        async def execute(
            self,
            arguments: dict[str, Any],
            context: ToolExecutionContext,
        ) -> ToolExecutionResult:
            self.started += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return ToolExecutionResult(output=f"done-{arguments['label']}")

    tool = CountingTool()
    calls = [
        ToolCall(call_id=f"call-{index}", name="counting", arguments={"label": str(index)})
        for index in range(10)
    ]
    provider = FakeProvider(
        [
            ModelTurn(response_id="resp-1", output_text="", tool_calls=calls),
            ModelTurn(response_id="resp-2", output_text="done", tool_calls=[]),
        ]
    )
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()

    try:
        tool_registry = ToolRegistry([tool])
        agent = BaseAgent(
            agent_config=AgentConfig(
                name="Lead",
                role="lead",
                personality="Direct",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=["counting"],
            ),
            prompt_builder=PromptBuilder(
                SkillCatalog(tmp_path / ".feather" / "skills"), tool_registry
            ),
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=tool_registry,
            max_parallel_tool_calls=3,
        )

        session_id = await agent.create_session()
        result = await agent.run(session_id, "run many")

        assert result.status == AgentOutcome.COMPLETED
        assert tool.started == 10
        assert tool.max_active <= 3
    finally:
        await session_store.close()


def _make_agent(
    *,
    tmp_path: Path,
    provider: BaseLLMProvider,
    session_store: SessionStore,
    input_queue: UserInputQueue | None,
) -> BaseAgent:
    """Build a minimal BaseAgent for queue-drain tests."""

    prompt_builder = PromptBuilder(
        SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
    )
    return BaseAgent(
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
        input_queue=input_queue,
    )


async def test_run_loop_keeps_alive_when_message_arrives_mid_turn(
    tmp_path: Path,
) -> None:
    """When the user enqueues a message while the model is answering, the
    loop must not exit on the first "no tool calls" — it must drain and
    continue so the agent can respond to the queued idea.
    """

    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    queue = UserInputQueue()

    class MidTurnInjectingProvider(BaseLLMProvider):
        """On turn 1 enqueue a message to the queue, simulating a user typing
        while the model is thinking. Turn 2 returns COMPLETED normally."""

        def __init__(self, q: UserInputQueue, session_id_holder: dict) -> None:
            self._q = q
            self._session_holder = session_id_holder
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
            self.calls.append({"input_items": input_items})
            n = len(self.calls)
            if n == 1:
                await self._q.enqueue(
                    self._session_holder["session_id"], "follow-up idea"
                )
                return ModelTurn(
                    response_id="resp-1", output_text="first", tool_calls=[]
                )
            return ModelTurn(
                response_id=f"resp-{n}", output_text="second", tool_calls=[]
            )

    try:
        holder: dict = {}
        provider = MidTurnInjectingProvider(queue, holder)
        agent = _make_agent(
            tmp_path=tmp_path,
            provider=provider,
            session_store=session_store,
            input_queue=queue,
        )
        session_id = await agent.create_session()
        holder["session_id"] = session_id

        events: list[RuntimeEvent] = []
        result = await agent.run(session_id, "hello", events.append)

        assert len(provider.calls) == 2, "loop should have kept going"
        # Second call must carry the queued user message.
        second_items = provider.calls[1]["input_items"]
        assert any(
            item.get("type") == "message"
            and item.get("role") == "user"
            and item["content"][0]["text"] == "follow-up idea"
            for item in second_items
        )
        assert result.status == AgentOutcome.COMPLETED
        assert result.assistant_text == "second"

        messages = await session_store.list_active_messages(session_id)
        user_texts = [m.content for m in messages if m.role == MessageRole.USER]
        assert "follow-up idea" in user_texts

        injected_events = [e for e in events if e.kind == "user_message_injected"]
        assert len(injected_events) == 1
        assert injected_events[0].text == "follow-up idea"
    finally:
        await session_store.close()


async def test_run_loop_drains_at_top_of_iteration(tmp_path: Path) -> None:
    """A message enqueued *before* run starts must be picked up on iteration 0."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    queue = UserInputQueue()

    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        agent = _make_agent(
            tmp_path=tmp_path,
            provider=provider,
            session_store=session_store,
            input_queue=queue,
        )
        session_id = await agent.create_session()

        # Enqueue BEFORE run starts.
        await queue.enqueue(session_id, "pre-run note")

        result = await agent.run(session_id, "first user line")
        assert result.status == AgentOutcome.COMPLETED

        # First provider call should see BOTH the original message and the
        # queued note as user input items.
        first_items = provider.calls[0]["input_items"]
        texts = [
            it["content"][0]["text"]
            for it in first_items
            if it.get("type") == "message" and it.get("role") == "user"
        ]
        assert "first user line" in texts
        assert "pre-run note" in texts
    finally:
        await session_store.close()


async def test_run_loop_without_queue_is_unaffected(tmp_path: Path) -> None:
    """Agents constructed without an input_queue must behave exactly as before."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()

    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        agent = _make_agent(
            tmp_path=tmp_path,
            provider=provider,
            session_store=session_store,
            input_queue=None,
        )
        session_id = await agent.create_session()
        result = await agent.run(session_id, "hi")
        assert result.status == AgentOutcome.COMPLETED
        assert len(provider.calls) == 1
    finally:
        await session_store.close()


async def test_keep_alive_is_bounded_to_prevent_infinite_loop(
    tmp_path: Path,
) -> None:
    """A queue that keeps producing messages every turn must not trap the
    agent forever; keep-alive must fall through to COMPLETED after a bound.
    """

    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    queue = UserInputQueue()

    class RelentlessProvider(BaseLLMProvider):
        """Enqueue a fresh message on every turn so the keep-alive branch
        would loop forever without a bound."""

        def __init__(self, q: UserInputQueue, session_holder: dict) -> None:
            self._q = q
            self._holder = session_holder
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
            self.calls.append({})
            n = len(self.calls)
            await self._q.enqueue(self._holder["sid"], f"extra-{n}")
            return ModelTurn(
                response_id=f"resp-{n}", output_text=f"t{n}", tool_calls=[]
            )

    try:
        holder: dict = {}
        provider = RelentlessProvider(queue, holder)
        agent = _make_agent(
            tmp_path=tmp_path,
            provider=provider,
            session_store=session_store,
            input_queue=queue,
        )
        session_id = await agent.create_session()
        holder["sid"] = session_id

        result = await agent.run(session_id, "first")
        # Bound is 3 keep-alive injections → turn 1 (original) + 3 keep-alive turns.
        assert len(provider.calls) == 4
        assert result.status == AgentOutcome.COMPLETED
        # The last message enqueued by the final turn stays in the queue
        # for the next .run() to pick up — nothing is silently lost.
        leftover = await queue.peek(session_id)
        assert leftover == ("extra-4",)
    finally:
        await session_store.close()


async def test_base_agent_passes_user_profile_into_prompt(tmp_path: Path) -> None:
    """When a profile store is wired, its rendered contents reach the provider prompt."""

    from feather.profile import UserProfileStore

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    profile_store = UserProfileStore(tmp_path / ".feather" / "user.md")
    await profile_store.create("name", "TimUser")

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        prompt_builder = PromptBuilder(
            SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
        )
        agent = BaseAgent(
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
            profile_store=profile_store,
        )
        session_id = await agent.create_session()
        await agent.run(session_id, "hello")
        instructions = provider.calls[0]["instructions"]
        assert "<user_profile>" in instructions
        assert "name: TimUser" in instructions
    finally:
        await session_store.close()


async def test_drain_failure_does_not_kill_turn(
    tmp_path: Path, caplog
) -> None:
    """An exception during queue.drain is logged but does not crash the run."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)

    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()

    class ExplodingQueue(UserInputQueue):
        async def drain(self, session_id: str) -> list[str]:
            raise RuntimeError("boom")

    try:
        provider = FakeProvider(
            [ModelTurn(response_id="resp-1", output_text="ok", tool_calls=[])]
        )
        agent = _make_agent(
            tmp_path=tmp_path,
            provider=provider,
            session_store=session_store,
            input_queue=ExplodingQueue(),
        )
        session_id = await agent.create_session()

        import logging as _logging

        caplog.set_level(_logging.ERROR)
        result = await agent.run(session_id, "hi")
        assert result.status == AgentOutcome.COMPLETED
        assert any(
            "user_input_queue.drain_error" in rec.message for rec in caplog.records
        )
    finally:
        await session_store.close()
