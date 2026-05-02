"""Tests for MemoryExtractor."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from feather.memory.config import MemoryOperationModelConfig
from feather.memory.extractor import MemoryExtractor
from feather.memory.models import AtomicMemory, MemoryWindow
from feather.memory.prompts.extraction_prompt import EXTRACTION_PROMPT
from feather.models import (
    MessageRole,
    ModelTurn,
    ProviderRequestConfig,
    SessionMessage,
)
from feather.providers.base import BaseLLMProvider


class _FakeProvider(BaseLLMProvider):
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text
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
        return ModelTurn(response_id="r", output_text=self._output_text)


def _msg(role: MessageRole, content: str, seq: int) -> SessionMessage:
    return SessionMessage(
        id=str(uuid4()),
        session_id="sess",
        role=role,
        content=content,
        file_ref=None,
        is_compact=False,
        sequence=seq,
        created_at="2026-04-16T00:00:00Z",
    )


def _window() -> MemoryWindow:
    msgs = [
        _msg(MessageRole.USER, "I'm a data scientist", 1),
        _msg(MessageRole.ASSISTANT, "Noted.", 2),
        _msg(MessageRole.USER, "I prefer Python for async", 3),
        _msg(MessageRole.ASSISTANT, "OK.", 4),
    ]
    return MemoryWindow(
        session_id="sess",
        start_message_id=msgs[0].id,
        end_message_id=msgs[-1].id,
        messages=msgs,
    )


# Happy path ------------------------------------------------------------------


async def test_extractor_returns_atomic_memories_from_structured_json() -> None:
    """The extractor parses ExtractionResponse JSON into AtomicMemory dataclasses."""
    provider = _FakeProvider(
        json.dumps(
            {
                "memories": [
                    {
                        "who": "the user",
                        "what": "is a data scientist",
                        "when": "ongoing",
                        "where": "unspecified",
                        "why": "unspecified",
                        "how": "unspecified",
                        "purpose": "tailor technical depth",
                        "content": "the user is a data scientist",
                    },
                    {
                        "who": "the user",
                        "what": "prefers Python for async work",
                        "when": "ongoing",
                        "where": "unspecified",
                        "why": "productivity",
                        "how": "async-first",
                        "purpose": "pick default library",
                        "content": "the user prefers Python for async work",
                    },
                ]
            }
        )
    )
    extractor = MemoryExtractor(
        provider=provider, prompt=EXTRACTION_PROMPT, cfg=MemoryOperationModelConfig()
    )

    out = await extractor.extract(_window(), agent_model="gpt-5-mini")

    assert len(out) == 2
    assert all(isinstance(m, AtomicMemory) for m in out)
    assert out[0].content == "the user is a data scientist"
    assert out[1].content == "the user prefers Python for async work"


async def test_extractor_sends_extraction_prompt_and_uses_structured_output() -> None:
    provider = _FakeProvider(json.dumps({"memories": []}))
    extractor = MemoryExtractor(
        provider=provider, prompt=EXTRACTION_PROMPT, cfg=MemoryOperationModelConfig()
    )
    await extractor.extract(_window(), agent_model="gpt-5-mini")
    call = provider.calls[0]
    assert call["instructions"] == EXTRACTION_PROMPT
    assert call["tools"] == []
    assert call["previous_response_id"] is None
    rc: ProviderRequestConfig = call["request_config"]
    assert rc.response_schema is not None
    assert rc.response_schema.__name__ == "ExtractionResponse"


async def test_extractor_inherits_agent_model_when_cfg_model_is_none() -> None:
    provider = _FakeProvider(json.dumps({"memories": []}))
    extractor = MemoryExtractor(
        provider=provider,
        prompt=EXTRACTION_PROMPT,
        cfg=MemoryOperationModelConfig(model=None),
    )
    await extractor.extract(_window(), agent_model="gpt-5-mini")
    assert provider.calls[0]["request_config"].model == "gpt-5-mini"


async def test_extractor_uses_override_model_when_configured() -> None:
    provider = _FakeProvider(json.dumps({"memories": []}))
    extractor = MemoryExtractor(
        provider=provider,
        prompt=EXTRACTION_PROMPT,
        cfg=MemoryOperationModelConfig(model="gpt-4.1-mini"),
    )
    await extractor.extract(_window(), agent_model="gpt-5-mini")
    assert provider.calls[0]["request_config"].model == "gpt-4.1-mini"


async def test_extractor_returns_empty_list_on_empty_response() -> None:
    provider = _FakeProvider(json.dumps({"memories": []}))
    extractor = MemoryExtractor(
        provider=provider, prompt=EXTRACTION_PROMPT, cfg=MemoryOperationModelConfig()
    )
    out = await extractor.extract(_window(), agent_model="gpt-5-mini")
    assert out == []


# Malformed input -------------------------------------------------------------


async def test_extractor_raises_validation_error_on_malformed_json() -> None:
    provider = _FakeProvider("this is not json")
    extractor = MemoryExtractor(
        provider=provider, prompt=EXTRACTION_PROMPT, cfg=MemoryOperationModelConfig()
    )
    with pytest.raises((ValidationError, ValueError)):
        await extractor.extract(_window(), agent_model="gpt-5-mini")


async def test_extractor_renders_transcript_with_role_prefixes() -> None:
    """The rendered transcript passed to the LLM is role-prefixed and preserves message order."""
    provider = _FakeProvider(json.dumps({"memories": []}))
    extractor = MemoryExtractor(
        provider=provider, prompt=EXTRACTION_PROMPT, cfg=MemoryOperationModelConfig()
    )
    await extractor.extract(_window(), agent_model="gpt-5-mini")
    rendered = provider.calls[0]["input_items"][0]["content"]
    assert "user: I'm a data scientist" in rendered
    assert "assistant: Noted." in rendered
    # Order preserved
    assert rendered.index("data scientist") < rendered.index("prefer Python")
