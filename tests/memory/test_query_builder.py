"""Tests for MemoryQueryBuilder."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from feather.memory.config import MemoryOperationModelConfig
from feather.memory.models import QueryDecision
from feather.memory.prompts.query_prompt import QUERY_PROMPT
from feather.memory.query_builder import MemoryQueryBuilder
from feather.models import (
    MessageRole,
    ModelTurn,
    ProviderRequestConfig,
    SessionMessage,
)
from feather.providers.base import BaseLLMProvider


class _FakeProvider(BaseLLMProvider):
    def __init__(self, *, output_text: str | None = None, exc: BaseException | None = None) -> None:
        self._output_text = output_text
        self._exc = exc
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
        if self._exc is not None:
            raise self._exc
        assert self._output_text is not None
        return ModelTurn(response_id="r", output_text=self._output_text)


def _recent_messages() -> list[SessionMessage]:
    msgs: list[SessionMessage] = []
    for i, (role, content) in enumerate(
        [
            (MessageRole.USER, "earlier thing"),
            (MessageRole.ASSISTANT, "earlier response"),
            (MessageRole.USER, "and the other one?"),
        ]
    ):
        msgs.append(
            SessionMessage(
                id=str(uuid4()),
                session_id="sess",
                role=role,
                content=content,
                file_ref=None,
                is_compact=False,
                sequence=i + 1,
                created_at="2026-04-16T00:00:00Z",
            )
        )
    return msgs


# Happy path ------------------------------------------------------------------


async def test_query_builder_returns_query_decision_on_successful_response() -> None:
    provider = _FakeProvider(
        output_text=json.dumps(
            {
                "query": "the user's second preference for X",
                "should_skip": False,
                "reasoning": "pronoun resolution",
            }
        )
    )
    qb = MemoryQueryBuilder(
        provider=provider, prompt=QUERY_PROMPT, cfg=MemoryOperationModelConfig()
    )
    decision = await qb.build(
        _recent_messages(), latest_user_text="and the other one?", agent_model="gpt-5-mini"
    )
    assert isinstance(decision, QueryDecision)
    assert decision.query == "the user's second preference for X"
    assert decision.should_skip is False
    assert decision.reasoning == "pronoun resolution"


async def test_query_builder_returns_skip_decision_when_llm_says_skip() -> None:
    provider = _FakeProvider(
        output_text=json.dumps(
            {"query": "", "should_skip": True, "reasoning": "greeting"}
        )
    )
    qb = MemoryQueryBuilder(
        provider=provider, prompt=QUERY_PROMPT, cfg=MemoryOperationModelConfig()
    )
    decision = await qb.build(
        _recent_messages(), latest_user_text="hi", agent_model="gpt-5-mini"
    )
    assert decision.should_skip is True
    assert decision.query == ""


async def test_query_builder_uses_query_prompt_and_response_schema() -> None:
    provider = _FakeProvider(
        output_text=json.dumps({"query": "x", "should_skip": False, "reasoning": "r"})
    )
    qb = MemoryQueryBuilder(
        provider=provider, prompt=QUERY_PROMPT, cfg=MemoryOperationModelConfig()
    )
    await qb.build(
        _recent_messages(), latest_user_text="q", agent_model="gpt-5-mini"
    )
    call = provider.calls[0]
    assert call["instructions"] == QUERY_PROMPT
    assert call["tools"] == []
    assert call["previous_response_id"] is None
    assert call["request_config"].response_schema.__name__ == "QueryBuildResponse"


async def test_query_builder_inherits_agent_model_when_override_is_none() -> None:
    provider = _FakeProvider(
        output_text=json.dumps({"query": "x", "should_skip": False, "reasoning": "r"})
    )
    qb = MemoryQueryBuilder(
        provider=provider,
        prompt=QUERY_PROMPT,
        cfg=MemoryOperationModelConfig(model=None),
    )
    await qb.build(
        _recent_messages(), latest_user_text="q", agent_model="gpt-5-mini"
    )
    assert provider.calls[0]["request_config"].model == "gpt-5-mini"


async def test_query_builder_uses_override_model_when_configured() -> None:
    provider = _FakeProvider(
        output_text=json.dumps({"query": "x", "should_skip": False, "reasoning": "r"})
    )
    qb = MemoryQueryBuilder(
        provider=provider,
        prompt=QUERY_PROMPT,
        cfg=MemoryOperationModelConfig(model="gpt-4.1-mini"),
    )
    await qb.build(
        _recent_messages(), latest_user_text="q", agent_model="gpt-5-mini"
    )
    assert provider.calls[0]["request_config"].model == "gpt-4.1-mini"


async def test_query_builder_renders_recent_conversation_for_llm() -> None:
    provider = _FakeProvider(
        output_text=json.dumps({"query": "x", "should_skip": False, "reasoning": "r"})
    )
    qb = MemoryQueryBuilder(
        provider=provider, prompt=QUERY_PROMPT, cfg=MemoryOperationModelConfig()
    )
    await qb.build(
        _recent_messages(), latest_user_text="and the other one?", agent_model="gpt-5-mini"
    )
    rendered = provider.calls[0]["input_items"][0]["content"]
    assert "earlier thing" in rendered
    assert "earlier response" in rendered
    assert "and the other one?" in rendered


# Fail-open paths -------------------------------------------------------------


async def test_query_builder_falls_back_to_latest_user_text_on_exception() -> None:
    provider = _FakeProvider(exc=RuntimeError("boom"))
    qb = MemoryQueryBuilder(
        provider=provider, prompt=QUERY_PROMPT, cfg=MemoryOperationModelConfig()
    )
    decision = await qb.build(
        _recent_messages(), latest_user_text="raw text", agent_model="gpt-5-mini"
    )
    assert decision.query == "raw text"
    assert decision.should_skip is False
    assert "fallback" in decision.reasoning.lower()


async def test_query_builder_falls_back_on_malformed_json() -> None:
    provider = _FakeProvider(output_text="not valid json")
    qb = MemoryQueryBuilder(
        provider=provider, prompt=QUERY_PROMPT, cfg=MemoryOperationModelConfig()
    )
    decision = await qb.build(
        _recent_messages(), latest_user_text="raw", agent_model="gpt-5-mini"
    )
    assert decision.query == "raw"
    assert decision.should_skip is False
