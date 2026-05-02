"""Tests for the recall_memory tool."""

from __future__ import annotations

from typing import Any, Sequence
from uuid import uuid4

from feather.memory.config import MemoryRetrievalConfig
from feather.memory.enums import EmbedType, MemoryOwner
from feather.memory.models import (
    MemoryPointPayload,
    MemorySearchResult,
)
from feather.memory.reader import MemoryReader
from feather.models import ToolExecutionContext
from feather.tools.recall_memory_tool import RecallMemoryTool


_CTX = ToolExecutionContext(session_id="sess-test", agent_name="Lead")


class _FakeReader(MemoryReader):
    def __init__(self, results: list[MemorySearchResult]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def augment_instructions(self, **_kw: object) -> str:  # pragma: no cover
        return ""

    async def recall(  # type: ignore[override]
        self,
        *,
        query: str,
        top_k: int,
        score_threshold: float,
        session_id: str | None,
        owner: MemoryOwner = MemoryOwner.USER,
    ) -> list[MemorySearchResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "session_id": session_id,
                "owner": owner,
            }
        )
        return list(self._results)


def _result(content: str, score: float = 0.8) -> MemorySearchResult:
    payload = MemoryPointPayload(
        type=EmbedType.MEMORY,
        memory_owner=MemoryOwner.USER,
        content=content,
        purpose="for tests",
        group_id=uuid4(),
        session_id=uuid4(),
        start_message_id=uuid4(),
        end_message_id=uuid4(),
    )
    return MemorySearchResult(payload=payload, score=score)


def _tool(reader: MemoryReader, *, session_id: str | None = "sess-x") -> RecallMemoryTool:
    return RecallMemoryTool(
        reader=reader,
        cfg=MemoryRetrievalConfig(top_k_tool=4, score_threshold=0.4),
        session_id_resolver=lambda: session_id,
    )


# Schema / metadata ----------------------------------------------------------


def test_tool_metadata() -> None:
    tool = _tool(_FakeReader([]))
    assert tool.name == "recall_memory"
    assert "long-term memory" in tool.description.lower()
    # OpenAI strict mode requires every property to appear in `required`;
    # optional behavior is encoded via `["T", "null"]` unions.
    assert set(tool.parameters_schema["required"]) == set(
        tool.parameters_schema["properties"].keys()
    )
    assert tool.parameters_schema["additionalProperties"] is False


# Execution ------------------------------------------------------------------


async def test_recall_uses_defaults_from_retrieval_config() -> None:
    reader = _FakeReader([_result("the user prefers Python", 0.7)])
    tool = _tool(reader)
    out = await tool.execute({"query": "the user's languages"}, _CTX)
    assert "the user prefers Python" in out.output
    call = reader.calls[0]
    assert call["top_k"] == 4
    assert call["score_threshold"] == 0.4
    # session_scoped defaults to False → no session_id passed.
    assert call["session_id"] is None


async def test_recall_session_scoped_passes_resolver_session_id() -> None:
    reader = _FakeReader([_result("x")])
    tool = _tool(reader, session_id="sess-42")
    await tool.execute({"query": "x", "session_scoped": True}, _CTX)
    assert reader.calls[0]["session_id"] == "sess-42"


async def test_recall_explicit_top_k_and_threshold_overrides_defaults() -> None:
    reader = _FakeReader([_result("x")])
    tool = _tool(reader)
    await tool.execute({"query": "x", "top_k": 7, "score_threshold": 0.9}, _CTX)
    assert reader.calls[0]["top_k"] == 7
    assert reader.calls[0]["score_threshold"] == 0.9


async def test_recall_returns_no_results_message_when_empty() -> None:
    tool = _tool(_FakeReader([]))
    out = await tool.execute({"query": "nothing"}, _CTX)
    assert "no memories" in out.output.lower()


async def test_recall_renders_score_and_content_for_each_hit() -> None:
    reader = _FakeReader(
        [
            _result("first memory", 0.91),
            _result("second memory", 0.55),
        ]
    )
    tool = _tool(reader)
    out = await tool.execute({"query": "x"}, _CTX)
    assert "first memory" in out.output
    assert "second memory" in out.output
    assert "0.91" in out.output
    assert "0.55" in out.output


async def test_recall_strips_query_whitespace() -> None:
    reader = _FakeReader([_result("x")])
    tool = _tool(reader)
    await tool.execute({"query": "   spaced query   "}, _CTX)
    assert reader.calls[0]["query"] == "spaced query"
