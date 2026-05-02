"""Tests for the Parallel AI web_fetch tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from feather.models import ParallelConfig, ToolExecutionContext
from feather.providers.parallel_client import (
    ParallelExtractHit,
    ParallelExtractResponse,
)
from feather.storage.tool_output_store import ToolOutputStore
from feather.tools.parallel_extract_tool import ParallelExtractTool


_CTX = ToolExecutionContext(session_id="sess-test", agent_name="Lead")


class _FakeParallelClient:
    """Fake client that records extract invocations."""

    def __init__(self, response: ParallelExtractResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []
        self.raise_on_call: Exception | None = None

    async def extract(self, **kwargs: Any) -> ParallelExtractResponse:
        self.calls.append(kwargs)
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return self._response


def _make_tool(
    client: _FakeParallelClient,
    tmp_path: Path,
    *,
    inline_full_content_threshold: int = 4000,
) -> ParallelExtractTool:
    config = ParallelConfig(
        api_key_env="UNUSED",
        inline_full_content_threshold=inline_full_content_threshold,
    )
    store = ToolOutputStore(tmp_path, ".feather/tmp")
    return ParallelExtractTool(client, config, store)  # type: ignore[arg-type]


def test_parallel_extract_schema_is_strict_compatible(tmp_path: Path) -> None:
    """All tool properties must be required for strict tool calling."""

    tool = _make_tool(
        _FakeParallelClient(ParallelExtractResponse(results=[], errors=[])),
        tmp_path,
    )
    schema = tool.parameters_schema
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == sorted(schema["properties"].keys())


async def test_parallel_extract_excerpts_mode_is_default_and_returns_inline(
    tmp_path: Path,
) -> None:
    """Default mode should be excerpts and return inline text without full content."""

    response = ParallelExtractResponse(
        results=[
            ParallelExtractHit(
                title="Example Article",
                url="https://example.com/a",
                excerpts=["key point one", "key point two"],
                full_content=None,
                publish_date=None,
            )
        ],
        errors=[],
    )
    client = _FakeParallelClient(response)
    tool = _make_tool(client, tmp_path)

    result = await tool.execute(
        {
            "url": "https://example.com/a",
            "objective": None,
            "mode": None,
        },
        _CTX,
    )

    call = client.calls[0]
    assert call["url"] == "https://example.com/a"
    assert call["include_full_content"] is False
    assert call["objective"] is None

    assert "Example Article" in result.output
    assert "key point one" in result.output
    assert "mode=excerpts" in result.output


async def test_parallel_extract_full_mode_small_content_stays_inline(
    tmp_path: Path,
) -> None:
    """Small full-content responses should be returned inline (below threshold)."""

    small_body = "# Heading\n\nBody text."
    response = ParallelExtractResponse(
        results=[
            ParallelExtractHit(
                title="Short",
                url="https://example.com/short",
                excerpts=[],
                full_content=small_body,
                publish_date=None,
            )
        ],
        errors=[],
    )
    client = _FakeParallelClient(response)
    tool = _make_tool(client, tmp_path, inline_full_content_threshold=4000)

    result = await tool.execute(
        {
            "url": "https://example.com/short",
            "objective": "what is the main idea",
            "mode": "full",
        },
        _CTX,
    )

    call = client.calls[0]
    assert call["include_full_content"] is True
    assert call["objective"] == "what is the main idea"

    assert "mode=full" in result.output
    assert small_body in result.output


async def test_parallel_extract_full_mode_overflow_saves_to_store(
    tmp_path: Path,
) -> None:
    """Large full-content responses should be saved to ToolOutputStore and referenced by path."""

    big_body = "x" * 8000
    response = ParallelExtractResponse(
        results=[
            ParallelExtractHit(
                title="Big",
                url="https://example.com/big",
                excerpts=["short excerpt"],
                full_content=big_body,
                publish_date=None,
            )
        ],
        errors=[],
    )
    client = _FakeParallelClient(response)
    tool = _make_tool(client, tmp_path, inline_full_content_threshold=4000)

    result = await tool.execute(
        {
            "url": "https://example.com/big",
            "objective": None,
            "mode": "full",
        },
        _CTX,
    )

    assert big_body not in result.output
    assert ".feather/tmp/web_fetch" in result.output
    assert "8000 chars" in result.output
    assert "read_file" in result.output.lower()

    stored_paths = list((tmp_path / ".feather" / "tmp" / "web_fetch").glob("*.output"))
    assert len(stored_paths) == 1
    assert stored_paths[0].read_text(encoding="utf-8") == big_body


async def test_parallel_extract_rejects_invalid_mode(tmp_path: Path) -> None:
    """Unknown mode values must fail fast."""

    tool = _make_tool(
        _FakeParallelClient(ParallelExtractResponse(results=[], errors=[])),
        tmp_path,
    )
    with pytest.raises(ValueError, match="mode"):
        await tool.execute(
            {
                "url": "https://example.com",
                "objective": None,
                "mode": "raw",
            },
            _CTX,
        )


async def test_parallel_extract_returns_error_message_on_client_failure(
    tmp_path: Path,
) -> None:
    """SDK exceptions must surface as tool output, not crash the agent loop."""

    client = _FakeParallelClient(ParallelExtractResponse(results=[], errors=[]))
    client.raise_on_call = RuntimeError("extract service unavailable")
    tool = _make_tool(client, tmp_path)

    result = await tool.execute(
        {
            "url": "https://example.com",
            "objective": None,
            "mode": None,
        },
        _CTX,
    )
    assert "extract service unavailable" in result.output
    assert result.output.lower().startswith("web_fetch failed")


async def test_parallel_extract_empty_results_reports_no_content(
    tmp_path: Path,
) -> None:
    """Empty results from Parallel should return a no-content message."""

    client = _FakeParallelClient(ParallelExtractResponse(results=[], errors=[]))
    tool = _make_tool(client, tmp_path)
    result = await tool.execute(
        {
            "url": "https://example.com/404",
            "objective": None,
            "mode": None,
        },
        _CTX,
    )
    assert "no content" in result.output.lower()
