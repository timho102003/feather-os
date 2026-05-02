"""Tests for the Parallel AI web_search tool."""

from __future__ import annotations

from typing import Any

import pytest

from feather.models import ParallelConfig, ToolExecutionContext
from feather.providers.parallel_client import (
    ParallelSearchHit,
    ParallelSearchResponse,
)
from feather.tools.parallel_search_tool import ParallelSearchTool


_CTX = ToolExecutionContext(session_id="sess-test", agent_name="Lead")


class _FakeParallelClient:
    """Fake client that records search invocations."""

    def __init__(self, response: ParallelSearchResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []
        self.raise_on_call: Exception | None = None

    async def search(self, **kwargs: Any) -> ParallelSearchResponse:
        self.calls.append(kwargs)
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return self._response


def _make_tool(
    client: _FakeParallelClient,
    *,
    max_results: int = 5,
    default_mode: str = "fast",
) -> ParallelSearchTool:
    config = ParallelConfig(
        api_key_env="UNUSED",
        default_search_mode=default_mode,
        max_results=max_results,
    )
    return ParallelSearchTool(client, config)  # type: ignore[arg-type]


def test_parallel_search_schema_is_strict_compatible() -> None:
    """All tool properties must appear in `required` for OpenAI strict tool calling."""

    tool = _make_tool(
        _FakeParallelClient(ParallelSearchResponse(mode="fast", results=[]))
    )
    schema = tool.parameters_schema
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == sorted(schema["properties"].keys())


async def test_parallel_search_uses_defaults_when_args_are_null() -> None:
    """Null optional args should fall back to the configured defaults."""

    response = ParallelSearchResponse(
        mode="fast",
        results=[
            ParallelSearchHit(
                title="United Nations - Wikipedia",
                url="https://en.wikipedia.org/wiki/United_Nations",
                excerpts=["The UN was established in 1945 after WWII."],
                publish_date="2024-12-01",
            )
        ],
    )
    client = _FakeParallelClient(response)
    tool = _make_tool(client, max_results=5, default_mode="fast")

    result = await tool.execute(
        {
            "objective": "When was the United Nations established?",
            "search_queries": None,
            "mode": None,
            "max_results": None,
            "include_domains": None,
        },
        _CTX,
    )

    assert client.calls == [
        {
            "objective": "When was the United Nations established?",
            "search_queries": None,
            "mode": "fast",
            "max_results": 5,
            "include_domains": None,
        }
    ]
    assert "United Nations - Wikipedia" in result.output
    assert "https://en.wikipedia.org/wiki/United_Nations" in result.output
    assert "1945" in result.output
    assert "mode=fast" in result.output


async def test_parallel_search_forwards_explicit_overrides() -> None:
    """Explicit arguments should override the configured defaults."""

    client = _FakeParallelClient(
        ParallelSearchResponse(mode="one-shot", results=[])
    )
    tool = _make_tool(client, max_results=5, default_mode="fast")

    await tool.execute(
        {
            "objective": "latest ai safety research",
            "search_queries": ["ai safety 2025", "alignment research"],
            "mode": "one-shot",
            "max_results": 8,
            "include_domains": ["anthropic.com", "openai.com"],
        },
        _CTX,
    )

    assert client.calls == [
        {
            "objective": "latest ai safety research",
            "search_queries": ["ai safety 2025", "alignment research"],
            "mode": "one-shot",
            "max_results": 8,
            "include_domains": ["anthropic.com", "openai.com"],
        }
    ]


async def test_parallel_search_rejects_invalid_mode() -> None:
    """Unknown modes must fail fast with a clear error."""

    tool = _make_tool(
        _FakeParallelClient(ParallelSearchResponse(mode="fast", results=[]))
    )
    with pytest.raises(ValueError, match="mode"):
        await tool.execute(
            {
                "objective": "x",
                "search_queries": None,
                "mode": "turbo",
                "max_results": None,
                "include_domains": None,
            },
            _CTX,
        )


async def test_parallel_search_reports_no_results() -> None:
    """Empty results should return a friendly no-results message."""

    client = _FakeParallelClient(
        ParallelSearchResponse(mode="fast", results=[])
    )
    tool = _make_tool(client)

    result = await tool.execute(
        {
            "objective": "nonsense query that finds nothing",
            "search_queries": None,
            "mode": None,
            "max_results": None,
            "include_domains": None,
        },
        _CTX,
    )
    assert "no results" in result.output.lower()


async def test_parallel_search_rejects_keyword_in_include_domains() -> None:
    """LLMs sometimes hallucinate `include_domains=['news']` thinking it's a category.
    The tool must validate before calling Parallel so the agent gets a clear,
    self-correcting error instead of a downstream HTTP 422.
    """

    client = _FakeParallelClient(
        ParallelSearchResponse(mode="fast", results=[])
    )
    tool = _make_tool(client)

    result = await tool.execute(
        {
            "objective": "latest crypto news",
            "search_queries": None,
            "mode": None,
            "max_results": None,
            "include_domains": ["news"],
        },
        _CTX,
    )
    # Tool returned a tool-output with a self-correcting hint;
    # the underlying client was never called.
    assert client.calls == []
    out_lower = result.output.lower()
    assert "include_domains" in result.output
    assert "'news'" in result.output or '"news"' in result.output
    assert "domain" in out_lower
    # Hint at the correct shape so the model can self-correct on next turn.
    assert "example.com" in result.output or ".gov" in result.output


async def test_parallel_search_rejects_url_with_scheme_in_include_domains() -> None:
    """Schemes/paths/ports are not allowed by Parallel — reject before hitting it."""

    client = _FakeParallelClient(
        ParallelSearchResponse(mode="fast", results=[])
    )
    tool = _make_tool(client)

    result = await tool.execute(
        {
            "objective": "x",
            "search_queries": None,
            "mode": None,
            "max_results": None,
            "include_domains": ["https://example.com/path"],
        },
        _CTX,
    )
    assert client.calls == []
    assert "include_domains" in result.output


async def test_parallel_search_accepts_valid_domains_and_extensions() -> None:
    """Plain domains and bare extensions like '.gov' must pass through unchanged."""

    client = _FakeParallelClient(
        ParallelSearchResponse(mode="fast", results=[])
    )
    tool = _make_tool(client)

    await tool.execute(
        {
            "objective": "x",
            "search_queries": None,
            "mode": None,
            "max_results": None,
            "include_domains": [
                "example.com",
                "subdomain.example.gov",
                ".edu",
                ".co.uk",
            ],
        },
        _CTX,
    )
    assert len(client.calls) == 1
    assert client.calls[0]["include_domains"] == [
        "example.com",
        "subdomain.example.gov",
        ".edu",
        ".co.uk",
    ]


async def test_parallel_search_returns_error_message_on_client_failure() -> None:
    """SDK exceptions must be converted to a tool output, not propagated."""

    client = _FakeParallelClient(
        ParallelSearchResponse(mode="fast", results=[])
    )
    client.raise_on_call = RuntimeError("rate limit exceeded")
    tool = _make_tool(client)

    result = await tool.execute(
        {
            "objective": "x",
            "search_queries": None,
            "mode": None,
            "max_results": None,
            "include_domains": None,
        },
        _CTX,
    )
    assert "rate limit exceeded" in result.output
    assert result.output.lower().startswith("web_search failed")
