"""Tests for the Parallel AI client wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from feather.models import ParallelConfig
from feather.providers.parallel_client import ParallelClient


class _FakeBeta:
    """Record invocations against a fake Parallel beta resource."""

    def __init__(self, search_return: Any, extract_return: Any) -> None:
        self._search_return = search_return
        self._extract_return = extract_return
        self.search_calls: list[dict[str, Any]] = []
        self.extract_calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Any:
        self.search_calls.append(kwargs)
        return self._search_return

    def extract(self, **kwargs: Any) -> Any:
        self.extract_calls.append(kwargs)
        return self._extract_return


class _FakeParallel:
    """Minimal stand-in for `parallel.Parallel` used in tests."""

    def __init__(self, beta: _FakeBeta) -> None:
        self.beta = beta


def _make_client(beta: _FakeBeta) -> ParallelClient:
    return ParallelClient(
        ParallelConfig(api_key_env="UNUSED_PARALLEL_KEY"),
        client=_FakeParallel(beta),
    )


async def test_search_passes_expected_kwargs_and_normalizes_results() -> None:
    """`search` should forward normalized kwargs and project results into hits."""

    search_return = SimpleNamespace(
        results=[
            SimpleNamespace(
                url="https://example.com/a",
                title="A",
                excerpts=["excerpt one", "excerpt two"],
                publish_date="2025-01-01",
            )
        ],
    )
    beta = _FakeBeta(search_return=search_return, extract_return=None)
    client = _make_client(beta)

    response = await client.search(
        objective="Find the UN founding year",
        search_queries=["UN founding year"],
        mode="fast",
        max_results=3,
        include_domains=["un.org"],
    )

    assert len(beta.search_calls) == 1
    call = beta.search_calls[0]
    assert call["objective"] == "Find the UN founding year"
    assert call["mode"] == "fast"
    assert call["max_results"] == 3
    assert call["search_queries"] == ["UN founding year"]
    assert call["source_policy"] == {"include_domains": ["un.org"]}

    assert response.mode == "fast"
    assert len(response.results) == 1
    hit = response.results[0]
    assert hit.url == "https://example.com/a"
    assert hit.title == "A"
    assert hit.excerpts == ["excerpt one", "excerpt two"]
    assert hit.publish_date == "2025-01-01"


async def test_search_omits_optional_kwargs_when_absent() -> None:
    """Optional search kwargs must not be sent when the caller leaves them empty."""

    beta = _FakeBeta(search_return=SimpleNamespace(results=[]), extract_return=None)
    client = _make_client(beta)

    await client.search(
        objective="anything",
        search_queries=None,
        mode="fast",
        max_results=5,
        include_domains=None,
    )

    call = beta.search_calls[0]
    assert "search_queries" not in call
    assert "source_policy" not in call


async def test_extract_requests_full_content_only_when_asked() -> None:
    """`extract` should forward `full_content` only when the caller opts in."""

    extract_return = SimpleNamespace(
        results=[
            SimpleNamespace(
                url="https://example.com/a",
                title="A",
                excerpts=["excerpt one"],
                full_content="# heading\n\nBody",
                publish_date=None,
            )
        ],
        errors=[],
    )
    beta = _FakeBeta(search_return=None, extract_return=extract_return)
    client = _make_client(beta)

    response = await client.extract(
        url="https://example.com/a",
        objective="what is the main idea",
        include_full_content=True,
    )

    call = beta.extract_calls[0]
    assert call["urls"] == ["https://example.com/a"]
    assert call["excerpts"] is True
    assert call["full_content"] is True
    assert call["objective"] == "what is the main idea"

    hit = response.results[0]
    assert hit.full_content == "# heading\n\nBody"
    assert hit.excerpts == ["excerpt one"]
    assert response.errors == []


async def test_extract_omits_full_content_kwarg_by_default() -> None:
    """When `include_full_content` is False, the kwarg must not be sent."""

    beta = _FakeBeta(
        search_return=None,
        extract_return=SimpleNamespace(results=[], errors=[]),
    )
    client = _make_client(beta)

    await client.extract(
        url="https://example.com",
        objective=None,
        include_full_content=False,
    )

    call = beta.extract_calls[0]
    assert "full_content" not in call
    assert "objective" not in call


def test_parallel_client_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing without a client or env var must fail loudly."""

    monkeypatch.delenv("MISSING_PARALLEL_KEY", raising=False)
    with pytest.raises(ValueError, match="MISSING_PARALLEL_KEY"):
        ParallelClient(ParallelConfig(api_key_env="MISSING_PARALLEL_KEY"))
