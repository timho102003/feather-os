"""Tests for the AgentCatalog discovery service."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from feather.core.agent_catalog import (
    AgentCatalog,
    AgentCatalogEntry,
    render_catalog_block,
)


def _write_agent_yaml(
    root: Path,
    slug: str,
    *,
    name: str,
    role: str,
    description: str = "",
    registered_tools: tuple[str, ...] = ("read_file",),
) -> None:
    (root / "config" / "agents").mkdir(parents=True, exist_ok=True)
    tools_yaml = "\n".join(f"  - {t}" for t in registered_tools)
    (root / "config" / "agents" / f"{slug}.yaml").write_text(
        f"""name: {name}
role: {role}
personality: Direct
description: {description}
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
registered_tools:
{tools_yaml}
""",
        encoding="utf-8",
    )


def test_agent_catalog_discovers_all_yamls_and_flags_builtins(tmp_path: Path) -> None:
    _write_agent_yaml(tmp_path, "lead", name="Lead", role="lead", description="root")
    _write_agent_yaml(tmp_path, "explore", name="Explore", role="explore", description="navigate")
    _write_agent_yaml(
        tmp_path,
        "reviewer-custom",
        name="Reviewer",
        role="custom",
        description="review PRs",
    )
    catalog = AgentCatalog(tmp_path)
    entries = catalog.list_entries()
    by_name = {e.name: e for e in entries}
    assert set(by_name) == {"lead", "explore", "reviewer-custom"}
    assert by_name["lead"].is_builtin is True
    assert by_name["explore"].is_builtin is True
    assert by_name["reviewer-custom"].is_builtin is False
    assert by_name["reviewer-custom"].description == "review PRs"


def test_agent_catalog_excludes_lead_from_dispatchable(tmp_path: Path) -> None:
    _write_agent_yaml(tmp_path, "lead", name="Lead", role="lead")
    _write_agent_yaml(tmp_path, "explore", name="Explore", role="explore")
    catalog = AgentCatalog(tmp_path)
    dispatchable = [e.name for e in catalog.list_dispatchable()]
    assert "lead" not in dispatchable
    assert "explore" in dispatchable


def test_agent_catalog_skips_broken_yaml_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_agent_yaml(tmp_path, "explore", name="Explore", role="explore")
    (tmp_path / "config" / "agents" / "broken.yaml").write_text(
        "::not valid yaml::\n  - [", encoding="utf-8"
    )
    caplog.set_level(logging.WARNING)
    catalog = AgentCatalog(tmp_path)
    names = [e.name for e in catalog.list_entries()]
    assert names == ["explore"]
    assert any("skipping unreadable agent yaml=broken.yaml" in rec.message for rec in caplog.records)


def test_agent_catalog_warns_on_custom_naming_mismatch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # role=custom but filename has no -custom suffix
    _write_agent_yaml(tmp_path, "weird", name="Weird", role="custom")
    # role!=custom but filename has -custom suffix
    _write_agent_yaml(tmp_path, "fake-custom", name="Fake", role="explore")
    caplog.set_level(logging.WARNING)
    AgentCatalog(tmp_path).list_entries()
    messages = [rec.message for rec in caplog.records]
    assert any("should be named with the '-custom' suffix" in m for m in messages)
    assert any("uses the custom filename suffix but" in m for m in messages)


def test_agent_catalog_is_valid_name_rejects_traversal() -> None:
    assert AgentCatalog.is_valid_name("reviewer-custom") is True
    assert AgentCatalog.is_valid_name("a_b_c") is True
    assert AgentCatalog.is_valid_name("") is False
    assert AgentCatalog.is_valid_name("../etc/passwd") is False
    assert AgentCatalog.is_valid_name("bad name") is False
    assert AgentCatalog.is_valid_name("bad.yaml") is False


def test_render_catalog_block_excludes_lead_and_formats_entries() -> None:
    entries = [
        AgentCatalogEntry(
            name="lead",
            role="lead",
            description="the lead",
            personality="",
            registered_tools=("spawn_agent",),
            is_builtin=True,
        ),
        AgentCatalogEntry(
            name="explore",
            role="explore",
            description="local nav",
            personality="",
            registered_tools=("read_file", "grep"),
            is_builtin=True,
        ),
        AgentCatalogEntry(
            name="reviewer-custom",
            role="custom",
            description="reviews PRs",
            personality="",
            registered_tools=("bash",),
            is_builtin=False,
        ),
    ]
    block = render_catalog_block(entries)
    assert "lead" not in block  # lead not dispatchable
    assert "`explore` [builtin, role=explore] — local nav | tools: read_file, grep" in block
    assert "`reviewer-custom` [custom, role=custom] — reviews PRs | tools: bash" in block


def test_render_catalog_block_empty_returns_empty_string() -> None:
    assert render_catalog_block([]) == ""
    # only non-dispatchable entries → empty
    only_lead = [
        AgentCatalogEntry(
            name="lead",
            role="lead",
            description="",
            personality="",
            registered_tools=(),
            is_builtin=True,
        )
    ]
    assert render_catalog_block(only_lead) == ""
