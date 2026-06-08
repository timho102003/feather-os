"""Tests for progressive MCP discovery and registration tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feather.integrations.mcp.client import MCPClientManager
from feather.models import MCPServerConfig, ToolExecutionContext
from feather.storage.session_store import SessionStore
from feather.tools.mcp_tools import ListMCPServersTool, RegisterMCPServerTool


async def test_list_mcp_servers_lists_only_agent_available_servers(
    tmp_path: Path,
) -> None:
    """The list tool should expose metadata, not remote tool descriptions."""

    store = SessionStore(tmp_path / "feather.db")
    await store.initialize()
    try:
        session = await store.create_session("Lead")
        servers = (
            MCPServerConfig(
                label="docs",
                server_url="https://example.test/mcp",
                server_description="Docs server",
                providers=("openrouter",),
                agents=("Lead",),
            ),
            MCPServerConfig(
                label="hidden",
                server_url="https://hidden.test/mcp",
                providers=("openrouter",),
                agents=("Research",),
            ),
        )
        tool = ListMCPServersTool(
            mcp_servers=servers[:1],
            provider_name="openrouter",
            session_store=store,
        )

        result = await tool.execute(
            {}, ToolExecutionContext(session_id=session.id, agent_name="Lead")
        )

        rows = json.loads(result.output)
        assert [row["label"] for row in rows] == ["docs"]
        assert rows[0]["activation"] == {
            "mode": "proxy_tool",
            "tool_name": "mcp_docs",
        }
    finally:
        await store.close()


async def test_register_mcp_server_activates_session_without_verification(
    tmp_path: Path,
) -> None:
    """Registering a server should persist session-scoped activation."""

    store = SessionStore(tmp_path / "feather.db")
    await store.initialize()
    manager = MCPClientManager()
    try:
        session = await store.create_session("Lead")
        servers = (
            MCPServerConfig(
                label="docs",
                server_url="https://example.test/mcp",
                providers=("openrouter",),
                agents=("Lead",),
            ),
        )
        tool = RegisterMCPServerTool(
            mcp_servers=servers,
            provider_name="openrouter",
            session_store=store,
            manager=manager,
        )

        result = await tool.execute(
            {"server_label": "docs", "verify_connection": False},
            ToolExecutionContext(session_id=session.id, agent_name="Lead"),
        )

        updated = await store.get_session(session.id)
        assert updated.active_mcp_servers == ["docs"]
        assert "mcp_docs" in result.output
    finally:
        await manager.aclose()
        await store.close()


async def test_register_mcp_server_rejects_approval_required_servers(
    tmp_path: Path,
) -> None:
    """Approval-required MCP servers should not be session-activated."""

    store = SessionStore(tmp_path / "feather.db")
    await store.initialize()
    manager = MCPClientManager()
    try:
        session = await store.create_session("Lead")
        servers = (
            MCPServerConfig(
                label="dangerous",
                server_url="https://example.test/mcp",
                require_approval="always",
            ),
        )
        tool = RegisterMCPServerTool(
            mcp_servers=servers,
            provider_name="openrouter",
            session_store=store,
            manager=manager,
        )

        with pytest.raises(ValueError, match="approval flows"):
            await tool.execute(
                {"server_label": "dangerous", "verify_connection": False},
                ToolExecutionContext(session_id=session.id, agent_name="Lead"),
            )

        updated = await store.get_session(session.id)
        assert updated.active_mcp_servers == []
    finally:
        await manager.aclose()
        await store.close()
