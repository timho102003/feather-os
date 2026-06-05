"""Tools for progressive MCP server discovery and activation."""

from __future__ import annotations

import json
from typing import Any

from feather.integrations.mcp.client import (
    MCPClientManager,
    MCPStreamableHTTPClient,
    mcp_proxy_tool_name,
    should_proxy_mcp_server,
)
from feather.models import MCPServerConfig, ToolExecutionContext, ToolExecutionResult
from feather.storage.session_store import SessionStore
from feather.tools.base import BaseTool


class ListMCPServersTool(BaseTool):
    """List MCP servers available to the current agent."""

    name = "list_mcp_servers"
    description = "List MCP servers available for this agent to register on demand."
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        mcp_servers: tuple[MCPServerConfig, ...],
        provider_name: str,
        session_store: SessionStore,
    ) -> None:
        self._mcp_servers = mcp_servers
        self._provider_name = provider_name
        self._session_store = session_store

    def get_prompt(self) -> str:
        """Describe the MCP discovery workflow."""

        return (
            "- `list_mcp_servers`: inspect available MCP server integrations. "
            "Use this before registering a server for the session."
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        """Return a compact JSON catalog of available MCP servers."""

        active = set(
            (await self._session_store.get_session(context.session_id)).active_mcp_servers
        )
        rows = [
            _server_metadata(server, self._provider_name, active=server.label in active)
            for server in self._mcp_servers
            if _is_supported_server(server)
        ]
        return ToolExecutionResult(output=json.dumps(rows, ensure_ascii=False))


class RegisterMCPServerTool(BaseTool):
    """Activate one MCP server for the current session."""

    name = "register_mcp_server"
    description = "Register one available MCP server for this session by exact label."
    parameters_schema = {
        "type": "object",
        "properties": {
            "server_label": {
                "type": "string",
                "description": "Exact label from list_mcp_servers.",
            },
            "verify_connection": {
                "type": ["boolean", "null"],
                "description": "Whether to initialize the MCP server and list tools before activating it. Defaults to true.",
            },
        },
        "required": ["server_label", "verify_connection"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        mcp_servers: tuple[MCPServerConfig, ...],
        provider_name: str,
        session_store: SessionStore,
        manager: MCPClientManager,
    ) -> None:
        self._mcp_servers = mcp_servers
        self._provider_name = provider_name
        self._session_store = session_store
        self._manager = manager

    def get_prompt(self) -> str:
        """Describe how to activate one MCP server."""

        return (
            "- `register_mcp_server`: activate one MCP server for this session "
            "after checking `list_mcp_servers`. Register only servers needed for the task."
        )

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        """Validate and activate one MCP server for this session."""

        server_label = str(arguments["server_label"]).strip()
        verify = arguments.get("verify_connection")
        should_verify = verify is not False
        server = self._find_server(server_label)
        _ensure_supported_server(server)
        tool_names: list[str] = []
        if should_verify:
            if should_proxy_mcp_server(server, self._provider_name):
                client = await self._manager.client_for(context.session_id, server)
                tools = await client.list_tools()
            else:
                async with MCPStreamableHTTPClient(server) as client:
                    tools = await client.list_tools()
            tools = _filter_allowed_tools(tools, server.allowed_tools)
            tool_names = [
                str(tool.get("name")) for tool in tools if tool.get("name")
            ]
        await self._session_store.append_active_mcp_server(
            context.session_id, server.label
        )
        metadata = _server_metadata(server, self._provider_name, active=True)
        metadata["remote_tools"] = tool_names
        return ToolExecutionResult(
            output=(
                f"Registered MCP server `{server.label}` for this session.\n"
                f"{json.dumps(metadata, ensure_ascii=False)}"
            )
        )

    def _find_server(self, label: str) -> MCPServerConfig:
        for server in self._mcp_servers:
            if server.label == label:
                return server
        raise ValueError(f"MCP server `{label}` is not available to this agent.")


def _server_metadata(
    server: MCPServerConfig, provider_name: str, *, active: bool
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "label": server.label,
        "description": server.server_description or "",
        "transport": server.transport,
        "active": active,
        "allowed_tools": list(server.allowed_tools),
    }
    if should_proxy_mcp_server(server, provider_name):
        metadata["activation"] = {
            "mode": "proxy_tool",
            "tool_name": mcp_proxy_tool_name(server),
        }
    else:
        metadata["activation"] = {"mode": "native_openai_mcp"}
    return metadata


def _filter_allowed_tools(
    tools: list[dict[str, Any]], allowed_tools: tuple[str, ...]
) -> list[dict[str, Any]]:
    if not allowed_tools:
        return tools
    allowed = set(allowed_tools)
    return [tool for tool in tools if tool.get("name") in allowed]


def _ensure_supported_server(server: MCPServerConfig) -> None:
    if not _is_supported_server(server):
        raise ValueError(
            f"MCP server `{server.label}` requires approval flows, which are "
            "not supported yet."
        )


def _is_supported_server(server: MCPServerConfig) -> bool:
    return server.require_approval in (None, "never")
