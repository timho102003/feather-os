"""Tests for the lightweight MCP HTTP client and proxy tool."""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from typing import Any

import httpx
import pytest

from feather.integrations.mcp.client import (
    MCPClientManager,
    MCPProxyTool,
    MCPStdioClient,
    MCPStreamableHTTPClient,
    mcp_servers_for,
    openai_mcp_tools,
)
from feather.models import MCPConfig, MCPServerConfig, ToolExecutionContext


@pytest.mark.asyncio
async def test_mcp_streamable_http_client_initializes_lists_and_calls_tools() -> None:
    """The client should perform the MCP initialize handshake once per session."""

    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        seen.append(payload)
        if payload.get("method") == "initialize":
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "sess-1"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "test", "version": "1"},
                    },
                },
            )
        if payload.get("method") == "notifications/initialized":
            assert request.headers["Mcp-Session-Id"] == "sess-1"
            return httpx.Response(202)
        if payload.get("method") == "tools/list":
            assert request.headers["Mcp-Session-Id"] == "sess-1"
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "lookup",
                                "description": "Look up a record.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"id": {"type": "string"}},
                                },
                            }
                        ]
                    },
                },
            )
        if payload.get("method") == "tools/call":
            assert payload["params"] == {"name": "lookup", "arguments": {"id": "42"}}
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "content": [{"type": "text", "text": "record 42"}],
                        "isError": False,
                    },
                },
            )
        raise AssertionError(f"unexpected MCP method: {payload}")

    server = MCPServerConfig(label="docs", server_url="https://mcp.example.test")
    async with MCPStreamableHTTPClient(
        server, transport=httpx.MockTransport(handler)
    ) as client:
        tools = await client.list_tools()
        result = await client.call_tool("lookup", {"id": "42"})

    assert tools[0]["name"] == "lookup"
    assert result == "record 42"
    assert [payload["method"] for payload in seen] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]


@pytest.mark.asyncio
async def test_mcp_streamable_http_client_rejects_repeated_tool_cursor() -> None:
    """Buggy MCP pagination should not spin forever."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        if payload.get("method") == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "test", "version": "1"},
                    },
                },
            )
        if payload.get("method") == "notifications/initialized":
            return httpx.Response(202)
        if payload.get("method") == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"tools": [], "nextCursor": "same"},
                },
            )
        raise AssertionError(f"unexpected MCP method: {payload}")

    server = MCPServerConfig(label="docs", server_url="https://mcp.example.test")
    async with MCPStreamableHTTPClient(
        server, transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(RuntimeError, match="repeated tool cursor"):
            await client.list_tools()


@pytest.mark.asyncio
async def test_mcp_client_manager_closes_one_session_clients() -> None:
    """Session cleanup should not wait for process-wide runtime shutdown."""

    class FakeClient:
        def __init__(self) -> None:
            self.closed = False

        async def initialize(self) -> None:
            pass

        async def aclose(self) -> None:
            self.closed = True

    class FakeManager(MCPClientManager):
        def __init__(self) -> None:
            super().__init__()
            self.built: list[FakeClient] = []

        def _build_client(self, server: MCPServerConfig) -> FakeClient:
            client = FakeClient()
            self.built.append(client)
            return client

    manager = FakeManager()
    first_server = MCPServerConfig(label="one", server_url="https://one.test")
    second_server = MCPServerConfig(label="two", server_url="https://two.test")
    first = await manager.client_for("s1", first_server)
    second = await manager.client_for("s2", second_server)

    await manager.close_session("s1")

    assert first.closed is True
    assert second.closed is False
    await manager.aclose()


def test_mcp_servers_for_requires_agent_match_for_scoped_servers() -> None:
    """Agent-scoped MCP servers must not leak into agent-agnostic calls."""

    scoped = MCPServerConfig(
        label="scoped",
        server_url="https://mcp.example.test",
        providers=("openai",),
        agents=("Lead",),
    )
    global_server = MCPServerConfig(
        label="global",
        server_url="https://global.example.test",
        providers=("openai",),
    )
    config = MCPConfig(enabled=True, servers=(scoped, global_server))

    assert mcp_servers_for(config, provider_name="openai", agent_name=None) == (
        global_server,
    )
    assert mcp_servers_for(config, provider_name="openai", agent_name="Lead") == (
        scoped,
        global_server,
    )


def test_openai_mcp_tools_rejects_unsupported_approval_flow() -> None:
    """Approval-requiring MCP servers must fail before an API call."""

    server = MCPServerConfig(
        label="dangerous",
        server_url="https://mcp.example.test",
        require_approval="always",
    )

    try:
        openai_mcp_tools((server,))
    except ValueError as exc:
        assert "approval flows are not supported" in str(exc)
    else:
        raise AssertionError("Expected approval-requiring MCP server to fail")


@pytest.mark.asyncio
async def test_mcp_stdio_client_initializes_lists_and_calls_tools(tmp_path) -> None:
    """Feather should launch stdio MCP servers without a user-started HTTP server."""

    server_script = tmp_path / "fake_mcp_server.py"
    server_script.write_text(
        textwrap.dedent(
            """
            import json
            import sys

            def read_message():
                line = sys.stdin.buffer.readline()
                if not line:
                    return None
                return json.loads(line.decode())

            def write_message(payload):
                body = json.dumps(payload, separators=(",", ":")).encode()
                sys.stdout.buffer.write(body + b"\\n")
                sys.stdout.buffer.flush()

            while True:
                message = read_message()
                if message is None:
                    break
                method = message.get("method")
                if method == "notifications/initialized":
                    continue
                if method == "initialize":
                    result = {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake", "version": "1"},
                    }
                elif method == "tools/list":
                    result = {
                        "tools": [
                            {"name": "lookup", "description": "Look up a record."}
                        ]
                    }
                elif method == "tools/call":
                    result = {
                        "content": [
                            {
                                "type": "text",
                                "text": "called " + message["params"]["name"],
                            }
                        ],
                        "isError": False,
                    }
                else:
                    write_message(
                        {
                            "jsonrpc": "2.0",
                            "id": message.get("id"),
                            "error": {"code": -32601, "message": "unknown"},
                        }
                    )
                    continue
                write_message({"jsonrpc": "2.0", "id": message.get("id"), "result": result})
            """
        ),
        encoding="utf-8",
    )
    server = MCPServerConfig(
        label="stdio",
        transport="stdio",
        command=sys.executable,
        args=(str(server_script),),
    )
    client = MCPStdioClient(server)
    try:
        tools = await client.list_tools()
        result = await client.call_tool("lookup", {"id": "42"})
    finally:
        await client.aclose()

    assert tools[0]["name"] == "lookup"
    assert result == "called lookup"


@pytest.mark.asyncio
async def test_mcp_stdio_client_does_not_inherit_secret_environment(
    tmp_path, monkeypatch
) -> None:
    """Stdio MCP servers should receive only minimal env plus explicit config."""

    monkeypatch.setenv("FEATHER_SECRET_TOKEN", "should-not-leak")
    server_script = tmp_path / "env_mcp_server.py"
    server_script.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys

            def read_message():
                line = sys.stdin.buffer.readline()
                if not line:
                    return None
                return json.loads(line.decode())

            def write_message(payload):
                sys.stdout.buffer.write(
                    json.dumps(payload, separators=(",", ":")).encode() + b"\\n"
                )
                sys.stdout.buffer.flush()

            while True:
                message = read_message()
                if message is None:
                    break
                method = message.get("method")
                if method == "notifications/initialized":
                    continue
                if method == "initialize":
                    result = {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "env", "version": "1"},
                    }
                elif method == "tools/call":
                    result = {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "explicit": os.environ.get("EXPLICIT_ALLOWED"),
                                        "secret_present": "FEATHER_SECRET_TOKEN" in os.environ,
                                    }
                                ),
                            }
                        ],
                        "isError": False,
                    }
                else:
                    result = {"tools": [{"name": "env"}]}
                write_message({"jsonrpc": "2.0", "id": message.get("id"), "result": result})
            """
        ),
        encoding="utf-8",
    )
    server = MCPServerConfig(
        label="stdio",
        transport="stdio",
        command=sys.executable,
        args=(str(server_script),),
        env={"EXPLICIT_ALLOWED": "yes"},
    )
    client = MCPStdioClient(server)
    try:
        result = await client.call_tool("env", {})
    finally:
        await client.aclose()

    payload = json.loads(result)
    assert payload == {"explicit": "yes", "secret_present": False}


@pytest.mark.asyncio
async def test_mcp_stdio_client_reinitializes_after_process_exit(tmp_path) -> None:
    """A crashed stdio server should be reinitialized before the next request."""

    server_script = tmp_path / "exit_after_list.py"
    server_script.write_text(
        textwrap.dedent(
            """
            import json
            import sys

            def read_message():
                line = sys.stdin.buffer.readline()
                if not line:
                    return None
                return json.loads(line.decode())

            def write_message(payload):
                sys.stdout.buffer.write(
                    json.dumps(payload, separators=(",", ":")).encode() + b"\\n"
                )
                sys.stdout.buffer.flush()

            while True:
                message = read_message()
                if message is None:
                    break
                method = message.get("method")
                if method == "notifications/initialized":
                    continue
                if method == "initialize":
                    result = {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "flaky", "version": "1"},
                    }
                elif method == "tools/list":
                    result = {"tools": [{"name": "lookup"}]}
                    write_message({"jsonrpc": "2.0", "id": message.get("id"), "result": result})
                    sys.exit(0)
                else:
                    result = {"content": [], "isError": False}
                write_message({"jsonrpc": "2.0", "id": message.get("id"), "result": result})
            """
        ),
        encoding="utf-8",
    )
    server = MCPServerConfig(
        label="stdio",
        transport="stdio",
        command=sys.executable,
        args=(str(server_script),),
    )
    client = MCPStdioClient(server)
    try:
        first = await client.list_tools()
        await asyncio.sleep(0.05)
        second = await client.list_tools()
    finally:
        await client.aclose()

    assert first == [{"name": "lookup"}]
    assert second == [{"name": "lookup"}]


@pytest.mark.asyncio
async def test_mcp_proxy_tool_enforces_configured_allowed_tools() -> None:
    """The OpenRouter proxy must deny tool names outside config allowlists."""

    server = MCPServerConfig(
        label="docs",
        server_url="https://mcp.example.test",
        allowed_tools=("search",),
    )
    tool = MCPProxyTool(server, transport=httpx.MockTransport(lambda _req: httpx.Response(500)))

    with pytest.raises(ValueError, match="not allowed"):
        await tool.execute(
            {"action": "call_tool", "tool_name": "fetch", "arguments": {}},
            ToolExecutionContext(session_id="s", agent_name="Lead"),
        )


def test_mcp_proxy_tool_uses_non_strict_schema_for_arbitrary_arguments() -> None:
    """Remote MCP arguments are arbitrary JSON and must not use strict mode."""

    server = MCPServerConfig(label="docs", server_url="https://mcp.example.test")
    tool = MCPProxyTool(server)

    assert tool.to_openai_tool()["strict"] is False


@pytest.mark.asyncio
async def test_mcp_proxy_tool_lists_tools_as_json() -> None:
    """The proxy tool gives OpenRouter agents a provider-neutral listing path."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        if payload.get("method") == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "test", "version": "1"},
                    },
                },
            )
        if payload.get("method") == "notifications/initialized":
            return httpx.Response(202)
        if payload.get("method") == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"tools": [{"name": "search", "description": "Search."}]},
                },
            )
        raise AssertionError(f"unexpected MCP method: {payload}")

    server = MCPServerConfig(label="docs", server_url="https://mcp.example.test")
    tool = MCPProxyTool(server, transport=httpx.MockTransport(handler))

    result = await tool.execute(
        {"action": "list_tools", "tool_name": None, "arguments": {}},
        ToolExecutionContext(session_id="s", agent_name="Lead"),
    )

    assert json.loads(result.output) == [{"name": "search", "description": "Search."}]
