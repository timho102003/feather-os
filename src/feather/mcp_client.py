"""Remote MCP server helpers and OpenRouter proxy tooling."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
from pathlib import Path
from typing import Any

import httpx

from feather.models import (
    MCPConfig,
    MCPServerConfig,
    ToolExecutionContext,
    ToolExecutionResult,
)
from feather.tools.base import BaseTool

logger = logging.getLogger(__name__)

_MCP_PROTOCOL_VERSION = "2025-06-18"
_MCP_MAX_TOOL_PAGES = 20
_MCP_MAX_TOOLS = 256
_MCP_PROXY_OUTPUT_LIMIT = 20_000
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
_STDIO_ENV_ALLOWLIST = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LOGNAME",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USER",
    "WINDIR",
)


def mcp_servers_for(
    config: MCPConfig,
    *,
    provider_name: str,
    agent_name: str | None,
) -> tuple[MCPServerConfig, ...]:
    """Return MCP servers enabled for one provider/agent pair.

    Args:
        config: Top-level MCP registry.
        provider_name: Provider key, such as ``openai`` or ``openrouter``.
        agent_name: Runtime agent display name. ``None`` only matches servers
            without an ``agents`` filter.

    Returns:
        Matching server configs in configured order.
    """

    if not config.enabled:
        return ()
    provider = provider_name.strip().lower()
    normalized_agent = agent_name.strip().lower() if agent_name else None
    matches: list[MCPServerConfig] = []
    for server in config.servers:
        if server.providers and provider not in server.providers:
            continue
        if server.agents:
            if normalized_agent is None:
                continue
            allowed_agents = {agent.strip().lower() for agent in server.agents}
            if normalized_agent not in allowed_agents:
                continue
        matches.append(server)
    return tuple(matches)


def openai_mcp_tools(servers: tuple[MCPServerConfig, ...]) -> list[dict[str, Any]]:
    """Serialize MCP server configs into OpenAI Responses remote-tool entries."""

    tools: list[dict[str, Any]] = []
    for server in servers:
        if server.transport != "http" or not server.server_url:
            raise ValueError(
                f"MCP server `{server.label}` cannot be sent as a native "
                "OpenAI remote MCP tool without an HTTP URL."
            )
        if server.require_approval not in (None, "never"):
            raise ValueError(
                "OpenAI MCP approval flows are not supported yet; set "
                f"mcp.servers.{server.label}.require_approval to `never`."
            )
        tool: dict[str, Any] = {
            "type": "mcp",
            "server_label": server.label,
            "server_url": server.server_url,
        }
        if server.server_description:
            tool["server_description"] = server.server_description
        if server.allowed_tools:
            tool["allowed_tools"] = list(server.allowed_tools)
        if server.require_approval is not None:
            tool["require_approval"] = server.require_approval
        headers = resolve_mcp_headers(server)
        if headers:
            tool["headers"] = headers
        tools.append(tool)
    return tools


def resolve_mcp_headers(server: MCPServerConfig) -> dict[str, str]:
    """Resolve static and environment-backed headers for one MCP server.

    Args:
        server: MCP server config.

    Returns:
        Header mapping ready to send on an HTTP request.

    Raises:
        ValueError: If a configured header environment variable is missing.
    """

    headers = dict(server.headers)
    for header_name, env_name in server.header_envs.items():
        value = os.getenv(env_name)
        if value is None or not value.strip():
            raise ValueError(
                f"Missing required environment variable for MCP server "
                f"{server.label}: {env_name}"
            )
        headers[header_name] = value
    return headers


def mcp_proxy_tool_name(server: MCPServerConfig) -> str:
    """Return the local proxy tool name for ``server``."""

    return f"mcp_{_sanitize_tool_label(server.label)}"


def should_proxy_mcp_server(server: MCPServerConfig, provider_name: str) -> bool:
    """Return whether Feather should expose ``server`` through a local proxy."""

    return provider_name.strip().lower() != "openai" or server.transport != "http"


class MCPStreamableHTTPClient:
    """Small JSON-RPC client for MCP Streamable HTTP servers."""

    def __init__(
        self,
        server: MCPServerConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._server = server
        if not server.server_url:
            raise ValueError(f"MCP HTTP server `{server.label}` is missing `server_url`.")
        self._next_id = 1
        self._session_id: str | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        timeout = httpx.Timeout(server.request_timeout_seconds, connect=30.0)
        client_kwargs: dict[str, Any] = {"timeout": timeout}
        if transport is not None:
            client_kwargs["transport"] = transport
        self._client = httpx.AsyncClient(**client_kwargs)

    async def __aenter__(self) -> "MCPStreamableHTTPClient":
        await self.initialize()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""

        await self._client.aclose()

    async def initialize(self) -> None:
        """Perform the MCP initialize handshake once for this client."""

        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._request(
                "initialize",
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "feather", "version": "0.1.0"},
                },
            )
            await self._notify("notifications/initialized")
            self._initialized = True

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return all tools reported by the remote MCP server."""

        await self.initialize()
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        pages = 0
        while True:
            pages += 1
            if pages > _MCP_MAX_TOOL_PAGES:
                raise RuntimeError(
                    f"MCP server {self._server.label} exceeded tool page limit"
                )
            params = {"cursor": cursor} if cursor else None
            result = await self._request("tools/list", params)
            tools.extend(result.get("tools") or [])
            if len(tools) > _MCP_MAX_TOOLS:
                raise RuntimeError(
                    f"MCP server {self._server.label} exceeded tool count limit"
                )
            cursor = result.get("nextCursor")
            if not cursor:
                return tools
            if cursor in seen_cursors:
                raise RuntimeError(
                    f"MCP server {self._server.label} returned repeated tool cursor"
                )
            seen_cursors.add(cursor)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call one remote MCP tool and render its result content."""

        await self.initialize()
        result = await self._request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        rendered = _render_mcp_content(result.get("content") or [])
        if result.get("isError"):
            raise RuntimeError(rendered or f"MCP tool `{name}` failed")
        return rendered

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        response = await self._client.post(
            str(self._server.server_url),
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()

    async def _request(
        self, method: str, params: dict[str, Any] | None
    ) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        response = await self._client.post(
            str(self._server.server_url),
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id
        data = _decode_jsonrpc_response(response)
        if data.get("id") != request_id:
            raise RuntimeError(
                f"MCP server {self._server.label} returned mismatched JSON-RPC id"
            )
        if data.get("error"):
            raise RuntimeError(
                f"MCP server {self._server.label} error: {data['error']}"
            )
        result = data.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(
                f"MCP server {self._server.label} returned invalid result"
            )
        return result

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": _MCP_PROTOCOL_VERSION,
            **resolve_mcp_headers(self._server),
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers


class MCPStdioClient:
    """Persistent JSON-RPC client for an MCP stdio server process."""

    def __init__(self, server: MCPServerConfig) -> None:
        self._server = server
        if not server.command:
            raise ValueError(f"MCP stdio server `{server.label}` is missing `command`.")
        self._next_id = 1
        self._process: asyncio.subprocess.Process | None = None
        self._request_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._stderr_task: asyncio.Task[None] | None = None

    async def aclose(self) -> None:
        """Terminate the child process and drain the stderr reader task."""

        process = self._process
        self._process = None
        self._initialized = False
        if process is not None and process.stdin is not None:
            process.stdin.close()
            with contextlib.suppress(Exception):
                await process.stdin.wait_closed()
        if process is not None and process.returncode is None:
            self._terminate_process(process, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._terminate_process(process, _SIGKILL)
                await process.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None

    async def initialize(self) -> None:
        """Start the stdio process and complete the MCP initialize handshake."""

        if self._initialized and self._process_is_alive():
            return
        async with self._init_lock:
            if self._initialized and self._process_is_alive():
                return
            self._initialized = False
            await self._ensure_process(restart=True)
            await self._request(
                "initialize",
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "feather", "version": "0.1.0"},
                },
            )
            await self._notify("notifications/initialized")
            self._initialized = True

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return all tools reported by the remote MCP server."""

        await self.initialize()
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        pages = 0
        while True:
            pages += 1
            if pages > _MCP_MAX_TOOL_PAGES:
                raise RuntimeError(
                    f"MCP server {self._server.label} exceeded tool page limit"
                )
            params = {"cursor": cursor} if cursor else None
            result = await self._request("tools/list", params)
            tools.extend(result.get("tools") or [])
            if len(tools) > _MCP_MAX_TOOLS:
                raise RuntimeError(
                    f"MCP server {self._server.label} exceeded tool count limit"
                )
            cursor = result.get("nextCursor")
            if not cursor:
                return tools
            if cursor in seen_cursors:
                raise RuntimeError(
                    f"MCP server {self._server.label} returned repeated tool cursor"
                )
            seen_cursors.add(cursor)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call one remote MCP tool and render its result content."""

        await self.initialize()
        result = await self._request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        rendered = _render_mcp_content(result.get("content") or [])
        if result.get("isError"):
            raise RuntimeError(rendered or f"MCP tool `{name}` failed")
        return rendered

    async def _ensure_process(self, *, restart: bool) -> None:
        if self._process_is_alive():
            return
        if self._process is not None and not restart:
            self._initialized = False
            raise RuntimeError(
                f"MCP stdio server `{self._server.label}` is not initialized."
            )
        self._initialized = False
        env = _stdio_child_env(self._server.env)
        cwd = str(Path(self._server.cwd).expanduser()) if self._server.cwd else None
        kwargs: dict[str, Any] = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        self._process = await asyncio.create_subprocess_exec(
            self._server.command,
            *self._server.args,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        await self._write_message(payload)

    async def _request(
        self, method: str, params: dict[str, Any] | None
    ) -> dict[str, Any]:
        async with self._request_lock:
            request_id = self._next_id
            self._next_id += 1
            payload: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }
            if params is not None:
                payload["params"] = params
            await self._write_message(payload)
            try:
                return await asyncio.wait_for(
                    self._read_response(request_id),
                    timeout=self._server.request_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                await self.aclose()
                raise TimeoutError(
                    f"MCP stdio server `{self._server.label}` did not respond "
                    f"to `{method}` within {self._server.request_timeout_seconds:.0f}s"
                ) from exc

    async def _read_response(self, request_id: int) -> dict[str, Any]:
        """Read messages until the response for ``request_id`` arrives."""

        while True:
            message = await self._read_message()
            if message.get("id") != request_id:
                continue
            if message.get("error"):
                raise RuntimeError(
                    f"MCP server {self._server.label} error: {message['error']}"
                )
            result = message.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"MCP server {self._server.label} returned invalid result"
                )
            return result

    async def _write_message(self, payload: dict[str, Any]) -> None:
        await self._ensure_process(restart=False)
        assert self._process is not None
        if self._process.stdin is None:
            raise RuntimeError(f"MCP server {self._server.label} stdin is closed")
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._process.stdin.write(data + b"\n")
        await self._process.stdin.drain()

    async def _read_message(self) -> dict[str, Any]:
        assert self._process is not None
        if self._process.stdout is None:
            raise RuntimeError(f"MCP server {self._server.label} stdout is closed")
        while True:
            line = await self._process.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP server {self._server.label} exited")
            stripped = line.strip()
            if not stripped:
                continue
            break
        try:
            message = json.loads(stripped.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"MCP server {self._server.label} wrote invalid stdout message"
            ) from exc
        if not isinstance(message, dict):
            raise RuntimeError(f"MCP server {self._server.label} returned non-object")
        return message

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            line = await process.stderr.readline()
            if not line:
                return
            logger.debug(
                "mcp_stdio_stderr server=%s text=%r",
                self._server.label,
                line.decode("utf-8", errors="replace").strip()[:500],
            )

    def _process_is_alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def _terminate_process(
        self, process: asyncio.subprocess.Process, sig: signal.Signals
    ) -> None:
        if os.name != "nt":
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, sig)
            return
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


class MCPClientManager:
    """Session-scoped cache for HTTP and stdio MCP clients."""

    def __init__(self) -> None:
        self._clients: dict[tuple[str, str], Any] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def client_for(self, session_id: str, server: MCPServerConfig) -> Any:
        """Return an initialized MCP client for ``session_id`` and ``server``."""

        key = (session_id, server.label)
        cached = self._clients.get(key)
        if cached is not None:
            return cached
        async with self._lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
        async with lock:
            cached = self._clients.get(key)
            if cached is not None:
                return cached
            client = self._build_client(server)
            try:
                await client.initialize()
            except Exception:
                await client.aclose()
                raise
            self._clients[key] = client
            return client

    async def aclose(self) -> None:
        """Close every cached MCP client."""

        clients = list(self._clients.values())
        self._clients.clear()
        self._locks.clear()
        for client in clients:
            await client.aclose()

    async def close_session(self, session_id: str) -> None:
        """Close cached MCP clients for one session."""

        clients: list[Any] = []
        async with self._lock:
            for key, client in list(self._clients.items()):
                if key[0] != session_id:
                    continue
                clients.append(client)
                self._clients.pop(key, None)
                self._locks.pop(key, None)
        for client in clients:
            await client.aclose()

    def _build_client(self, server: MCPServerConfig) -> Any:
        if server.transport == "stdio":
            return MCPStdioClient(server)
        return MCPStreamableHTTPClient(server)


class MCPProxyTool(BaseTool):
    """Local function-tool facade for an MCP server.

    OpenRouter's Chat Completions endpoint consumes function tools, so this
    proxy gives the model a normal Feather tool that can list and call remote
    MCP tools through Feather.
    """

    def __init__(
        self,
        server: MCPServerConfig,
        *,
        manager: MCPClientManager | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._server = server
        self._manager = manager
        self._transport = transport
        self.name = mcp_proxy_tool_name(server)
        self.description = (
            f"List or call tools from the remote MCP server `{server.label}`."
        )
        if server.server_description:
            self.description += f" {server.server_description}"
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_tools", "call_tool"],
                    "description": "Use list_tools to inspect available MCP tools, then call_tool to invoke one.",
                },
                "tool_name": {
                    "type": ["string", "null"],
                    "description": "Remote MCP tool name. Required for call_tool.",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the remote MCP tool.",
                    "additionalProperties": True,
                },
            },
            "required": ["action", "tool_name", "arguments"],
            "additionalProperties": False,
        }

    def get_prompt(self) -> str:
        """Describe the remote MCP server for prompt assembly."""

        allowed = (
            f" Allowed remote tools: {', '.join(self._server.allowed_tools)}."
            if self._server.allowed_tools
            else ""
        )
        return (
            f"- `{self.name}`: list and call tools from MCP server "
            f"`{self._server.label}`.{allowed}"
        )

    def to_openai_tool(self) -> dict[str, Any]:
        """Serialize the proxy as a non-strict function tool.

        The remote MCP call arguments are intentionally arbitrary JSON, which
        cannot be represented by OpenAI's strict structured-output subset.
        """

        tool = super().to_openai_tool()
        tool["strict"] = False
        return tool

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        """List or call a remote MCP server tool."""

        action = arguments["action"]
        tool_name = ""
        tool_args: dict[str, Any] = {}
        if action == "call_tool":
            tool_name = str(arguments.get("tool_name") or "").strip()
            if not tool_name:
                raise ValueError("`tool_name` is required for call_tool.")
            _ensure_allowed_tool(tool_name, self._server.allowed_tools)
            raw_tool_args = arguments.get("arguments") or {}
            if not isinstance(raw_tool_args, dict):
                raise ValueError("`arguments` must be an object.")
            tool_args = raw_tool_args
        elif action != "list_tools":
            raise ValueError(f"Unsupported MCP proxy action: {action}")

        if self._manager is not None:
            client = await self._manager.client_for(context.session_id, self._server)
            if action == "list_tools":
                tools = await client.list_tools()
                return ToolExecutionResult(
                    output=_cap_mcp_output(
                        json.dumps(
                            _filter_allowed_tools(tools, self._server.allowed_tools),
                            ensure_ascii=False,
                        )
                    )
                )
            output = await client.call_tool(tool_name, tool_args)
            return ToolExecutionResult(output=_cap_mcp_output(output))

        async with MCPStreamableHTTPClient(
            self._server, transport=self._transport
        ) as client:
            if action == "list_tools":
                tools = await client.list_tools()
                return ToolExecutionResult(
                    output=_cap_mcp_output(
                        json.dumps(
                            _filter_allowed_tools(tools, self._server.allowed_tools),
                            ensure_ascii=False,
                        )
                    )
                )
            output = await client.call_tool(tool_name, tool_args)
            return ToolExecutionResult(output=_cap_mcp_output(output))
        raise RuntimeError("MCP proxy client exited before producing a result")


def _decode_jsonrpc_response(response: httpx.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type.lower():
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("MCP server returned a non-object JSON-RPC response")
        return payload

    text = response.text.replace("\r\n", "\n")
    for raw_event in text.split("\n\n"):
        data_lines = [
            line[len("data: ") :]
            for line in raw_event.splitlines()
            if line.startswith("data: ")
        ]
        if not data_lines:
            continue
        payload_text = "\n".join(data_lines)
        if payload_text == "[DONE]":
            continue
        payload = json.loads(payload_text)
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("MCP SSE response did not contain a JSON-RPC payload")


def _render_mcp_content(content: list[Any]) -> str:
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
        else:
            parts.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(part for part in parts if part)


def _cap_mcp_output(output: str) -> str:
    if len(output) <= _MCP_PROXY_OUTPUT_LIMIT:
        return output
    return (
        output[:_MCP_PROXY_OUTPUT_LIMIT]
        + f"\n[MCP output truncated to {_MCP_PROXY_OUTPUT_LIMIT} characters]"
    )


def _stdio_child_env(configured: dict[str, str]) -> dict[str, str]:
    env = {
        key: value
        for key in _STDIO_ENV_ALLOWLIST
        if (value := os.environ.get(key)) is not None
    }
    env.update(configured)
    return env


def _sanitize_tool_label(label: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", label.strip().lower())
    normalized = normalized.strip("_")
    return normalized or "server"


def _filter_allowed_tools(
    tools: list[dict[str, Any]], allowed_tools: tuple[str, ...]
) -> list[dict[str, Any]]:
    if not allowed_tools:
        return tools
    allowed = set(allowed_tools)
    return [tool for tool in tools if tool.get("name") in allowed]


def _ensure_allowed_tool(tool_name: str, allowed_tools: tuple[str, ...]) -> None:
    if allowed_tools and tool_name not in set(allowed_tools):
        raise ValueError(f"MCP tool `{tool_name}` is not allowed by config.")
