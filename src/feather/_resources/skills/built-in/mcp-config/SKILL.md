---
name: mcp-config
description: Use when the user asks in plain English to add, connect, configure, register, or troubleshoot an MCP server integration. Guides the agent to collect missing server details, write the `mcp:` section in `~/.feather/config/app.yaml`, avoid leaking secrets, and explain how the user can activate the MCP on demand with `list_mcp_servers` and `register_mcp_server`.
---

# MCP Config

Use this skill when a user says things like "connect Playwright MCP",
"add the GitHub MCP", "make this MCP available to OpenRouter", or "set up an
MCP server for the lead agent."

Goal: turn plain English into a safe `~/.feather/config/app.yaml` `mcp:` entry. The user
should not need to know the YAML shape up front.

## Workflow

1. Identify the requested MCP server.
2. If any required detail is missing, ask one focused question.
3. Edit `~/.feather/config/app.yaml`.
4. Verify config parsing and relevant MCP tests.
5. Tell the user the server label and how the agent will activate it.

## Required Details

Collect only what is missing:

- **label**: short stable id, lowercase if possible, e.g. `playwright`.
- **transport**:
  - `stdio` when Feather should launch a local command like `npx`.
  - `http` when the MCP server already exposes a Streamable HTTP URL.
- **command and args** for stdio, or **url** for HTTP.
- **providers**: usually `[openai, openrouter]`, or narrower if requested.
- **agents**: usually `[Lead]`, or the specific agent names that need it.
- **description**: one short sentence for `list_mcp_servers`.
- **allowed_tools**: add when the remote server exposes many tools and the user
  only needs a few.
- **secrets**: never put secret values in YAML. Use `env` for stdio variable
  names/values only when safe, or `header_envs` for HTTP headers backed by
  environment variables.

Feather currently supports `require_approval: never` for MCP activation. If the
user needs approval-gated remote actions, explain that approval flows are not
supported yet and keep the server out of active config until support exists.

## Config Patterns

Stdio server launched by Feather:

```yaml
mcp:
  enabled: true
  servers:
    playwright:
      command: npx
      args: ["-y", "@playwright/mcp@latest"]
      description: Browser automation through Playwright
      providers: [openai, openrouter]
      agents: [Lead]
      require_approval: never
```

HTTP server passed natively to OpenAI and proxied for OpenRouter:

```yaml
mcp:
  enabled: true
  servers:
    openai_docs:
      url: https://developers.openai.com/mcp
      description: OpenAI developer documentation
      providers: [openai, openrouter]
      agents: [Lead]
      allowed_tools: [search_openai_docs, fetch_openai_doc]
      require_approval: never
```

HTTP server with an authorization header from an environment variable:

```yaml
mcp:
  enabled: true
  servers:
    private_docs:
      url: https://example.com/mcp
      description: Private company docs
      providers: [openai]
      agents: [Lead]
      header_envs:
        Authorization: PRIVATE_DOCS_AUTH_HEADER
      require_approval: never
```

## How Feather Uses The Config

- Agents do not connect to every MCP at startup.
- Configured servers appear through the small `list_mcp_servers` tool.
- The agent calls `register_mcp_server` only when the task needs that server.
- Session state records the active label in `active_mcp_servers`.
- OpenAI + HTTP MCP uses native `type: "mcp"` tools.
- OpenRouter or stdio MCP uses a local proxy tool named `mcp_<label>`.

## Safety Rules

- Do not invent commands, package names, URLs, headers, or tool allowlists.
- Prefer official install/config docs when the requested MCP package is unknown.
- Do not embed API keys or bearer tokens in `~/.feather/config/app.yaml`.
- For stdio, only configure commands the user trusts to run locally.
- Keep providers and agents scoped as narrowly as the user needs.
- Add `allowed_tools` when the server is broad or action-capable.

## Verification

After editing `~/.feather/config/app.yaml`, run the narrow checks first:

```bash
env PYTHONPATH=src ../.venv/bin/pytest tests/test_mcp_config.py tests/test_mcp_tools.py
```

If the change touches agent wiring or examples, also run:

```bash
env PYTHONPATH=src ../.venv/bin/pytest tests/test_agent_factory.py tests/test_mcp_client.py
```

For a stdio server the user explicitly wants tested, smoke test by registering
or by using `MCPStdioClient` with a short timeout. Report success as "Feather
started the MCP command and listed tools", not as a guarantee that every remote
tool action is safe.

## Response Shape

When done, keep the user-facing answer short:

- State the label that was added.
- State whether it is stdio or HTTP.
- State who can use it: providers and agents.
- Explain: "Ask the agent to use `<label>`; it will list and register the MCP
  only when needed."
