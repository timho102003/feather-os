# MCP Servers

MCP stands for **Model Context Protocol**. It is the protocol Anthropic
introduced for letting an LLM safely call tools that live in a separate
process. Feather can use any MCP server, whether it speaks over a
local stdio pipe (`npx ...`, `docker run ...`) or over an HTTP endpoint
(remote SaaS, internal service).

You do not need to understand the protocol to use it. Pick a server,
add a few lines of YAML, and tell the agent to use it.

## The fast path

Ask the lead in chat:

> "Add the Playwright MCP server so I can drive a browser."

The lead loads the built-in `mcp-config` skill and asks any clarifying
questions, writes the YAML for you, and tells you what to do next.

The rest of this guide is for when you'd rather edit the file by hand
or want to know what's going on under the hood.

## Where the config lives

`~/.feather/config/app.yaml`, in the `mcp:` section.

```yaml
mcp:
  enabled: true
  servers: {}              # one entry per server
```

Set `enabled: true` to expose the discovery tools (`list_mcp_servers`,
`register_mcp_server`) to the agent. Without that, no MCP work happens
even if you wrote server entries.

## Stdio servers

Use stdio when the MCP server is a command Feather should launch
locally. Example: Playwright via `npx`.

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

What each field does:

* `command` and `args`: what Feather runs to start the server. The
  process is started only when the agent registers the server, not at
  startup.
* `description`: what the agent sees when it lists available servers.
  One sentence.
* `providers`: which model providers can use this server. Usually
  both, sometimes just one.
* `agents`: which agents can use it. Usually `[Lead]` only, since the
  lead is in conversation with you.
* `require_approval`: must be `never`. Approval flows are not
  implemented yet.

## HTTP servers

Use HTTP when the server already exposes a Streamable HTTP endpoint.

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

The fields are the same except `url` replaces `command`/`args`.
`allowed_tools` is optional. Use it when the remote server exposes a
lot of tools and you only want to expose a few of them to the agent.

## HTTP servers with auth headers

Never paste secrets into YAML. Reference an environment variable
instead:

```yaml
mcp:
  enabled: true
  servers:
    private_docs:
      url: https://example.com/mcp
      description: Internal company docs
      providers: [openai]
      agents: [Lead]
      header_envs:
        Authorization: PRIVATE_DOCS_AUTH_HEADER
      require_approval: never
```

Then put the actual value in `~/.feather/.env`:

```
PRIVATE_DOCS_AUTH_HEADER=Bearer eyJhbGciOi...
```

`header_envs` accepts any header name and any env var name. Whatever is
in the env at runtime gets sent as that header. The full string goes
through, so include the `Bearer ` (or `Token `, etc.) prefix in the env
value.

## How the agent activates a server

Feather does not connect to every server at startup. Instead, it
registers two tools the agent can call:

1. `list_mcp_servers`: show every server in the config that this
   agent and provider are allowed to use.
2. `register_mcp_server`: actually open the connection for the
   current session.

Once registered, the server's tools become callable for the rest of
the session. The session DB remembers which servers are active so a
resumed session reconnects.

The activation pattern matters because some MCP servers are slow to
start (Playwright launches a browser, for example). Spinning them up
only on demand keeps cold-starts fast.

## Per-provider transport

* **OpenAI** can receive HTTP MCP servers natively as a remote tool.
* **OpenRouter** does not (yet) support remote MCP. For OpenRouter
  Feather wraps every active server as a local proxy tool named
  `mcp_<label>` and forwards calls itself.
* **Stdio** servers always go through the local proxy regardless of
  provider.

You don't need to think about this; Feather picks the right path. It
matters when you write `providers:` because you might want to scope a
server to a single provider.

## Listing what's wired up

In a chat:

* Ask the agent: "what MCP servers can you use?" The agent will call
  `list_mcp_servers` and print the catalog.
* `/integrations` shows which messaging integrations are connected.
  This is separate from MCP but lives nearby in your mental model.

## Things to avoid

* **Do not put secrets in YAML.** Use `header_envs` for HTTP, or
  reference env vars in `args` for stdio.
* **Do not enable a stdio command you do not trust.** It runs on your
  machine with your permissions.
* **Do not over-broaden `agents`.** Most MCP servers should only be
  available to the lead. Sub-agents are dispatched for narrow jobs;
  giving them broad MCP access is a foot-gun.

## Next

* Want to write a skill that bundles "use this MCP for X"? See
  [skills.md](skills.md).
* Want to see every config knob in `mcp:`? See
  [configuration.md](configuration.md#mcp).
