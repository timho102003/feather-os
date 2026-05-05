# Configuration Reference

Every knob Feather exposes, grouped by section. The shipped defaults
work for most people; come here when you want to change something
specific.

## How config layering works

Three layers, bottom-up:

1. **Packaged defaults** baked into the wheel at
   `feather/_resources/config/app.yaml`. Always loaded.
2. **Global overrides** at `~/.feather/config/app.yaml`. Whatever you
   put here deep-merges over the packaged defaults.
3. **Project-local config** is *not* a thing for `app.yaml`. The same
   global config applies to every project. If you need different
   settings for a specific project, run that project under a different
   `FEATHER_HOME` (see "Paths" below).

For agent YAMLs (`config/agents/<name>.yaml`) the layering is
**first-hit-wins**: the project's `config/agents/<name>.yaml` (if any)
beats the global `~/.feather/config/agents/<name>.yaml`, which beats
the packaged default.

## Database

```yaml
database:
  path: .feather/db/feather.db
```

Where the SQLite file lives, relative to the project root. Holds
sessions, messages, tasks, cron jobs, integrations, agent message
mailbox. In global-only mode the runtime ignores this path and uses
`~/.feather/state/sessions.db` instead.

## Storage

```yaml
storage:
  temp_directory: .feather/tmp
```

Where overflow tool output goes when a tool returns more text than the
inline cap. Each overflow becomes a file under this directory; the
chat row stores a reference to the file path.

## Logging

```yaml
logging:
  path: .feather/logs/feather.log
  level: INFO
```

Path is relative to the project root in project mode, or to
`~/.feather/state/` in global mode. Levels: `DEBUG`, `INFO`,
`WARNING`, `ERROR`.

## Compaction

```yaml
compaction:
  enabled: true
  trigger_ratio: 0.8
  context_window_tokens: 400000
  model:
  max_output_tokens: 2000
  temperature: 0.2
```

When the chat fills 80% of the model's context window, Feather
summarizes the older messages with a separate model call and continues
the conversation from the summary. `model:` blank means "use the
current conversation model"; set it to pin a different (cheaper) model
for compaction.

## Skills

```yaml
skills:
  directory: .feather/skills
```

Where to look for project-local skills. Global skills live at
`~/.feather/skills/` and don't need a config entry.

## Scheduler

```yaml
scheduler:
  enabled: true
  poll_interval_seconds: 2
  failure_retry_seconds: 30
  max_due_jobs_per_tick: 10
```

Drives the cron tools (see [scheduling.md](scheduling.md)). Set
`enabled: false` to freeze every scheduled job.

## active_provider

```yaml
active_provider: openai           # or: openrouter
```

Top-level switch. Decides which `*_provider` block the runtime builds.
Switching here flips every agent that does not pin its own provider.

## openai

```yaml
openai:
  api_key_env: OPENAI_API_KEY
  model: gpt-5-mini
  max_output_tokens: 16000
  temperature: 1.0
  parallel_tool_calls: true
  prompt_cache_key: feather-lead
  prompt_cache_retention: in_memory
  store: true
  reasoning:
    effort: low                   # minimal | low | medium | high
    summary: auto
  stream_idle_timeout_seconds: 90
```

* `api_key_env`: env var name to read the key from. Change only if you
  use a non-default name.
* `model`: any OpenAI Responses-API model.
* `temperature`: silently dropped on GPT-5 family models.
* `prompt_cache_*`: enables prefix caching across turns. Big win for
  long-running sessions.
* `reasoning.effort`: only meaningful on reasoning models (gpt-5*).
  Higher effort = better answers, slower and more expensive.

## openrouter

```yaml
openrouter:
  api_key_env: OPEN_ROUTER_API_KEY
  base_url: https://openrouter.ai/api/v1
  http_referer: https://github.com/timho102003/feather-os
  app_title: Feather
  model: qwen/qwen3.6-plus
  max_output_tokens: 32000
  temperature: 0.7
  parallel_tool_calls: false
  stream_idle_timeout_seconds: 120
  request_timeout_seconds: 180
  max_attempts: 3
  reasoning:
    effort: medium
  cache_strategy: anthropic_breakpoint
  provider_preferences:
    only: ["Alibaba"]              # restrict to specific upstreams
    require_parameters: true       # drop providers that ignore tools
    allow_fallbacks: false         # surface 503s instead of re-routing
  fallback_models:
    - qwen/qwen3.5-plus-02-15
  tracing:                          # optional; default off
    enabled: false                  # flip to true to broadcast trace metadata
    user: ops@example.com           # ≤128 chars; identifies you in trace UI
    metadata:                       # static keys merged into trace object
      env: prod
      build_sha: abc123
```

* `cache_strategy: anthropic_breakpoint`: wraps the system prompt in
  a content block with `cache_control: ephemeral` for Anthropic-style
  providers (Claude, Z.ai, DeepSeek, Moonshot).
* `provider_preferences.require_parameters: true`: strongly
  recommended whenever the agent uses tools.
* `fallback_models`: tried in order if the primary model is
  unavailable.
* `tracing`: opt-in observability metadata. When `enabled: true`,
  every OpenRouter request body carries `session_id`, an optional
  `user`, and a structured `trace` object that OpenRouter forwards
  to every observability destination configured on its dashboard
  (Comet Opik, Langfuse, OTel, Sentry, Grafana, webhooks). Default
  off; the wire body is byte-identical to prior behaviour for anyone
  who has not opted in. Operator-supplied `metadata` values are
  clamped to OpenRouter's published limits (16 keys, 64-char keys,
  512-char values). Reserved Feather identity keys (`trace_name`,
  `generation_name`, `feather_app`, `feather_agent_name`,
  `feather_agent_role`, `feather_session_id`) always win over
  operator metadata of the same name. See
  [providers.md](providers.md#sending-traces-to-comet-opik-and-other-observability-platforms)
  for the full walkthrough.

The packaged `openrouter-examples/` folder has tested, drop-in blocks
for popular models. See [providers.md](providers.md).

## parallel

```yaml
parallel:
  api_key_env: PARALLEL_API_KEY
  default_search_mode: fast        # fast | one-shot | agentic
  max_results: 5
  inline_full_content_threshold: 4000
```

Web search via Parallel AI. Without `PARALLEL_API_KEY` in your env,
the `web_search` and `web_fetch` tools are silently disabled.

## mcp

```yaml
mcp:
  enabled: false
  servers:
    # one entry per server; see docs/mcp.md
```

Set `enabled: true` to expose the MCP discovery tools to the agents.
Each server entry can have `command`/`args` (stdio) or `url` (HTTP),
`description`, `providers`, `agents`, `allowed_tools`,
`require_approval`, and `header_envs`. See [mcp.md](mcp.md) for the
full schema and examples.

## memory

```yaml
memory:
  enabled: true                    # global gate; the marker is the user-facing switch
  qdrant:
    url: http://localhost:6333     # env QDRANT_URL wins over this
    api_key_env: QDRANT_API_KEY
    collection_name: feather_memory_v2
    embedding_dims: 3072
    hnsw_m: 32
    hnsw_ef_construct: 256
    hnsw_ef_search: 128
    hnsw_full_scan_threshold: 10000
    indexing_threshold: 20000
    default_segment_number: 2
    on_disk_vectors: false
    on_disk_payload: false
    prefer_grpc: false
    request_timeout_s: 15.0
  embedding:
    provider: gemini
    model: gemini-embedding-2-preview
    output_dimensionality: 3072
    task_type_document: RETRIEVAL_DOCUMENT
    task_type_query: RETRIEVAL_QUERY
    normalize_reduced_dims: true
    request_timeout_s: 30.0
    max_retries: 3
    retry_backoff_s: 1.5
  chunking:
    chunk_size_tokens: 1000
    chunk_overlap_tokens: 100
    tokenizer: tiktoken
    tokenizer_encoding: o200k_base
  retrieval:
    enabled: true
    top_k_prompt_injection: 5
    top_k_tool: 10
    score_threshold: 0.5
    classifier_top_k: 3
    classifier_score_threshold: 0.75
    retrieval_timeout_s: 15.0
    query_builder_enabled: true
    query_builder_recent_messages: 8
  trigger:
    enabled: true
    trigger_turns: 10              # extract every N user turns
    skip_compact_messages: true
    background: true
    shutdown_timeout_s: 120.0
    max_concurrent_extractions_per_session: 1
  extraction:
    provider: openai               # null = inherit active_provider
    model: gpt-5.4-nano
    max_output_tokens: 2000
    temperature: 0.1
  classification:
    provider: openai
    model: gpt-5.4-nano
    max_output_tokens: 400
    temperature: 0.0
  query_builder:
    provider: openai
    model: gpt-5.4-nano
    max_output_tokens: 200
    temperature: 0.0
```

The three sub-blocks at the bottom (`extraction`, `classification`,
`query_builder`) let you pick a different provider/model for each
memory operation than for the conversation itself. The shipped
defaults pin them to cheap OpenAI nano because they need
structured output, which nano handles well at low cost.

See [memory.md](memory.md) for what each operation does.

## Per-agent YAML

`~/.feather/config/agents/<name>.yaml` (or
`./.feather/config/agents/<name>.yaml` for project scope). The shape:

```yaml
name: Lead                         # display name
role: lead                         # internal role key
personality: Decisive, structured, business-like.
provider: openai                   # optional, overrides app-level active_provider
model: gpt-5                       # optional, overrides openai.model / openrouter.model
reasoning:
  effort: medium
  summary: auto
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
  - feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT
memory_enabled: true
registered_tools:
  - read_file
  - write_file
  - bash
  - load_skill
  # ... see docs/tools-and-commands.md
```

`prompt_modules` is a list of dotted `module:symbol` references. The
runtime imports each one and concatenates the resulting strings to
form the system prompt for that agent.

`registered_tools` controls which tools the agent has. Lead-only tools
will be rejected if you list them in a sub-agent.

## Paths

| Env var | Default | What it sets |
|---|---|---|
| `FEATHER_HOME` | `~/.feather` | The global config root. |
| `FEATHER_PROJECT_ROOT` | (walk-up) | The project root, skipping the walk-up search. |

Path resolution rules:

* If `FEATHER_PROJECT_ROOT` is set, that's the project root.
  Otherwise, Feather walks up from your current directory looking for
  the first folder that contains a `.feather/` subdirectory.
* Walk-up stops at your home directory or the filesystem root.
* If no `.feather/` is found, Feather runs in **global-only** mode and
  stores sessions in `~/.feather/state/sessions.db`.
* `feather init` inside any folder creates `./.feather/` there and
  registers it in the global projects index at
  `~/.feather/state/projects.json`.

Full path layout is documented in
[getting-started.md](getting-started.md#where-things-live).

## Environment variables Feather reads

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI provider auth. Required for the default config. |
| `OPEN_ROUTER_API_KEY` | OpenRouter provider auth. |
| `GEMINI_API_KEY` | Gemini embeddings for long-term memory. |
| `QDRANT_URL` | Where Qdrant lives. Beats the marker and `app.yaml`. |
| `QDRANT_API_KEY` | Qdrant auth header (cloud Qdrant). |
| `PARALLEL_API_KEY` | Parallel AI key for `web_search` / `web_fetch`. |
| `FEATHER_HOME` | Override the global state root. |
| `FEATHER_PROJECT_ROOT` | Skip the walk-up search and pin a project. |
| `FEATHER_USE_LEAD_WORKER` | Opt in to running the lead agent as a separate worker subprocess. See [Lead worker mode (opt-in)](#lead-worker-mode-opt-in) below. |

Feather loads `~/.feather/.env` first, then `./.env` from the project
root with override-on. So a project `.env` wins over the global one.

## Lead worker mode (opt-in)

Default: **off**. With the env var unset, the lead agent runs in the
same Python process as the Textual TUI — the long-standing behavior. No
configuration change is needed for typical use.

Set `FEATHER_USE_LEAD_WORKER=1` (also accepted: `true`, `yes`, `on`) to
spawn the lead as a separate `python -m feather.lead_worker_entry`
subprocess. The TUI becomes the supervisor: it talks to the worker via
the worker's stdin/stdout (one JSON line per command/event) and watches
a new `worker_heartbeats` SQLite table for liveness. This mode is the
substrate for upcoming self-repair (the lead patches its own code and
asks for a clean restart) and for out-of-band hang detection.

Defaults that govern the worker's heartbeat cadence and the supervisor's
staleness threshold are not user-configurable in `app.yaml` yet — they
ship as 1 s and 5 s respectively (`feather.core.lead_supervisor`).

**Known limitations of worker mode in this release** — both are
deliberate guards that the runtime enforces automatically when the env
flag is set:

* The cron scheduler is not started. Cron jobs build their own
  in-process `BaseAgent` and would race the worker on the shared
  `sessions` row.
* Messaging integrations (Telegram, LINE, WhatsApp) are not started for
  the same reason — their inbound queue is the TUI-process input queue
  the worker can't see.

If you need scheduled jobs or messaging webhooks in this session, leave
the env flag unset.
