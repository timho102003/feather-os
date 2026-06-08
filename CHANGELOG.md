# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **Prompt caching now caches only the stable prefix.** The system-prompt
  cache breakpoint is anchored at the end of the static section (identity,
  tools, skill/MCP catalogs, dispatch catalog) instead of wrapping the whole
  prompt, so per-turn dynamic content (recalled memory, loaded skill bodies)
  no longer invalidates the cached prefix every turn. Claude additionally
  caches the growing conversation via a rolling `cache_control` breakpoint on
  the last message. The static dispatchable-agent catalog moved into the
  cached prefix. Per-turn cache read/write/hit-rate is logged (`cache.usage`)
  so caching can be verified at runtime.

### Fixed

- **`cache_strategy: gemini_breakpoint` now actually enables caching.** It was
  silently treated as "no caching" (only `anthropic_breakpoint` was handled);
  Gemini honors the `cache_control` hint via OpenRouter.

## [0.1.1] - 2026-05-08

### Added

- **Anthropic Claude (Messages API) provider.** Third LLM provider at
  feature parity with OpenAI Responses and OpenRouter Chat: streaming,
  multi-turn, parallel tool calls, prompt caching (`cache_control`
  ephemeral breakpoints), extended thinking (`thinking.budget_tokens`),
  multimodal (image + PDF), structured outputs via the forced
  single-tool pattern, retry on 408/409/429/500/503/504/529 with
  `retry-after` honoring, and idle/wall-clock stream timeouts. Opt in
  via `active_provider: claude` plus a `claude:` block in `app.yaml`
  and `ANTHROPIC_API_KEY` in your env. Default model is
  `claude-opus-4-7`. Per-agent overrides via `provider:` /
  `model:` in the agent YAML are honored. The translator gates
  `temperature` for models that reject it (Opus 4.7+, Mythos), mirroring
  OpenAI's gpt-5 reasoning-model handling.
- **Three-provider onboarding flow.** The onboarding wizard now asks
  three independent yes/no questions ("wire OpenAI?", "wire OpenRouter?",
  "wire Claude?") and prompts for each chosen provider's API key. When
  more than one is wired the wizard asks which should be the default
  `active_provider`. When none is chosen it aborts with a clear "you
  need to wire at least one" message. `.env.example` lists all three
  keys with section headers.
- OpenRouter trace metadata broadcast. Opt-in via the new
  `openrouter.tracing` block in `app.yaml`. When enabled, every
  OpenRouter turn carries `session_id`, an optional `user`, and a
  structured `trace` object (with `trace_name`, `generation_name`, and
  `feather_*` identity keys) that OpenRouter forwards to every
  configured observability destination — Comet Opik, Langfuse, OTel
  collectors, and so on. Default is off; the wire body is byte-identical
  to prior behaviour for anyone who hasn't opted in.
- Operator-supplied static `trace.metadata` values are clamped to
  OpenRouter's published limits (16 keys, 64-char keys, 512-char values)
  so a typo can't trigger a 400. Reserved Feather identity keys always
  win over operator metadata of the same name.
- Lead worker subprocess substrate. Opt-in via the new
  `FEATHER_USE_LEAD_WORKER=1` env var. When enabled, the lead agent
  runs as a separate `python -m feather.lead_worker_entry` subprocess
  with the Textual TUI as its supervisor; they communicate via a
  JSONL command/event protocol over stdin/stdout plus a new
  `worker_heartbeats` SQLite table. Default is off; default users see
  byte-identical behaviour. Substrate for upcoming out-of-band hang
  detection (the supervisor's `is_stale()` is wired but the UI banner
  ships in a follow-up) and self-repair restart-resume (a
  `request_restart` tool ships in a follow-up). See
  [docs/architecture.md](docs/architecture.md#lead-worker-mode-opt-in)
  and [docs/configuration.md](docs/configuration.md#lead-worker-mode-opt-in).
- Worker-mode runtime guards: when `FEATHER_USE_LEAD_WORKER=1` is set,
  `FeatherRuntime.start_background_services` skips the cron scheduler
  and the messaging adapters. Both build their own in-process
  `BaseAgent` and would race the worker on the shared `sessions` row,
  silently corrupting `last_response_id`, `pending_inputs`, and
  `messages.sequence`.
- Hang-detection banner. In worker mode, the TUI polls
  `LeadSupervisor.is_stale()` every 2 s and surfaces a red
  `Lead unresponsive` conversation marker on transition into stale
  (heartbeat older than 5 s) and a green `Lead recovered` marker on
  recovery. State machine fires on transitions only, so a sustained
  hang produces one alert rather than one per tick.
- `/restart-lead` slash command. Manual recovery hook for the hang
  banner and for "I just patched the lead's code, please reload it."
  Cancels any in-flight run, calls `LeadSupervisor.restart()` (which
  does the SIGTERM → SIGKILL dance + respawns on the same
  `--session-id`). Worker-mode only; surfaces a clear "worker mode is
  off" notice when called against the in-process default path.
- Error-triage bot. In worker mode, the supervisor tails
  `.feather/logs/feather.log` for ERROR-level entries and posts a
  single summary message into the lead's mailbox so the lead can
  investigate. Two-layer dedup (high-water mark + bounded recency
  set), auto-resets on log rotation (inode change), reads off-thread
  via `asyncio.to_thread` so the supervisor's event loop stays
  responsive.
- `request_restart` tool. Self-repair primitive: the lead calls it
  after patching `feather/*` modules and verifying the change. The
  tool itself does not kill any process — it writes a new
  `sessions.restart_requested_at` flag. The supervisor's restart
  watcher polls the flag, calls `LeadSupervisor.restart()` (serialized
  via a dedicated `_restart_lock` so it can't race a concurrent
  `/restart-lead`), and posts a "restart succeeded / failed" message
  back into the lead's inbox so the conversation continues naturally.
- Install-mode detection. `feather.core.install_mode.detect_install_mode`
  classifies the running install as `editable`, `wheel`, or `read_only`.
  Surfaced in `request_restart`'s response so the model can warn the
  user when a patch will be silently overwritten on the next
  `pip install --upgrade`.
- `submit_github_report` tool + skill. File an issue (PR support
  deferred) on the upstream repo via the `gh` CLI. The
  `submit-github-report` skill (loaded on demand) carries the issue
  template, hard rules ("never auto-submit", "one bug per report",
  "tests must pass first"), and the "ask the user before sending"
  flow. Body length is checked in UTF-8 bytes (not chars) so
  CJK/emoji-heavy bodies don't slip past Feather's check just to be
  rejected by GitHub. The URL is extracted by scanning for the first
  `https://github.com/*` line so a `gh` release-update warning before
  the URL doesn't break parsing.
- New `sessions.restart_requested_at` and `sessions.restart_reason`
  columns (idempotent migration via `ensure_column`). Pre-existing
  flags on launch are cleared so a SIGKILLed prior session can't
  trigger a surprise restart of a freshly-spawned worker.
- Onboarding wizard asks once whether to enable the **self-repair
  safety net** (the user-facing name for the lead-worker subprocess
  feature) and writes the answer to `self_repair.enabled` in
  `app.yaml`. New `SelfRepairConfig` dataclass + parser layer the
  three signals: env var (`FEATHER_USE_LEAD_WORKER`) wins, then YAML
  setting, then default-off. The env var also accepts explicit
  falsy values (`0`, `false`, `no`, `off`) so a user with the YAML
  setting on can disable the feature for a single launch.

### Changed

- **Cron scheduler now routes through the agent-message mailbox.**
  When a job fires, the scheduler drops one message into the lead's
  inbox via `agent_message_store.send` and marks the job succeeded.
  The lead's existing `resume_on_inbox` path picks it up under the
  lead's own session lock. Old design built an in-process `BaseAgent`
  per fire and called `agent.run` synchronously, which raced the
  worker on shared session state when self-repair was on — that was
  the only reason cron was paused in safety-net mode. **Cron now
  runs in both modes.** Behavior diff: the cron job is marked
  succeeded as soon as the mailbox row commits, not after the agent
  finishes the turn; an LLM failure on the queued turn shows up via
  the agent's normal turn-completion UI rather than as a
  `scheduled_task_failed` cron event. The
  `scheduled_task_completed` event no longer fires (cron does not
  observe completion).
- `write_file` now writes anywhere inside the workspace (the discovered
  project root, or the directory `feather` was launched from when no
  project is detected), matching the `bash` tool's `cwd` constraint.
  Previously only `.feather/` and `config/` subdirs were writable,
  which prevented authoring deliverables in the user's own working
  directory after `pip install feather-agent-os`. `~/.feather/config/`
  and `~/.feather/skills/` remain whitelisted for global agent / skill
  installs. Paths that escape the workspace via `..` or symlinks are
  still rejected.
- `agent-creator` skill and the shared output-artifact protocol now
  recommend `write_file` first for deliverable / YAML writes; the bash
  heredoc patterns are kept as a fallback for chained shell steps.
- `read_file` now also reads anywhere under the global root
  (`~/.feather/`, or `$FEATHER_HOME` when set) when `FeatherPaths` is
  wired through the agent factory, so agents can inspect global agent
  YAMLs, skill bodies, `user.md`, and state markers without shelling
  out. The workspace remains the default sandbox; paths that escape
  every allowed root via `..` or symlinks are still rejected. As
  defense-in-depth, any filename starting with `.env` (e.g. `.env`,
  `.env.local`, `.env_backup`, `.envrc`) is refused by `read_file` to
  avoid leaking API keys into chat — use the `bash` tool if you
  genuinely need to inspect them. The deny is name-based after symlink
  resolution; hardlinks with non-`.env` names are NOT caught (best-
  effort defense in depth, not a sandbox).
- `read_file` now expands `~` in paths via `$HOME` so `~/.feather/...`
  works as written; reads outside the workspace but under `$HOME`
  render in the output with a `~/` prefix so the user's real `$HOME`
  is not echoed into chat.

### Fixed

- **Synthetic system messages from `log_triage_bot`,
  `restart_watcher`, and the new cron path were stranded forever
  due to a case-sensitive SQL filter on the lead's inbox.** Senders
  were addressing `to_agent_name="lead"` (lowercase) but the lead's
  `BaseAgent` filters by `agent_config.name` which is `"Lead"`
  (capital L, from `lead.yaml`), and SQLite string equality is
  case-sensitive. Result: every triage notification, restart
  acknowledgement, and (after this PR's mailbox-routed cron) cron
  prompt was silently dropped. Centralised the canonical name as
  `feather.core.constants.LEAD_AGENT_NAME` and routed all three
  senders through it. Added an end-to-end test that pins the
  invariant by querying both the canonical name (must return the
  message) and the wrong case (must return nothing).
- **Textual TUI failed to start with `AttributeError: 'FeatherRuntime'
  object has no attribute 'config'`.** The TUI's `on_mount` reads
  `self._runtime.config.self_repair.enabled` to decide whether to
  launch the lead in a worker subprocess, but `FeatherRuntime` only
  kept `app_config` as a local in `create()` — it was never persisted
  on the instance. The runtime now stores the loaded `AppConfig` and
  exposes it via a `config` property; a regression test pins the exact
  attribute chain so a future `AppConfig` refactor can't silently
  break TUI startup.

## [0.1.0] - 2026-05-02

First public release on PyPI as `feather-agent-os`.

### Highlights

- Single lead agent driving the OpenAI Responses API or OpenRouter
  Chat Completions, with streaming, prompt caching, reasoning effort
  control, and per-agent provider overrides.
- Textual TUI as the default chat surface, plus a Rich streaming CLI
  for plain terminals.
- Built-in tools: `read_file`, `read_pdf`, `write_file`, `grep`,
  `bash`, `ask_user`, `load_skill`, `web_search`, `web_fetch`,
  `recall_memory`, `manage_memory`, `user_info`, cron tools,
  sub-agent tools, task tools, and MCP discovery tools.
- Background sub-agents (`explore`, `research`, `validate`, plus your
  own custom ones) dispatched as Python subprocesses with a SQLite
  inter-agent mailbox.
- Optional long-term memory backed by Qdrant and Gemini embeddings,
  with a separate provider call for fact extraction and classification.
- Optional MCP server integration (stdio and HTTP), activated on
  demand per session.
- Optional messaging integrations for Telegram, LINE, and WhatsApp.
- Cron-style scheduled prompts that fire back into the lead session.
- Skill catalog with progressive loading from packaged, global, and
  project sources; later sources override by name.
- Layered configuration: packaged defaults, user-global overrides at
  `~/.feather/config/app.yaml`, and per-project agent overrides.
- Hybrid state layout: personal config and skills live in `~/.feather/`,
  per-project sessions and artifacts live in `./.feather/`. Walk-up
  detection chooses one automatically.
- CLI subcommands: `feather init`, `feather init-memory`,
  `feather stop-memory`, `feather remove-memory`, `feather onboard`,
  `feather cli`, `feather tui`, plus `--version`, `--project`,
  `--session-id`, and `FEATHER_HOME` / `FEATHER_PROJECT_ROOT`
  environment overrides.
- Onboarding wizard for first-run setup of identity and API keys.
- PEP 561 typed package (`py.typed`); Python 3.12 and 3.13 supported.

### Build and release

- Hatchling build backend with `hatch-vcs` for git-tag-driven
  versioning.
- GitHub Actions release workflow publishes to PyPI via OIDC
  Trusted Publisher; no API tokens stored.
