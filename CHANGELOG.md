# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

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
