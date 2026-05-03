# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
