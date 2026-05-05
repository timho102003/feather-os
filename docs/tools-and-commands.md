# Tools, Slash Commands, and Keyboard Shortcuts

This is the reference page. Three sections, three tables.

* **Tools** are the things the agent itself can call. You don't invoke
  them directly; the agent does.
* **Slash commands** are things *you* type into the input box during a
  chat.
* **Keyboard shortcuts** drive the Textual TUI.

## Tools the agent can call

These are the tools wired to the lead agent by default. Sub-agents see
a smaller subset (see [agents.md](agents.md) for the per-agent tool
list).

### File and shell

| Tool | What it does |
|---|---|
| `read_file` | Read a UTF-8 text file, optionally a slice by line range. Reads anywhere in the workspace and anywhere under `~/.feather/` (global config, agent YAMLs, skills, `user.md`, state markers). Files whose name starts with `.env` (e.g. `.env`, `.env.local`, `.envrc`) are refused — use `bash` if you genuinely need them. Paths outside every allowed root are rejected. |
| `read_pdf` | Extract readable text from a PDF. Modes: `auto`, `text`, `opendataloader_hybrid`. |
| `write_file` | Write a UTF-8 file anywhere inside the workspace (the project root, or the working directory `feather` was launched from). Paths that escape via `..` or symlinks are rejected. Also whitelists `~/.feather/config/` and `~/.feather/skills/` for global agent / skill installs. |
| `grep` | Regex search across the workspace, scoped by path. |
| `bash` | Run a short bash command. Default 10 s timeout, 4000 char output cap. |

### Conversation control

| Tool | What it does |
|---|---|
| `ask_user` | Pause the loop and ask you a focused question. Used when the agent is genuinely stuck. |
| `load_skill` | Load the full body of a skill from the catalog. |

### Web

| Tool | What it does |
|---|---|
| `web_search` | Search the web via Parallel AI. Modes: `fast`, `one-shot`, `agentic`. Optional `source_policy` (allow/deny domains). |
| `web_fetch` | Fetch and clean one URL via Parallel AI Extract. Modes: `excerpts` (default), `full`. |

Both web tools require `PARALLEL_API_KEY` in `~/.feather/.env`.

### Memory

| Tool | What it does |
|---|---|
| `recall_memory` | Search long-term memory for a fact or preference. |
| `manage_memory` | Create, update, or delete a memory item on direct user instruction. |
| `user_info` | Maintain the structured profile in `user.md` (name, role, etc). Lead-only. |

These three are no-ops if memory is disabled. See [memory.md](memory.md).

### Scheduling

| Tool | What it does |
|---|---|
| `create_cron` | Create a recurring or one-time scheduled prompt. |
| `update_cron` | Change schedule, prompt, or status of an existing job. |
| `delete_cron` | Remove a job permanently. |
| `list_crons` | List the session's scheduled jobs. |

Lead-only. See [scheduling.md](scheduling.md).

### Sub-agents and tasks

| Tool | What it does |
|---|---|
| `spawn_agent` | Launch a sub-agent in the background. Lead-only. |
| `terminate_agent` | Kill a running sub-agent. Lead-only. |
| `send_message` | Send a message to another agent's inbox. Available to every agent. |
| `task_create` | Open a durable task. Lead-only. |
| `task_list` | List tasks for the session. |
| `task_get` | Read full detail on one task. |
| `task_update` | Change status or notes. |
| `task_output` | Attach a final report. |
| `task_stop` | Abandon a task. |
| `task_resume` | Pick a stopped task back up. |
| `request_input` | A sub-agent uses this to ask the lead for clarification. |

See [agents.md](agents.md) for context.

### MCP

| Tool | What it does |
|---|---|
| `list_mcp_servers` | List MCP servers configured for this agent and provider. |
| `register_mcp_server` | Activate one server for the current session. |

Visible only when `mcp.enabled: true`. See [mcp.md](mcp.md).

### Self-repair and upstream reporting

| Tool | What it does |
|---|---|
| `request_restart` | Queue a clean restart of the lead worker subprocess so patched `feather/*` modules reload. The current session continues on the new worker; conversation history is preserved. Worker mode only (`FEATHER_USE_LEAD_WORKER=1`). |
| `submit_github_report` | File an issue (PR support deferred) on the upstream Feather repo via the `gh` CLI. Always reads the `submit-github-report` skill first; never auto-submits. |

The `request_restart` tool surfaces the install mode in its response —
in wheel installs it warns that the patch will be overwritten on the
next `pip install --upgrade` and recommends `submit_github_report` to
preserve the fix. See [configuration.md → Lead worker mode](configuration.md#lead-worker-mode-opt-in).

## Slash commands

Type these into the input box during a chat. The TUI shows
autocomplete as you type `/`.

### Session and view

| Command | Aliases | What it does |
|---|---|---|
| `/help` | `/?` | List every command. |
| `/exit` | `/quit` | Leave the session. |
| `/onboard` | | Re-run the first-run wizard. |
| `/clear` | | Clear the on-screen transcript. History is preserved. |
| `/copy` | | Copy the transcript to the clipboard. |
| `/queue` | | Show messages you typed while the agent was busy. |
| `/session` | | Show the session ID, agent name, and how full the context is. |
| `/restart-lead` | `/restart_lead` | Respawn the lead worker subprocess. Worker mode only — does nothing in default in-process mode. Use after a "Lead unresponsive" banner or to force-reload patched lead code. Conversation history is preserved across restarts. |

### Inspection

| Command | Aliases | What it does |
|---|---|---|
| `/agents` | `/agent` | List currently running sub-agents. |
| `/tasks` | `/task` | List durable tasks for this session. |
| `/skills` | | List every skill the agent could load. |

### Memory

| Command | What it does |
|---|---|
| `/qdrant status` | Is the local memory container running? |
| `/qdrant start` | Start the local memory container. |
| `/qdrant stop` | Stop it. |
| `/qdrant remove` | Stop and remove it. |
| `/qdrant help` | Detail on the above. |

### Messaging integrations

| Command | What it does |
|---|---|
| `/integrations` | Show the connection state for telegram, line, whatsapp. |
| `/telegram status` | Is the Telegram bot connected? |
| `/telegram connect <token>` | Connect a Telegram bot using its API token. |
| `/telegram disconnect` | Disconnect. |
| `/line status` | Is the LINE channel connected? |
| `/line connect <secret> <token>` | Connect a LINE Messaging API channel. |
| `/line disconnect` | Disconnect. |
| `/whatsapp status` | Is the WhatsApp number connected? |
| `/whatsapp connect <phone_id> <token> <verify_token> <app_secret>` | Connect a WhatsApp Cloud API number. |
| `/whatsapp disconnect` | Disconnect. |

See [messaging.md](messaging.md) for what each value means and how to
get it.

## Keyboard shortcuts (Textual TUI)

The TUI splits the screen into a transcript pane (the chat) and a work
pane (tool calls). Most shortcuts apply to the transcript; Shift +
shortcut applies to the work pane.

### Composer

| Key | Action |
|---|---|
| Enter | Send the message you typed. |
| Esc | Interrupt the agent if it is mid-response. |

### Scrolling

| Key | Action |
|---|---|
| Page Up | Scroll the transcript one page up (older). |
| Page Down | Scroll one page down (newer). |
| Home | Jump to the top of the transcript. |
| End | Jump to the latest message. |
| Shift + Page Up | Same, but for the work pane. |
| Shift + Page Down | Same, but for the work pane. |
| Shift + Home | Top of the work pane. |
| Shift + End | Latest entry in the work pane. |

### Copy

| Key | Action |
|---|---|
| Ctrl + C | Copy the current selection, or the whole transcript if nothing is selected. |
| Ctrl + Y | Copy the entire transcript. |

## Drag-drop attachments

In the TUI, drag a file from your file manager into the chat. The path
becomes part of your message and Feather attaches the file. See
[attachments.md](attachments.md).
