# Sub-agents

The agent you chat with in the terminal is the **lead**. It can dispatch
work to specialist sub-agents that run in the background, report back
when done, and cleanly hand control back to the lead.

This guide covers the four sub-agents that ship with Feather, how to
add your own, and how the lead talks to them.

## Built-in agents

| Name | Personality | Tools it has | When the lead calls it |
|---|---|---|---|
| `lead` | Decisive, structured. | Everything. | This is you. The agent you chat with. |
| `explore` | Precise, file-grounded. | `read_file`, `grep`, `bash`, `web_*`, `read_pdf`. No write. | Survey an unfamiliar codebase, locate something, list files. |
| `research` | Rigorous, source-aware. | `web_search`, `web_fetch`, `read_file`, `read_pdf`, `write_file`. | Pull together facts from the web. |
| `validate` | Skeptical, refuses irreversible actions. | `bash`, `read_file`, `grep`, `write_file`. | Re-check a claim, run tests, audit a fix. |

The lead picks one based on the kind of work you ask for. You can also
say "use the explore agent to find every place we set up Express
middleware" and the lead will dispatch it explicitly.

## How dispatch works

Three tools, all of which the lead has by default.

* **`spawn_agent`** launches a sub-agent as a subprocess. Returns
  immediately. The sub-agent runs to completion in the background.
* **`send_message`** drops a message into a sub-agent's inbox while it
  is running. The sub-agent reads its inbox at the start of every
  turn.
* **`terminate_agent`** kills a sub-agent that is taking too long or
  has gone off-track.

When the sub-agent finishes, a **final report** lands in the lead's
inbox. The lead reads it on its next turn and either summarizes it for
you or acts on it.

You can see the live sub-agents at any time with `/agents` in the TUI.

## Tasks: durable work units

For multi-step work, the lead can create a **task**. A task is a
persistent record (title, description, success criteria, notes,
artifacts, status) that survives across sessions and across crashes.

Tools, all available to the lead:

* `task_create`: open a new task.
* `task_list`: list tasks for this session.
* `task_get`: full detail on one task.
* `task_update`: record progress, change status.
* `task_output`: attach a final report.
* `task_stop`: abandon a task.
* `task_resume`: pick a stopped task back up.

Sub-agents read and append to tasks but do not create them.

You can list tasks at any time with `/tasks` in the TUI.

## Add a custom sub-agent

Drop a YAML file at `~/.feather/config/agents/<slug>-custom.yaml`.
Picking a slug ending in `-custom` is a convention so the agent catalog
shows it as user-defined, not packaged.

Minimum viable example, a "release notes" agent:

```yaml
name: ReleaseNotesWriter
role: release-notes
personality: Concise, factual, allergic to marketing language.
reasoning:
  effort: low
  summary: auto
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
  - feather.core.prompts.agent_messaging_protocol:AGENT_MESSAGING_PROTOCOL_PROMPT
memory_enabled: false
registered_tools:
  - read_file
  - bash
  - load_skill
  - send_message
  - task_get
  - task_update
  - task_output
  - request_input
```

After writing the file, ask the lead in chat: "you have a new agent
called release-notes. Use it to draft notes for v0.4.0." The catalog
is rescanned on demand, so no restart is needed.

Want a guided way to write the YAML? Tell the lead "create a
sub-agent for X". It will load the built-in `agent-creator` skill and
walk you through the questions.

## What sub-agents are NOT allowed

A few tools are **lead-only**. Sub-agents that try to register them
will fail to start.

* `spawn_agent`, `terminate_agent`: only the lead orchestrates.
* `create_cron`, `update_cron`, `delete_cron`, `list_crons`: scheduling
  is a lead-only concern.
* `manage_memory`, `user_info`: only the lead is in direct
  conversation with you, so only it gets to mutate your profile or
  long-term memory. (Sub-agents can still *read* memory with
  `recall_memory` if you give it to them.)

The `agent-creator` skill knows the allow-list. If you write the YAML
by hand and accidentally include a lead-only tool, Feather will refuse
to dispatch the agent and tell you which tool to remove.

## Sending the lead a message from a sub-agent

Sub-agents have `send_message` so they can stream updates back to the
lead during long jobs. The recipient is addressed by `(agent_name,
session_id)`. The lead's session ID is in the launch environment that
Feather injects into every sub-agent.

You will rarely write this code yourself. The shipped agents already
know how to send progress updates. If you write a custom sub-agent,
the prompt modules referenced in your YAML
(`agent_messaging_protocol`) explain the protocol to the model.

## Useful slash commands

* `/agents`: list live sub-agents.
* `/tasks`: list current tasks.

## Next

* Want the lead to *schedule* a sub-agent run? See
  [scheduling.md](scheduling.md).
* Want to write a skill that the agent-creator skill calls on top of?
  See [skills.md](skills.md).
