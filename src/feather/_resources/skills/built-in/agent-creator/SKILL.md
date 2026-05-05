---
name: agent-creator
description: Use when the user asks to create, add, customize, or register a new specialist sub-agent. Generates a YAML config file under ~/.feather/config/agents/ that the lead can immediately spawn via spawn_agent.
---

# Agent Creator

Use this procedure to turn a user request like *"create an agent that summarises
git history"* into a real, dispatchable sub-agent. The output is a YAML file at
`~/.feather/config/agents/<slug>-custom.yaml` that the agent catalog picks up on the next
lead turn.

## 1. Collect the requirements

Use one focused `ask_user` exchange if details are missing. Keep it short. You
need:

- **Purpose** — one sentence: what should this agent do?
- **Tool needs** — which tools does it need? Pick from the allow-list below.
- **Behavior boundaries** — anything it should *not* do (e.g. "never write to
  the repo", "never hit the network").
- **Output format** — what should its final report look like?

If the user's ask is already unambiguous, skip the question and proceed.

## 2. Pick a slug and display name

- **slug**: lowercase-hyphenated, short, descriptive. E.g. `git-log-summarizer`,
  `release-notes-writer`, `security-reviewer`.
- **display name**: short PascalCase or Title Case (shown in the CLI header
  when the agent runs). E.g. `GitLogSummarizer`, `Release Notes Writer`.
- The YAML **filename must end with `-custom.yaml`** (e.g.
  `git-log-summarizer-custom.yaml`). The catalog uses this suffix as a visual
  cue that the agent is user-defined.

## 3. Pick tools (allow-list for sub-agents)

A sub-agent should list only the tools it genuinely needs. Lead-only tools
must not appear in a sub-agent YAML.

| Tool          | When to include                                                    |
|---------------|--------------------------------------------------------------------|
| `read_file`   | reading local files                                                |
| `grep`        | searching the local repo                                           |
| `write_file`  | writing deliverables / artifacts inside the workspace; prefer over `bash` heredocs because writes are atomic and shell-free |
| `bash`        | running local commands — tests, lint, git, small scripts           |
| `web_search`  | Parallel AI Search for quick/iterative external lookups            |
| `web_fetch`   | Parallel AI Extract for pulling one specific authoritative page    |
| `load_skill`  | loading another skill's body on demand — include this by default   |
| `send_message`| sending agent-to-agent messages (status updates back to the lead, etc.); include by default |

**Do not include** `spawn_agent`, `create_cron` / `update_cron` / `delete_cron`
/ `list_crons`, `ask_user`, or `recall_memory`. Those are lead-only. The
subprocess has no user to ask; giving a sub-agent `spawn_agent` would enable
recursive spawning.

## 4. Draft the YAML

Use this template. Keep `role: custom` exactly — the factory routes on it.

```yaml
name: <DisplayName>
role: custom
description: <one-line purpose; appears in the lead's dispatch catalog>
personality: <two-to-three adjectives, same style as built-ins>
memory_enabled: false
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
  - feather.core.prompts.agent_messaging_protocol:AGENT_MESSAGING_PROTOCOL_PROMPT
inline_prompt: |
  <custom_agent_identity>
  You are Feather's <DisplayName> sub-agent. <one-line mission>.
  </custom_agent_identity>

  <custom_agent_responsibilities>
  - <bullet: what you do>
  - <bullet: how you handle ambiguity>
  - <bullet: scope boundaries>
  </custom_agent_responsibilities>

  <custom_agent_completion_rules>
  - Finish with ONE final assistant turn containing a structured report.
  - Your final report must contain: `Task:` (one sentence restating the ask),
    `Findings:` (the answer, grounded in concrete evidence), and any agent-
    specific sections the user requested.
  - No filler, no meta-commentary. Execute the task and stop.
  </custom_agent_completion_rules>

  <custom_agent_tool_discipline>
  - <bullet per registered tool, explaining when/how to use it>
  </custom_agent_tool_discipline>
registered_tools:
  - load_skill
  - send_message
  - <other tools from the allow-list>
```

Fill every `<…>` placeholder. Do not ship template placeholders.

## 5. Write the file

### Preferred: the `write_file` tool

`write_file` writes the YAML atomically with no shell layer, so backticks,
`$variables`, and `$()` in your prose pass through as literals — no
quoting hazards at all. The global agents directory
(`~/.feather/config/agents/`) is on its whitelist, so an absolute path
works directly:

```
write_file(
  path="~/.feather/config/agents/git-log-summarizer-custom.yaml",
  content="name: GitLogSummarizer\nrole: custom\n... (the rest of the YAML)\n",
  overwrite=False,
  create_parents=True,
)
```

Use this path unless you specifically need to validate the YAML inline
before persisting (in which case use the Python alternative below).

### Fallback: bash heredoc

If you must use `bash` (e.g. you want to chain a parse-check in the same
call), the rules below apply. The `bash` tool passes the `command`
argument to `/bin/bash` as a single string — there is **no outer shell**
pre-parsing it. That means you must write the command so bash itself
does zero substitution inside the file content.

### The safe pattern — single-quoted heredoc, no `bash -c` wrapper

```
cat > ~/.feather/config/agents/git-log-summarizer-custom.yaml <<'EOF'
name: GitLogSummarizer
role: custom
description: Summarise recent git history for a branch or path.
... (the rest of the YAML — prose can contain $variables, `backticks`,
    "quotes", whatever — the quoted 'EOF' marker disables ALL expansion
    inside the body)
EOF
```

- The heredoc marker is `<<'EOF'` **with single quotes**. Single-quoting
  `EOF` tells bash: do NOT perform variable expansion, command
  substitution, or backtick expansion anywhere in the body. This is the
  single most important detail.
- Do NOT wrap the whole thing in `bash -c "..."`. That adds a second
  shell-parsing pass where the outer double quotes DO perform backtick
  and `$()` substitution on your YAML content **before** the inner
  heredoc ever sees it. This is the exact pattern that silently shreds
  YAML files.

### The cautionary tale

If your agent's prose contains ` ``git commit`` ` or ` ``$HOME`` ` or
similar inline-code markup with backticks or `$`, and you wrap the
write in `bash -c "..."`, the outer shell will:

- Run `git commit` as a command and splice its stdout / stderr into your
  YAML.
- Replace `$HOME` with the actual home directory.
- Collapse entire sections of the file into garbage.

The result is a YAML file that looks plausible at a glance and breaks
on parse because a `git status` dump was pasted mid-block-scalar. If
you catch yourself typing `bash -c "..."` followed by a heredoc, stop
and use the unwrapped form above instead.

### The Python alternative (when the YAML is very long)

For long bodies, a Python heredoc is safer AND lets you verify the
YAML parses cleanly in the same tool call:

```
python3 - <<'PY'
from pathlib import Path
import yaml

content = r"""name: GitLogSummarizer
role: custom
description: Summarise recent git history for a branch or path.
... (rest of the YAML)
"""

# Validate before writing — catch parse errors locally instead of letting
# the agent catalog discover them on the next turn.
yaml.safe_load(content)

Path("~/.feather/config/agents/git-log-summarizer-custom.yaml").write_text(content)
print("WROTE")
PY
```

- `<<'PY'` with single quotes — same rule as `'EOF'`: no outer shell
  substitution inside the body.
- Python's triple-quoted `r"""..."""` string is a raw literal — no
  Python escape processing either.
- `yaml.safe_load(content)` raises on syntax errors. If the tool output
  does not end with `WROTE`, the file was not written and the error
  above it tells you exactly what's wrong.

### Inline-code markup inside the YAML prose

YAML block scalars (`inline_prompt: |`) accept almost anything, but
backticks and `$()` are a risk factor **only when the command writing
the file goes through an unsafe shell-parsing layer**. With the safe
patterns above (unwrapped `<<'EOF'` or `python3 - <<'PY'`), the body is
inert — backticks pass through as literal backticks. You do not need
to strip them from your prose.

If you find yourself needing to strip them anyway (because a lead is
forced to use an unsafe pattern for some reason), use ordinary quotes
or nothing: write `"git commit"` or `git commit` instead of
`` `git commit` ``.

## 6. Verify

After writing:

1. **Parse-check the YAML.** Run `python3 -c "import yaml, sys;
   yaml.safe_load(open('~/.feather/config/agents/<slug>-custom.yaml'))"`. Exit code
   0 = parses cleanly; non-zero = the file is broken and you must fix
   it before telling the user the agent was created.
   (If you used the Python-alternative write pattern above, the parse
   check already ran — skip this step.)
2. `read_file` the new YAML to confirm the content matches what you
   intended — specifically scan for: lines starting at column 0 inside
   the `inline_prompt: |` block (those kill the block scalar),
   stray `On branch master` / `Your branch is up to date` text
   (evidence that a `git status` subshell fired), and any `<placeholder>`
   strings you forgot to fill.
3. Report the creation to the user in 2–3 lines: the agent name, what
   it does, and how to invoke it (`spawn_agent agent_name=<slug>-custom
   task=...`).
4. Do **not** test-spawn the new agent unless the user asks. The catalog
   will refresh on your next turn regardless.

### If the parse check fails

- `could not find expected ':'` near a specific line → almost always a
  block-scalar indentation break OR a shell-substitution artifact. Open
  the file around that line, look for text starting at column 0 that
  should be indented, and rewrite the file with the safe pattern.
- `mapping values are not allowed here` → a colon inside an unquoted
  scalar. Wrap the scalar in single quotes: `description: 'Text with:
  colons inside'`.
- Unexpected content that looks like command output (`fatal:`,
  `On branch ...`, `Untracked files:`) → you used the unsafe `bash -c`
  wrapper. Delete the file and rewrite with the safe pattern.

## 7. When to skip / refuse

- The request is ambiguous and `ask_user` is unsafe to skip — clarify first.
- The user asks for an agent that would require a lead-only tool — push back:
  either redesign the agent without that tool, or keep the work inside the
  lead.
- The user asks for an agent that duplicates a built-in role with no added
  value — suggest using the built-in instead.
