# Skills

A **skill** is a small markdown file that teaches the agent how to do
one specific thing. Skills are loaded only when needed, so they do not
bloat the prompt.

If you have used Claude Code, this is the same idea. The agent sees a
catalog of skill names and one-line descriptions. When it spots one
that matches the task, it calls a tool to load the full file.

## How discovery works

Feather looks in three places, in this order:

1. **Packaged skills.** Shipped with the wheel under
   `feather/_resources/skills/built-in/`. You cannot edit these
   directly, and that's fine; the next two layers can override them.
2. **Global skills.** `~/.feather/skills/`. Anything you drop here is
   available to every project.
3. **Project skills.** `./.feather/skills/`. Skills that should only
   apply when you are working in this repo.

If two layers define a skill with the same name, the later layer wins.
So a global skill with the same name as a built-in replaces the
built-in. A project skill with the same name as a global one replaces
the global. This is on purpose: it lets you customize built-ins
without forking the package.

## What ships out of the box

| Name | What it teaches |
|---|---|
| `pdf-reading` | When to use the `read_pdf` tool, how to pick a mode, how to handle big documents. |
| `mcp-config` | How to add an MCP server to `app.yaml` without leaking secrets. |
| `planning` | How to break a complicated request into smaller tasks. |
| `agent-creator` | How to draft a new sub-agent YAML. |
| `repo-navigation` | How to explore an unfamiliar codebase efficiently. |

Type `/skills` in the TUI to see the live list (including any global
or project additions you made).

## Write your own skill

A skill is a folder with a `SKILL.md` file inside. Minimum viable
skill:

```
~/.feather/skills/release-notes/SKILL.md
```

```markdown
---
name: release-notes
description: Use when the user asks to write release notes, draft a changelog entry, or summarize what shipped in a version. Reads recent commits and groups them by type.
---

# Release Notes

When the user asks for release notes, follow these steps:

1. Run `git log --oneline <previous-tag>..HEAD` to gather commits.
2. Group commits into Features, Fixes, and Internal.
3. Write the summary in the project's existing changelog style. Look at
   `CHANGELOG.md` for the format if one exists.
4. Use plain English. No marketing fluff.
```

That's it. Two rules that matter:

* The frontmatter (`name`, `description`) is required. The agent uses
  the description to decide whether the skill is relevant. Be specific
  about *when* to use it.
* The folder name and the `name` field should match. Stick to
  lowercase-hyphenated names.

After saving the file, restart the chat. The agent picks the new skill
up at session start.

## How the agent uses a skill

In every prompt, the agent sees a compact catalog like this:

```
Available skills:
- pdf-reading: When the user shares a PDF, ...
- mcp-config: When the user asks to add or configure an MCP server, ...
- release-notes: Use when the user asks to write release notes, ...
```

When the agent decides one is relevant, it calls the `load_skill` tool
with the exact name. The full body becomes part of the next prompt and
stays loaded for the rest of the session.

You can ask the agent to load a skill directly: "load the release-notes
skill and write the notes for v0.3.0."

## Things to avoid

* **Do not write giant skills.** The whole point is progressive
  disclosure. If you find yourself writing a 1000-line skill, split it
  into a `SKILL.md` summary that points at companion files in the same
  folder. The agent can read those with `read_file` once the skill is
  loaded.
* **Do not put secrets in `SKILL.md`.** Skill files become part of the
  prompt. Anything in there is sent to the model.
* **Do not duplicate content already in the global agent prompt.** The
  agent already knows it should be terse, accurate, and honest. Skills
  should add domain-specific knowledge, not rehash basics.

## Built-in skill examples worth reading

Open the shipped files for reference:

* `src/feather/_resources/skills/built-in/repo-navigation/SKILL.md`. A
  short top-level skill that points at a longer `reference.md` in the
  same folder. Good template for splitting big skills.
* `src/feather/_resources/skills/built-in/agent-creator/SKILL.md`. Walks
  the agent through generating a new sub-agent YAML. Good example of a
  procedural skill.

## Next

* Want a skill that creates a custom sub-agent? See
  [agents.md](agents.md).
* Want a skill that drives an MCP server? See [mcp.md](mcp.md).
