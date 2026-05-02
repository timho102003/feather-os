---
name: planning
description: Lead-only playbook for turning a user request into correctly-scoped sub-agent dispatches. Use on every non-trivial user request (multi-faceted, open-ended, ambiguous scope). Covers the full flow — read user's literal words → ground unknown named terms → run the planning sweep → pass the task-string confidence gate → write a plan artifact only when warranted → dispatch. Skip only when the request is narrow and concrete and uses no unknown named terms.
---

# Planning

The lead's job on a non-trivial request is to convert the user's words into
task strings that sub-agents can execute without ever needing to ask a
follow-up. Most bad sub-agent outputs trace back to bad task strings. Bad
task strings trace back to two root causes: **the lead paraphrased the user
instead of holding their literal words**, or **the lead planned around a
term it couldn't actually define**. This playbook prevents both.

Core principle: **grounding and the task-string confidence gate ARE the
planning**. A written plan artifact is a secondary deliverable, not the
point. Write one when it's worth recording; skip it when it isn't.

## When to load this skill

Load it on the FIRST turn of any non-trivial request, BEFORE calling
`spawn_agent`. Signals:

- The ask uses a named term, product, library, URL, acronym, or
  project-specific vocab you can't confidently define AND point to a
  canonical source for.
- The ask is multi-faceted or open-ended ("help me…", "thinking of…",
  "comprehensive tutorial on…", "what are my options for…",
  "how should I build…").
- The plan would reasonably produce 2+ parallel dispatches.

Skip the skill (and skip this playbook) when:

- The ask is narrow, concrete, and uses no unknown named terms.
- The user has explicitly delegated ("you decide", "skip planning",
  "don't ask questions").
- You already ran planning on a prior turn in this session and are
  simply dispatching or re-dispatching inside an existing plan.

## Step 0 — Hold the user's exact words

Re-read the user's message. Hold the literal phrasing in mind. Do NOT
paraphrase yet.

Identify every NAMED TERM — products, libraries, vendors, URLs, acronyms,
proper nouns, project-specific vocabulary. For each:

- Can you write a one-sentence definition RIGHT NOW, without guessing?
- Can you point to the CANONICAL source (URL, repo, spec)?

If the answer to either is "no" for any term, you have an ambiguity. Go to
Step 1. Do NOT skip ahead.

**The anti-pattern this step exists to prevent:** silently translating the
user's phrasing ("ChatGPT MCP app") into your own interpretation ("MCP app
architectures integrating ChatGPT-like models") and then planning on top of
the interpretation. Sub-agents only see the task strings you write; if you
paraphrased the user wrong, they inherit the error with no way to recover.

## Step 1 — Grounding pass

Trigger: Step 0 surfaced any term you can't both define AND point at a
canonical source for.

Resolve in this order, stopping as soon as the ambiguity is closed:

### 1a. `recall_memory`
Cheapest — zero network cost, zero user turns. If the user has discussed
this topic in prior sessions, memory collapses the ambiguity instantly.
Always try first.

### 1b. `ask_user` — branch (a), text-only
Use when the term has 2–3 plausible meanings and a word from the user will
collapse the ambiguity. One sentence, cite the candidates explicitly:

> *"Are you referring to OpenAI's Apps SDK
> (`developers.openai.com/apps-sdk`), the Model Context Protocol
> (`modelcontextprotocol.io`), or something else?"*

Stop and wait for the answer on that turn. Do NOT call `spawn_agent`.

### 1c. Inline scouting — ≤2 tool calls
Use when the user can't easily answer (e.g., a factual disambiguation like
"what's the canonical source for X?"), or when asking would feel slow for a
low-stakes resolution. Budget:

- ONE `web_search` with a targeted query — use `site:` and quoted phrases
  (e.g. `"chatgpt mcp app" site:openai.com`, `"openclaw" github`).
- Optionally ONE `web_fetch` on the top authoritative hit, in **`mode='full'`**
  so you get the cleaned markdown INCLUDING the sidebar / TOC / nav
  (excerpts mode hides the tree structure).

Stop at 2 tool calls. Scouting is reconnaissance, not research. If 2 calls
aren't enough to disambiguate, fall back to 1b (ask the user).

### Grounding outputs

By the end of Step 1 you must have:

- **Canonical name** (e.g. "OpenAI Apps SDK").
- **Canonical URL** (e.g. `https://developers.openai.com/apps-sdk/`).
- **One-paragraph "what is it"** definition.
- **Tree map** when the source is a doc site, repo, or API reference: the
  sidebar / TOC items as a list. This is the natural decomposition map
  for Step 3's 5-question sweep — a doc site with 20 sidebar pages is 20
  candidate facets, not one source to read.

## Step 2 — Classify the request

Given the grounded context:

- **Trivial** — single narrow concrete ask. Signals: "look up X's docs",
  "what does `config.foo` do", "run the tests on branch Y". One dispatch
  at most (often zero — just answer inline). Skip to Step 5 (dispatch).
- **Non-trivial** — multi-faceted, multi-dispatch, or genuinely ambiguous.
  Continue to Step 3.

## Step 3 — The 5-question planning sweep

Silent internal reasoning. Each question is a trigger: if the honest answer
exposes an ambiguity, the plan has to resolve it (either by asking the user
or by recording the chosen interpretation under `assumptions`).

1. **What does the user actually want?** Restate in one sentence using the
   user's vocabulary (not yours). Name the two most plausible
   interpretations if the ask is still ambiguous.
2. **What scope decisions am I about to make on the user's behalf?**
   Enumerate: implementation language, platform, depth, coverage,
   deliverable format, recency cutoff, budget / latency, safety (paper vs
   live, destructive operations y/n). Any silent decision is a guess;
   surface it.
3. **Which sub-agents and what task per sub-agent?** Prefer 1–3 focused
   dispatches over one sprawling task. When the grounded source is a tree,
   map each facet to specific URLs / sub-pages. Name a success criterion
   per dispatch.
4. **What's the whole-request success criterion?** How do I know the
   user's question is answered end-to-end, versus one sub-agent happens to
   have finished?
5. **What are the risks of wasted work?** Research whose answer is in a
   doc the user already named; deliverables the user won't use; dispatches
   that duplicate an active sub-agent; plans that assume state you
   haven't verified.

## Step 4 — Task-string confidence gate

For EACH intended dispatch, self-test before calling `spawn_agent`:

> "If a fresh colleague walked into the room and saw ONLY this task
> string — no conversation, no project context, no user history — could
> they execute it without asking me a follow-up?"

If the answer is NO for any dispatch, DO NOT call `spawn_agent`. Instead:

- **Grounding-related vagueness** (WHAT the topic actually is) → go back
  to Step 1, scout more.
- **User-intent vagueness** (WHAT the user wants) → branch (a), ask.
- **Decomposition vagueness** (WHICH facets matter) → branch (b), present
  the plan and let the user redirect.

### Signals that a task string FAILS the gate

- Vague qualifiers: "common architectures", "typical patterns", "general
  principles", "best practices" — without naming the specific product,
  source, or use case.
- Implicit success criteria ("produce a report") — the sub-agent has to
  guess what "good" looks like.
- No canonical URL — if the user named or implied a source, or grounding
  identified one, it MUST appear in the task string verbatim.
- No scope fence — sub-agent can't tell what's IN vs OUT of scope.

### What a passing task string looks like

> *"Produce a concrete 8-step getting-started walkthrough for building an
> OpenAI Apps SDK app. Source: fetch these 5 pages from
> `developers.openai.com/apps-sdk/` in `web_fetch` `mode='full'` —
> `/quickstart`, `/set-up-your-server`, `/define-tools`,
> `/build-your-chatgpt-ui`, `/deploy-your-app`. Each step must cite the
> source page and quote the relevant code snippet verbatim. Scope:
> getting-started only; EXCLUDE deep dives on authentication,
> monetization, and app submission review — separate dispatches handle
> those."*

Compact, concrete, canonical. A fresh colleague executes it without
asking you anything.

## Step 5 — Pick the action

Exactly ONE of three next actions. Never mix them on the same turn — each
has a different output shape.

### (a) Ask a focused clarifying question
Use when user-intent is ambiguous in a way that materially changes which
sub-agents dispatch. One sentence or a numbered list of ≤4 options.

- Response is **text-only, ZERO `spawn_agent` calls**. (Grounding scout
  calls from Step 1 happened earlier; if you're on this turn, they're
  already done.)
- Dispatch on the NEXT turn once the answer arrives.

### (b) Present a one-paragraph plan and ask for confirmation
Use when the decomposition is complex enough to warrant multiple
sub-agents AND the scope is mostly clear but benefits from a sanity check.

- Response is **text-only**. ~5 lines naming dispatches + deliverable,
  ending with "Does this look right?".
- Write the plan artifact on this turn (per Step 6) even before the user
  confirms — the written plan is the durable record.
- Dispatch on the next turn.

### (c) Dispatch immediately
Use when grounding is clear, task-string gate passes on every dispatch,
and the user hasn't flagged a scope question.

- Response calls `spawn_agent` (possibly in parallel for independent
  tasks) and tells the user which dispatches went out + the plan
  filepath if one was written.

### Defaults

- Open-ended multi-facet request after successful grounding → (c) is
  usually right if task-string gate passes; (b) if any task string is
  marginal.
- Narrow concrete request → skip this skill entirely; dispatch or answer
  inline.
- User explicitly delegated, memory confirms preference → (c).

## Step 6 — Plan artifact: CONDITIONAL

Write a plan file to `.feather/artifacts/plan/<YYYY-MM-DD>-<slug>.md` ONLY
when any of these hold:

- **Parallel dispatches.** You're about to spawn 2+ sub-agents in parallel.
  The plan coordinates them — records `correlation_id`s, per-dispatch
  success criteria, and the merge plan.
- **Architectural / multi-step work.** The request is a design,
  implementation, or refactor that plays out over multiple turns or
  sessions. The plan is the durable record of the approach.
- **Non-trivial grounding.** Step 1 surfaced canonical URLs, caveats, or
  assumptions worth recording for the user or a future session to audit.

SKIP the plan file when:

- Single-dispatch trivial task. One sub-agent, concrete task, no
  coordination. Spawn directly.
- Pure info lookup the lead can answer inline from existing context or a
  quick grounding scout.
- Trivial responses ("what does this config do", "run the tests on branch Y").

The goal is not "write a plan every time" — it's "write a plan when there
is something worth recording that won't otherwise survive the turn".
Over-writing plan files turns `.feather/artifacts/plan/` into noise and
deprioritizes the plans that matter.

### Frontmatter schema (when you do write)

```yaml
---
title: <short human-readable title>
slug: <lowercase-hyphenated id; matches the filename>
goal: <one-sentence whole-request success criterion>
status: draft          # draft | awaiting_user | approved | in_progress | complete | abandoned
created_at: <YYYY-MM-DD>
updated_at: <YYYY-MM-DD>
session_id: <current session id>
stakeholders:
  - the user
  # add named stakeholders when the conversation identifies them
grounding:             # filled from Step 1 — omit section only if Step 1 was skipped
  user_term: "<the literal phrase from the user's ask>"
  canonical_name: "<what it actually refers to>"
  canonical_url: "<authoritative source>"
  definition: >
    One-paragraph "what is this, concretely" write-up.
  tree:                # list of sub-pages / TOC items when the canonical source is a tree
    - "<page slug or section>"
    - "<page slug or section>"
problem: >
  2–4 sentence statement of the issue / motivation — the "why" behind
  this request, not the solution.
summary: >
  3–6 sentence prose summary of the whole plan, headline first.
dispatches:
  - name: <short slug for this dispatch>
    agent: <research | explore | validate | custom agent name>
    task: >
      The task string you will pass to spawn_agent verbatim. MUST pass
      the Step 4 confidence gate — name canonical URLs, scope fences,
      and a concrete success criterion.
    success_criterion: >
      One sentence naming what evidence/output makes this dispatch done.
    depends_on: []     # list of other dispatch names that must finish first
    status: planned    # planned | running | complete | failed
open_questions:
  - <question>         # ambiguities you would resolve by asking the user
assumptions:
  - <assumption>       # scope decisions you silently made because the user did not specify
risks:
  - risk: <short>
    mitigation: <short>
---
```

### Body sections (below the frontmatter)

```markdown
# <Title>

## Problem
<Expanded problem statement.>

## Goal & Success Criteria
<Expanded goal + checklist the lead uses at delivery time.>

## Grounding
<When Step 1 surfaced non-trivial facts, expand here with fetched URLs,
sidebar enumeration, definitions, and anything else worth preserving.>

## Approach
<Prose describing the decomposition: which sub-agents, in what order /
parallel, and why this decomposition over alternatives you considered.>

## Dispatches
<Table or detailed list mirroring frontmatter dispatches. Include the
full task string for each — the lead copies this into the spawn_agent
call.>

## Open Questions
<Items blocking dispatch if unresolved.>

## Assumptions
<Interpretive choices the lead silently made; surfaced to the user at
delivery time.>

## Risks & Mitigations
<What could go wrong and how the plan compensates.>

## Definition of Done
<Final user-facing deliverable: format, length, required sections.>
```

### Updating the plan as work progresses

As dispatches return, rewrite the plan:

1. Flip `dispatches[].status` to `running` / `complete` / `failed`.
2. Update top-level `status` and `updated_at`.
3. Append `## Results — <dispatch name>` sections to the body linking to
   any output artifacts under `.feather/artifacts/outputs/`.

Rewrite the full file; do not patch in place.

## Output-artifact protocol (user-facing files)

Separate from the plan artifact: user-facing deliverables (reports, code,
notes) go to `.feather/artifacts/outputs/<YYYY-MM-DD>-<slug>.<ext>`. The
base-agent prompt holds the full contract — every agent (lead + sub-agent)
follows it.

## Reporting the plan to the user

When you do write a plan file, end your response with a one-liner that
names the exact filepath:

> *"Plan written to
> `.feather/artifacts/plan/2026-04-24-openai-apps-sdk-tutorial.md`.
> Dispatched 4 research sub-agents in parallel — I'll synthesize when
> they return."*

This is the durable pointer: the user (or a future session resuming this
work) can `read_file` the plan to pick up where you left off.

When you do NOT write a plan (per Step 6), don't fabricate a filepath —
just dispatch and report.
