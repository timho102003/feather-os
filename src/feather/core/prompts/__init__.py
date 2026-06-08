"""Prompt templates for Feather agents.

Authoring guide — distilled from the published prompt-engineering guidance of
Anthropic, OpenAI, and Google (Gemini). These are the rules every prompt in
this package already follows; keep them in mind when editing.

**Structure**
- One agent = one ``*_AGENT_PROMPT`` string of XML-tagged sections. Use
  consistent, semantic tags (``<identity>``, ``<mission>``, ``<…_rules>``,
  ``<…_completion_rules>``); all three vendors recommend explicit delimiters,
  and Claude in particular was trained with XML. Don't mix XML and Markdown
  headers as section boundaries.
- Order from most general/stable to most specific: identity → mission →
  operating rules → tool discipline → output contract. The shared
  ``BASE_AGENT_PROMPT`` renders before the role prompt for the same reason.
- State the output contract explicitly (the ``*_completion_rules`` sections):
  the exact sections, citation format, and what "done" looks like.

**Writing**
- Explain the *why* behind a rule — models generalize from rationale (e.g.
  "a text-only first turn is a WASTED SPAWN because the runtime flags zero-tool
  exits as failures"). A bare imperative generalizes worse.
- Prefer telling the model what TO do over what NOT to do.
- Avoid contradictory or vague instructions: reasoning models (gpt-5 / Opus)
  burn tokens reconciling conflicts, which hurts quality and latency.

**Caching — the load-bearing constraint (see PromptBuilder.build_sections)**
- These constants are STATIC: they go in the cached prefix
  (``<static_cached_prefix>``). Per-turn content (recalled memory, loaded
  skill bodies) goes in the dynamic suffix, after the cache breakpoint.
- NEVER interpolate per-turn or per-day values — ``datetime.now()``, a UUID,
  a turn counter, a session id — into a prompt constant or anywhere in the
  cached prefix. A single changing byte before the breakpoint invalidates the
  whole cached prefix and silently drops the runtime cache-hit rate to zero.
  Dynamic context (e.g. the current date) belongs in the conversation
  messages, not the system prompt. ``feather.observability.cache_stats``
  surfaces the hit rate so a regression here is visible.
"""
