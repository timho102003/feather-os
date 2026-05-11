# Config TUI + Claude Tool-Schema Sanitizer — Design

**Status:** Draft
**Date:** 2026-05-11
**Branch:** `feature/config-tui`

## Background

Two issues motivate this spec, intentionally bundled:

1. **Claude provider is broken end-to-end.** When `active_provider: claude`, the
   first turn fails with `tools.0.custom: For 'integer' type, property 'minimum'
   is not supported`. ~10 shipped tools (`bash`, `read_file`, `grep`,
   `pdf`, `recall_memory`, `task_*`, `cron_*`, `parallel_search`) declare
   `parameters_schema` with `minimum:` on integer-typed fields and
   `"type": ["integer", "null"]`. The Anthropic Messages API tool validator
   rejects both. The translator at `feather/providers/claude_translator.py`
   passes `tool["parameters"]` straight into `input_schema` with no sanitizer.
   When this happens mid-session, every subsequent turn (including the lead
   supervisor's recovery attempts) fails the same way — there is no in-app
   path to switch providers.

2. **`app.yaml` is the only escape hatch and it's painful.** Switching the
   active provider, swapping a model, tuning reasoning effort, or adjusting
   memory thresholds all require quitting the TUI, opening
   `~/.feather/config/app.yaml` in an editor, and rebooting. The packaged
   default also has organisational debt — `active_provider` sits between
   `self_repair` and `openai`, the MCP block is 90% commented-out examples,
   and the three memory-operation overrides
   (`extraction`/`classification`/`query_builder`) live as siblings of every
   other memory subsection rather than grouped.

The fix for (1) is a small translator-side sanitizer. The fix for (2) is a
config service backed by a typed schema, accessible via slash commands and
a Textual modal. Bundling them lets us prove the value of the config service
on the first demo: a user whose `claude` provider rejects a schema
gets a sanitizer-fixed retry; if a different config knob needs flipping
(e.g. drop to `claude-haiku-4.5`), the modal makes it possible without
quitting.

## Goals

- Editable, validated, comment-preserving access to every operationally
  relevant `app.yaml` field and the Lead agent's per-agent fields, from
  inside the TUI and from headless slash commands.
- Per-field reload semantics so most edits apply at the next turn boundary,
  expensive ones only after `/restart-lead`, and the few that affect
  long-lived singletons require a full TUI restart.
- A worker-aware reload protocol so `self_repair: true` users are not
  excluded from in-app config changes.
- Anthropic-compatible tool schemas, regardless of how individual tools
  declare their JSON Schema.

## Non-goals (this spec)

- Editing `.env` / API keys from the modal. Different file, different
  security posture.
- Reloading TUI-process singletons (memory stack, MCP clients, messaging
  service, scheduler) without quitting `feather`. These remain
  `RESTART_APP`.
- Adding, deleting, or cloning agent definitions. Phase 1 shows only the
  Lead tab; the codebase is "lead agent only" today (see CLAUDE.md).
- Config version history / undo.
- Migrating the legacy `memory.{extraction,classification,query_builder}`
  flat layout out of users' files. The loader continues to read both
  shapes; only the canonical writer emits the new
  `memory.operations.*` shape.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ feather/config_schema.py                                         │
│   ConfigField(path, type, widget, enum, validator,               │
│               description, reload_class, sensitive, default)     │
│   REGISTRY: tuple[ConfigField, ...]                              │
│   IGNORED_PATHS: frozenset[str]                                  │
│   Drift tripwire test asserts every AppConfig + AgentConfig      │
│   leaf is in REGISTRY ∪ IGNORED_PATHS.                           │
└────────────────────┬────────────────────────────────────────────┘
                     │ reads
┌────────────────────▼────────────────────────────────────────────┐
│ feather/config_service.py                                        │
│   ConfigService(paths, schema, app_config, agent_loader)         │
│   .get(path) -> ConfigValue                                      │
│   .set(path, value, *, scope) -> WriteResult                     │
│   .list(section) -> list[ConfigRow]                              │
│   .diff() -> dict[path, (old, new)]                              │
│   .reset(path, *, scope) -> WriteResult                          │
│   .validate(path, value) -> Ok | Error(msg)                      │
│ Single entry point for headless /config and the modal.           │
└──────┬──────────────────────────────────────────┬───────────────┘
       │                                          │ delegates write
       │                                          ▼
       │           ┌─────────────────────────────────────────────┐
       │           │ feather/config_writer.py                    │
       │           │   Strict line-walker rewrite (preserves      │
       │           │     comments) for known-shape leaf paths.   │
       │           │   ruamel.yaml round-trip fallback when the  │
       │           │     line walker can't resolve (e.g. first   │
       │           │     write of a nested key not yet present). │
       │           │   Atomic via tmp + rename.                  │
       │           └─────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│ feather/runtime.py (additions)                                   │
│   .reload_config()              — re-read disk, swap _app_config │
│   .rebuild_agent(name)          — fresh agent via agent_factory  │
│   .apply_config_change(paths)   — fans out by reload class       │
│      ├─ in-process: reload + rebuild as the worst class demands │
│      └─ worker mode: supervisor.request_config_reload(...)      │
└──────┬──────────────────────────────────────────────────────────┘
       │ control envelope
┌──────▼──────────────────────────────────────────────────────────┐
│ LeadSupervisor + lead_worker_core (envelope additions)           │
│   request:  {type:"reload_config", correlation_id,               │
│              changed_paths: [...], reload_class: "..."}         │
│   reply:    {type:"reload_config_ack", correlation_id, ok,       │
│              applied_paths: [...] | error: "..."}                │
│   Worker defers until between turns. Validates by attempting a   │
│   throwaway agent rebuild before swapping `_app_config`. Rolls   │
│   back on validation failure. Never half-applies.                │
└──────────────────────────────────────────────────────────────────┘
```

### Component responsibilities

- **`config_schema.py`** — declarative registry of every editable field.
  One source of truth for: dotted addressability, type, widget hint,
  allowed enum values, validation, human-readable description, reload
  class, sensitivity, and default. Plus a notion of *scoped paths*: an
  entry for `agents.<name>.model` resolves at runtime to one ConfigField
  per agent name.
- **`config_service.py`** — pure-Python orchestration layer used by both
  the TUI modal and the headless `/config` subcommands. Exposes `get`,
  `set`, `list`, `diff`, `reset`, `validate`. Does not import any UI
  framework.
- **`config_writer.py`** — generalises
  `feather/onboarding.py::apply_app_yaml_toggles` (strict line walker)
  to arbitrary dotted paths. Falls back to `ruamel.yaml` round-trip when
  the line walker cannot find a target line (e.g. inserting a key under
  a section that exists but does not yet contain that leaf). Always
  preserves trailing comments, blank lines, and surrounding indentation.
  Atomic via tmp + rename.
- **`config_paths.py`** — dotted-path resolver that maps
  `openai.reasoning.effort` → `(file=~/.feather/config/app.yaml,
  yaml_path=["openai","reasoning","effort"])` and
  `agents.Lead.model` → `(file=~/.feather/config/agents/Lead.yaml,
  yaml_path=["model"])`. Honours the project-vs-global scope flag.
- **`runtime.py`** additions — three new methods:
  - `reload_config()` re-runs `load_app_config(root, paths)` and
    swaps `self._app_config`.
  - `rebuild_agent(name)` reconstructs the agent via
    `self._agent_factory.build(name, self._app_config)` and **also
    reconstructs the agent's provider client** so per-provider
    `NEXT_TURN` fields (model, reasoning, parallel_tool_calls,
    thinking config) actually take effect. The session cursor
    (`last_response_id`) is preserved across the rebuild — the new
    agent picks up the in-flight conversation. Long-lived per-process
    HTTP transports (timeouts, base URLs, API-key env names) are NOT
    reconstructed; those are `RESTART_LEAD`.
  - `apply_config_change(changed_paths)` is the modal's single call
    site. It looks up each path's reload class, takes the strictest,
    and:
    - in-process mode: calls `reload_config()` and (if any class is
      `NEXT_TURN`) `rebuild_agent(name)`.
    - worker mode: calls `self._supervisor.request_config_reload(...)`.
    - any `RESTART_LEAD`: returns a struct that tells the modal to
      offer the respawn prompt.
    - any `RESTART_APP`: returns a struct that tells the modal to
      surface a "quit and restart" banner.
- **Supervisor + worker** — new envelope types:
  - Request: `{type: "reload_config", correlation_id, changed_paths,
    reload_class}`
  - Reply: `{type: "reload_config_ack", correlation_id, ok,
    applied_paths | error}`
  - The worker handler defers to the next turn boundary by enqueuing
    the reload after any in-flight `run` / `resume_on_inbox` finishes.
    Validation rebuilds a throwaway agent with the new config; on
    failure, the existing config and agent are kept and the ack
    carries the error message verbatim.

## Reload classes

Every `ConfigField` carries one of:

| Class            | Definition                                         | Examples |
|------------------|----------------------------------------------------|----------|
| `LIVE`           | Read on every use; `reload_config()` is enough.   | `memory.retrieval.top_k_tool`, `memory.retrieval.score_threshold`, `compaction.trigger_ratio` |
| `NEXT_TURN`      | Captured at agent build time; needs `rebuild_agent` (which also reconstructs the provider). | `active_provider`, `openai.model`, `openai.reasoning.effort`, `claude.model`, `claude.thinking.*`, `claude.anthropic_beta`, per-provider `parallel_tool_calls`, per-agent `personality`, per-agent `registered_tools`, per-agent `model`/`provider` overrides |
| `RESTART_LEAD`   | Baked into long-lived clients within the worker (HTTP clients, compaction provider) that `rebuild_agent` does not reconstruct; needs `/restart-lead`. | per-provider `*_timeout_seconds`, per-provider `base_url`, per-provider `api_key_env`, `compaction.model`, `openrouter.provider_preferences`, `openrouter.fallback_models`, `openrouter.cache_strategy` |
| `RESTART_APP`   | Captured by TUI-process singletons or affects worker-vs-in-process topology. | `database.path`, `logging.path`, `logging.level`, `memory.qdrant.url`, `memory.qdrant.collection_name`, `memory.embedding.provider`, `memory.embedding.model`, `mcp.servers.*`, `self_repair.enabled` |

**Special carve-out:** `self_repair.enabled` is `RESTART_APP` and the
modal refuses to save it without an explicit `--force` confirmation
(both in the headless `/config set` and in the modal's save flow).
Mid-session topology flips have no clean recovery — the in-process
agent owns the session and cannot be transplanted into a worker.

## Slash command surface

Registered in `feather/slash_commands.py`. New entries:

```
/config                                           # opens modal
/config get <path>                                # print current value + source
/config set <path> <value> [--project|--global]   # write + apply
/config list [section]                            # tree of fields, current values
/config diff                                      # global vs packaged-default delta,
                                                   # OR uncommitted modal edits
/config reset <path> [--project|--global]         # remove the override
```

Subcommand parsing follows the existing convention used by `/qdrant`,
`/telegram`, `/line`, `/whatsapp` (handler signature
`(args: str) -> None`, splits on whitespace internally). Headless paths
delegate every operation to `ConfigService`, which is the same surface
the modal calls — there is no second code path.

## TUI modal

Lives in a new module `feather/textual_config_screen.py`. Mounted via
`app.push_screen` from the `/config` slash handler. Uses Textual's
`ModalScreen` (first use in Feather; the existing TUI renders all
slash-command output inline).

```
┌──── /config ─────────────────────────────────────────────────────┐
│  [App]   Lead                                       (←→ tabs)     │
├──────────────┬───────────────────────────────────────────────────┤
│ provider     │ active_provider                  [global]   [LIVE]│
│ compaction   │   one of: openai, openrouter, claude              │
│ scheduler    │   ▸ openrouter                                    │
│ self_repair  │   The provider every agent routes through unless  │
│ openai       │   the agent overrides it.                         │
│▶openrouter   │                                                   │
│ claude       │ openai.model                     [default] [NEXT] │
│ parallel     │   ▸ gpt-5-mini                                    │
│ mcp          │                                                   │
│ memory       │ openai.reasoning.effort          [default] [NEXT] │
│   qdrant     │   one of: minimal, low, medium, high              │
│   embedding  │   ▸ low                                           │
│   chunking   │                                                   │
│   retrieval  │ openai.reasoning.summary         [default] [NEXT] │
│   trigger    │   one of: auto, concise, detailed                 │
│   operations │   ▸ auto                                          │
├──────────────┴───────────────────────────────────────────────────┤
│ 0 dirty         s=save  d=diff  r=reset  /=focus search  esc=close│
└───────────────────────────────────────────────────────────────────┘
```

### Interactions

- **`←` / `→`** — cycle top tabs (`App`, `Lead`).
- **`↑` / `↓`** — move focus within the sidebar (subsections) or the
  form (fields).
- **`Tab` / `Shift+Tab`** — jump between sidebar and form.
- **`Enter`** — open inline editor for the focused field. Editor type
  matches widget hint: dropdown for enum, numeric stepper for int/float,
  toggle for bool, text input for str, list editor for `list[str]`.
- **`s`** — save. Runs every dirty field through `ConfigService.set()`,
  then one `runtime.apply_config_change(...)` with the union. Surfaces
  the per-class banner (see below).
- **`d`** — show a diff popup of dirty fields (`old → new`).
- **`r`** — reset focused field to its inherited value.
- **`/`** — focus a search input that filters the form to fields
  whose path or description matches.
- **`Esc`** — close. If dirty, prompts to discard or stay.

### Field row anatomy

```
<path>                              [<source>]  [<reload-class>]
  one of: <enum values>   |   range: [<min>, <max>]   |   <type>
  ▸ <current value>
  <description>
  <validation error if any, in red>
```

- **`[source]` badges:** `[default]` = packaged default; `[global]` =
  global overlay value; `[project]` = project-staged value;
  `[sensitive]` = env-var indirection (read-only in modal).
- **`[reload-class]` badges:** `[LIVE]` (green), `[NEXT]` (cyan),
  `[RESTART-LEAD]` (yellow), `[RESTART-APP]` (red).

### Save flow

```
Saved 4 fields. 2 applied live. 1 applies on next turn.
1 needs restart-lead — restart now? [y/n]
```

If any field is `RESTART_APP`, no save happens; the modal shows the
quit-and-restart banner with the list of impeding fields. The
`self_repair.enabled` carve-out additionally requires `--force` /
explicit modal confirmation.

### Agent tab (`Lead`)

Fields: `personality`, `memory_enabled`, `provider` (override),
`model` (override), `reasoning.effort`, `reasoning.summary`,
`registered_tools` (list editor — checkbox per tool name from the live
tool registry). `prompt_modules` is read-only in Phase 1 (changing it
risks loading a non-existent module).

If the agent's YAML has been written to the global overlay, the tab
header shows `Lead [shadowed by global override]` so the user knows
packaged-default updates for that agent will not reach them until the
override is removed.

## Phase 0 — Claude tool-schema sanitizer

Single helper added to `feather/providers/claude_translator.py`,
called by the existing `translate_tools` immediately before the
passthrough branch.

### Sanitization rules

Recursive over `properties.*`, `items`, `anyOf`, `oneOf`, `allOf`:

1. **Strip unsupported integer constraints:** `minimum`, `maximum`,
   `exclusiveMinimum`, `exclusiveMaximum`. (Anthropic's tool validator
   rejects these on `integer` types.)
2. **Normalise array-typed `type`:** when `"type"` is a list, the
   sanitizer:
   - drops `"null"` from the list (Anthropic does not honour
     nullable union types in tool input schemas);
   - if the remaining list has length 1, replaces with the bare scalar
     (`["integer"]` → `"integer"`);
   - if the remaining list has length > 1, leaves it as-is and emits a
     `logger.warning` so the developer knows a multi-type union is
     reaching the wire untouched.
3. **Idempotent.** Running the sanitizer on an already-clean schema is
   a no-op. Safe to call on Anthropic-native (`input_schema`-shaped)
   tools too.
4. **Non-mutating.** Operates on a deep copy of the input dict.

### Justification for dropping `null`

Every shipped tool that uses `"type": ["integer", "null"]` treats `null`
as "use the tool's default" inside the body — they all start with `value
= arguments.get("foo") or DEFAULT`. The `null` member of the union is a
schema-level convenience to let the model omit the field; making the
field optional via `required: [...]` (which all the tool schemas
already do correctly) preserves that semantics on Anthropic's side
without smuggling `null` through the union.

### Tests

- Unit: each shipped tool's sanitized schema satisfies the Anthropic
  tool-validator constraints (assert `minimum` not present anywhere
  recursively; `type` is never a list including `null`).
- Unit: idempotence — sanitizing twice returns the same dict.
- Unit: non-mutation — input dict is unchanged after the call.
- Integration: `claude_provider.complete(...)` against a mocked
  Anthropic endpoint that runs the same schema validator the real API
  applies. Body validates green.

## `app.yaml` cleanup

Mechanical reorganisation of the packaged default and a single
loader-side change to read the new `memory.operations.*` shape.

### Reorder for narrative flow

Top-down:

1. Infrastructure: `database`, `storage`, `logging`.
2. Behavioural defaults: `compaction`, `skills`, `scheduler`,
   `self_repair`.
3. Routing: `active_provider`.
4. Provider blocks: `openai`, `openrouter`, `claude`, `parallel`.
5. Integration registries: `mcp`.
6. Memory subsystem: `memory.{enabled,qdrant,embedding,chunking,retrieval,trigger,operations}`.

### Group memory operations

Replace flat siblings with one nested block:

```yaml
memory:
  ...existing subsections...
  operations:
    extraction:    { provider: openai, model: gpt-5.4-nano, ... }
    classification:{ provider: openai, model: gpt-5.4-nano, ... }
    query_builder: { provider: openai, model: gpt-5.4-nano, ... }
```

The loader (`feather/config.py::_parse_memory_config`) reads
`raw.get("operations") or raw` so the legacy flat shape continues to
work. New `ConfigService.set()` writes through the nested shape only.
Deprecation log line emitted once per process when the flat shape is
detected.

### MCP example comments

The 90 lines of commented-out example MCP servers move to
`src/feather/_resources/config/examples/mcp.example.yaml`. The live
file gets:

```yaml
mcp:
  enabled: false
  servers: {}
  # See _resources/config/examples/mcp.example.yaml for HTTP and
  # stdio MCP server templates.
```

## Tests

### Schema layer

- `tests/test_config_schema_drift.py` — walks `dataclasses.fields()` of
  `AppConfig` and `AgentConfig` recursively. Asserts every leaf path
  is in `REGISTRY` or `IGNORED_PATHS`. Failure message names the
  missing path so a contributor knows what to add.
- `tests/test_config_schema.py` — every `ConfigField`'s default
  matches the dataclass default (or is explicitly marked
  `inherits_default=True`); enum values are non-empty; validators are
  callable.

### Service layer

- `tests/test_config_service.py` — `get`/`set`/`list`/`diff`/`reset`
  round-trips against a tmpdir-scoped `FeatherPaths`. Edits write to
  the global overlay by default; `--project` flag flips. `validate`
  rejects malformed values with a useful message. `reset` removes the
  overlay key entirely (not "set to default").

### Writer

- `tests/test_config_writer.py` — line-walker preserves comments,
  blank lines, indentation. Round-trip fallback inserts new nested
  keys in the right section. Atomic write semantics (interrupted write
  leaves the original intact).

### Reload protocol

- `tests/test_runtime_apply_config_change.py` — in-process branch
  calls `reload_config` + (when class is `NEXT_TURN`) `rebuild_agent`.
- `tests/test_supervisor_reload_envelope.py` — supervisor sends the
  envelope, awaits ack, surfaces error-on-rollback to caller. Mid-turn
  defer is verified by sending a reload while a `run` is in flight and
  asserting the reload only completes after the run.
- `tests/test_worker_reload_validation.py` — worker rejects an
  invalid `claude.model` (typo) and keeps the prior config; ack
  carries the error.

### TUI modal

- `tests/test_textual_config_screen.py` — modal mounts; `←`/`→`
  cycles tabs; `↑`/`↓` cycles fields; `Enter` opens editor; `s`
  triggers save flow with the right banner for the worst reload
  class; `Esc` prompts on dirty.

### Claude sanitizer (Phase 0)

- `tests/test_claude_tool_schema_sanitizer.py` — every shipped tool
  schema, post-sanitize: no `minimum`/`maximum` anywhere recursively;
  no `type` list containing `null`. Idempotence + non-mutation.
- `tests/test_claude_provider_tool_payload.py` — integration with a
  recorded Anthropic-shaped validator; body passes.

### Slash command

- `tests/test_slash_commands.py` — `/config` registered, all
  subcommands recognised, `accepts_args` flag set. Existing dropdown
  test extended.

## Phasing

- **Phase 0 — Claude tool-schema sanitizer.** ~50 LOC + tests. Single
  commit. Unblocks claude provider immediately. No dependency on the
  rest of the work.
- **Phase 1 — Headless config + reload plumbing.** `config_schema`
  registry (covering ~80% of `app.yaml` fields and all Lead-agent
  fields), `config_service`, `config_writer`, `config_paths`,
  `runtime.reload_config` / `rebuild_agent` / `apply_config_change`,
  supervisor envelope, drift tripwire test, `/config get|set|list|
  diff|reset` headless commands. App.yaml cleanup (reorder + memory
  operations grouping + MCP example extraction) ships in this phase
  so the registry has a clean target shape.
- **Phase 2 — Modal.** `feather/textual_config_screen.py` with tabs,
  sidebar, form rendering, save flow with per-class banner and
  restart-lead prompt. Uses the Phase 1 service unchanged.
- **Phase 3 (out of this spec).** Agent CRUD, secrets editing in
  `.env`, registry coverage for the remaining 20% of fields,
  `/config undo`, project-vs-global field-level visualization beyond
  the source badge.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Drift between `models.py` and the registry. | CI tripwire test fails the build. Adding a field requires either a registry entry or an explicit `IGNORED_PATHS` opt-in. |
| Worker reload ack times out. | Use the same correlation-id timeout as the existing supervisor envelopes. On timeout, surface a banner: "Worker did not acknowledge reload — try `/restart-lead`." |
| YAML round-trip clobbers user comments. | Default to the strict line-walker for known leaf shapes; fall back to `ruamel.yaml` only when the line walker can't find the target. Snapshot tests on the packaged default ensure idempotent re-write. |
| Modal `Enter` editing collides with composer keybindings. | Modal is a `ModalScreen` push, captures all input until dismissed; composer is unmounted while open. |
| `self_repair.enabled` flipped mid-session destabilises agent. | Force-only save, modal banner explains the topology change requires full restart, and the loader continues to honour `FEATHER_USE_LEAD_WORKER=1` env override for one-off testing. |
| Sanitizer hides a real schema bug in a tool. | A new `tests/test_tool_schemas_anthropic_compatible.py` runs every shipped tool through the sanitizer and asserts the post-sanitize shape is what the registry expects — i.e. the tool author can see exactly what Anthropic will see. |

## Open questions for review

- Should the sanitizer also handle `string`-typed unsupported keywords
  (e.g. `pattern`, `format`)? Spec scope today says no — none of the
  shipped tools use them — but a future tool could trip it. Decision:
  add a one-line warning in the sanitizer log when an unknown
  unsupported keyword passes through, so we learn before the user
  hits a 400.
- Should `/config diff` show `global vs packaged-default` or
  `dirty modal state vs disk`? Decision: both, selected by flag —
  no-flag default is `dirty modal state` while the modal is open and
  `global vs packaged-default` from headless invocation.
- Whether to expose `compaction.model` as a settable enum drawn from
  the active provider's catalogue (closed list) or as freeform string.
  Decision: freeform string in Phase 1; the enum requires a
  per-provider model catalogue we don't have yet.
