# Providers and Models

Feather can call models through two providers: OpenAI directly, or
OpenRouter (which gives you access to dozens of models from many vendors
behind one API key). Pick the one that fits your situation, or run a
mix where the chat uses one provider and the memory pipeline uses
another.

## Quick decisions

* **Just want to try Feather?** Use OpenAI. Get a key, paste it during
  `feather onboard`, done.
* **Want to try Anthropic, Qwen, DeepSeek, Grok, etc. from one place?**
  Use OpenRouter.
* **Want to mix them?** You can. The lead agent might run on OpenRouter
  while memory extraction runs on cheap OpenAI nano. See "Per-agent and
  per-task overrides" below.

## OpenAI

You need an account at <https://platform.openai.com/>. Create a key at
<https://platform.openai.com/api-keys>.

Put the key in `~/.feather/.env`:

```
OPENAI_API_KEY=sk-...
```

`feather onboard` does that for you. If you ever want to change the
key, edit the file directly or re-run `feather onboard --force`.

The default model is `gpt-5-mini`. Reasoning effort defaults to `low`.
To change either, drop a partial YAML at `~/.feather/config/app.yaml`:

```yaml
active_provider: openai
openai:
  model: gpt-5
  reasoning:
    effort: medium
    summary: auto
  prompt_cache_key: feather-lead
  prompt_cache_retention: in_memory
```

You only need to write the keys you want to override. Anything you
leave out keeps its packaged default.

Note: GPT-5 family models do not accept `temperature`. Feather drops it
automatically for those models.

## OpenRouter

OpenRouter is a single API that routes to many providers (OpenAI,
Anthropic, Google, xAI, Alibaba, DeepSeek, and more). One key, many
models, often cheaper, and you can swap models without changing
anything else in your config.

Get a key at <https://openrouter.ai/keys>. Add credit to your account
under <https://openrouter.ai/credits> before you start chatting. Then
put the key in `~/.feather/.env`:

```
OPEN_ROUTER_API_KEY=sk-or-...
```

Switch Feather to OpenRouter by setting `active_provider: openrouter`
in `~/.feather/config/app.yaml`. The shipped default already does this,
so if you ran `feather onboard` and chose OpenRouter, you're done.

A minimal OpenRouter block looks like this:

```yaml
active_provider: openrouter

openrouter:
  api_key_env: OPEN_ROUTER_API_KEY
  model: qwen/qwen3.6-plus
  max_output_tokens: 32000
  temperature: 0.7
  reasoning:
    effort: medium
  cache_strategy: anthropic_breakpoint
  provider_preferences:
    require_parameters: true
    allow_fallbacks: true
```

`require_parameters: true` filters out cheap upstream providers that
silently drop tool definitions. Always keep it on when the agent uses
tools.

`cache_strategy: anthropic_breakpoint` enables prompt caching on
Anthropic-style providers (Claude, Z.ai GLM, DeepSeek, Moonshot). It is
ignored by providers that don't support caching, so it's safe to keep
on always.

### Sending traces to Comet Opik (and other observability platforms)

OpenRouter can broadcast every request to one or more observability
backends: Comet Opik, Langfuse, OTel collectors, W&B Weave, Sentry,
Grafana Cloud, generic webhooks. Set this up once on the OpenRouter
dashboard under **Settings → Observability**, then turn on tracing in
Feather so every turn carries the metadata your dashboard needs to
group sessions cleanly.

The walkthrough below uses Comet Opik. Other destinations work the
same way: only the credentials you paste into OpenRouter change.

#### Step 1: Get your Comet Opik credentials

OpenRouter's Comet Opik destination requires three values:

| Field | What it is |
|---|---|
| API Key | Comet API key, starts with `opik_...` |
| Workspace | Your Comet workspace name |
| Project Name | The Opik project where traces should land |

How to collect them:

1. Sign up at <https://www.comet.com/signup> if you do not already
   have a Comet account. The free tier is enough to get started.
2. Sign in at <https://www.comet.com/opik>.
3. Click your avatar in the top-right corner and pick **Account
   Settings**. Copy your API key from the **API Keys** panel. It
   starts with `opik_`.
4. Your workspace name is the slug shown next to your account name
   in the top-left corner, and it is also in the URL after
   `/opik/`. If you only have one workspace, that is the one to use.
5. Open the Opik UI sidebar and click **Projects**. Either pick an
   existing project or click **New project** and give it a name
   like `feather`. The exact string you use becomes the **Project
   Name** value below.

Note: this works only with Comet's hosted Opik (cloud). The
self-hosted open-source build does not issue API keys, so OpenRouter
cannot broadcast to it.

#### Step 2: Add Opik as a broadcast destination on OpenRouter

1. Open <https://openrouter.ai/settings/integrations>.
2. Find **Comet Opik** in the list of observability destinations and
   click **Configure**.
3. Paste the three values from Step 1 (API Key, Workspace, Project
   Name).
4. Click **Test Connection**. OpenRouter only saves the
   configuration if the test succeeds, so a green check means the
   credentials are valid.
5. Toggle **Enable Broadcast**.

OpenRouter is now set up. From this point every chat completion
request you send through your OpenRouter API key has its metadata
forwarded to Opik, regardless of which client made the call.

#### Step 3: Enable tracing in Feather

Open `~/.feather/config/app.yaml` (or your project-local
`.feather/config/app.yaml` if you ran `feather init`) and add a
`tracing:` block under `openrouter:`:

```yaml
openrouter:
  # ... your existing settings ...
  tracing:
    enabled: true
    user: ops@example.com           # optional, identifies you in the UI
    metadata:                       # optional static fields, merged into trace
      deployment: prod
      build_sha: abc123
```

When this is on, Feather adds three things to every OpenRouter request:

* `session_id`: the Feather session UUID, so all turns of one chat
  cluster together.
* `user`: the operator-supplied identifier (only if you set it).
* `trace`: a small object carrying `trace_name = "feather/<agent>"`,
  `generation_name = <model>`, `feather_app`, `feather_agent_name`,
  `feather_agent_role`, `feather_session_id`, plus any static metadata
  you put in the YAML.

OpenRouter forwards all of that to Opik. In the Opik UI you will see
traces named `feather/<agent_name>`, grouped by `session_id`, with the
`feather_*` keys and your operator metadata available as searchable
facets.

Tracing is opt-in. With no `tracing:` block, the request body is
identical to what Feather sent before this feature existed.

#### Verifying it works

Send one message through `feather tui` and then check, in this order:

1. **OpenRouter Activity tab.** Open
   <https://openrouter.ai/activity>, click the most recent request,
   and look at the Request panel. You should see `session_id`,
   `user`, and a `trace` object with the `feather_*` keys at the
   top level. If they are missing, your `tracing.enabled` did not
   take effect (wrong config file, or `active_provider` is not
   `openrouter`).
2. **Opik UI.** In your Opik project, refresh the Traces view. A new
   trace named `feather/<agent_name>` should appear within a few
   seconds. If the OpenRouter Activity panel shows the fields but
   Opik does not, return to Step 2 and re-test the connection from
   the OpenRouter dashboard.

### Ready-made OpenRouter examples

The package ships drop-in `openrouter:` blocks for popular models. Look
under the installed package or browse them in the source repo at
`src/feather/_resources/config/openrouter-examples/`:

* `moonshot-kimi-k2.6.yaml`: agentic-tuned, 256K context.
* `zai-glm-4.7.yaml`: newer GLM line, 202K context.
* `deepseek-v3.2.yaml`: strong reasoning at low cost.
* `qwen3.6-plus.yaml`: 1 M context, multimodal input.
* `xai-grok-4.1-fast.yaml`: 2 M context, fast time-to-first-token.
* `minimax-m2.7.yaml`: agentic-tuned, multi-provider routing.

To use one of these, copy its mapping into the `openrouter:` block of
`~/.feather/config/app.yaml`.

## Switching providers later

There is no special command to switch providers. Edit
`~/.feather/config/app.yaml` and change `active_provider`. The next
chat picks up the new setting.

If you flip to `openrouter` and there is no `openrouter:` block in your
config, Feather will refuse to start instead of silently falling back.
Add the block (or copy one of the examples above) and try again.

## Per-agent and per-task overrides

The shipped agents (lead, explore, research, validate) can each pin
their own provider, model, and reasoning settings. The agent YAML wins
over `app.yaml`.

Example: keep the lead on OpenRouter Qwen but run the explore sub-agent
on cheap OpenAI nano. Drop a custom file at
`~/.feather/config/agents/explore.yaml` that overrides the packaged
default:

```yaml
name: Explore
role: explore
provider: openai
model: gpt-5.4-nano
reasoning:
  effort: low
  summary: auto
```

The same per-agent override works for any agent file you ship.

Memory operations (extracting facts from chat, classifying them, and
building retrieval queries) are routed separately. Each has its own
`provider:` and `model:` knob in the `memory:` section of `app.yaml`.
This is how the shipped scenarios pair an expensive conversation model
with a cheap structured-output model for memory.

## Scenario presets

The package also ships five end-to-end scenarios under
`src/feather/_resources/config/scenarios/` that illustrate full
configurations:

1. **default-qwen-plus-openai-memory**: Qwen3.6-plus on OpenRouter for
   chat, OpenAI nano for memory. The shipped default.
2. **all-openai-tiered-reasoning**: pure OpenAI; the lead and research
   agents use `gpt-5` at high effort, the others use `gpt-5-mini` at
   low effort.
3. **cost-optimized-mixed-providers**: DeepSeek for the lead loop,
   Qwen for big-corpus research, OpenAI for strict validation.
4. **latency-2M-context-grok**: Grok-4.1-fast for every agent, OpenAI
   nano for memory.
5. **anthropic-conversation-openai-memory**: Claude Sonnet 4.6 via
   OpenRouter for chat, OpenAI nano for memory.

These are YAML fragments, not full files. Copy the sections you want
into `~/.feather/config/app.yaml`.

## Quick reference of knobs

| Setting | Where | What it does |
|---|---|---|
| `active_provider` | `app.yaml` top level | `openai` or `openrouter`. |
| `openai.model` | `app.yaml` | Model used when `active_provider` is openai. |
| `openai.reasoning.effort` | `app.yaml` | `minimal`, `low`, `medium`, or `high`. |
| `openrouter.model` | `app.yaml` | Model slug, e.g. `qwen/qwen3.6-plus`. |
| `openrouter.provider_preferences` | `app.yaml` | Strict routing rules; see the OpenRouter docs. |
| `provider:` and `model:` | `~/.feather/config/agents/<name>.yaml` | Per-agent override. |
| `memory.extraction.provider` | `app.yaml` | Provider for the memory-extraction model. |

The full reference is in [configuration.md](configuration.md).

## Next

* Memory pipeline details: [memory.md](memory.md).
* Where keys come from and how they are stored:
  [getting-started.md](getting-started.md).
* What to do when an OpenRouter call fails:
  [troubleshooting.md](troubleshooting.md).
