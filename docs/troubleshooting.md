# Troubleshooting

Common things that break, and how to fix them.

## "command not found: feather"

The `pip install feather-agent-os` step worked but the `feather` command is
not on your `PATH`. Three usual culprits:

* You used `pip install --user` and your user-bin directory is not on
  `PATH`. On Linux that's usually `~/.local/bin`. On macOS Homebrew
  Python it's `~/Library/Python/3.12/bin`.
* You installed inside a virtual environment that is no longer active.
  Reactivate the venv or use `pipx`/`uv tool` to install once and
  forget.
* The shell hasn't picked up the new entry. Try `hash -r` (bash/zsh)
  or open a new terminal.

The cleanest install for a casual user is:

```bash
pipx install feather-agent-os
```

`pipx` puts the entry on your PATH and isolates dependencies.

## "OPENAI_API_KEY is not set" or 401 from OpenAI

Run `feather onboard --force` and paste the key when asked. Or edit
`~/.feather/.env` directly:

```
OPENAI_API_KEY=sk-...
```

The key must start with `sk-`. If it doesn't, you copied something
wrong from the dashboard.

To verify the key is being picked up:

```bash
grep OPENAI_API_KEY ~/.feather/.env
```

## "active_provider=openrouter but no openrouter: block in app.yaml"

You flipped `active_provider: openrouter` but did not paste an
`openrouter:` block. Either:

* Switch back to OpenAI: edit `~/.feather/config/app.yaml` and set
  `active_provider: openai`.
* Or paste a minimal openrouter block (see
  [providers.md](providers.md#openrouter)).

## OpenRouter HTTP 402 ("insufficient credits")

You ran out of credit. Top up at <https://openrouter.ai/credits>. If
you want to be warned earlier, set a billing alert in the OpenRouter
dashboard.

## OpenRouter HTTP 503 with strict routing

The model you pinned is temporarily unavailable from the upstream you
restricted to. Two options:

* Add a `fallback_models:` list in your `openrouter:` block. Feather
  will retry with the next model on the list.
* Loosen routing: `provider_preferences.allow_fallbacks: true` lets
  OpenRouter pick a different upstream for the same model.

## "Docker is not installed or the daemon is not reachable"

The `feather init-memory` command needs Docker. Install it from
<https://docs.docker.com/get-docker/> and start it:

* macOS / Windows: launch Docker Desktop.
* Linux: `sudo systemctl start docker` (or `service docker start`).

Test with `docker ps`. If you see a list (even an empty one), you're
good.

If you don't want Docker, use a remote Qdrant URL instead. See
[memory.md](memory.md#use-a-remote-qdrant-instead).

## "Qdrant did not respond at .../readyz within 60s"

The container started but didn't pass the health check fast enough.
Check that the container is actually running:

```bash
docker ps --filter name=feather-qdrant
```

If it's there, look at its logs:

```bash
docker logs feather-qdrant --tail 50
```

Common causes:

* Port 6333 was in use. Pick a different port:
  `feather init-memory --port 7333`.
* The image failed to pull (no internet, registry rate limit). Try
  `docker pull qdrant/qdrant:latest` directly.
* Disk full. `docker system df` and clean up if needed.

If the container is *not* listed, run `docker logs feather-qdrant`
anyway to see why it exited.

## "Memory marker exists but the agent doesn't remember things"

Memory needs three things to work:

1. The marker file at `~/.feather/state/memory.json`.
2. Qdrant reachable at the URL in the marker (or in `QDRANT_URL`).
3. `GEMINI_API_KEY` set in the environment.

Verify:

```bash
cat ~/.feather/state/memory.json
curl http://localhost:6333/readyz
grep GEMINI_API_KEY ~/.feather/.env
```

If any of those fails, fix that first. Then check the log:

```bash
tail -100 ~/.feather/state/logs/feather.log
```

Search for `memory` and look for embedding errors.

## TUI shows garbled characters or weird spacing

Feather's TUI is built on Textual, which expects a true-color terminal
with UTF-8. Try:

* `export TERM=xterm-256color`
* `export LANG=en_US.UTF-8`
* Run inside a modern terminal (iTerm2, Alacritty, GNOME Terminal,
  Windows Terminal). The macOS default Terminal.app works but is
  cramped.

If the TUI is still unhappy, fall back to the streaming console:

```bash
feather cli
```

It works in any terminal that supports basic colors.

## Telegram bot connects but doesn't respond

Two things to check:

* The bot is in **private chat** with you, not a group. Group support
  is not implemented.
* Telegram delivers messages with a slight delay. Wait a few seconds
  before declaring it dead.

If it really is dead, look at the log
(`~/.feather/state/logs/feather.log` or
`./.feather/logs/feather.log`) and search for `telegram`.

## LINE / WhatsApp webhook never fires

Verification checklist:

* Is the tunnel running? `ngrok` shows the live URL in its dashboard
  at <http://localhost:4040>.
* Is the webhook URL in the platform's console exactly
  `https://<tunnel>/messaging/webhook/{line,whatsapp}`?
* For WhatsApp, did Meta successfully verify the endpoint? The Meta
  console shows the verification status.
* Is Feather actually listening? Look for
  `messaging.webhook.started` lines in the log.

## Tools the agent should have are missing

If the agent says "I don't have access to web search", check:

* `PARALLEL_API_KEY` is set. Without it, `web_search` and `web_fetch`
  are not registered.
* The tool is in the agent's `registered_tools` list. Look at
  `~/.feather/config/agents/<agent>.yaml` (if you have a custom
  override) or the packaged `feather/_resources/config/agents/<agent>.yaml`.

## Sessions are missing or chat history starts fresh

Feather walks up from your current directory looking for `.feather/`.
If it finds none, it falls back to the global session DB at
`~/.feather/state/sessions.db`. So if you started chatting from inside
a project (with its own `.feather/`) and now you're chatting from
outside, you're talking to the global database, not the project one.

To confirm where you are:

```bash
feather --help            # the parser doesn't change behavior
ls ./.feather             # is there a project-local DB here?
```

To pin a project: from inside the project root, run `feather init`.

To force a specific session:

```bash
feather --session-id <uuid>
```

## "No skills found"

Feather looks in three places:

1. Packaged skills inside the wheel.
2. `~/.feather/skills/`
3. `./.feather/skills/`

If `/skills` shows fewer than five skills, the packaged set isn't
loading. That usually means the install is broken; reinstall:

```bash
pip install --upgrade --force-reinstall feather-agent-os
```

## "Lead unresponsive" banner appears in worker mode

You see a red `Lead unresponsive` marker mid-conversation. This means
the supervisor (the TUI process) hasn't seen a heartbeat row from the
lead worker subprocess for over 5 seconds. The worker is either stuck
in a long blocking call (e.g. a `bash` tool with no timeout, an
external HTTP call hung, the LLM provider stalled) or the worker
crashed entirely.

To recover:

1. Type `/restart-lead`. The supervisor SIGTERMs the worker, then
   SIGKILLs as a fallback after a 2 s grace, then respawns it on the
   same `--session-id`. Conversation history is preserved.
2. If `/restart-lead` itself fails (very rare — usually a Python
   import error in the worker boot path), type `/exit` and relaunch
   `feather --session-id <uuid>` (the session id is shown by
   `/session`).

The banner only appears when `FEATHER_USE_LEAD_WORKER=1` is set —
in default in-process mode the lead and TUI share an event loop, so
"the lead is hung" means the TUI is also hung and there's no banner
to draw.

## `request_restart` says "wheel install" — what does that mean?

The lead patched a file under `site-packages/feather/`. The fix works
for the current process and any restart, but the next
`pip install --upgrade feather-agent-os` (or `pipx upgrade`, or any
reinstall) will silently overwrite it. To make the fix durable:

* **Recommended:** ask the lead to call `submit_github_report` with
  `kind="issue"`. The bug + fix lands upstream and ships in the next
  release.
* **Alternative:** reinstall feather editable (`pip install -e .`
  from a clone) and re-apply the patch — future patches will then
  survive upgrades.

## Still stuck

Feather's runtime log is the best clue. Find it at:

* Project mode: `./.feather/logs/feather.log`
* Global mode: `~/.feather/state/logs/feather.log`

Open an issue at <https://github.com/timho102003/feather-os/issues>
with the log line that mentions the problem and whatever steps got you
there.
