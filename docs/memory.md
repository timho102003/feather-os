# Long-term Memory

Long-term memory lets the agent remember facts about you and your work
across sessions. Without memory, every chat starts fresh. With memory,
when you mention a project, a teammate, or a preference, the agent
quietly stores it. Next week when you bring it up again, the agent
already knows.

Memory is **optional and off by default**. You opt in with one command.

## What is under the hood

* **Qdrant** is a vector database. Feather uses it to store and search
  small chunks of remembered text.
* **Gemini embeddings** turn text into vectors. The agent calls Gemini
  whenever it stores or recalls a memory.
* **A marker file** at `~/.feather/state/memory.json` is the on/off
  switch. If the file is there, memory is on. If not, memory is off.
  Everything else (the wizard, the runtime) reads that file.

You will need:

* Docker installed and running (only if you want Feather to manage the
  Qdrant container for you), or a remote Qdrant URL you already have.
* A Gemini API key from <https://aistudio.google.com/apikey>.

## Turn it on

Run the command from anywhere:

```bash
feather init-memory
```

Feather will:

1. Pull the Qdrant Docker image (only on first run).
2. Start a container called `feather-qdrant` listening on port 6333.
3. Wait until Qdrant answers a readiness probe.
4. Write the marker file.
5. Print the URL it ended up using.

If you already had a Gemini key in `~/.feather/.env`, the next chat
just works. If not, run `feather onboard` once and it will see the
marker, skip the "do you want memory?" question, and go straight to
asking for the Gemini key.

## Pause it

Stop the container without losing any data:

```bash
feather stop-memory
```

The marker stays. The container is gone but the docker volume
`feather-qdrant-data` is preserved. Run `feather init-memory` again to
bring it back up.

## Remove it

Stop and remove the container:

```bash
feather remove-memory
```

The marker file is deleted. The data volume is **kept** so you do not
lose memories by accident. The next `feather onboard` run will ask the
memory question again.

To wipe stored memories permanently, add `--purge`:

```bash
feather remove-memory --purge
```

This deletes the docker volume too. There is no undo.

## Use a remote Qdrant instead

If you already run Qdrant somewhere (your own server, Qdrant Cloud,
etc.), tell Feather where to find it before you run anything else:

```bash
echo 'QDRANT_URL=https://your-qdrant.example.com' >> ~/.feather/.env
echo 'QDRANT_API_KEY=...' >> ~/.feather/.env  # only if your endpoint needs one
echo 'GEMINI_API_KEY=...' >> ~/.feather/.env
```

Then either run `feather init-memory` (which will skip the Docker step
and just write the marker) or simply create the marker file by hand:

```bash
echo '{"version":1,"url":"https://your-qdrant.example.com","mode":"cloud"}' \
  > ~/.feather/state/memory.json
```

Feather reads `QDRANT_URL` from the environment first, then the marker,
then the URL inside `app.yaml`. The first one it finds wins.

## What the agent does with memory

The agent has two memory tools and one background process.

**`recall_memory`.** The agent calls this when it needs a specific
fact. You don't normally see it; it just runs. You can also ask the
agent directly: "what do you remember about my Postgres project?"

**`manage_memory`.** When you say "remember that I prefer Black for
Python formatting", the agent stores it. When you say "forget that I
hate yaml", it removes it. When you say "actually, my preferred name is
Tim now", it updates.

**Background extractor.** Every ten user turns by default, Feather runs
a small extraction call against the recent chat to pull out things
worth remembering (preferences, ongoing projects, durable facts). This
is why long-term memory needs the Gemini key even when you do not
explicitly ask the agent to remember anything.

## Tuning what gets remembered and recalled

The whole memory pipeline is in the `memory:` block of
`~/.feather/config/app.yaml`. The shipped defaults are sensible. The
knobs you might want to touch:

```yaml
memory:
  enabled: true
  qdrant:
    url: http://localhost:6333         # env QDRANT_URL wins if set
    collection_name: feather_memory_v2
  embedding:
    provider: gemini
    model: gemini-embedding-2-preview
    output_dimensionality: 3072
  retrieval:
    top_k_prompt_injection: 5          # how many memories appear in the prompt
    top_k_tool: 10                     # how many recall_memory returns by default
    score_threshold: 0.5               # below this, results are dropped
  trigger:
    trigger_turns: 10                  # extract every N user turns
    background: true                   # do it without blocking the chat
  extraction:
    provider: openai
    model: gpt-5.4-nano                # cheap, structured-output-friendly
  classification:
    provider: openai
    model: gpt-5.4-nano
```

The full reference is in [configuration.md](configuration.md#memory).

## Inspect what is stored

The Qdrant Web UI at <http://localhost:6333/dashboard> (or the URL of
your remote Qdrant) lists every collection and lets you browse the raw
vectors. The Feather collection is `feather_memory_v2` by default.

In a chat, ask the agent: "what do you remember about me?" or "list
your top ten memories." The agent will call `recall_memory` and walk
you through what it finds.

## Useful slash commands

In the TUI:

* `/qdrant status`: is the local container running?
* `/qdrant start`: start it.
* `/qdrant stop`: stop it.
* `/qdrant remove`: remove it.

These are the same operations as the CLI commands above, surfaced from
inside chat for convenience.

## Troubleshooting

* **`Docker is not installed or the daemon is not reachable.`** Install
  Docker Desktop or the Docker Engine and start it, then try again. On
  Linux, check `systemctl status docker`.
* **`Qdrant did not respond at .../readyz within 60s`.** Check whether
  the container actually started: `docker ps -a --filter
  name=feather-qdrant`. Look at its logs: `docker logs feather-qdrant`.
  Port 6333 might be in use; stop whatever else is listening or pick a
  different port with `feather init-memory --port 7333`.
* **The agent keeps saying it does not know things you told it.** Check
  that memory is actually on (`/qdrant status` in chat, or
  `ls ~/.feather/state/memory.json`). Check `~/.feather/.env` has
  `GEMINI_API_KEY`. Check the Feather log at
  `~/.feather/state/logs/feather.log` for embedding errors.

More fixes in [troubleshooting.md](troubleshooting.md).
