# Getting Started

This guide walks you from "I have never used Feather" to "I am chatting
with the agent" in under five minutes.

## What you need

* **Python 3.12 or 3.13.** Check with `python3 --version`. If you don't
  have it, install from <https://www.python.org/downloads/> or use your
  OS package manager.
* **An OpenAI API key.** Get one at
  <https://platform.openai.com/api-keys>. The wizard will ask for it.
* **(Optional) Docker.** Only needed if you want long-term memory and you
  want Feather to spin up Qdrant for you. Skip this for now if you just
  want to try the chat.

That's everything required to start.

## Step 1. Install

Pick the install method that matches how you usually install Python
tools.

```bash
pip install feather-agent-os
```

```bash
pipx install feather-agent-os
```

```bash
uv tool install feather-agent-os
```

Any of those puts a `feather` command on your `PATH`. Verify with:

```bash
feather --version
```

You should see something like `feather 0.2.0`.

## Step 2. Run the onboarding wizard

```bash
feather onboard
```

The wizard asks for, in order:

1. Your name. Required. The agent uses this to address you.
2. Optional details: preferred name, role, what you work on, a short
   bio. Press Enter to skip any of them.
3. Your OpenAI API key. Required. Pasted as a hidden input so it does
   not land in your shell history.
4. Provider choice. Press Enter to keep OpenAI. Type `2` to use
   OpenRouter instead (you will be asked for an OpenRouter key).
5. Whether to enable long-term memory. If you say yes, the wizard offers
   to start a local Qdrant Docker container for you, or to use one you
   already have, or to point at a remote Qdrant URL. You will also need
   a Gemini key for embeddings.
6. Whether to enable web search. If yes, you will be asked for a
   Parallel AI key.

When the wizard finishes you'll see a confirmation line that tells you
where your profile and `.env` ended up. By default that's
`~/.feather/user.md` and `~/.feather/.env`.

You only need to run `feather onboard` once per machine. Re-run it any
time with `feather onboard --force` if you want to change answers.

## Step 3. Start chatting

```bash
feather
```

The terminal UI opens. You see a header at the top, a transcript area
in the middle, and an input box at the bottom. Type a message and press
Enter. The agent will stream a reply. When it uses a tool you see the
tool call and its output inline.

Try something like:

* `read the file pyproject.toml and tell me what version of python it requires`
* `find every TODO comment in this folder`
* `what is the largest file under src/`

When you are done, type `/exit`.

## Resume a previous chat

Every chat has a session ID. To pick up where you left off:

```bash
feather --session-id <the-uuid-from-before>
```

You can find the ID at the top of the chat (`Feather session: ...`) or
in the header of the TUI.

## Where things live

By default Feather keeps two things separate.

**Personal, follows you across projects:**

* `~/.feather/.env`: your API keys
* `~/.feather/user.md`: what the agent knows about you
* `~/.feather/config/app.yaml`: your settings
* `~/.feather/config/agents/`: custom sub-agents you wrote
* `~/.feather/skills/`: skills you installed
* `~/.feather/state/memory.json`: memory marker
* `~/.feather/state/onboarded.json`: wizard completion marker
* `~/.feather/state/sessions.db`: chats started outside any project

**Per project, lives next to the code:**

* `./.feather/db/feather.db`: chat history for that project
* `./.feather/tmp/`: overflow tool output written to disk
* `./.feather/attachments/`: files you dropped into chat
* `./.feather/skills/`: skills that should only apply here
* `./.feather/user.md`: optional persona override for this project

To pin a project so Feather always uses its local `.feather/`, run
`feather init` inside the project's top-level folder. After that, any
`feather` invocation from inside the folder (or any subfolder) uses the
local state.

To run from anywhere without pinning a project, just type `feather`. If
no `.feather/` is found by walking up from the current directory,
Feather uses the global location for sessions. This is fine for casual
chats; pin a project when you start serious work in a repo.

## Two ways to chat

* `feather` (no arguments) opens the **Textual TUI**, the colorful
  full-screen interface. Default and recommended.
* `feather cli` opens the **streaming console**, a simpler scrolling
  experience that works well over flaky SSH connections or in plain
  terminals that do not handle full-screen apps.

Both share the same chat history and tools. Pick whichever feels
better.

## Slash commands you might want right away

Inside a chat:

| Command | What it does |
|---|---|
| `/help` | List every slash command. |
| `/exit` (`/quit`) | Leave the chat. |
| `/clear` | Clear the on-screen transcript (history is preserved). |
| `/copy` | Copy the visible transcript to your clipboard. |
| `/session` | Show the current session ID and how full the context window is. |
| `/skills` | Show what skills the agent can load. |
| `/qdrant` | Manage the local memory container. |

The full slash command list is in
[tools-and-commands.md](tools-and-commands.md).

## Next

* Want a different model or provider? See [providers.md](providers.md).
* Want long-term memory? See [memory.md](memory.md).
* Stuck? See [troubleshooting.md](troubleshooting.md).
