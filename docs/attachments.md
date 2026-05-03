# Attachments

You can drop files into a chat to give the agent something concrete to
work with. Text, source code, configuration files, images, and PDFs
are all supported.

## How to attach a file

In the Textual TUI, drag a file from your file manager onto the
terminal window. The path appears in your message. You can also paste
an absolute path or a `file://` URL by hand.

Examples that work:

```
look at /home/me/Downloads/report.pdf and pull out the dates
fix the bug in /home/me/code/api/server.py
file:///Users/tim/photos/screenshot.png  what is wrong with this UI?
```

You can attach more than one file in a single message. Feather parses
each absolute path it finds and adds them as attachments.

## What gets attached

When you drop a file, Feather:

1. Validates that the path exists and is readable.
2. Checks the file is not on a sensitive blocklist (see below).
3. Decides what kind of attachment it is (text, image, PDF, or "binary"
   that the agent will need a tool to read).
4. Stores a copy under `./.feather/attachments/` so the original cannot
   change underneath the chat.
5. Sends the file to the model in the right shape (inline text, base64
   image, or a path the agent can open with `read_pdf` or `read_file`).

## Supported types and limits

| Kind | What works | Cap |
|---|---|---|
| Plain text and source code | `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.md`, `.csv`, `.html`, `.toml`, `.xml`, `.css`, `.ini`, `.log`, `.txt`, and more | 1 MiB and 120,000 chars per inline attachment |
| Images | `.png`, `.jpeg`, `.webp`, `.gif` | 50 MiB per file |
| PDFs | `.pdf` | 50 MiB per file |
| Anything else | A path attachment the agent can read with `read_file` or `read_pdf` | 50 MiB per file |

If a text file is larger than the inline cap, Feather attaches it as a
file the agent can read on demand instead of slamming the whole thing
into the prompt.

## What is blocked

To keep secrets out of the prompt, Feather refuses to attach files
that look sensitive:

* `.env`, `.git-credentials`, `.netrc`, `.npmrc`, `.pypirc`
* SSH keys: `id_rsa`, `id_ed25519`, `id_ecdsa`, `id_dsa`, anything
  ending in `.key`, `.pem`, `.p12`, `.pfx`
* Anything inside system directories: `/bin`, `/boot`, `/dev`, `/etc`,
  `/lib`, `/lib64`, `/proc`, `/root`, `/run`, `/sbin`, `/sys`, `/usr`,
  `/var`

If you really need to share one of these, copy the relevant lines into
a fresh file in your project and attach that. Or paste the content
directly into the message.

## PDFs

Feather has a dedicated `read_pdf` tool. For most PDFs the default
mode works fine: it pulls the text layer out with `pypdf`. The agent
will pick the mode based on what you ask for.

The three modes:

* `auto`: default. Try the text layer first, fall back to other
  methods if needed.
* `text`: text layer only. Fast. Fails on scanned PDFs.
* `opendataloader_hybrid`: uses the `opendataloader-pdf` hybrid
  pipeline (text layer plus OCR plus layout). Best for scanned or
  complex PDFs but only available if you installed the optional
  dependency.

To install the hybrid extras:

```bash
pip install 'feather-agent-os[pdf-hybrid]'
```

Or if you already installed Feather, add the extra:

```bash
pip install --upgrade 'feather-agent-os[pdf-hybrid]'
```

Then ask the agent: "use the opendataloader hybrid mode to read
report.pdf, this one is scanned."

## Images

Images are sent to the model as base64-encoded inline content. Most
recent OpenAI and Anthropic vision models can read them directly. The
agent describes what it sees, finds text, identifies UI bugs, etc.

If the model you picked does not support images, the call fails and
the agent will tell you. Switch to a vision-capable model (see
[providers.md](providers.md)).

## Where attachments live

`./.feather/attachments/<sha256>/<original-filename>`. The hash dedup
prevents storing the same file twice across sessions in the same
project. Delete the folder to free space; nothing depends on the files
still being there once the chat has used them.

## Things to avoid

* **Don't drop entire directories.** Feather attaches files, not
  directories. If you drop a folder, only the path is captured. Tell
  the agent to walk it with `bash` or `grep` instead.
* **Don't paste secrets.** Attaching is blocked, but pasting the body
  of a `.env` file directly will end up in the model's context.
  Strip secrets first.
* **Don't expect images to work on every model.** Vision support is
  per model; see your provider's docs.

## Next

* See [tools-and-commands.md](tools-and-commands.md) for `read_pdf`
  and `read_file` parameters.
* See [getting-started.md](getting-started.md#where-things-live) for
  where attachments live on disk.
