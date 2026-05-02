"""Real-CLI smoke driver for OpenRouter.

Run manually — NOT picked up by ``uv run pytest``. Mirrors the earlier
``feather_cli_terminate_test.py`` style:

- Spawn the real ``feather`` CLI as a subprocess.
- Send realistic prompts that force tool calls.
- Scan captured stdout for expected markers.

Pre-reqs:

- ``OPEN_ROUTER_API_KEY`` set in the environment.
- ``config/app.yaml`` has ``active_provider: openrouter`` (or, if the
  default is still ``openai``, patch a temporary copy and set
  ``FEATHER_CONFIG_DIR`` — out of scope for this first pass, so for now
  the operator flips the YAML before running).

Run::

    export OPEN_ROUTER_API_KEY=sk-or-...
    python3 tests/integration/cli_openrouter_test.py

Exits 0 when every expected marker is present in the transcript.
"""

from __future__ import annotations

import asyncio
import os
import sys


PROMPTS = [
    "Use read_file to open pyproject.toml. Reply with exactly 3 short lines: "
    "(1) required Python version, (2) top dependency, (3) total dependency count.",
    "Use read_file in parallel to open both pyproject.toml and README.md at the "
    "same time, then reply with exactly 2 short lines — one fact from each file.",
    "Say the single word 'ready' and nothing else.",
]

EXPECTED_MARKERS = [
    "read_file",        # the agent invoked the tool at least once
    "openrouter",       # provider name visible somewhere in logs / output
    "ready",            # sanity prompt went through
]


async def main() -> int:
    if not os.getenv("OPEN_ROUTER_API_KEY"):
        print("OPEN_ROUTER_API_KEY not set", file=sys.stderr)
        return 2

    proc = await asyncio.create_subprocess_exec(
        "uv", "run", "feather",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdin is not None and proc.stdout is not None

    captured: list[bytes] = []

    async def reader() -> None:
        while True:
            line = await proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                return
            captured.append(line)
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()

    reader_task = asyncio.create_task(reader())

    async def send(text: str) -> None:
        proc.stdin.write((text + "\n").encode())  # type: ignore[union-attr]
        await proc.stdin.drain()  # type: ignore[union-attr]

    try:
        await asyncio.sleep(3.0)
        for prompt in PROMPTS:
            await send(prompt)
            await asyncio.sleep(35.0)

        await send("/exit")
        try:
            await asyncio.wait_for(proc.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            proc.terminate()
            await proc.wait()
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass

    transcript = b"".join(captured).decode(errors="replace")
    print("\n===== SESSION END =====", file=sys.stderr)
    missing: list[str] = []
    for marker in EXPECTED_MARKERS:
        found = marker in transcript
        print(f"  {marker!r} in output: {found}", file=sys.stderr)
        if not found:
            missing.append(marker)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
