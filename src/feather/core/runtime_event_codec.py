"""Wire-format codec for streaming :class:`RuntimeEvent`s between processes.

The lead worker subprocess emits one JSON line per :class:`RuntimeEvent`
to its ``stdout``; the supervisor (the Textual TUI process that spawned
it) reads the stream line-by-line and replays the events through the
same in-process handler that drove the UI before the split. This keeps
the rendering glue identical on both sides of the process boundary —
the boundary is a serialization concern, nothing more.

The codec is deliberately tiny:

* Each event is one UTF-8 JSON object on its own line.
* Default-``None`` fields are dropped so the wire stays small for the
  hot path (assistant text deltas).
* Decode failures surface as :class:`EventCodecError` so the supervisor
  can log and skip a malformed line without tearing down the worker.

Why not pickle / msgpack / protobuf? Because JSON keeps the stream
human-readable for ``tail``-style debugging, the worker has no native
serializer dependency, and we never round-trip more than ~1 KB per line
in practice.
"""

from __future__ import annotations

import json
from typing import Any

from feather.models import RuntimeEvent


class EventCodecError(ValueError):
    """Raised when a JSON line cannot be decoded as a RuntimeEvent."""


def encode_event(event: RuntimeEvent) -> str:
    """Encode ``event`` as a single JSON line (no trailing newline)."""

    payload: dict[str, Any] = {"kind": event.kind}
    if event.text is not None:
        payload["text"] = event.text
    if event.tool_name is not None:
        payload["tool_name"] = event.tool_name
    if event.payload is not None:
        payload["payload"] = event.payload
    # ensure_ascii=False keeps unicode small and human-readable; separators
    # trim the per-line overhead since this can fire 100s of times per turn.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def decode_event(line: str) -> RuntimeEvent:
    """Decode one JSON line into a :class:`RuntimeEvent`.

    Raises :class:`EventCodecError` for any malformed input — empty
    lines, non-JSON, non-object payloads, or missing/invalid ``kind``.
    """

    stripped = line.strip()
    if not stripped:
        raise EventCodecError("empty line")
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise EventCodecError(f"invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise EventCodecError(
            f"expected JSON object, got {type(raw).__name__}"
        )
    kind = raw.get("kind")
    if not isinstance(kind, str):
        raise EventCodecError("missing or non-string 'kind' field")
    return RuntimeEvent(
        kind=kind,
        text=raw.get("text"),
        tool_name=raw.get("tool_name"),
        payload=raw.get("payload"),
    )
