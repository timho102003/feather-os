"""Serialize a RuntimeEvent into the JSON shape clients consume.

Mirrors the worker-stream codec so the web client renders exactly what the TUI
renders (thinking deltas, tool started/finished, awaiting-user, sub-agent
messages, compaction/scheduler markers, usage updates).
"""

from __future__ import annotations

from typing import Any

from feather.models import RuntimeEvent

__all__ = ("event_to_dict",)


def event_to_dict(event: RuntimeEvent) -> dict[str, Any]:
    """Return the wire dict for one runtime event (drops default-None fields)."""

    data: dict[str, Any] = {"kind": event.kind}
    if event.text is not None:
        data["text"] = event.text
    if event.tool_name is not None:
        data["tool_name"] = event.tool_name
    if event.payload is not None:
        data["payload"] = event.payload
    return data
