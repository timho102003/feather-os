"""Pure render/format helpers for the Textual TUI.

No ``FeatherTextualApp`` import — this is the heaviest unit-test surface and is
kept App-free so tests import it without spinning up a running Textual app.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.geometry import Region

from feather.integrations.attachments.parse import (
    parse_attachment_drops,
    render_attachment_message,
)
from feather.models import RuntimeEvent, TaskRecord, TaskStatus
from feather.tui import _indent_lines, _status_style, preview_inline


_LEAD_STATUS_GLYPH = {"running": "●", "awaiting user": "◐", "idle": "○"}


def build_lead_strip(
    leads: tuple[tuple[str, str | None, str], ...],
    active_name: str,
) -> Text:
    """Render the switchable-lead strip.

    ``leads`` is ``(display_name, emoji, status)`` per lead in display order.
    The active lead is highlighted; each lead shows a status glyph so a busy
    background lead is visible without switching to it.
    """

    strip = Text()
    strip.append("leads ", style="dim")
    for index, (display_name, emoji, status) in enumerate(leads):
        if index:
            strip.append(" · ", style="dim")
        glyph = _LEAD_STATUS_GLYPH.get(status, "○")
        label = f"{emoji + ' ' if emoji else ''}{display_name} {glyph}"
        is_active = display_name == active_name
        strip.append(
            f"[{label}]" if is_active else label,
            style="bold white" if is_active else "grey58",
        )
    strip.append("   ctrl+←/→", style="dim")
    return strip


def build_header_text(
    *,
    agent_name: str,
    status: str,
    context_ratio: float | None,
    queue_depth: int,
    active_tool: str | None,
    session_id: str,
    lead_strip: Text | None = None,
) -> Text:
    """Build the Textual header renderable."""

    ctx = "ctx --" if context_ratio is None else f"ctx {round(context_ratio * 100)}%"
    active = f"active {active_tool}" if active_tool else "no active tool"
    header = Text.assemble(
        ("Feather", "bold white"),
        "  ",
        (agent_name, "white"),
        "  ",
        (status, _status_style(status)),
        "  ",
        (ctx, "white"),
        "  ",
        (f"queued {queue_depth}", "magenta" if queue_depth else "dim"),
        "  ",
        (active, "grey70" if active_tool else "dim"),
        "\n",
        (f"lead session {session_id}", "dim"),
    )
    if lead_strip is not None:
        header.append("\n")
        header.append_text(lead_strip)
    return header


def build_work_text(
    *,
    queue_depth: int,
    queued_messages: tuple[str, ...],
    running_agents: tuple[str, ...],
    task_rows: tuple[str, ...] = (),
    task_updates: tuple[str, ...] = (),
) -> Text:
    """Build the queued/future-work renderable."""

    rows = Text()
    has_rows = False
    if running_agents:
        has_rows = True
        rows.append("Live sub-agents\n", style="bold white")
        for agent in running_agents:
            rows.append(_indent_lines(agent), style="white")
    if queue_depth:
        if has_rows:
            rows.append("\n")
        has_rows = True
        rows.append("Queued queries\n", style="bold white")
        for index, message in enumerate(queued_messages, 1):
            rows.append(_indent_lines(f"{index}. {message}"), style="white")
    if task_rows:
        if has_rows:
            rows.append("\n")
        has_rows = True
        rows.append("Tasks\n", style="bold white")
        for task in task_rows:
            rows.append(_indent_lines(task), style="white")
    if task_updates:
        if has_rows:
            rows.append("\n")
        has_rows = True
        rows.append("Recent task updates\n", style="bold white")
        for update in task_updates[-3:]:
            rows.append(_indent_lines(update), style="white")
    if not has_rows:
        rows.append(
            "No queued queries, tracked tasks, live sub-agents, or recent task updates.",
            style="dim",
        )
    return rows


def format_task_row(
    task: TaskRecord,
    *,
    live_sessions: frozenset[str] = frozenset(),
) -> str:
    """Render one task row for the Textual monitor."""

    label = {
        "queued": "QUEUE",
        "running": "RUN",
        "blocked_needs_input": "BLOCK",
        "completed_with_report": "DONE",
        "completed_with_artifacts": "DONE",
        "completed_without_artifacts": "DONE",
        "failed": "FAIL",
        "stopped": "STOP",
    }.get(task.status.value, task.status.value.upper())
    if (
        task.status == TaskStatus.RUNNING
        and task.responsible_session_id in live_sessions
    ):
        label = "LIVE"
    agent = task.responsible_agent_name or "-"
    session = (
        task.responsible_session_id[:8]
        if task.responsible_session_id is not None
        else "-"
    )
    detail = task.blocked_question or task.error or task.title
    return (
        f"{label:<5} {agent:<10} {session:<8} "
        f"{preview_inline(detail, limit=92)}"
    )


def format_transcript_block(title: str, body: str) -> str:
    """Render one conversation block as plain text for copying."""

    body = body.strip()
    if not body:
        return title.strip()
    return f"{title.strip()}\n{_indent_lines(body).rstrip()}"


def build_transcript_text(blocks: tuple[str, ...]) -> str:
    """Build the plain-text transcript copied from the TUI."""

    return "\n\n".join(block.strip() for block in blocks if block.strip())


def summarize_user_input_for_display(text: str, root: Path) -> str:
    """Render dropped local paths as compact attachment placeholders."""

    draft = parse_attachment_drops(text, root=root)
    if not draft.attachments:
        return text.strip()
    effective_text = draft.text or "Please review the attached file(s)."
    return render_attachment_message(effective_text, draft.attachments)


def normalize_pasted_attachment_text(text: str, root: Path) -> str:
    """Convert pasted external file paths into explicit file URIs.

    Terminal drag/drop usually arrives as a bracketed paste containing one or
    more absolute file paths. Core agent parsing still ignores plain
    out-of-workspace absolute paths; this TUI-only conversion marks a paste/drop
    as explicit without changing normal typed path mentions.
    """

    stripped = text.strip()
    if not stripped:
        return text
    try:
        tokens = shlex.split(stripped, posix=True)
    except ValueError:
        return text
    if not tokens:
        return text

    converted: list[str] = []
    changed = False
    for token in tokens:
        converted_token = _normalize_pasted_attachment_token(token, root=root)
        if converted_token is None:
            return text
        converted.append(converted_token)
        changed = changed or converted_token != token
    return " ".join(converted) if changed else text


def _normalize_pasted_attachment_token(token: str, *, root: Path) -> str | None:
    if token.startswith("file://"):
        return token
    candidate = Path(token).expanduser()
    if not candidate.is_absolute():
        return None
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.is_file():
        return None
    root = root.resolve()
    if resolved == root or root in resolved.parents:
        return token
    return resolved.as_uri()


def is_exit_command(text: str) -> bool:
    """Return whether composer text requests leaving the TUI."""

    return text.strip() == "/exit"


def region_contains_point(region: Region, x: float | int, y: float | int) -> bool:
    """Return whether a Textual region contains screen coordinates."""

    return region.contains_point((int(x), int(y)))


def format_agent_message_event(event: RuntimeEvent) -> str:
    """Render an agent-message event with full delivered bodies when available."""

    payload = event.payload or {}
    bodies = payload.get("bodies")
    if not isinstance(bodies, list) or not bodies:
        return event.text or ""

    sender = _agent_message_sender(payload)
    count = len(bodies)
    plural = "message" if count == 1 else "messages"
    header = f"{sender} delivered {count} {plural}."
    rendered: list[str] = [header]
    for index, body in enumerate(bodies, 1):
        content = str(body).strip() or "(empty body)"
        if count == 1:
            rendered.append(content)
        else:
            rendered.append(f"Message {index}\n{content}")
    return "\n\n".join(rendered)


def summarize_agent_message_update(event: RuntimeEvent) -> str:
    """Build a one-line completion summary for the work panel."""

    payload = event.payload or {}
    sender = _agent_message_sender(payload)
    count = payload.get("count")
    total_chars = payload.get("total_chars")
    if isinstance(count, int) and isinstance(total_chars, int):
        plural = "message" if count == 1 else "messages"
        return f"completed {sender} ({count} {plural}, {total_chars} chars)"
    return f"completed {preview_inline(event.text or 'sub-agent report', limit=80)}"


def _agent_message_sender(payload: dict[str, Any]) -> str:
    name = payload.get("from_agent_name")
    session = payload.get("from_session_id")
    if isinstance(name, str) and isinstance(session, str) and session:
        return f"{name} {session[:8]}"
    if isinstance(name, str):
        return name
    return "sub-agent"
