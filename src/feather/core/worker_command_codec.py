"""Wire-format codec for streaming commands from the supervisor to the worker.

The supervisor (the Textual TUI process that spawned the lead worker)
writes one JSON line per command to the worker's ``stdin``. The worker
parses each line and dispatches the typed command:

* :class:`RunCommand` — start one ``BaseAgent.run`` cycle with the user
  text the operator just typed.
* :class:`ResumeOnInboxCommand` — wake the worker on an inbox push (a
  message arrived from a sub-agent or an external integration).
* :class:`EnqueueUserInputCommand` — append a mid-turn user message to
  the worker's in-process input queue, so the existing
  :meth:`BaseAgent._drain_user_input_queue` machinery picks it up
  between iterations without restarting the run.
* :class:`ShutdownCommand` — request a graceful shutdown.
* :class:`ConfigReloadCommand` — instruct the worker to reload the on-disk
  config and, when ``reload_class`` is ``"next_turn"``, also rebuild any
  cached agents.  The worker acknowledges via a ``_config_reload_ack``
  control event on stdout (see :mod:`feather.core.runtime_event_codec`).

Decoding is strict: malformed input raises :class:`CommandCodecError`
so the worker can log + skip rather than misinterpret a partial line.
The codec is symmetric with :mod:`feather.core.runtime_event_codec`,
so both directions of the worker pipe share the same JSONL grammar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class CommandCodecError(ValueError):
    """Raised when a JSON line cannot be decoded as a worker command."""


@dataclass(slots=True, frozen=True)
class RunCommand:
    """Run a single ``BaseAgent.run`` cycle with the supplied user text."""

    session_id: str
    incoming_text: str


@dataclass(slots=True, frozen=True)
class ResumeOnInboxCommand:
    """Wake the worker to drain pending inbox messages for the session."""

    session_id: str


@dataclass(slots=True, frozen=True)
class EnqueueUserInputCommand:
    """Append a mid-turn user message to the worker's input queue."""

    session_id: str
    text: str


@dataclass(slots=True, frozen=True)
class ShutdownCommand:
    """Request a graceful worker shutdown (drains, writes final heartbeat)."""


@dataclass(slots=True, frozen=True)
class ConfigReloadCommand:
    """Instruct the worker to reload config from disk and optionally rebuild agents.

    Attributes:
        correlation_id: Opaque string echoed back in the ``_config_reload_ack``
            control event so the supervisor can match the response to the
            originating call.
        changed_paths: Dotted config paths that were written to disk
            (e.g. ``["app.active_provider", "app.openai.model"]``).
        reload_class: ``"live"`` triggers a config file re-read only;
            ``"next_turn"`` additionally rebuilds cached agents.
    """

    correlation_id: str
    changed_paths: list[str]
    reload_class: str  # "live" or "next_turn"


WorkerCommand = (
    RunCommand
    | ResumeOnInboxCommand
    | EnqueueUserInputCommand
    | ShutdownCommand
    | ConfigReloadCommand
)

_RUN = "run"
_RESUME = "resume_on_inbox"
_ENQUEUE = "enqueue_user_input"
_SHUTDOWN = "shutdown"
_RELOAD_CONFIG = "reload_config"

# Control-event kind emitted by the worker as the ack for ConfigReloadCommand.
# Defined here (close to the command) so both sides import from one place.
CONFIG_RELOAD_ACK_KIND = "_config_reload_ack"


def encode_command(command: WorkerCommand) -> str:
    """Encode ``command`` as a single JSON line (no trailing newline)."""

    payload: dict[str, Any]
    match command:
        case RunCommand(session_id=session_id, incoming_text=incoming_text):
            payload = {
                "cmd": _RUN,
                "session_id": session_id,
                "incoming_text": incoming_text,
            }
        case ResumeOnInboxCommand(session_id=session_id):
            payload = {"cmd": _RESUME, "session_id": session_id}
        case EnqueueUserInputCommand(session_id=session_id, text=text):
            payload = {
                "cmd": _ENQUEUE,
                "session_id": session_id,
                "text": text,
            }
        case ShutdownCommand():
            payload = {"cmd": _SHUTDOWN}
        case ConfigReloadCommand(
            correlation_id=correlation_id,
            changed_paths=changed_paths,
            reload_class=reload_class,
        ):
            payload = {
                "cmd": _RELOAD_CONFIG,
                "correlation_id": correlation_id,
                "changed_paths": list(changed_paths),
                "reload_class": reload_class,
            }
        case _:
            raise CommandCodecError(
                f"unknown command type: {type(command).__name__}"
            )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _require_str(raw: dict[str, Any], key: str, *, cmd: str) -> str:
    """Return ``raw[key]`` if it is a string, else raise ``CommandCodecError``."""

    value = raw.get(key)
    if not isinstance(value, str):
        raise CommandCodecError(f"{cmd!r} requires string {key}")
    return value


def _require_str_list(raw: dict[str, Any], key: str, *, cmd: str) -> list[str]:
    """Return ``raw[key]`` as a list of strings, else raise ``CommandCodecError``."""

    value = raw.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CommandCodecError(f"{cmd!r} requires list[str] {key}")
    return value


def decode_command(line: str) -> WorkerCommand:
    """Decode one JSON line into a typed worker command.

    Raises :class:`CommandCodecError` on any malformed input.
    """

    stripped = line.strip()
    if not stripped:
        raise CommandCodecError("empty line")
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise CommandCodecError(f"invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise CommandCodecError(
            f"expected JSON object, got {type(raw).__name__}"
        )
    cmd = raw.get("cmd")
    if not isinstance(cmd, str):
        raise CommandCodecError("missing or non-string 'cmd' field")

    if cmd == _RUN:
        return RunCommand(
            session_id=_require_str(raw, "session_id", cmd=cmd),
            incoming_text=_require_str(raw, "incoming_text", cmd=cmd),
        )
    if cmd == _RESUME:
        return ResumeOnInboxCommand(
            session_id=_require_str(raw, "session_id", cmd=cmd)
        )
    if cmd == _ENQUEUE:
        return EnqueueUserInputCommand(
            session_id=_require_str(raw, "session_id", cmd=cmd),
            text=_require_str(raw, "text", cmd=cmd),
        )
    if cmd == _SHUTDOWN:
        return ShutdownCommand()
    if cmd == _RELOAD_CONFIG:
        return ConfigReloadCommand(
            correlation_id=_require_str(raw, "correlation_id", cmd=cmd),
            changed_paths=_require_str_list(raw, "changed_paths", cmd=cmd),
            reload_class=_require_str(raw, "reload_class", cmd=cmd),
        )
    raise CommandCodecError(f"unknown 'cmd' value: {cmd!r}")
