"""First-run onboarding wizard for Feather.

The wizard captures the minimum viable configuration: identity, OpenAI
API key, optional provider/memory/web-search choices. It writes:

- ``.env`` (append-only, ``0o600``) — captured API keys.
- ``.feather/user.md`` (via :class:`feather.profile.UserProfileStore`) —
  identity facts that the lead agent will see in every prompt.
- ``config/app.yaml`` toggles for ``active_provider`` and
  ``memory.enabled`` (regex line rewrite to preserve comments).
- ``.feather/onboarded.json`` — completion marker so the next run skips
  the wizard.

I/O callables (:meth:`OnboardingWizard.input_fn`,
:meth:`OnboardingWizard.output_fn`) are injected so tests can drive the
prompts deterministically without a real terminal.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# ---- Local Qdrant Docker management -------------------------------------

_QDRANT_CONTAINER_NAME = "feather-qdrant"
_QDRANT_IMAGE = "qdrant/qdrant:latest"
_QDRANT_VOLUME = "feather-qdrant-data"
_QDRANT_PORT = 6333
_QDRANT_LOCAL_URL = f"http://localhost:{_QDRANT_PORT}"


class NoProviderConfigured(RuntimeError):
    """Raised when the wizard finishes provider prompts with nothing wired.

    Feather requires at least one LLM provider (OpenAI, OpenRouter, or
    Claude) to start. A user who declines all three exits the wizard
    without writing the completion marker, so the next ``feather`` run
    re-prompts rather than booting into a half-configured state.
    """


class DockerNotAvailable(RuntimeError):
    """Raised when the ``docker`` CLI cannot be invoked."""


class QdrantStartFailed(RuntimeError):
    """Raised when the local Qdrant container cannot be started."""


@dataclass(slots=True)
class QdrantContainerStatus:
    """Snapshot of the container state used by the wizard."""

    state: str  # "running", "stopped", or "absent"


# ``CompletedProcess`` proxy used by the runner abstraction.
_DockerRunner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]
_ReadyChecker = Callable[[float], bool]


def _default_docker_runner(cmd: list[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _default_ready_checker(timeout_s: float) -> bool:
    """Poll the Qdrant ``/readyz`` endpoint until 200 or timeout."""

    deadline = time.monotonic() + timeout_s
    url = f"{_QDRANT_LOCAL_URL}/readyz"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return False


def _probe_qdrant_url(url: str, *, timeout_s: float = 3.0) -> bool:
    """Return True when ``<url>/readyz`` responds with 2xx.

    Used by the onboarding wizard's cloud-URL prompt to fail fast on
    typos like ``https://qdrant:6333`` against an HTTP-only server,
    where the SSL handshake error otherwise crashes the next TUI run.
    """

    target = f"{url.rstrip('/')}/readyz"
    try:
        with urllib.request.urlopen(target, timeout=timeout_s) as resp:  # noqa: S310
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError):
        return False


def docker_available(
    *, runner: _DockerRunner = _default_docker_runner
) -> bool:
    """Return True when the ``docker`` CLI is callable on this host."""

    try:
        result = runner(["docker", "--version"])
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def qdrant_container_state(
    *, runner: _DockerRunner = _default_docker_runner
) -> QdrantContainerStatus:
    """Return whether the named container is running, stopped, or absent."""

    try:
        result = runner(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name=^{_QDRANT_CONTAINER_NAME}$",
                "--format",
                "{{.State}}",
            ]
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return QdrantContainerStatus(state="absent")
    if result.returncode != 0:
        return QdrantContainerStatus(state="absent")
    raw = (result.stdout or "").strip().lower()
    if raw == "running":
        return QdrantContainerStatus(state="running")
    if raw in {"exited", "created", "paused", "dead", "restarting"}:
        return QdrantContainerStatus(state="stopped")
    return QdrantContainerStatus(state="absent")


def stop_local_qdrant_container(
    *,
    say: Callable[[str], None] = print,
    runner: _DockerRunner = _default_docker_runner,
) -> str:
    """Stop the named local Qdrant container.

    Idempotent — returns the post-call state label even if the container
    was already stopped or absent.

    Returns:
        ``"stopped"``, ``"absent"``.

    Raises:
        DockerNotAvailable: ``docker`` CLI not installed or daemon down.
        QdrantStartFailed: ``docker stop`` returned non-zero.
    """

    if not docker_available(runner=runner):
        raise DockerNotAvailable(
            "the 'docker' CLI is not available on this host"
        )
    status = qdrant_container_state(runner=runner)
    if status.state == "absent":
        say(f"No '{_QDRANT_CONTAINER_NAME}' container found.")
        return "absent"
    if status.state == "stopped":
        say(f"Container '{_QDRANT_CONTAINER_NAME}' was already stopped.")
        return "stopped"
    say(f"Stopping container '{_QDRANT_CONTAINER_NAME}'...")
    result = runner(["docker", "stop", _QDRANT_CONTAINER_NAME])
    if result.returncode != 0:
        raise QdrantStartFailed(
            (result.stderr or result.stdout or "docker stop failed").strip()
        )
    return "stopped"


def remove_local_qdrant_container(
    *,
    say: Callable[[str], None] = print,
    runner: _DockerRunner = _default_docker_runner,
) -> str:
    """Stop (if running) then remove the named local Qdrant container.

    The persistent volume ``feather-qdrant-data`` is intentionally NOT
    removed — recreating the container later picks up the same indexes
    and points. Use ``docker volume rm feather-qdrant-data`` to wipe
    data; we keep that step manual since it is destructive.

    Returns:
        ``"removed"`` or ``"absent"`` when the container was not present.

    Raises:
        DockerNotAvailable: ``docker`` CLI not installed or daemon down.
        QdrantStartFailed: ``docker stop`` or ``docker rm`` returned non-zero.
    """

    if not docker_available(runner=runner):
        raise DockerNotAvailable(
            "the 'docker' CLI is not available on this host"
        )
    status = qdrant_container_state(runner=runner)
    if status.state == "absent":
        say(f"No '{_QDRANT_CONTAINER_NAME}' container found.")
        return "absent"
    if status.state == "running":
        say(f"Stopping container '{_QDRANT_CONTAINER_NAME}'...")
        result = runner(["docker", "stop", _QDRANT_CONTAINER_NAME])
        if result.returncode != 0:
            raise QdrantStartFailed(
                (result.stderr or result.stdout or "docker stop failed").strip()
            )
    say(f"Removing container '{_QDRANT_CONTAINER_NAME}'...")
    result = runner(["docker", "rm", _QDRANT_CONTAINER_NAME])
    if result.returncode != 0:
        raise QdrantStartFailed(
            (result.stderr or result.stdout or "docker rm failed").strip()
        )
    say(
        f"Container removed. Volume 'feather-qdrant-data' was kept; "
        f"run 'docker volume rm feather-qdrant-data' to wipe stored vectors."
    )
    return "removed"


def ensure_local_qdrant_container(
    *,
    say: Callable[[str], None] = print,
    runner: _DockerRunner = _default_docker_runner,
    ready_checker: _ReadyChecker = _default_ready_checker,
    ready_timeout_s: float = 60.0,
) -> str:
    """Start (or reuse) a local Qdrant Docker container.

    Args:
        say: Sink for human-readable status lines.
        runner: Subprocess runner. Tests inject a recording fake.
        ready_checker: Callable that polls the Qdrant ``/readyz`` URL.
        ready_timeout_s: Max wait for the container to accept requests.

    Returns:
        The local URL the wizard should record (``http://localhost:6333``).

    Raises:
        DockerNotAvailable: ``docker`` CLI not installed or daemon down.
        QdrantStartFailed: Container failed to start or never became ready.
    """

    if not docker_available(runner=runner):
        raise DockerNotAvailable(
            "the 'docker' CLI is not available on this host"
        )
    status = qdrant_container_state(runner=runner)
    if status.state == "running":
        say("Local Qdrant container is already running.")
    elif status.state == "stopped":
        say("Starting existing local Qdrant container...")
        result = runner(["docker", "start", _QDRANT_CONTAINER_NAME])
        if result.returncode != 0:
            raise QdrantStartFailed(
                (result.stderr or result.stdout or "docker start failed").strip()
            )
    else:
        say(
            "Pulling Qdrant image and starting container "
            "(may take a minute on first run)..."
        )
        result = runner(
            [
                "docker",
                "run",
                "-d",
                "--name",
                _QDRANT_CONTAINER_NAME,
                "-p",
                f"127.0.0.1:{_QDRANT_PORT}:{_QDRANT_PORT}",
                "-v",
                f"{_QDRANT_VOLUME}:/qdrant/storage",
                "--restart",
                "unless-stopped",
                _QDRANT_IMAGE,
            ]
        )
        if result.returncode != 0:
            raise QdrantStartFailed(
                (result.stderr or result.stdout or "docker run failed").strip()
            )

    if not ready_checker(ready_timeout_s):
        raise QdrantStartFailed(
            f"Qdrant did not respond at {_QDRANT_LOCAL_URL}/readyz "
            f"within {int(ready_timeout_s)}s"
        )
    say(f"Qdrant ready at {_QDRANT_LOCAL_URL}.")
    return _QDRANT_LOCAL_URL


@dataclass(slots=True)
class OnboardingAnswers:
    """Captured answers from the wizard."""

    name: str = ""
    preferred_name: str = ""
    role: str = ""
    expertise: str = ""
    about: str = ""
    openai_api_key: str = ""
    provider: str = "openai"
    openrouter_api_key: str = ""
    claude_api_key: str = ""
    memory_enabled: bool = False
    # "" when memory disabled; otherwise one of:
    # "local-docker" — wizard spun up a local Qdrant Docker container,
    # "local-existing" — user already has Qdrant on http://localhost:6333,
    # "cloud" — remote Qdrant URL pasted by the user.
    qdrant_mode: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    gemini_api_key: str = ""
    web_search_enabled: bool = False
    parallel_api_key: str = ""
    # Experimental: when True, the TUI runs the lead agent as a separate
    # background process so it can detect hangs and let the agent reload
    # its own patched code. Trade-off: cron and messaging integrations
    # are paused. ``FEATHER_USE_LEAD_WORKER=1`` overrides this at runtime.
    self_repair_enabled: bool = False

    def collect_secrets(self) -> dict[str, str]:
        """Return only the non-empty (KEY, value) pairs for ``.env``.

        Empty values are dropped so we never write ``KEY=`` lines that
        would later override a real value loaded by ``feather.env``.
        """

        candidates = {
            "OPENAI_API_KEY": self.openai_api_key,
            "OPEN_ROUTER_API_KEY": self.openrouter_api_key,
            "ANTHROPIC_API_KEY": self.claude_api_key,
            "QDRANT_URL": self.qdrant_url,
            "QDRANT_API_KEY": self.qdrant_api_key,
            "GEMINI_API_KEY": self.gemini_api_key,
            "PARALLEL_API_KEY": self.parallel_api_key,
        }
        return {key: value for key, value in candidates.items() if value}


def write_env_file(path: Path, secrets: dict[str, str]) -> list[str]:
    """Persist wizard answers to a ``.env`` file atomically.

    Wizard re-runs are authoritative: any key whose value is supplied in
    ``secrets`` overwrites the existing line in ``.env`` (whether that
    line was a ``KEY=`` placeholder or a stale ``KEY=oldvalue``). Keys
    not present in ``secrets`` are left untouched, so a partial wizard
    pass (e.g. user skips Parallel AI) does not erase unrelated keys.

    Empty ``secrets[key]`` values are dropped — the wizard already
    omits keys it didn't ask about, so an empty value here means the
    user explicitly skipped the prompt and we must not write
    ``KEY=`` (which would later override a real env value).

    Args:
        path: ``.env`` path.
        secrets: Mapping of KEY -> value. Empty values are skipped.

    Returns:
        Ordered list of keys that were written (overwritten or appended).
    """

    existing_text = path.read_text(encoding="utf-8") if path.exists() else ""
    keys_with_values, placeholder_keys = _classify_env_keys(existing_text)
    existing_keys = keys_with_values | placeholder_keys
    added: list[str] = []
    overwrites: dict[str, str] = {}
    new_lines: list[str] = []
    for key, value in secrets.items():
        if not value:
            continue
        if key in existing_keys:
            overwrites[key] = value
            added.append(key)
            continue
        new_lines.append(f"{key}={value}")
        added.append(key)
    if not added:
        if not path.exists():
            _atomic_write_text(path, "")
            _maybe_chmod(path, 0o600)
        return added
    base_text = (
        _rewrite_env_placeholders(existing_text, overwrites)
        if overwrites
        else existing_text
    )
    if base_text and not base_text.endswith("\n"):
        base_text += "\n"
    final = base_text + ("\n".join(new_lines) + "\n" if new_lines else "")
    _atomic_write_text(path, final)
    _maybe_chmod(path, 0o600)
    return added


def _classify_env_keys(text: str) -> tuple[set[str], set[str]]:
    """Split env-file keys into (has-non-empty-value, has-empty-placeholder).

    A key that appears both ways takes the non-empty value as authoritative.
    """

    with_value: set[str] = set()
    placeholder: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        if value.strip():
            with_value.add(key)
        else:
            placeholder.add(key)
    return with_value, placeholder - with_value


def _rewrite_env_placeholders(text: str, overwrites: dict[str, str]) -> str:
    """Replace ``KEY=`` lines (empty or stale) with ``KEY=value`` lines.

    Walks every line, and for any line whose key is in ``overwrites``,
    replaces it. Lines without an ``=`` (comments, blanks) are left
    intact. Used for wizard re-runs where the user's new answer must
    win over whatever the file previously held.
    """

    out_lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(raw)
            continue
        body = stripped[len("export ") :].lstrip() if stripped.startswith("export ") else stripped
        prefix = stripped[: len(stripped) - len(body)]
        key, _, _value = body.partition("=")
        key = key.strip()
        if key not in overwrites:
            out_lines.append(raw)
            continue
        out_lines.append(f"{prefix}{key}={overwrites[key]}")
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


_ACTIVE_PROVIDER_LINE_RE = re.compile(
    r"^(?P<indent>\s*)active_provider:\s*"
    r"(?P<value>[A-Za-z0-9_./-]+)?"
    r"(?P<trailing>\s*(?:#.*)?)$"
)
_ENABLED_LINE_RE = re.compile(
    r"^(?P<indent>\s+)enabled:\s*"
    r"(?P<value>true|false|yes|no|on|off)"
    r"(?P<trailing>\s*(?:#.*)?)$",
    re.IGNORECASE,
)


def apply_self_repair_toggle(path: Path, *, enabled: bool) -> bool:
    """Rewrite ``self_repair.enabled`` in ``app.yaml`` while preserving comments.

    Mirrors the conservative line-walker used for ``memory.enabled``: only
    a strict ``  enabled: <bool> [# comment]`` line under a top-level
    ``self_repair:`` key is rewritten. Returns ``True`` iff the value
    was changed (so callers can warn the operator when the file shape
    is unexpected).
    """

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    in_block = False
    awaiting_enabled = False
    changed = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not in_block:
            if stripped == "self_repair:":
                in_block = True
                awaiting_enabled = True
            continue
        if not awaiting_enabled:
            break
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            # Sibling top-level key — left the self_repair block without
            # finding the enabled line. Stop without rewriting.
            break
        match = _ENABLED_LINE_RE.match(line)
        if match:
            indent_str = match.group("indent")
            trailing = match.group("trailing") or ""
            new_value = "true" if enabled else "false"
            lines[index] = f"{indent_str}enabled: {new_value}{trailing}"
            changed = True
            awaiting_enabled = False
            break

    if changed:
        new_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        _atomic_write_text(path, new_text)
    return changed


def apply_app_yaml_toggles(
    path: Path, *, active_provider: str, memory_enabled: bool
) -> dict[str, bool]:
    """Rewrite ``active_provider`` and ``memory.enabled`` while preserving
    surrounding comments and skipping look-alike lines (e.g. ``enabled:
    true`` inside a block-scalar).

    The line-by-line walker stays conservative: it only rewrites a line
    that matches the strict ``key: scalar [# comment]`` shape. Anything
    else (multi-line scalars, mappings, custom values) is left intact and
    reported as not-rewritten so the caller can warn the operator.
    """

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    changed_provider = False
    changed_memory = False
    in_memory_block = False
    awaiting_memory_enabled = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not in_memory_block:
            if not changed_provider:
                provider_match = _ACTIVE_PROVIDER_LINE_RE.match(line)
                # A top-level ``active_provider:`` only — indent must be empty
                # so we don't mis-target a deeper key with the same name.
                if provider_match and provider_match.group("indent") == "":
                    trailing = provider_match.group("trailing") or ""
                    lines[index] = f"active_provider: {active_provider}{trailing}"
                    changed_provider = True
                    continue
            if stripped == "memory:":
                in_memory_block = True
                awaiting_memory_enabled = True
                continue
        else:
            if awaiting_memory_enabled and stripped and not stripped.startswith("#"):
                # Either we are still inside the ``memory:`` block (indented)
                # or a sibling top-level key has begun. The first non-comment
                # line tells us which.
                indent = len(line) - len(line.lstrip(" "))
                if indent == 0:
                    in_memory_block = False
                    awaiting_memory_enabled = False
                    # Recurse-once for top-level active_provider on this line.
                    if not changed_provider:
                        provider_match = _ACTIVE_PROVIDER_LINE_RE.match(line)
                        if provider_match and provider_match.group("indent") == "":
                            trailing = provider_match.group("trailing") or ""
                            lines[index] = (
                                f"active_provider: {active_provider}{trailing}"
                            )
                            changed_provider = True
                    continue
                if not changed_memory:
                    enabled_match = _ENABLED_LINE_RE.match(line)
                    if enabled_match:
                        indent_str = enabled_match.group("indent")
                        trailing = enabled_match.group("trailing") or ""
                        new_value = "true" if memory_enabled else "false"
                        lines[index] = f"{indent_str}enabled: {new_value}{trailing}"
                        changed_memory = True
                        awaiting_memory_enabled = False

    if changed_provider or changed_memory:
        new_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        _atomic_write_text(path, new_text)
    return {
        "active_provider": bool(changed_provider),
        "memory_enabled": bool(changed_memory),
    }


def is_onboarded(marker_path: Path) -> bool:
    """Return True when the completion marker exists on disk."""

    return marker_path.exists()


def mark_onboarded(
    marker_path: Path,
    *,
    openai_key_configured: bool,
    memory_enabled: bool,
    web_search_enabled: bool,
) -> None:
    """Write the completion marker; idempotent overwrite."""

    payload = {
        "version": 1,
        "completed_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "openai_key_configured": openai_key_configured,
        "memory_enabled": memory_enabled,
        "web_search_enabled": web_search_enabled,
    }
    _atomic_write_text(marker_path, json.dumps(payload, indent=2) + "\n")


@dataclass(slots=True)
class OnboardingWizard:
    """Interactive first-run wizard.

    The wizard accepts injected ``input_fn`` and ``output_fn`` callables
    so tests can drive the prompts without a real terminal. The default
    constructor uses :func:`input` / :func:`print` for live runs.
    """

    root: Path
    input_fn: object = field(default=input)
    output_fn: object = field(default=print)
    # Secret-input callable (used for API keys). Defaults to ``getpass``
    # so terminal echo is suppressed and the key never reaches scrollback
    # or shell history. Tests inject a recording fake.
    secret_input_fn: object = field(default=getpass.getpass)
    # Local Qdrant launcher. Defaults to the Docker-based implementation
    # in this module; tests inject a stub so the wizard never shells out.
    qdrant_launcher: Callable[[Callable[[str], None]], str] | None = None
    # Optional :class:`feather.paths.FeatherPaths` for the new layered
    # state model. When provided, the memory question is *not* asked
    # interactively — instead, the existence of the global memory marker
    # written by ``feather init-memory`` decides whether memory is on.
    # Legacy callers (notably the existing test fixtures) leave this as
    # ``None`` and get the original interactive flow.
    feather_paths: object = field(default=None)

    async def run(self) -> OnboardingAnswers:
        """Run the full wizard and persist the captured answers.

        Returns:
            The captured :class:`OnboardingAnswers` dataclass.
        """

        answers = OnboardingAnswers()
        self._say("\n=== Feather first-run setup ===")
        self._say(
            "We will collect a few facts about you and the API keys "
            "Feather needs. Everything is stored locally in "
            "`.feather/user.md` and `.env` — nothing leaves your machine.\n"
        )

        # Identity
        answers.name = self._ask_required("Your name")
        answers.preferred_name = self._ask("Preferred name (Enter to skip)")
        answers.role = self._ask("Your role / profession (Enter to skip)")
        answers.expertise = self._ask(
            "Topics or languages you're focused on (Enter to skip)"
        )
        answers.about = self._ask("Short bio (Enter to skip)")

        # Providers — Feather supports OpenAI Responses, OpenRouter Chat
        # Completions, and Anthropic Claude Messages. The wizard prompts
        # for each independently so a user can wire as many as they have
        # keys for; at least one must be enabled for the runtime to
        # start. When more than one is wired we ask which should be the
        # session-wide default (``active_provider`` in app.yaml); a single
        # wired provider becomes active automatically.
        self._say(
            "\nLLM provider keys:\n"
            "Feather can route through OpenAI, OpenRouter, and Anthropic "
            "Claude. Wire whichever you have keys for — at least one is "
            "required."
        )
        if self._ask_yes_no("Wire OpenAI?"):
            self._say("Get an OpenAI key at https://platform.openai.com/api-keys")
            answers.openai_api_key = self._reuse_or_ask_secret("OPENAI_API_KEY")
        if self._ask_yes_no("Wire OpenRouter?"):
            self._say("Get an OpenRouter key at https://openrouter.ai/keys")
            answers.openrouter_api_key = self._reuse_or_ask_secret(
                "OPEN_ROUTER_API_KEY"
            )
        if self._ask_yes_no("Wire Claude (Anthropic)?"):
            self._say("Get an Anthropic key at https://console.anthropic.com/")
            answers.claude_api_key = self._reuse_or_ask_secret("ANTHROPIC_API_KEY")

        wired = [
            name
            for name, key in (
                ("openai", answers.openai_api_key),
                ("openrouter", answers.openrouter_api_key),
                ("claude", answers.claude_api_key),
            )
            if key
        ]
        if not wired:
            self._say(
                "\nNo provider was wired. Feather needs at least one of "
                "OpenAI / OpenRouter / Claude to run. Aborting onboarding — "
                "re-run `feather onboard` once you have a key ready."
            )
            raise NoProviderConfigured(
                "onboarding aborted: at least one LLM provider must be wired"
            )
        if len(wired) == 1:
            answers.provider = wired[0]
        else:
            choices = " / ".join(wired)
            while True:
                pick = (
                    self._ask(
                        f"Default provider [{choices}] (default {wired[0]})"
                    )
                    or wired[0]
                ).strip().lower()
                if pick in wired:
                    answers.provider = pick
                    break
                self._say(f"Pick one of: {choices}.")

        # Memory — when paths is provided, the marker decides instead of
        # an interactive yes/no. This keeps the new flow honest:
        # `feather init-memory` is the explicit opt-in that switches
        # memory on, and re-running `feather onboard` after it never
        # asks for the Gemini key it would now need.
        marker_decided = self._memory_decided_by_marker()
        if marker_decided is not None:
            if marker_decided is False:
                self._say(
                    "Long-term memory is OFF (no memory marker found). "
                    "Run `feather init-memory` later to enable it."
                )
                answers.memory_enabled = False
            else:
                marker_url = self._memory_url_from_marker()
                answers.memory_enabled = True
                answers.qdrant_mode = "preconfigured"
                answers.qdrant_url = marker_url or _QDRANT_LOCAL_URL
                self._say(
                    f"Long-term memory is ON (using {answers.qdrant_url} from "
                    "the memory marker). You'll still need a Gemini key for "
                    "embeddings."
                )
                self._say("Gemini key: https://aistudio.google.com/apikey")
                answers.gemini_api_key = self._ask_secret_required("GEMINI_API_KEY")
        elif self._ask_yes_no("Enable long-term memory (Qdrant + Gemini)?"):
            answers.memory_enabled = True
            preconfigured_url = (
                os.environ.get("QDRANT_URL", "") or ""
            ).strip()
            if preconfigured_url:
                # Compose / explicit-env path: trust whatever started us.
                # Don't write QDRANT_URL or QDRANT_API_KEY back to .env —
                # the environment must remain authoritative across
                # restarts so compose-bound URLs survive.
                answers.qdrant_mode = "preconfigured"
                answers.qdrant_url = ""
                answers.qdrant_api_key = ""
                self._say(
                    f"Using QDRANT_URL={preconfigured_url} from the environment "
                    "(skipping deployment-choice prompt)."
                )
            else:
                self._say(
                    "\nQdrant deployment:\n"
                    "  [1] Local — start a Docker container automatically (recommended)\n"
                    "  [2] Local — Qdrant is already running on http://localhost:6333\n"
                    "  [3] Remote / cloud — paste the URL"
                )
                choice = (self._ask("Choose 1, 2, or 3 (default 1)") or "1").strip()
                if choice == "3":
                    answers.qdrant_mode = "cloud"
                    answers.qdrant_url = self._ask_qdrant_cloud_url()
                    answers.qdrant_api_key = self._ask_secret(
                        "QDRANT_API_KEY (Enter to skip if your cloud doesn't need one)"
                    )
                elif choice == "2":
                    answers.qdrant_mode = "local-existing"
                    answers.qdrant_url = _QDRANT_LOCAL_URL
                else:
                    # Default + "1": auto-spin a local Docker container.
                    answers.qdrant_mode = "local-docker"
                    launcher = self.qdrant_launcher or (
                        lambda say: ensure_local_qdrant_container(say=say)
                    )
                    try:
                        answers.qdrant_url = launcher(self._say)
                    except DockerNotAvailable:
                        self._say(
                            "Docker is not available on this host. Falling back to "
                            "manual mode — start Qdrant yourself before the next "
                            f"`feather tui` run. URL recorded as {_QDRANT_LOCAL_URL}."
                        )
                        answers.qdrant_mode = "local-existing"
                        answers.qdrant_url = _QDRANT_LOCAL_URL
                    except QdrantStartFailed as exc:
                        self._say(
                            f"Could not start the Qdrant container: {exc}\n"
                            f"Recording {_QDRANT_LOCAL_URL} as the URL — start it "
                            "manually before the next `feather tui` run."
                        )
                        answers.qdrant_mode = "local-existing"
                        answers.qdrant_url = _QDRANT_LOCAL_URL
            self._say("Gemini key: https://aistudio.google.com/apikey")
            answers.gemini_api_key = self._reuse_or_ask_secret("GEMINI_API_KEY")

        # Web search
        if self._ask_yes_no("Enable web search via Parallel AI?"):
            answers.web_search_enabled = True
            self._say("Parallel AI: https://parallel.ai/")
            answers.parallel_api_key = self._reuse_or_ask_secret("PARALLEL_API_KEY")

        # Self-repair safety net (experimental, advanced)
        self._say(
            "\nSelf-repair safety net (experimental, advanced)\n"
            "  When ON, Feather runs your agent in a separate background\n"
            "  process so it can:\n"
            "    • detect when the agent stops responding and offer a\n"
            "      recovery action,\n"
            "    • let the agent fix bugs in its own code and reload\n"
            "      itself, so a session-breaking issue doesn't cost you\n"
            "      the whole chat.\n"
            "  Trade-off: in this mode, scheduled reminders and messaging\n"
            "  integrations (Telegram, LINE, WhatsApp) are paused — they\n"
            "  share state with the agent and would conflict.\n"
            "  Recommended: leave OFF unless you know you want it.\n"
            "  You can change this later in app.yaml."
        )
        answers.self_repair_enabled = self._ask_yes_no(
            "Enable experimental self-repair safety net?"
        )

        await self._persist(answers)
        targets = self._resolve_persist_targets()
        self._say(
            f"\nOnboarding complete. Profile: {targets['profile']}  Keys: {targets['env']}\n"
        )
        return answers

    async def _persist(self, answers: OnboardingAnswers) -> None:
        """Write all four artifacts.

        Write targets are global (``~/.feather/...``) when the wizard
        was constructed with ``feather_paths``; otherwise they fall back
        to project-scoped paths under ``self.root`` for back-compat with
        existing tests + flows that haven't migrated yet.

        The completion marker is written last on purpose: if any earlier
        step crashes, the marker is absent and the next ``feather`` run
        re-prompts instead of treating the partial state as success.
        """

        from feather.profile import UserProfileStore

        targets = self._resolve_persist_targets()

        write_env_file(targets["env"], answers.collect_secrets())

        store = UserProfileStore(targets["profile"])
        existing_fields = set(store.load().fields.keys())
        identity_fields = {
            "name": answers.name,
            "preferred_name": answers.preferred_name,
            "role": answers.role,
            "expertise": answers.expertise,
        }
        for key, value in identity_fields.items():
            if not value:
                continue
            if key in existing_fields:
                await store.update(key, value)
            else:
                await store.create(key, value)
        if answers.about:
            await store.append_note(answers.about)

        yaml_path = targets["app_yaml"]
        # Materialize the packaged default at the global path on first
        # run so the regex toggles below have a file to rewrite. This
        # only kicks in for the new (paths-aware) flow — legacy callers
        # already had a project app.yaml staged or they didn't, and the
        # original behavior of silently skipping the toggle is preserved.
        if self.feather_paths is not None and not yaml_path.exists():
            from feather.resources import packaged_app_yaml_text

            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_path.write_text(packaged_app_yaml_text(), encoding="utf-8")
        if yaml_path.exists():
            applied = apply_app_yaml_toggles(
                yaml_path,
                active_provider=answers.provider,
                memory_enabled=answers.memory_enabled,
            )
            if not applied["active_provider"] or not applied["memory_enabled"]:
                logger.warning(
                    "app.yaml regex toggle missed; user must flip "
                    "active_provider/memory.enabled manually."
                )
            self_repair_changed = apply_self_repair_toggle(
                yaml_path, enabled=answers.self_repair_enabled
            )
            # Only WARN if the user actually opted IN and we couldn't
            # write it — leaving the default false in a default file is
            # not interesting.
            if answers.self_repair_enabled and not self_repair_changed:
                logger.warning(
                    "app.yaml self_repair toggle missed; user must flip "
                    "self_repair.enabled to true manually."
                )

        marker = targets["marker"]
        marker.parent.mkdir(parents=True, exist_ok=True)
        mark_onboarded(
            marker,
            openai_key_configured=bool(answers.openai_api_key),
            memory_enabled=answers.memory_enabled,
            web_search_enabled=answers.web_search_enabled,
        )

    def _resolve_persist_targets(self) -> dict:
        """Pick global vs project paths for the four wizard artifacts."""

        if self.feather_paths is not None:
            paths = self.feather_paths
            paths.ensure_global_dirs()  # type: ignore[attr-defined]
            return {
                "env": paths.env_file,  # type: ignore[attr-defined]
                "profile": paths.global_user_md,  # type: ignore[attr-defined]
                "app_yaml": paths.global_config_dir / "app.yaml",  # type: ignore[attr-defined]
                "marker": paths.onboarded_marker,  # type: ignore[attr-defined]
            }
        return {
            "env": self.root / ".env",
            "profile": self.root / ".feather" / "user.md",
            "app_yaml": self.root / "config" / "app.yaml",
            "marker": self.root / ".feather" / "onboarded.json",
        }

    # -- prompt helpers --------------------------------------------------

    def _say(self, message: str) -> None:
        self.output_fn(message)  # type: ignore[misc]

    def _ask(self, prompt: str) -> str:
        return str(self.input_fn(f"{prompt}: ")).strip()  # type: ignore[misc]

    def _ask_required(self, prompt: str) -> str:
        while True:
            answer = self._ask(prompt)
            if answer:
                return answer
            self._say("This field is required.")

    def _ask_secret(self, prompt: str) -> str:
        """Read a secret without echoing it to the terminal.

        After capture we print a short masked confirmation
        (``abc******xyz``) so the user can verify they pasted
        *something* without exposing the full key to scrollback.
        """

        answer = str(self.secret_input_fn(f"{prompt}: ")).strip()  # type: ignore[misc]
        if answer:
            self._say(f"  captured: {_mask_secret(answer)}")
        return answer

    def _reuse_or_ask_secret(self, env_var: str) -> str:
        """Reuse a secret already in ``os.environ``, otherwise prompt.

        Returning the existing value (without re-prompting) means
        re-onboarding doesn't ask the user to paste keys they already
        configured. We still echo a masked confirmation so the user
        can see what we picked up. Wizard-supplied values get persisted
        unchanged via :func:`write_env_file`.
        """
        existing = (os.environ.get(env_var) or "").strip()
        if existing:
            self._say(
                f"  reusing {env_var} from environment "
                f"({_mask_secret(existing)})"
            )
            return existing
        return self._ask_secret_required(env_var)

    def _ask_secret_required(self, prompt: str) -> str:
        while True:
            answer = self._ask_secret(prompt)
            if answer:
                return answer
            self._say("This field is required.")

    def _ask_yes_no(self, prompt: str) -> bool:
        answer = self._ask(f"{prompt} [y/N]").lower()
        return answer in {"y", "yes"}

    def _memory_decided_by_marker(self) -> "bool | None":
        """Return marker-driven memory state, or None to fall back to a question.

        Returns:
            ``True`` if the global marker exists (memory is on),
            ``False`` if ``feather_paths`` is set but no marker exists
            (memory is off), or ``None`` to indicate that the legacy
            interactive flow should run.
        """

        if self.feather_paths is None:
            return None
        from feather.cli_commands import memory_enabled_via_marker

        return memory_enabled_via_marker(self.feather_paths)

    def _memory_url_from_marker(self) -> "str | None":
        if self.feather_paths is None:
            return None
        from feather.cli_commands import memory_url_from_marker

        return memory_url_from_marker(self.feather_paths)

    def _ask_qdrant_cloud_url(self) -> str:
        """Prompt for a remote Qdrant URL with scheme + reachability checks.

        Re-prompts on missing scheme. On reachability failure, surfaces
        the most common cause (wrong scheme — Qdrant rarely speaks TLS
        without explicit setup) and asks whether to use the URL anyway,
        so a user with a not-yet-running endpoint can still proceed.
        """

        while True:
            raw = self._ask("QDRANT_URL")
            url = (raw or _QDRANT_LOCAL_URL).strip()
            if not (url.startswith("http://") or url.startswith("https://")):
                self._say(
                    f"URL must start with http:// or https:// (got {url!r})."
                )
                continue
            self._say(f"Probing {url}/readyz ...")
            if _probe_qdrant_url(url):
                self._say("Reachable.")
                return url
            self._say(
                f"Could not reach {url}/readyz. Common causes:\n"
                "  - wrong scheme — Qdrant typically speaks http:// unless "
                "you've explicitly set up TLS\n"
                "  - service not running yet\n"
                "  - network / firewall blocking the connection"
            )
            if self._ask_yes_no("Use this URL anyway?"):
                return url
            # else loop back and re-prompt


async def maybe_run_onboarding(
    root: Path,
    *,
    wizard_factory=None,
    force: bool = False,
    skip: bool = False,
    paths: object = None,
) -> OnboardingAnswers | None:
    """Decide whether to launch the wizard and do so if needed.

    Args:
        root: Repository root. ``.env``, ``.feather/`` live underneath.
        wizard_factory: Callable returning a wizard. Defaults to
            :class:`OnboardingWizard`. Tests inject stubs.
        force: Ignore the completion marker.
        skip: Hard skip; never run.

    Returns:
        :class:`OnboardingAnswers` if the wizard ran, otherwise ``None``.
    """

    if skip:
        return None
    factory = wizard_factory or OnboardingWizard
    # Marker location follows the new global model when ``paths`` is
    # provided; otherwise use the legacy project-scoped path so existing
    # tests + flows behave unchanged.
    if paths is not None:
        marker = paths.onboarded_marker  # type: ignore[attr-defined]
        profile_required = False
    else:
        marker = root / ".feather" / "onboarded.json"
        profile_required = True
    profile = root / ".feather" / "user.md"
    if not force and is_onboarded(marker) and (
        not profile_required or profile.exists()
    ):
        return None
    if paths is not None:
        wizard = factory(root=root, feather_paths=paths)
    else:
        wizard = factory(root=root)
    return await wizard.run()


# -- internal helpers -----------------------------------------------------


def _mask_secret(value: str) -> str:
    """Return a partial-mask view of a secret for terminal confirmation.

    Long values render as ``<first3>******<last3>`` so the user can confirm
    a paste worked and the key family is correct (``sk-…``, ``gem-…``)
    without exposing the body. Short values redact entirely so we never
    leak more than half of a credential.
    """

    if len(value) < 8:
        return "***"
    return f"{value[:3]}******{value[-3:]}"


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via tempfile + rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _maybe_chmod(path: Path, mode: int) -> None:
    """Best-effort ``chmod``; swallow on filesystems that don't support it."""

    try:
        os.chmod(path, mode)
    except OSError:
        pass
