"""Implementation of the new top-level ``feather`` subcommands.

Lives outside :mod:`feather.cli` so the wiring stays small and so unit
tests can call individual handlers without going through ``argparse``.
Each public function takes a :class:`feather.paths.FeatherPaths` and
returns a Unix-style exit code (``0`` on success, non-zero on failure).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from feather.onboarding import (
    DockerNotAvailable,
    QdrantStartFailed,
    _QDRANT_CONTAINER_NAME,
    _QDRANT_IMAGE,
    _QDRANT_LOCAL_URL,
    _QDRANT_VOLUME,
    docker_available,
    ensure_local_qdrant_container,
    qdrant_container_state,
    remove_local_qdrant_container,
    stop_local_qdrant_container,
)
from feather.paths import FeatherPaths

try:
    from feather import __version__
except Exception:  # pragma: no cover — only fires before packaging
    __version__ = "0.0.0+unknown"


_MARKER_VERSION = 1


# ---------------------------------------------------------------------------
# init — create a project-scoped .feather/ in the current directory
# ---------------------------------------------------------------------------


def init_project(paths: FeatherPaths, *, say: Callable[[str], None] = print) -> int:
    """Create ``./.feather/`` in the working directory and register it.

    Idempotent — re-running on an already-initialized project is a no-op
    that just refreshes the projects index.

    Args:
        paths: Resolved paths. Must already be in project mode (the
            caller decides what "project root" means via
            :meth:`FeatherPaths.for_project`).
        say: User-facing print function (replaceable for tests).

    Returns:
        Exit code: ``0`` on success.
    """

    if not paths.is_project_mode:
        say("feather init: no project root resolved (this should not happen).")
        return 2

    paths.ensure_global_dirs()
    paths.ensure_project_dirs()
    _register_project(paths)
    say(f"Initialized feather project at {paths.project_feather_dir}")
    return 0


def _register_project(paths: FeatherPaths) -> None:
    """Add ``paths.project_root`` to the global projects index."""

    index = paths.projects_index
    existing: list[str] = []
    if index.exists():
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("projects"), list):
                existing = [str(p) for p in data["projects"]]
        except (json.JSONDecodeError, OSError):
            existing = []
    project = str(paths.project_root)
    if project not in existing:
        existing.append(project)
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        json.dumps(
            {"projects": sorted(set(existing))}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# init-memory — start qdrant + write memory marker
# ---------------------------------------------------------------------------


def init_memory(
    paths: FeatherPaths,
    *,
    say: Callable[[str], None] = print,
) -> int:
    """Start a local Qdrant container and record the marker.

    Idempotent: if the marker already exists and the container is
    healthy, the call is a no-op that prints the current URL.

    Args:
        paths: Resolved paths (project optional; this command only
            touches global state).
        say: User-facing print function.

    Returns:
        Exit code: ``0`` on success, ``1`` on a Docker / startup failure.
    """

    paths.ensure_global_dirs()
    if paths.memory_marker.exists() and qdrant_container_state().state == "running":
        marker = _read_marker(paths)
        url = (marker or {}).get("url", _QDRANT_LOCAL_URL)
        say(f"Memory already running at {url}")
        return 0

    if not docker_available():
        say(
            "Docker is not installed or the daemon is not reachable. "
            "Install Docker (or start Docker Desktop) and try again."
        )
        return 1

    try:
        url = ensure_local_qdrant_container(say=say)
    except DockerNotAvailable as exc:
        say(f"Docker unavailable: {exc}")
        return 1
    except QdrantStartFailed as exc:
        say(f"Qdrant failed to start: {exc}")
        return 1

    _write_marker(
        paths,
        {
            "version": _MARKER_VERSION,
            "url": url,
            "mode": "local-docker",
            "container_name": _QDRANT_CONTAINER_NAME,
            "image": _QDRANT_IMAGE,
            "volume": _QDRANT_VOLUME,
            "started_at": _utcnow_iso(),
            "feather_version": __version__,
        },
    )
    say(f"Memory ready at {url}")
    say(f"Marker: {paths.memory_marker}")
    return 0


# ---------------------------------------------------------------------------
# stop-memory / remove-memory
# ---------------------------------------------------------------------------


def stop_memory(
    paths: FeatherPaths,
    *,
    say: Callable[[str], None] = print,
) -> int:
    """Stop the local Qdrant container; marker is preserved."""

    if not docker_available():
        say("Docker is not installed or the daemon is not reachable.")
        return 1
    state = stop_local_qdrant_container(say=say)
    say(f"Memory container: {state}")
    if paths.memory_marker.exists():
        say(f"Marker preserved at {paths.memory_marker}")
    return 0


def remove_memory(
    paths: FeatherPaths,
    *,
    purge: bool = False,
    say: Callable[[str], None] = print,
) -> int:
    """Stop + remove the container; delete the marker.

    Args:
        paths: Resolved paths.
        purge: When ``True``, also delete the named docker volume so
            stored vectors are gone for good.
        say: User-facing print function.
    """

    if not docker_available():
        say("Docker is not installed or the daemon is not reachable.")
        return 1
    state = remove_local_qdrant_container(say=say)
    say(f"Memory container: {state}")
    if paths.memory_marker.exists():
        paths.memory_marker.unlink()
        say(f"Marker removed: {paths.memory_marker}")
    if purge:
        # Best-effort volume removal; subprocess errors get reported but
        # don't fail the command since the container is already gone.
        from subprocess import CompletedProcess, run

        result: CompletedProcess = run(
            ["docker", "volume", "rm", _QDRANT_VOLUME],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            say(f"Volume purged: {_QDRANT_VOLUME}")
        else:
            stderr = (result.stderr or "").strip()
            say(f"Volume purge skipped ({stderr or 'unknown error'})")
    return 0


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------


def _read_marker(paths: FeatherPaths) -> dict[str, Any] | None:
    """Read and JSON-parse the memory marker, or return ``None`` if absent."""
    if not paths.memory_marker.exists():
        return None
    try:
        return json.loads(paths.memory_marker.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_marker(paths: FeatherPaths, payload: dict[str, Any]) -> None:
    """Atomically write the memory marker JSON."""
    paths.memory_marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.memory_marker.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(paths.memory_marker)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def memory_enabled_via_marker(paths: FeatherPaths) -> bool:
    """True when a memory marker exists on disk.

    The wizard consults this to decide whether to ask memory questions.
    """
    return paths.memory_marker.is_file()


def memory_url_from_marker(paths: FeatherPaths) -> str | None:
    """Return the URL recorded in the marker, if any."""
    marker = _read_marker(paths)
    if not marker:
        return None
    url = marker.get("url")
    return str(url) if url else None
