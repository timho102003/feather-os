"""Minimal `.env` loading for local development."""

from __future__ import annotations

import os
import shlex
from pathlib import Path


def load_dotenv(path: Path, *, override: bool = False) -> dict[str, str]:
    """Load environment variables from a `.env` file.

    Args:
        path: Path to the dotenv file.
        override: Whether file values should overwrite existing environment values.

    Returns:
        Mapping of variables loaded into the process environment.
    """

    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, raw_value = line.split("=", maxsplit=1)
        key = key.strip()
        value = _parse_value(raw_value.strip())
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


def _parse_value(value: str) -> str:
    """Parse one dotenv value.

    Args:
        value: Raw string after the first `=`.

    Returns:
        Parsed environment value.
    """

    if not value:
        return ""
    if value[0] in {'"', "'"}:
        parsed = shlex.split(value, posix=True)
        return parsed[0] if parsed else ""
    if " #" in value:
        value = value.split(" #", maxsplit=1)[0]
    return value.strip()
