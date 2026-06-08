"""Default entrypoint for ``fastapi run`` / ``uvicorn feather.api.main:app``.

Roots the server at the current working directory. For programmatic use (and
tests), call :func:`feather.api.server.create_app` directly with an explicit
root and optional provider factory.
"""

from __future__ import annotations

from pathlib import Path

from feather.api.server import create_app

app = create_app(Path.cwd())
