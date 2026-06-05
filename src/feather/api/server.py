"""FastAPI app factory for the Feather parity layer."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from feather.api.hub import ApiHub
from feather.api.routes import router

__all__ = ("create_app",)

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    root: Path,
    *,
    provider_factory: Callable[[Any], Any] | None = None,
) -> FastAPI:
    """Build the Feather API app rooted at ``root``.

    ``provider_factory`` lets tests inject a fake LLM provider so the server
    runs without real API keys.

    Security note: this layer is **unauthenticated** and intended to be bound
    to localhost for a single operator (it mirrors the local TUI). It can
    create leads (writing project YAML) and read session transcripts, so do
    not expose it on an untrusted network without adding an auth layer.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.hub = await ApiHub.create(root, provider_factory=provider_factory)
        try:
            yield
        finally:
            await app.state.hub.close()

    app = FastAPI(title="Feather API", version="1", lifespan=lifespan)
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    return app
