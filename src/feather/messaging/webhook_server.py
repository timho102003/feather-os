"""Shared aiohttp webhook server for LINE + WhatsApp adapters.

The server is reference-counted: the first adapter to register a route
starts the listener; the last adapter to unregister stops it. This
keeps the network exposure scoped to "as long as someone is actually
listening". The default bind is ``127.0.0.1:8765`` so users must
explicitly tunnel (ngrok / cloudflared) before public-internet platform
events can reach it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from aiohttp import web

logger = logging.getLogger(__name__)


WebhookHandler = Callable[[web.Request], Awaitable[web.StreamResponse]]


class WebhookServer:
    """A minimal aiohttp server with dynamically registered handlers.

    Args:
        host: Bind interface. Defaults to ``127.0.0.1`` so the server
            is not reachable from anywhere off-host without an explicit
            tunnel.
        port: TCP port. Defaults to ``8765``.
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._routes: dict[tuple[str, str], WebhookHandler] = {}
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        """Return the local URL clients should tunnel to."""

        return f"http://{self._host}:{self._port}"

    @property
    def is_running(self) -> bool:
        """Return True when the listener is bound."""

        return self._site is not None

    async def register(
        self,
        method: str,
        path: str,
        handler: WebhookHandler,
    ) -> None:
        """Register a webhook handler. Starts the server on first call."""

        method_upper = method.upper()
        async with self._lock:
            self._routes[(method_upper, path)] = handler
            self._app.router.add_route(method_upper, path, handler)
            if self._runner is None:
                self._runner = web.AppRunner(self._app)
                await self._runner.setup()
                self._site = web.TCPSite(self._runner, self._host, self._port)
                await self._site.start()
                logger.info(
                    "messaging.webhook.started host=%s port=%s",
                    self._host,
                    self._port,
                )

    async def unregister(self, method: str, path: str) -> None:
        """Remove a webhook handler. Stops the server when empty."""

        method_upper = method.upper()
        async with self._lock:
            if (method_upper, path) not in self._routes:
                return
            del self._routes[(method_upper, path)]
            # aiohttp doesn't expose route removal cleanly, so when the
            # adapter set changes we rebuild the router from scratch.
            new_app = web.Application()
            for (m, p), h in self._routes.items():
                new_app.router.add_route(m, p, h)
            self._app = new_app
            if self._routes:
                # Rebind: stop old runner, start a new one with the
                # rebuilt app, on the same port. Brief drop in
                # availability — acceptable since adapters disconnect
                # rarely.
                await self._stop_locked()
                self._runner = web.AppRunner(self._app)
                await self._runner.setup()
                self._site = web.TCPSite(self._runner, self._host, self._port)
                await self._site.start()
                return
            await self._stop_locked()

    async def stop(self) -> None:
        """Force-stop the server regardless of registered routes."""

        async with self._lock:
            self._routes.clear()
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        """Inner helper that assumes the lock is already held."""

        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:  # noqa: BLE001
                logger.exception("messaging.webhook.site_stop_failed")
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:  # noqa: BLE001
                logger.exception("messaging.webhook.runner_cleanup_failed")
            self._runner = None


__all__ = ("WebhookHandler", "WebhookServer")
