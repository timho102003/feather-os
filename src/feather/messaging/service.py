"""Lifecycle manager that owns the running messaging adapters.

The :class:`MessagingService` is the only thing the rest of the runtime
talks to: slash commands ask it to ``connect_telegram``,
``disconnect_line``, etc.; it instantiates the right
:class:`BaseMessagingAdapter`, persists credentials, and starts/stops
adapters. It also owns the shared :class:`WebhookServer` that LINE and
WhatsApp share.

On runtime startup the service auto-restores any previously-connected
adapters from ``messaging_credentials`` so a user only needs to type
``/telegram connect …`` once.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import httpx

from feather.messaging.base import BaseMessagingAdapter
from feather.messaging.models import (
    AdapterState,
    AdapterStatus,
    Platform,
)
from feather.messaging.router import MessagingRouter
from feather.messaging.store import MessagingStore
from feather.messaging.webhook_server import WebhookServer

logger = logging.getLogger(__name__)


_AdapterFactory = Callable[
    ["MessagingService", dict[str, Any]],
    BaseMessagingAdapter,
]


class MessagingService:
    """Coordinate adapter lifecycle for the lead-agent runtime.

    Args:
        store: Persistent credential / chat-mapping store.
        router: Router that converts inbound messages into agent runs.
        webhook_server: Shared webhook listener used by LINE +
            WhatsApp.
        http_client: Shared async HTTP client used by adapters for all
            outbound API calls. The service does not own its lifecycle
            unless ``own_http_client`` is True (used in standalone
            tests).
        adapter_factories: Mapping from platform to a callable that
            builds the adapter for that platform. Tests inject stubs;
            production wiring uses the platform-specific defaults.
        own_http_client: When True, ``shutdown`` will close
            ``http_client`` too.
    """

    def __init__(
        self,
        *,
        store: MessagingStore,
        router: MessagingRouter,
        webhook_server: WebhookServer,
        http_client: httpx.AsyncClient,
        adapter_factories: dict[Platform, _AdapterFactory],
        own_http_client: bool = False,
    ) -> None:
        self._store = store
        self._router = router
        self._webhook_server = webhook_server
        self._http_client = http_client
        self._adapter_factories = adapter_factories
        self._adapters: dict[Platform, BaseMessagingAdapter] = {}
        self._lock = asyncio.Lock()
        self._own_http_client = own_http_client
        self._started = False

    # ---- Accessors ------------------------------------------------------

    @property
    def store(self) -> MessagingStore:
        """Return the underlying messaging store."""

        return self._store

    @property
    def router(self) -> MessagingRouter:
        """Return the shared router."""

        return self._router

    @property
    def webhook_server(self) -> WebhookServer:
        """Return the shared webhook server."""

        return self._webhook_server

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Return the shared async HTTP client."""

        return self._http_client

    def adapter(self, platform: Platform) -> BaseMessagingAdapter | None:
        """Return the running adapter for ``platform``, if any."""

        return self._adapters.get(platform)

    async def status(self) -> dict[Platform, AdapterStatus]:
        """Return the current status of every supported platform.

        Disconnected platforms are reported with ``DISCONNECTED`` state
        so ``/integrations`` can show the full picture.
        """

        out: dict[Platform, AdapterStatus] = {}
        for platform in self._adapter_factories.keys():
            adapter = self._adapters.get(platform)
            if adapter is None:
                out[platform] = AdapterStatus(
                    platform=platform,
                    state=AdapterState.DISCONNECTED,
                    detail="not connected",
                    connected_chat_count=await self._store.count_chats_for_platform(
                        platform
                    ),
                )
            else:
                out[platform] = adapter.status
        return out

    # ---- Lifecycle ------------------------------------------------------

    async def start(self) -> None:
        """Restore previously-connected adapters from the store."""

        if self._started:
            return
        self._started = True
        records = await self._store.list_credentials()
        for record in records:
            if not record.enabled:
                continue
            try:
                await self._spawn_adapter(record.platform, record.config)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "messaging.service.restore_failed platform=%s",
                    record.platform.value,
                )

    async def shutdown(self) -> None:
        """Stop every running adapter and the shared webhook server."""

        async with self._lock:
            for adapter in list(self._adapters.values()):
                try:
                    await adapter.stop()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "messaging.service.stop_failed platform=%s",
                        adapter.platform.value,
                    )
                self._router.unregister_adapter(adapter.platform)
            self._adapters.clear()
            await self._webhook_server.stop()
            if self._own_http_client:
                try:
                    await self._http_client.aclose()
                except Exception:  # noqa: BLE001
                    logger.exception("messaging.service.http_close_failed")

    # ---- Connect / disconnect ------------------------------------------

    async def connect(
        self, platform: Platform, config: dict[str, Any]
    ) -> AdapterStatus:
        """Persist credentials and bring the adapter online.

        Args:
            platform: Which adapter to start.
            config: Platform-specific configuration. The factory
                validates required keys and raises
                ``ValueError`` on missing fields; the slash command
                surfaces the exception directly.

        Returns:
            The post-start status snapshot.
        """

        async with self._lock:
            existing = self._adapters.pop(platform, None)
            if existing is not None:
                try:
                    await existing.stop()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "messaging.service.preempt_stop_failed platform=%s",
                        platform.value,
                    )
                self._router.unregister_adapter(platform)

            adapter = await self._spawn_adapter(platform, config)
            await self._store.save_credentials(platform, config)
            return adapter.status

    async def disconnect(self, platform: Platform) -> None:
        """Stop the adapter (if any) and forget its credentials."""

        async with self._lock:
            adapter = self._adapters.pop(platform, None)
            if adapter is not None:
                try:
                    await adapter.stop()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "messaging.service.disconnect_stop_failed platform=%s",
                        platform.value,
                    )
                self._router.unregister_adapter(platform)
            await self._store.delete_credentials(platform)

    # ---- Internal -------------------------------------------------------

    async def _spawn_adapter(
        self, platform: Platform, config: dict[str, Any]
    ) -> BaseMessagingAdapter:
        factory = self._adapter_factories.get(platform)
        if factory is None:
            raise ValueError(f"no factory registered for {platform.value}")
        adapter = factory(self, config)
        await adapter.start()
        self._adapters[platform] = adapter
        self._router.register_adapter(platform, adapter.send_outgoing)
        return adapter


__all__ = ("MessagingService",)
