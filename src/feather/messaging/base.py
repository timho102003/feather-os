"""Base messaging adapter contract.

Each platform's adapter is a self-contained subclass:
- :meth:`start` brings the adapter online (long-polling task, webhook
  registration, …).
- :meth:`stop` cleans up.
- :meth:`send_outgoing` delivers an :class:`OutgoingMessage` to the
  platform.

All adapters share a single ``MessagingRouter`` reference; the router
owns the agent-side dispatch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from feather.messaging.models import (
    AdapterState,
    AdapterStatus,
    OutgoingMessage,
    Platform,
)

if TYPE_CHECKING:
    from feather.messaging.router import MessagingRouter


class BaseMessagingAdapter(ABC):
    """Abstract messaging adapter shared by Telegram / LINE / WhatsApp.

    Concrete adapters are constructed with their platform-specific
    config plus a router reference. Lifecycle is driven by
    ``MessagingService``; concrete adapters do not call
    ``router.handle_incoming`` from any thread other than the running
    event loop.

    Subclasses MUST set :attr:`platform` to identify themselves.
    """

    platform: Platform

    def __init__(self, router: "MessagingRouter") -> None:
        """Initialize the adapter.

        Args:
            router: Shared router that turns incoming messages into
                agent runs and routes replies back through
                :meth:`send_outgoing`.
        """

        self._router = router
        self._status = AdapterStatus(platform=self.platform)

    @property
    def status(self) -> AdapterStatus:
        """Return the latest known status of this adapter."""

        return self._status

    def _set_state(
        self,
        state: AdapterState,
        *,
        detail: str = "",
        last_error: str | None = None,
    ) -> None:
        """Update the cached status (called by subclasses)."""

        self._status = AdapterStatus(
            platform=self.platform,
            state=state,
            detail=detail,
            last_error=last_error,
            connected_chat_count=self._status.connected_chat_count,
        )

    def _set_chat_count(self, count: int) -> None:
        """Update the connected-chat counter on the cached status."""

        self._status = AdapterStatus(
            platform=self._status.platform,
            state=self._status.state,
            detail=self._status.detail,
            last_error=self._status.last_error,
            connected_chat_count=max(0, count),
        )

    @abstractmethod
    async def start(self) -> None:
        """Bring the adapter online.

        Implementations must transition status through
        ``STARTING → RUNNING`` (or ``ERROR``) and raise on fatal config
        errors so the slash command can show the user the problem
        without persisting bad credentials.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Tear the adapter down. Idempotent."""

    @abstractmethod
    async def send_outgoing(self, outgoing: OutgoingMessage) -> None:
        """Deliver a reply to the platform.

        Implementations should chunk text that exceeds platform limits
        rather than failing.
        """


__all__ = ("BaseMessagingAdapter",)
