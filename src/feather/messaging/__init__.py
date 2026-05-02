"""Messaging integrations for Feather (Telegram, LINE, WhatsApp).

Public API:
- :class:`feather.messaging.models.Platform`
- :class:`feather.messaging.models.IncomingMessage`
- :class:`feather.messaging.models.OutgoingMessage`
- :class:`feather.messaging.models.AdapterStatus`
- :class:`feather.messaging.base.BaseMessagingAdapter`
- :class:`feather.messaging.store.MessagingStore`
- :class:`feather.messaging.router.MessagingRouter`
- :class:`feather.messaging.service.MessagingService`
"""

from feather.messaging.base import BaseMessagingAdapter
from feather.messaging.models import (
    AdapterStatus,
    IncomingMessage,
    OutgoingMessage,
    Platform,
)
from feather.messaging.router import MessagingRouter
from feather.messaging.service import MessagingService
from feather.messaging.store import MessagingStore

__all__ = (
    "AdapterStatus",
    "BaseMessagingAdapter",
    "IncomingMessage",
    "MessagingRouter",
    "MessagingService",
    "MessagingStore",
    "OutgoingMessage",
    "Platform",
)
