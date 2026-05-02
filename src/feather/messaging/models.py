"""Canonical data shapes shared across all messaging adapters.

Adapters convert their platform-native event payloads into
:class:`IncomingMessage` instances; the router converts agent replies
into :class:`OutgoingMessage` instances. Keeping the canonical shape
small forces adapters to do the platform-specific parsing once, at the
edge, instead of leaking platform fields through the rest of the stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Platform(str, Enum):
    """Supported messaging platforms."""

    TELEGRAM = "telegram"
    LINE = "line"
    WHATSAPP = "whatsapp"


@dataclass(slots=True, frozen=True)
class IncomingMessage:
    """A message delivered into Feather from an external chat platform.

    Attributes:
        platform: Source platform.
        chat_id: Platform-specific chat identifier (Telegram chat id, LINE
            user id, WhatsApp wa_id). Used as the partition key for
            ``messaging_chats``.
        sender_display_name: Best-effort human-readable sender label.
            Falls back to the chat id when the platform does not include
            a name.
        text: UTF-8 message body. Empty when the platform sent something
            non-textual (e.g. sticker); adapters drop these before they
            reach the router.
        native_message_id: Platform's own unique id for this delivery.
            Used for inbound dedup.
        timestamp_ms: Unix epoch milliseconds when the platform observed
            the message. ``None`` when the platform omitted it.
        reply_context: Opaque platform-specific token an adapter may need
            to reply (e.g. LINE replyToken). Routed straight back to the
            adapter via :class:`OutgoingMessage.reply_context`.
    """

    platform: Platform
    chat_id: str
    sender_display_name: str
    text: str
    native_message_id: str
    timestamp_ms: int | None = None
    reply_context: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class OutgoingMessage:
    """An assistant reply that should be delivered back to a chat.

    Attributes:
        platform: Target platform.
        chat_id: Same chat id supplied in the originating
            :class:`IncomingMessage`.
        text: Body to send. Routers truncate per-platform if needed.
        reply_context: The original incoming reply context, so adapters
            can prefer "reply" semantics (LINE replyToken, Telegram
            reply_to_message_id) before falling back to "push" semantics.
    """

    platform: Platform
    chat_id: str
    text: str
    reply_context: dict[str, Any] | None = None


class AdapterState(str, Enum):
    """Adapter lifecycle states."""

    DISCONNECTED = "disconnected"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(slots=True)
class AdapterStatus:
    """Current health/state of one adapter, surfaced in slash commands.

    Attributes:
        platform: Which adapter this describes.
        state: Lifecycle state.
        detail: Free-form one-line description (e.g. "polling as
            @feather_bot", "webhook listening on /line/webhook", or the
            most recent error message).
        last_error: The most recent error message, if any. Cleared once
            the adapter recovers.
        connected_chat_count: Number of distinct chats currently mapped
            to a session for this platform.
    """

    platform: Platform
    state: AdapterState = AdapterState.DISCONNECTED
    detail: str = ""
    last_error: str | None = None
    connected_chat_count: int = 0

    @property
    def is_running(self) -> bool:
        """Return True when the adapter is actively running."""

        return self.state == AdapterState.RUNNING


@dataclass(slots=True, frozen=True)
class CredentialRecord:
    """A persisted credential row from the ``messaging_credentials`` table.

    The opaque ``config`` dict carries platform-specific fields. Each
    adapter is responsible for validating the keys it needs.
    """

    platform: Platform
    config: dict[str, Any]
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True, frozen=True)
class ChatMappingRecord:
    """A persisted row from the ``messaging_chats`` table."""

    platform: Platform
    chat_id: str
    session_id: str
    display_name: str = ""
    created_at: str = ""
    updated_at: str = ""


__all__ = (
    "AdapterState",
    "AdapterStatus",
    "ChatMappingRecord",
    "CredentialRecord",
    "IncomingMessage",
    "OutgoingMessage",
    "Platform",
)
