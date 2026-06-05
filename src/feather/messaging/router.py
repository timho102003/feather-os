"""Bridge between platform adapters and the lead Feather agent.

For each :class:`IncomingMessage`:

1. Drop the message if its native id was already seen (idempotency
   against platform redelivery).
2. Resolve or create the Feather session bound to ``(platform, chat_id)``.
3. Run the lead agent for that session — when it's already busy,
   enqueue via the existing :class:`UserInputQueue` so messages serialise
   instead of racing.
4. Convert the agent's final assistant text into an
   :class:`OutgoingMessage` and hand it back to the originating adapter.

The router never holds a reference to the agent loop directly — it goes
through whatever ``run`` callable :class:`feather.runtime.FeatherRuntime`
provides. Tests inject a stub runtime to exercise the routing logic
without spinning up a real agent.
"""

from __future__ import annotations

import logging
from typing import Protocol

from feather.core.session.input_queue import UserInputQueue
from feather.messaging.models import (
    IncomingMessage,
    OutgoingMessage,
    Platform,
)
from feather.messaging.store import MessagingStore
from feather.models import AgentOutcome, AgentRunResult

logger = logging.getLogger(__name__)


class _AdapterSendCallable(Protocol):
    async def __call__(self, outgoing: OutgoingMessage) -> None: ...


class _SessionFactory(Protocol):
    async def __call__(self) -> str: ...


class _AgentRunCallable(Protocol):
    async def __call__(
        self, session_id: str, text: str
    ) -> AgentRunResult: ...


class _SessionBusyCallable(Protocol):
    async def __call__(self, session_id: str) -> bool: ...


class MessagingRouter:
    """Route inbound platform messages through the lead agent.

    Args:
        store: Persistent chat-mapping + dedup storage.
        create_session: Async factory that builds a fresh Feather
            session id (typically ``agent.create_session``).
        run_agent: Async callable that runs the agent for a session
            with new user text and returns the outcome.
        is_session_busy: Returns True when a run is currently in flight
            for the session (so we should enqueue rather than spawn).
        input_queue: The shared :class:`UserInputQueue` used by the rest
            of the runtime to inject text between turns.
    """

    def __init__(
        self,
        *,
        store: MessagingStore,
        create_session: _SessionFactory,
        run_agent: _AgentRunCallable,
        is_session_busy: _SessionBusyCallable,
        input_queue: UserInputQueue,
    ) -> None:
        self._store = store
        self._create_session = create_session
        self._run_agent = run_agent
        self._is_session_busy = is_session_busy
        self._input_queue = input_queue
        self._adapter_senders: dict[Platform, _AdapterSendCallable] = {}
        # Track which sessions currently have a router-spawned run in
        # flight. Used to ensure that two concurrent inbound messages
        # for the same chat serialise correctly (review fix C3+M2):
        # only the agent-side ``SessionRunCoordinator`` would have
        # serialised them otherwise, but the second message would
        # spawn its own ``agent.run`` and produce a separate reply,
        # bypassing the ``UserInputQueue`` coalescing path. Tracking
        # locally — and removing the entry on completion — avoids the
        # unbounded ``dict[str, asyncio.Lock]`` leak that an
        # ever-growing per-session lock store would cause.
        self._active_sessions: set[str] = set()

    def register_adapter(
        self,
        platform: Platform,
        sender: _AdapterSendCallable,
    ) -> None:
        """Register an adapter's send-callback so the router can reply."""

        self._adapter_senders[platform] = sender

    def unregister_adapter(self, platform: Platform) -> None:
        """Forget the adapter (e.g. on disconnect)."""

        self._adapter_senders.pop(platform, None)

    async def handle_incoming(self, message: IncomingMessage) -> None:
        """Process one inbound message end to end.

        Logs and swallows exceptions so adapter loops never die from a
        single bad payload. Real bugs surface as ``ERROR``-level log
        entries that the operator can inspect via ``.feather/logs/``.

        Inbound dedup is claimed up-front to drop platform redeliveries.
        On exception the dedup row is released so a retry from the
        platform side can still be processed (review fix M4).
        """

        if not message.text or not message.text.strip():
            return

        claimed = await self._store.claim_inbound(
            message.platform, message.native_message_id
        )
        if not claimed:
            logger.info(
                "messaging.router.duplicate platform=%s native_id=%s",
                message.platform.value,
                message.native_message_id,
            )
            return

        try:
            await self._handle_claimed(message)
        except Exception:  # noqa: BLE001
            # Release the dedup so a redelivery can be processed.
            await self._store.release_inbound(
                message.platform, message.native_message_id
            )
            logger.exception(
                "messaging.router.handle_failed platform=%s chat=%s",
                message.platform.value,
                message.chat_id,
            )

    async def _handle_claimed(self, message: IncomingMessage) -> None:
        mapping = await self._store.get_chat_mapping(
            message.platform, message.chat_id
        )
        if mapping is None:
            session_id = await self._create_session()
            await self._store.upsert_chat_mapping(
                platform=message.platform,
                chat_id=message.chat_id,
                session_id=session_id,
                display_name=message.sender_display_name,
            )
        else:
            session_id = mapping.session_id
            if mapping.display_name != message.sender_display_name:
                await self._store.upsert_chat_mapping(
                    platform=message.platform,
                    chat_id=message.chat_id,
                    session_id=session_id,
                    display_name=message.sender_display_name,
                )

        # Cheap router-side coalescing: if another inbound message is
        # already running through this same chat's session, queue rather
        # than spawn a parallel agent run. The agent's
        # ``SessionRunCoordinator`` would serialise correctly in any
        # case, but the queue path produces one consolidated reply
        # instead of N replies for N pile-on messages.
        if session_id in self._active_sessions or await self._is_session_busy(
            session_id
        ):
            ok = await self._input_queue.enqueue(session_id, message.text)
            logger.info(
                "messaging.router.enqueued platform=%s chat=%s session=%s ok=%s",
                message.platform.value,
                message.chat_id,
                session_id,
                ok,
            )
            return

        self._active_sessions.add(session_id)
        try:
            result = await self._run_agent(session_id, message.text)
        finally:
            self._active_sessions.discard(session_id)

        await self._dispatch_reply(message, session_id, result)

    async def _dispatch_reply(
        self,
        incoming: IncomingMessage,
        session_id: str,
        result: AgentRunResult,
    ) -> None:
        sender = self._adapter_senders.get(incoming.platform)
        if sender is None:
            logger.warning(
                "messaging.router.no_sender platform=%s session=%s",
                incoming.platform.value,
                session_id,
            )
            return

        body = _select_reply_body(result)
        if not body.strip():
            return
        outgoing = OutgoingMessage(
            platform=incoming.platform,
            chat_id=incoming.chat_id,
            text=body,
            reply_context=incoming.reply_context,
        )
        try:
            await sender(outgoing)
        except Exception:  # noqa: BLE001
            logger.exception(
                "messaging.router.send_failed platform=%s chat=%s",
                incoming.platform.value,
                incoming.chat_id,
            )


def _select_reply_body(result: AgentRunResult) -> str:
    """Pick the text to send back to the chat from an agent outcome."""

    text = (result.assistant_text or "").strip()
    if result.status == AgentOutcome.AWAITING_USER and result.question:
        question = result.question.strip()
        if not text:
            return question
        return f"{text}\n\n{question}"
    return text


__all__ = ("MessagingRouter",)
