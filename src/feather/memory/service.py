"""End-to-end memory write-path orchestrator.

One public method — :meth:`MemoryService.extract_and_store` — composes the
window builder, extractor, classifier, chunker, embedder, and vector store
into a single deterministic pipeline. Each invocation is serialized per
session via an :class:`asyncio.Lock`; per-atom failures are isolated so a
single bad classification can't sink an entire batch.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from feather.memory.chunker import Chunker, TextChunk
from feather.memory.classifier import CrudClassifier
from feather.memory.config import MemoryConfig
from feather.memory.embedding.base import BaseEmbeddingProvider
from feather.memory.enums import EmbedType, MemoryOp, MemoryOwner
from feather.memory.extractor import MemoryExtractor
from feather.memory.models import (
    AppliedOp,
    AtomicMemory,
    ClassifiedOp,
    MemoryExtractionReport,
    MemoryPointPayload,
    MemorySearchResult,
    MemoryWindow,
)
from feather.memory.store.base import BaseVectorStore
from feather.models import AttachmentKind, AttachmentRecord, MessageRole, SessionMessage

# Default similarity floor for proactive UPDATE/DELETE target lookup. Looser
# than the auto-classifier's threshold because the user named the target
# explicitly — we just need a reasonable best match, not a confident one.
_PROACTIVE_DEFAULT_SCORE_THRESHOLD = 0.4
_PROACTIVE_SEARCH_TOP_K = 3

if TYPE_CHECKING:
    from feather.storage.session_store import SessionStore

logger = logging.getLogger(__name__)


def _user_turn_count(messages: list[SessionMessage]) -> int:
    """Return the number of USER messages in ``messages``."""
    return sum(1 for m in messages if m.role == MessageRole.USER)


def _slice_window_messages(
    messages: list[SessionMessage], trigger_turns: int
) -> list[SessionMessage] | None:
    """Return the window message slice, or ``None`` if below threshold.

    The window ends with the last message before the (trigger_turns+1)th
    USER message. If no such next user message exists yet, the window
    extends to the very last message in ``messages``.
    """
    user_count = 0
    nth_user_index = -1
    for i, msg in enumerate(messages):
        if msg.role == MessageRole.USER:
            user_count += 1
            if user_count == trigger_turns:
                nth_user_index = i
                break
    if nth_user_index < 0:
        return None  # below threshold
    end_inclusive = len(messages) - 1
    for j in range(nth_user_index + 1, len(messages)):
        if messages[j].role == MessageRole.USER:
            end_inclusive = j - 1
            break
    return messages[: end_inclusive + 1]


class MemoryService:
    """Write-path orchestrator. One public coroutine: :meth:`extract_and_store`."""

    def __init__(
        self,
        *,
        cfg: MemoryConfig,
        store: BaseVectorStore,
        embedder: BaseEmbeddingProvider,
        chunker: Chunker,
        extractor: MemoryExtractor,
        classifier: CrudClassifier,
        session_store: "SessionStore",
    ) -> None:
        self._cfg = cfg
        self._store = store
        self._embedder = embedder
        self._chunker = chunker
        self._extractor = extractor
        self._classifier = classifier
        self._session_store = session_store
        self._session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def initialize(self) -> None:
        """Create the Qdrant collection + indexes if absent (idempotent)."""
        await self._store.ensure_schema()

    # -- public entry point ---------------------------------------------------

    async def extract_and_store(
        self,
        session_id: str,
        *,
        agent_model: str,
        owner: MemoryOwner = MemoryOwner.USER,
    ) -> MemoryExtractionReport:
        """Extract memories from the current 10-turn window and persist.

        Args:
            session_id: Session whose messages drive extraction.
            agent_model: The agent's current conversation model — used as
                the default for any per-operation ``model=None`` config.
            owner: Memory owner; today only ``USER`` is produced.

        Returns:
            A structured report summarizing what happened. Empty reports
            carry a ``reason`` (e.g., ``"below_turn_threshold"``).
        """
        async with self._session_locks[session_id]:
            window = await self._build_window(session_id)
            if window is None:
                logger.info(
                    "memory.extract.skip",
                    extra={"session_id": session_id, "reason": "below_turn_threshold"},
                )
                return MemoryExtractionReport.empty(
                    session_id, reason="below_turn_threshold"
                )

            correlation_id = str(uuid4())
            logger.info(
                "memory.extract.start",
                extra={
                    "session_id": session_id,
                    "correlation_id": correlation_id,
                    "start_message_id": window.start_message_id,
                    "end_message_id": window.end_message_id,
                    "message_count": len(window.messages),
                },
            )

            atoms = await self._extractor.extract(window, agent_model)
            if not atoms:
                logger.info(
                    "memory.extract.empty",
                    extra={
                        "session_id": session_id,
                        "correlation_id": correlation_id,
                    },
                )
                return MemoryExtractionReport.empty(
                    session_id, reason="no_atoms", correlation_id=correlation_id
                )

            applied_ops: list[AppliedOp] = []
            for atom in atoms:
                try:
                    classified = await self._classifier.classify(
                        atom,
                        agent_model=agent_model,
                        owner=owner,
                        session_id=session_id,
                    )
                    applied = await self._apply_op(
                        classified, atom, window, owner=owner, session_id=session_id
                    )
                    applied_ops.append(applied)
                except Exception as exc:
                    logger.exception(
                        "memory.apply.failed",
                        extra={
                            "session_id": session_id,
                            "correlation_id": correlation_id,
                        },
                    )
                    applied_ops.append(AppliedOp.failed(MemoryOp.CREATE, str(exc)))

            report = MemoryExtractionReport(
                session_id=session_id,
                correlation_id=correlation_id,
                window=window,
                applied_ops=applied_ops,
            )
            logger.info("memory.extract.done", extra=report.to_log_fields())
            return report

    # -- proactive CRUD (direct CRUD without extractor/classifier) -----------

    async def proactive_create(
        self,
        *,
        content: str,
        purpose: str,
        session_id: str,
        owner: MemoryOwner = MemoryOwner.USER,
    ) -> AppliedOp:
        """Persist a memory the user explicitly asked the agent to remember.

        Bypasses the extractor + classifier. Chunk + embed + upsert directly,
        anchored to the session's most recent ``USER`` message for provenance.

        Args:
            content: The memory text to embed (one declarative sentence).
            purpose: How a future agent could use this memory.
            session_id: Active session id; required so the new point carries
                valid provenance and lock-serialization on the session works.
            owner: Memory owner; only ``USER`` is populated today.

        Returns:
            ``AppliedOp(op=CREATE, group_id=<new>, chunk_count=N)``.

        Raises:
            ValueError: ``content`` or ``purpose`` is empty after strip, or
                the session has no ``USER`` message yet (so no anchor).
        """
        content_clean = content.strip()
        purpose_clean = purpose.strip()
        if not content_clean:
            raise ValueError("proactive_create: content must be non-empty")
        if not purpose_clean:
            raise ValueError("proactive_create: purpose must be non-empty")

        async with self._session_locks[session_id]:
            anchor_msg_id = await self._latest_user_message_id(session_id)
            window = MemoryWindow(
                session_id=session_id,
                start_message_id=anchor_msg_id,
                end_message_id=anchor_msg_id,
                messages=[],
            )
            atom = AtomicMemory(
                who="the user",
                what=content_clean,
                when="proactive",
                where="proactive",
                why="user explicit request",
                how="manage_memory tool",
                purpose=purpose_clean,
                content=content_clean,
            )
            group_id = uuid4()
            chunks = self._chunker.chunk(atom.content)
            vectors = await self._embedder.embed_documents(
                [c.text for c in chunks]
            )
            payloads = self._build_payloads(
                chunks=chunks,
                group_id=group_id,
                session_id=session_id,
                owner=owner,
                window=window,
                atom=atom,
            )
            await self._store.upsert_group(payloads, vectors)
            logger.info(
                "memory.proactive.created",
                extra={
                    "session_id": session_id,
                    "group_id": str(group_id),
                    "chunk_count": len(chunks),
                },
            )
            return AppliedOp(
                op=MemoryOp.CREATE, group_id=group_id, chunk_count=len(chunks)
            )

    async def proactive_update(
        self,
        *,
        target_query: str,
        content: str,
        purpose: str,
        session_id: str,
        owner: MemoryOwner = MemoryOwner.USER,
        score_threshold: float = _PROACTIVE_DEFAULT_SCORE_THRESHOLD,
    ) -> AppliedOp:
        """Replace an existing memory selected by ``target_query``.

        Cross-session search (owner-scoped). The top hit must score at least
        ``score_threshold`` or the call returns a failed ``AppliedOp`` so the
        caller can decide whether to fall back to ``proactive_create``.

        Returns ``AppliedOp(op=UPDATE, group_id=<matched>, chunk_count=N)``
        on success; ``AppliedOp.failed(UPDATE, "no match …")`` otherwise.
        """
        target_query_clean = target_query.strip()
        content_clean = content.strip()
        purpose_clean = purpose.strip()
        if not target_query_clean:
            raise ValueError("proactive_update: target_query must be non-empty")
        if not content_clean:
            raise ValueError("proactive_update: content must be non-empty")
        if not purpose_clean:
            raise ValueError("proactive_update: purpose must be non-empty")

        async with self._session_locks[session_id]:
            best = await self._find_best_match(
                target_query=target_query_clean,
                owner=owner,
                score_threshold=score_threshold,
            )
            if best is None:
                return AppliedOp.failed(
                    MemoryOp.UPDATE,
                    f"no match for target_query={target_query_clean!r} above {score_threshold}",
                )

            anchor_msg_id = await self._latest_user_message_id(session_id)
            window = MemoryWindow(
                session_id=session_id,
                start_message_id=anchor_msg_id,
                end_message_id=anchor_msg_id,
                messages=[],
            )
            atom = AtomicMemory(
                who="the user",
                what=content_clean,
                when="proactive",
                where="proactive",
                why="user explicit update",
                how="manage_memory tool",
                purpose=purpose_clean,
                content=content_clean,
            )
            group_id = best.payload.group_id
            chunks = self._chunker.chunk(atom.content)
            vectors = await self._embedder.embed_documents(
                [c.text for c in chunks]
            )
            await self._store.delete_group(group_id)
            payloads = self._build_payloads(
                chunks=chunks,
                group_id=group_id,
                session_id=session_id,
                owner=owner,
                window=window,
                atom=atom,
            )
            await self._store.upsert_group(payloads, vectors)
            logger.info(
                "memory.proactive.updated",
                extra={
                    "session_id": session_id,
                    "group_id": str(group_id),
                    "matched_score": best.score,
                    "chunk_count": len(chunks),
                },
            )
            return AppliedOp(
                op=MemoryOp.UPDATE, group_id=group_id, chunk_count=len(chunks)
            )

    async def proactive_delete(
        self,
        *,
        target_query: str,
        session_id: str,
        owner: MemoryOwner = MemoryOwner.USER,
        score_threshold: float = _PROACTIVE_DEFAULT_SCORE_THRESHOLD,
    ) -> AppliedOp:
        """Delete the memory whose top vector match for ``target_query``
        scores at least ``score_threshold``.

        Cross-session search (owner-scoped). Returns ``AppliedOp.failed`` if
        no candidate beats the threshold so the caller can ask the user to
        rephrase.
        """
        target_query_clean = target_query.strip()
        if not target_query_clean:
            raise ValueError("proactive_delete: target_query must be non-empty")

        async with self._session_locks[session_id]:
            best = await self._find_best_match(
                target_query=target_query_clean,
                owner=owner,
                score_threshold=score_threshold,
            )
            if best is None:
                return AppliedOp.failed(
                    MemoryOp.DELETE,
                    f"no match for target_query={target_query_clean!r} above {score_threshold}",
                )
            group_id = best.payload.group_id
            await self._store.delete_group(group_id)
            logger.info(
                "memory.proactive.deleted",
                extra={
                    "session_id": session_id,
                    "group_id": str(group_id),
                    "matched_score": best.score,
                },
            )
            return AppliedOp(op=MemoryOp.DELETE, group_id=group_id, chunk_count=0)

    async def index_attachment(
        self,
        attachment: AttachmentRecord,
        *,
        content: str,
        owner: MemoryOwner = MemoryOwner.USER,
    ) -> AppliedOp:
        """Embed and store chunks for one saved chat attachment.

        Args:
            attachment: Persisted attachment metadata.
            content: Text representation to embed. For images this is
                currently metadata/caption text, not raw pixels.
            owner: Memory owner for the vector payload.

        Returns:
            Applied operation summary.
        """

        cleaned = content.strip()
        if not cleaned:
            return AppliedOp.failed(MemoryOp.CREATE, "empty attachment content")
        async with self._session_locks[attachment.session_id]:
            group_id = UUID(attachment.id)
            await self._store.delete_group(group_id)
            chunks = self._chunker.chunk(cleaned)
            vectors = await self._embedder.embed_documents(
                [chunk.text for chunk in chunks]
            )
            embed_type = _attachment_embed_type(attachment)
            ids = [uuid4() for _ in chunks]
            payloads: list[MemoryPointPayload] = []
            for index, (chunk, point_id) in enumerate(zip(chunks, ids, strict=True)):
                payloads.append(
                    MemoryPointPayload(
                        id=point_id,
                        type=embed_type,
                        memory_owner=owner,
                        content=chunk.text,
                        purpose=(
                            f"Retrieve content from attachment {attachment.original_name}."
                        ),
                        filepath=attachment.filepath,
                        group_id=group_id,
                        previous_chunk_id=ids[index - 1] if index > 0 else None,
                        next_chunk_id=ids[index + 1] if index < len(ids) - 1 else None,
                        chunk_index=index,
                        session_id=UUID(attachment.session_id),
                        start_message_id=UUID(attachment.message_id),
                        end_message_id=UUID(attachment.message_id),
                    )
                )
            await self._store.upsert_group(payloads, vectors)
            logger.info(
                "attachment.indexed",
                extra={
                    "session_id": attachment.session_id,
                    "attachment_id": attachment.id,
                    "chunk_count": len(chunks),
                    "filepath": attachment.filepath,
                },
            )
            return AppliedOp(
                op=MemoryOp.CREATE, group_id=group_id, chunk_count=len(chunks)
            )

    # -- helpers --------------------------------------------------------------

    async def _latest_user_message_id(self, session_id: str) -> str:
        """Return the id of the most recent ``USER`` message for provenance.

        Raises ``ValueError`` if the session has no user message — proactive
        ops fundamentally need a user turn to anchor against.
        """
        messages = await self._session_store.list_messages(session_id)
        for msg in reversed(messages):
            if msg.role is MessageRole.USER:
                return msg.id
        raise ValueError(
            f"session {session_id!r} has no user message to anchor a proactive memory against"
        )

    async def _find_best_match(
        self,
        *,
        target_query: str,
        owner: MemoryOwner,
        score_threshold: float,
    ) -> MemorySearchResult | None:
        """Vector-search for ``target_query`` and return the top hit above
        ``score_threshold``, or ``None``.

        Search runs cross-session (matching the read-path's default for user
        memory) so a 'forget' request lands the same way regardless of which
        session originally produced the memory.
        """
        query_vec = await self._embedder.embed_query(target_query)
        results = await self._store.search(
            query=query_vec,
            top_k=_PROACTIVE_SEARCH_TOP_K,
            filters={
                "type": EmbedType.MEMORY.value,
                "memory_owner": owner.value,
            },
            session_id=None,  # cross-session, owner-scoped
        )
        if not results:
            return None
        # ``results`` is already sorted by Qdrant; defensively pick the max.
        best = max(results, key=lambda r: r.score)
        if best.score < score_threshold:
            return None
        return best

    async def _build_window(self, session_id: str) -> MemoryWindow | None:
        """Return the next extraction window or None if below threshold."""
        latest = await self._store.latest_memory_for_session(session_id)
        anchor_id: str | None = (
            str(latest.end_message_id) if latest is not None else None
        )
        messages = await self._session_store.get_non_compact_after(
            session_id, after_message_id=anchor_id
        )
        sliced = _slice_window_messages(messages, self._cfg.trigger.trigger_turns)
        if sliced is None or not sliced:
            return None
        return MemoryWindow(
            session_id=session_id,
            start_message_id=sliced[0].id,
            end_message_id=sliced[-1].id,
            messages=sliced,
        )

    async def _apply_op(
        self,
        classified: ClassifiedOp,
        atom: AtomicMemory,
        window: MemoryWindow,
        *,
        owner: MemoryOwner,
        session_id: str,
    ) -> AppliedOp:
        op = classified.op
        if op is MemoryOp.NO_OP:
            return AppliedOp(op=MemoryOp.NO_OP, group_id=None, chunk_count=0)

        if op is MemoryOp.DELETE:
            assert classified.target_group_id is not None
            target = UUID(classified.target_group_id)
            await self._store.delete_group(target)
            return AppliedOp(op=MemoryOp.DELETE, group_id=target, chunk_count=0)

        # CREATE or UPDATE: chunk + embed + write.
        if op is MemoryOp.UPDATE:
            assert classified.target_group_id is not None
            group_id = UUID(classified.target_group_id)
        else:
            group_id = uuid4()

        chunks = self._chunker.chunk(atom.content)
        vectors = await self._embedder.embed_documents([c.text for c in chunks])

        if op is MemoryOp.UPDATE:
            await self._store.delete_group(group_id)

        payloads = self._build_payloads(
            chunks=chunks,
            group_id=group_id,
            session_id=session_id,
            owner=owner,
            window=window,
            atom=atom,
        )
        await self._store.upsert_group(payloads, vectors)
        return AppliedOp(op=op, group_id=group_id, chunk_count=len(chunks))

    @staticmethod
    def _build_payloads(
        *,
        chunks: list[TextChunk],
        group_id: UUID,
        session_id: str,
        owner: MemoryOwner,
        window: MemoryWindow,
        atom: AtomicMemory,
    ) -> list[MemoryPointPayload]:
        """Produce one ``MemoryPointPayload`` per chunk with prev/next links."""
        # Pre-allocate ids so we can wire prev/next before constructing payloads.
        ids = [uuid4() for _ in chunks]
        payloads: list[MemoryPointPayload] = []
        for i, (chunk, point_id) in enumerate(zip(chunks, ids)):
            payloads.append(
                MemoryPointPayload(
                    id=point_id,
                    type=EmbedType.MEMORY,
                    memory_owner=owner,
                    content=chunk.text,
                    purpose=atom.purpose,
                    filepath=None,
                    group_id=group_id,
                    previous_chunk_id=ids[i - 1] if i > 0 else None,
                    next_chunk_id=ids[i + 1] if i < len(ids) - 1 else None,
                    chunk_index=i,
                    session_id=UUID(session_id),
                    start_message_id=UUID(window.start_message_id),
                    end_message_id=UUID(window.end_message_id),
                )
            )
        return payloads


__all__ = ["MemoryService"]


def _attachment_embed_type(attachment: AttachmentRecord) -> EmbedType:
    if attachment.kind == AttachmentKind.IMAGE:
        return EmbedType.ATTACHMENT_IMAGE
    if attachment.mime_type == "application/pdf":
        return EmbedType.ATTACHMENT_PDF
    return EmbedType.ATTACHMENT_TEXT
