"""CRUD classifier for new atomic memories.

For each freshly-extracted :class:`AtomicMemory` the classifier:

1. Embeds the memory content as a *query* and looks up the top-K most
   similar existing memories across all sessions (memory is per-user, not
   per-session).
2. If no candidate exceeds the configured cosine-similarity threshold, the
   op is ``CREATE`` and no LLM call is made.
3. Otherwise a single strict-JSON LLM call classifies the op as
   ``CREATE`` / ``UPDATE`` / ``DELETE`` / ``NO_OP`` and, for UPDATE/DELETE,
   names the target group_id from the candidate set.

A deterministic guardrail coerces invalid ``target_group_id`` values (not
in the candidate set, or missing for UPDATE/DELETE) into ``NO_OP`` + a
warning log — a hallucinated id must never corrupt the store.
"""

from __future__ import annotations

import logging
from typing import Sequence

from feather.memory.config import MemoryOperationModelConfig, MemoryRetrievalConfig
from feather.memory.embedding.base import BaseEmbeddingProvider
from feather.memory.enums import EmbedType, MemoryOp, MemoryOwner
from feather.memory.models import (
    AtomicMemory,
    ClassificationResponse,
    ClassifiedOp,
    MemorySearchResult,
)
from feather.memory.store.base import BaseVectorStore
from feather.models import ProviderRequestConfig
from feather.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def _render_for_classifier(
    atom: AtomicMemory, candidates: Sequence[MemorySearchResult]
) -> str:
    """Render the new atom + candidates for the classification LLM call."""
    lines: list[str] = []
    lines.append("=== NEW memory ===")
    lines.append(f"who: {atom.who}")
    lines.append(f"what: {atom.what}")
    lines.append(f"when: {atom.when}")
    lines.append(f"where: {atom.where}")
    lines.append(f"why: {atom.why}")
    lines.append(f"how: {atom.how}")
    lines.append(f"purpose: {atom.purpose}")
    lines.append(f"content: {atom.content}")
    lines.append("")
    lines.append("=== EXISTING candidates (highest similarity first) ===")
    for i, cand in enumerate(candidates, 1):
        p = cand.payload
        lines.append(
            f"[{i}] group_id={p.group_id}  similarity={cand.score:.3f}"
        )
        lines.append(f"    content: {p.content}")
        lines.append(f"    purpose: {p.purpose}")
    return "\n".join(lines)


class CrudClassifier:
    """Decide CREATE / UPDATE / DELETE / NO_OP for one new atomic memory."""

    def __init__(
        self,
        *,
        provider: BaseLLMProvider,
        prompt: str,
        cfg: MemoryOperationModelConfig,
        store: BaseVectorStore,
        embedder: BaseEmbeddingProvider,
        retrieval_cfg: MemoryRetrievalConfig,
        default_model: str | None = None,
    ) -> None:
        self._provider = provider
        self._prompt = prompt
        self._cfg = cfg
        self._store = store
        self._embedder = embedder
        self._retrieval_cfg = retrieval_cfg
        self._default_model = default_model

    async def classify(
        self,
        atom: AtomicMemory,
        *,
        agent_model: str,
        owner: MemoryOwner,
        session_id: str,
    ) -> ClassifiedOp:
        # 1. Embed + retrieve.
        q_vec = await self._embedder.embed_query(atom.content)
        candidates = await self._store.search(
            query=q_vec,
            top_k=self._retrieval_cfg.classifier_top_k,
            filters={
                "type": EmbedType.MEMORY.value,
                "memory_owner": owner.value,
            },
            # Intentionally cross-session: memory is per-user.
            session_id=None,
        )

        # 2. Short-circuit CREATE when no candidate is similar enough.
        if (
            not candidates
            or candidates[0].score < self._retrieval_cfg.classifier_score_threshold
        ):
            return ClassifiedOp(
                op=MemoryOp.CREATE, target_group_id=None, candidates=candidates
            )

        # 3. LLM decides the op.
        rendered = _render_for_classifier(atom, candidates)
        request_config = ProviderRequestConfig(
            model=self._cfg.model or self._default_model or agent_model,
            max_output_tokens=self._cfg.max_output_tokens,
            temperature=self._cfg.temperature,
            response_schema=ClassificationResponse,
        )
        turn = await self._provider.complete(
            instructions=self._prompt,
            input_items=[{"role": "user", "content": rendered}],
            tools=[],
            previous_response_id=None,
            request_config=request_config,
        )
        parsed = ClassificationResponse.model_validate_json(turn.output_text)
        op = MemoryOp(parsed.op)
        target = parsed.target_group_id

        # 4. Guardrail: for UPDATE/DELETE, the target MUST be a known group_id.
        if op in (MemoryOp.UPDATE, MemoryOp.DELETE):
            valid_ids = {str(c.payload.group_id) for c in candidates}
            if not target or target not in valid_ids:
                logger.warning(
                    "memory.classifier.invalid_target",
                    extra={
                        "returned_op": op.value,
                        "returned_target": target,
                        "valid_ids": list(valid_ids),
                        "session_id": session_id,
                    },
                )
                op = MemoryOp.NO_OP
                target = None

        # 5. CREATE / NO_OP must not carry a target.
        if op in (MemoryOp.CREATE, MemoryOp.NO_OP):
            target = None

        return ClassifiedOp(op=op, target_group_id=target, candidates=candidates)
