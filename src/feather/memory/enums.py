"""Enums shared by the memory subsystem.

These values are persisted to Qdrant payloads and used as literal values in
strict-JSON-schema responses from the LLM; they must remain stable once
points are written. Prefer additive evolution (new members) over renames.
"""

from __future__ import annotations

from enum import Enum


class EmbedType(str, Enum):
    """Primary tenancy key in the Qdrant payload.

    A single collection holds multiple logical corpora partitioned by this
    field; the Qdrant payload index on ``type`` is marked ``is_tenant=True``
    so HNSW clusters vectors per tenant for dramatic recall/latency gains.
    """

    MEMORY = "memory"
    ATTACHMENT_TEXT = "attachment:text"
    ATTACHMENT_IMAGE = "attachment:image"
    ATTACHMENT_PDF = "attachment:pdf"


class MemoryOwner(str, Enum):
    """Semantic ownership of a memory.

    - ``USER``: facts, preferences, and decisions about the Feather user.
      This is the only owner produced today.
    - ``LEAD``: reserved for future lead-agent self-knowledge memories.
    """

    USER = "user"
    LEAD = "lead"


class MemoryOp(str, Enum):
    """CRUD op produced by the classification LLM.

    Values are uppercase because they appear verbatim in the
    ``ClassificationResponse`` Pydantic schema's ``Literal[...]`` — strict
    JSON schema rejects any other casing.
    """

    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NO_OP = "NO_OP"


class EmbeddingTaskType(str, Enum):
    """Asymmetric task-type hints for Gemini embeddings.

    ``RETRIEVAL_DOCUMENT`` is used when indexing memory content; matching
    searches MUST use ``RETRIEVAL_QUERY`` or retrieval quality degrades
    silently (per the ``gemini-embedding`` skill).
    """

    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
