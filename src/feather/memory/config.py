"""Configuration dataclasses for the memory subsystem.

These are loaded from ``config/app.yaml`` under a top-level ``memory:`` block
and injected into the runtime via ``AppConfig``. Every tunable parameter for
extraction, classification, retrieval, chunking, Qdrant wiring, and the
async trigger lives here — the subsystem intentionally surfaces no magic
numbers or in-code defaults that bypass user configuration.

Each sub-config is a dedicated ``@dataclass(slots=True)`` so it can be
constructed, tested, and overridden in isolation. Top-level ``MemoryConfig``
holds one instance of each.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# -----------------------------------------------------------------------------
# Per-operation model override (extraction / classification / query_builder)
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryOperationModelConfig:
    """Provider + model override for a single memory LLM operation.

    Both ``provider`` and ``model`` may be set independently. ``None`` (i.e.
    YAML ``~``) on either field means "inherit":

    - ``provider=None``: use the app-default provider built from
      ``active_provider`` in ``app.yaml``. The same shared instance the
      agent loop uses — no extra httpx client.
    - ``provider="openai"`` or ``"openrouter"``: build (or reuse) an
      alternate provider for this op. Lets a memory operation run on a
      different LLM client than the calling agent (e.g. cheap OpenAI
      extractor while the conversation runs on OpenRouter).
    - ``model=None`` paired with ``provider=None``: resolve at call-time
      to the calling agent's current conversation model.
    - ``model=None`` paired with an alternate provider: resolve to that
      provider's configured default model (``app.openai.model`` /
      ``app.openrouter.model``). Inheriting ``agent_model`` across
      provider boundaries would mis-route slugs (e.g. an OpenRouter
      slug sent to the OpenAI client), which is why this rule fires
      automatically.
    """

    model: str | None = None
    max_output_tokens: int = 2000
    temperature: float = 0.1
    provider: str | None = None


# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryChunkingConfig:
    """Token-level chunking parameters for long atomic memories.

    The tokenizer is pluggable so environments without ``tiktoken`` can fall
    back to a cheaper estimator. ``tiktoken`` is strictly better for chunking
    because it allows token-id-accurate slicing; the other estimators only
    count, which forces a word-level greedy packer.
    """

    chunk_size_tokens: int = 1000
    chunk_overlap_tokens: int = 100
    tokenizer: str = "tiktoken"
    tokenizer_encoding: str = "o200k_base"


# -----------------------------------------------------------------------------
# Embedding provider
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryEmbeddingConfig:
    """Embedding-provider config.

    The default model ``gemini-embedding-2-preview`` is Gemini 2.0 multimodal
    with an 8192-token input ceiling. ``output_dimensionality=3072`` is the
    full vector; the Gemini API L2-normalizes only at this dimensionality, so
    ``normalize_reduced_dims`` is a no-op at 3072 but enforced for any
    Matryoshka down-shift (e.g. 768).
    """

    provider: str = "gemini"
    model: str = "gemini-embedding-2-preview"
    output_dimensionality: int = 3072
    task_type_document: str = "RETRIEVAL_DOCUMENT"
    task_type_query: str = "RETRIEVAL_QUERY"
    normalize_reduced_dims: bool = True
    request_timeout_s: float = 30.0
    max_retries: int = 3
    retry_backoff_s: float = 1.5


# -----------------------------------------------------------------------------
# Qdrant wiring
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryQdrantConfig:
    """Qdrant client + collection + HNSW tuning.

    Defaults are tuned for search quality over raw speed: ``hnsw_m=32`` and
    ``ef_construct=256`` build denser graphs than Qdrant's defaults, and
    ``hnsw_ef_search=128`` raises per-query recall. Payload stays in RAM
    (``on_disk_payload=False``) so filter-during-search is fast; flip to
    ``True`` once the collection grows large.
    """

    url: str | None = None
    api_key_env: str = "QDRANT_API_KEY"
    collection_name: str = "feather_memory"
    embedding_dims: int = 3072
    hnsw_m: int = 32
    hnsw_ef_construct: int = 256
    hnsw_ef_search: int = 128
    hnsw_full_scan_threshold: int = 10_000
    indexing_threshold: int = 20_000
    default_segment_number: int = 2
    on_disk_vectors: bool = False
    on_disk_payload: bool = False
    prefer_grpc: bool = False
    request_timeout_s: float = 15.0
    quantization: str | None = None  # reserved; None = off


# -----------------------------------------------------------------------------
# Retrieval (read path)
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryRetrievalConfig:
    """Read-path knobs shared by proactive injection, recall_memory tool,
    and the CRUD-classifier's initial similarity lookup."""

    enabled: bool = True
    top_k_prompt_injection: int = 5
    top_k_tool: int = 10
    score_threshold: float = 0.5
    classifier_top_k: int = 3
    classifier_score_threshold: float = 0.75
    retrieval_timeout_s: float = 2.0
    query_builder_enabled: bool = True
    query_builder_recent_messages: int = 8


# -----------------------------------------------------------------------------
# Trigger (write-path scheduling)
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryTriggerConfig:
    """When and how the async background extraction job fires."""

    enabled: bool = True
    trigger_turns: int = 10
    skip_compact_messages: bool = True
    background: bool = True
    shutdown_timeout_s: float = 30.0
    max_concurrent_extractions_per_session: int = 1


# -----------------------------------------------------------------------------
# Top-level container
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryConfig:
    """Root memory-subsystem configuration.

    ``enabled`` is a global kill switch; per-agent ``memory_enabled`` (on
    ``AgentConfig``) gates participation further, and the runtime additionally
    requires ``QDRANT_URL`` + a Gemini API key and honors the
    ``FEATHER_MEMORY_DISABLED`` environment variable.
    """

    enabled: bool = False
    qdrant: MemoryQdrantConfig = field(default_factory=MemoryQdrantConfig)
    embedding: MemoryEmbeddingConfig = field(default_factory=MemoryEmbeddingConfig)
    chunking: MemoryChunkingConfig = field(default_factory=MemoryChunkingConfig)
    retrieval: MemoryRetrievalConfig = field(default_factory=MemoryRetrievalConfig)
    trigger: MemoryTriggerConfig = field(default_factory=MemoryTriggerConfig)
    extraction: MemoryOperationModelConfig = field(default_factory=MemoryOperationModelConfig)
    classification: MemoryOperationModelConfig = field(default_factory=MemoryOperationModelConfig)
    query_builder: MemoryOperationModelConfig = field(default_factory=MemoryOperationModelConfig)
