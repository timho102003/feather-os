"""Tests for parsing the `memory:` config block and env-var precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.config import load_agent_config, load_app_config
from feather.memory.config import MemoryConfig


_MIN_YAML_NO_MEMORY = """database:
  path: .feather/db/feather.db

storage:
  temp_directory: .feather/tmp

logging:
  path: .feather/logs/feather.log

compaction:
  enabled: true
  trigger_ratio: 0.8
  context_window_tokens: 400000
  max_output_tokens: 2000
  temperature: 0.2

skills:
  directory: .feather/skills

openai:
  api_key_env: OPENAI_API_KEY
  model: gpt-5-mini
  max_output_tokens: 4000
  temperature: 1.0
  parallel_tool_calls: true
"""


def _write_yaml(tmp_path: Path, extra: str = "") -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "app.yaml").write_text(
        _MIN_YAML_NO_MEMORY + extra,
        encoding="utf-8",
    )


# AppConfig.memory wiring -----------------------------------------------------


def test_app_config_defaults_memory_to_default_memory_config(tmp_path: Path) -> None:
    """Missing memory: block yields a fully-defaulted MemoryConfig (disabled by default)."""
    _write_yaml(tmp_path)
    cfg = load_app_config(tmp_path)
    assert isinstance(cfg.memory, MemoryConfig)
    assert cfg.memory.enabled is False
    assert cfg.memory.qdrant.embedding_dims == 3072
    assert cfg.memory.embedding.output_dimensionality == 3072
    assert cfg.memory.chunking.chunk_size_tokens == 1000
    assert cfg.memory.trigger.trigger_turns == 10


def test_app_config_parses_full_memory_block(tmp_path: Path) -> None:
    """All fields across every sub-config should round-trip from YAML."""
    extra = """
memory:
  enabled: true
  qdrant:
    url: http://qdrant:6333
    api_key_env: MY_QDRANT_KEY
    collection_name: custom_mem
    embedding_dims: 3072
    hnsw_m: 48
    hnsw_ef_construct: 512
    hnsw_ef_search: 256
    hnsw_full_scan_threshold: 5000
    indexing_threshold: 15000
    default_segment_number: 4
    on_disk_vectors: true
    on_disk_payload: true
    prefer_grpc: true
    request_timeout_s: 10.0
    quantization: int8
  embedding:
    provider: gemini
    model: gemini-embedding-2-preview
    output_dimensionality: 3072
    task_type_document: RETRIEVAL_DOCUMENT
    task_type_query: RETRIEVAL_QUERY
    normalize_reduced_dims: false
    request_timeout_s: 20.0
    max_retries: 5
    retry_backoff_s: 2.0
  chunking:
    chunk_size_tokens: 800
    chunk_overlap_tokens: 80
    tokenizer: char4
    tokenizer_encoding: cl100k_base
  retrieval:
    enabled: true
    top_k_prompt_injection: 7
    top_k_tool: 15
    score_threshold: 0.6
    classifier_top_k: 5
    classifier_score_threshold: 0.8
    retrieval_timeout_s: 3.0
    query_builder_enabled: false
    query_builder_recent_messages: 6
  trigger:
    enabled: true
    trigger_turns: 20
    skip_compact_messages: true
    background: false
    shutdown_timeout_s: 60.0
    max_concurrent_extractions_per_session: 2
  extraction:
    model: gpt-5
    max_output_tokens: 4000
    temperature: 0.2
  classification:
    model: gpt-5-mini
    max_output_tokens: 300
    temperature: 0.0
  query_builder:
    model: gpt-4.1-mini
    max_output_tokens: 150
    temperature: 0.0
"""
    _write_yaml(tmp_path, extra)
    cfg = load_app_config(tmp_path)
    m = cfg.memory
    assert m.enabled is True
    assert m.qdrant.url == "http://qdrant:6333"
    assert m.qdrant.api_key_env == "MY_QDRANT_KEY"
    assert m.qdrant.collection_name == "custom_mem"
    assert m.qdrant.hnsw_m == 48
    assert m.qdrant.hnsw_ef_search == 256
    assert m.qdrant.on_disk_vectors is True
    assert m.qdrant.quantization == "int8"
    assert m.embedding.max_retries == 5
    assert m.embedding.normalize_reduced_dims is False
    assert m.chunking.chunk_size_tokens == 800
    assert m.chunking.tokenizer == "char4"
    assert m.retrieval.classifier_score_threshold == 0.8
    assert m.retrieval.query_builder_enabled is False
    assert m.retrieval.query_builder_recent_messages == 6
    assert m.trigger.trigger_turns == 20
    assert m.trigger.background is False
    assert m.extraction.model == "gpt-5"
    assert m.classification.model == "gpt-5-mini"
    assert m.query_builder.model == "gpt-4.1-mini"


def test_app_config_partial_memory_block_uses_defaults_for_missing_fields(tmp_path: Path) -> None:
    """A partial memory block should fill in unspecified fields with defaults."""
    extra = """
memory:
  enabled: true
  qdrant:
    url: http://q:6333
"""
    _write_yaml(tmp_path, extra)
    cfg = load_app_config(tmp_path)
    assert cfg.memory.enabled is True
    assert cfg.memory.qdrant.url == "http://q:6333"
    # Unspecified fields retain defaults
    assert cfg.memory.qdrant.collection_name == "feather_memory"
    assert cfg.memory.qdrant.hnsw_m == 32
    assert cfg.memory.chunking.chunk_size_tokens == 1000
    assert cfg.memory.trigger.trigger_turns == 10


# Env-var precedence ---------------------------------------------------------


def test_qdrant_url_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """QDRANT_URL env var must win over the YAML value."""
    extra = """
memory:
  enabled: true
  qdrant:
    url: http://yaml:6333
"""
    _write_yaml(tmp_path, extra)
    monkeypatch.setenv("QDRANT_URL", "http://env:6333")
    cfg = load_app_config(tmp_path)
    assert cfg.memory.qdrant.url == "http://env:6333"


def test_qdrant_url_env_unset_leaves_yaml_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When QDRANT_URL is unset, the YAML value is retained."""
    extra = """
memory:
  enabled: true
  qdrant:
    url: http://yaml:6333
"""
    _write_yaml(tmp_path, extra)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    cfg = load_app_config(tmp_path)
    assert cfg.memory.qdrant.url == "http://yaml:6333"


def test_qdrant_url_env_empty_string_is_treated_as_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty env var should NOT nuke the YAML value (whitespace counts as unset)."""
    extra = """
memory:
  enabled: true
  qdrant:
    url: http://yaml:6333
"""
    _write_yaml(tmp_path, extra)
    monkeypatch.setenv("QDRANT_URL", "")
    cfg = load_app_config(tmp_path)
    assert cfg.memory.qdrant.url == "http://yaml:6333"


def test_qdrant_url_env_sets_value_when_yaml_omits_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Common deployment: YAML leaves url null, env provides it."""
    _write_yaml(tmp_path)
    monkeypatch.setenv("QDRANT_URL", "http://only-env:6333")
    cfg = load_app_config(tmp_path)
    assert cfg.memory.qdrant.url == "http://only-env:6333"


# AgentConfig.memory_enabled -------------------------------------------------


def test_agent_config_memory_enabled_defaults_to_false(tmp_path: Path) -> None:
    """Agents must default to memory_enabled=False — only lead opts in explicitly."""
    (tmp_path / "config" / "agents").mkdir(parents=True)
    (tmp_path / "config" / "agents" / "worker.yaml").write_text(
        """name: Worker
role: worker
personality: Direct
prompt_modules:
  - feather.core.prompts.default_agent_prompt:DEFAULT_AGENT_PROMPT
registered_tools:
  - grep
""",
        encoding="utf-8",
    )
    cfg = load_agent_config(tmp_path, "worker")
    assert cfg.memory_enabled is False


def test_agent_config_memory_enabled_read_from_yaml(tmp_path: Path) -> None:
    """A YAML with memory_enabled: true must be surfaced on AgentConfig."""
    (tmp_path / "config" / "agents").mkdir(parents=True)
    (tmp_path / "config" / "agents" / "lead.yaml").write_text(
        """name: Lead
role: lead
personality: Decisive
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
registered_tools:
  - grep
memory_enabled: true
""",
        encoding="utf-8",
    )
    cfg = load_agent_config(tmp_path, "lead")
    assert cfg.memory_enabled is True
