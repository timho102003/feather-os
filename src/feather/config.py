"""Load Feather configuration from YAML files."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

_FLAT_MEMORY_OPS_WARNED = False

import yaml

from feather.memory.config import (
    MemoryChunkingConfig,
    MemoryConfig,
    MemoryEmbeddingConfig,
    MemoryOperationModelConfig,
    MemoryQdrantConfig,
    MemoryRetrievalConfig,
    MemoryTriggerConfig,
)
from feather.models import (
    AgentConfig,
    AppConfig,
    ClaudeConfig,
    ClaudeThinkingConfig,
    CompactionConfig,
    DatabaseConfig,
    LoggingConfig,
    MCPConfig,
    MCPServerConfig,
    OpenAIConfig,
    OpenRouterConfig,
    OpenRouterTracingConfig,
    ParallelConfig,
    ReasoningConfig,
    SchedulerConfig,
    SelfRepairConfig,
    StorageConfig,
    SkillsConfig,
)
from feather.resources import (
    has_packaged_agent,
    packaged_agent_yaml_text,
    packaged_app_yaml_dict,
)

if TYPE_CHECKING:
    from feather.paths import FeatherPaths


def load_app_config(
    root: Path,
    paths: "FeatherPaths | None" = None,
) -> AppConfig:
    """Load the central application config with layered overrides.

    Resolution order:

    1. **Base config:** if ``<root>/config/app.yaml`` exists it becomes
       the authoritative base — this preserves the semantics that test
       fixtures and project-staged configs have always relied on (a
       staged file is the full config, not a partial overlay). When no
       project-staged file exists, the packaged default shipped in the
       wheel is the base, so a fresh ``pip install`` works without any
       config files on disk.
    2. **Global overlay:** if ``paths`` is provided and
       ``paths.global_config_dir/app.yaml`` exists, it is deep-merged
       on top so user-global tweaks (e.g. preferred model, reasoning
       effort) override the base without having to copy the whole file.

    Args:
        root: Working directory whose ``config/app.yaml`` (if present)
            acts as the authoritative base.
        paths: Optional :class:`feather.paths.FeatherPaths` whose global
            config layer is merged on top.

    Returns:
        Parsed application configuration.
    """

    project_app_yaml = root / "config" / "app.yaml"
    if project_app_yaml.exists():
        raw = _read_yaml(project_app_yaml)
    else:
        raw = packaged_app_yaml_dict()
    if paths is not None:
        global_app_yaml = paths.global_config_dir / "app.yaml"
        if global_app_yaml.exists():
            raw = _deep_merge(raw, _read_yaml(global_app_yaml))
    compaction_raw = raw.get("compaction") or {}
    scheduler_raw = raw.get("scheduler") or {}
    self_repair_raw = raw.get("self_repair") or {}
    reasoning_raw = raw["openai"].get("reasoning") or None
    parallel_raw = raw.get("parallel") or None
    active_provider = str(raw.get("active_provider") or "openai").strip().lower()
    openrouter_raw = raw.get("openrouter") or None
    openrouter_cfg: OpenRouterConfig | None = None
    if openrouter_raw is not None:
        or_reasoning_raw = openrouter_raw.get("reasoning") or None
        or_fallback_raw = openrouter_raw.get("fallback_models") or None
        or_prefs_raw = openrouter_raw.get("provider_preferences") or None
        or_tracing_raw = openrouter_raw.get("tracing") or None
        or_tracing_cfg: OpenRouterTracingConfig | None = None
        if or_tracing_raw is not None:
            tracing_metadata_raw = or_tracing_raw.get("metadata") or None
            or_tracing_cfg = OpenRouterTracingConfig(
                enabled=bool(or_tracing_raw.get("enabled", False)),
                user=or_tracing_raw.get("user"),
                metadata=(
                    dict(tracing_metadata_raw)
                    if isinstance(tracing_metadata_raw, dict)
                    else None
                ),
            )
        openrouter_cfg = OpenRouterConfig(
            api_key_env=openrouter_raw.get("api_key_env", "OPEN_ROUTER_API_KEY"),
            base_url=openrouter_raw.get("base_url", "https://openrouter.ai/api/v1"),
            http_referer=openrouter_raw.get("http_referer"),
            app_title=openrouter_raw.get("app_title"),
            model=openrouter_raw.get("model", "anthropic/claude-sonnet-4.6"),
            max_output_tokens=int(openrouter_raw.get("max_output_tokens", 32_000)),
            temperature=float(openrouter_raw.get("temperature", 1.0)),
            parallel_tool_calls=bool(openrouter_raw.get("parallel_tool_calls", True)),
            reasoning=(
                ReasoningConfig(
                    effort=or_reasoning_raw.get("effort"),
                    summary=or_reasoning_raw.get("summary"),
                )
                if or_reasoning_raw is not None
                else None
            ),
            provider_preferences=(dict(or_prefs_raw) if or_prefs_raw else None),
            fallback_models=(list(or_fallback_raw) if or_fallback_raw else None),
            cache_strategy=str(
                openrouter_raw.get("cache_strategy", "anthropic_breakpoint")
            ),
            stream_idle_timeout_seconds=float(
                openrouter_raw.get("stream_idle_timeout_seconds", 90.0)
            ),
            request_timeout_seconds=float(
                openrouter_raw.get("request_timeout_seconds", 120.0)
            ),
            max_attempts=int(openrouter_raw.get("max_attempts", 3)),
            supports_multimodal=bool(openrouter_raw.get("supports_multimodal", True)),
            max_stream_wall_seconds=float(
                openrouter_raw.get("max_stream_wall_seconds", 600.0)
            ),
            tracing=or_tracing_cfg,
        )
    parallel_config = (
        ParallelConfig(
            api_key_env=parallel_raw["api_key_env"],
            default_search_mode=parallel_raw.get("default_search_mode", "fast"),
            max_results=int(parallel_raw.get("max_results", 5)),
            inline_full_content_threshold=int(
                parallel_raw.get("inline_full_content_threshold", 4000)
            ),
        )
        if parallel_raw is not None
        else None
    )
    claude_raw = raw.get("claude") or None
    claude_cfg: ClaudeConfig | None = None
    if claude_raw is not None:
        thinking_raw = claude_raw.get("thinking") or None
        thinking_cfg: ClaudeThinkingConfig | None = None
        if thinking_raw is not None:
            thinking_cfg = ClaudeThinkingConfig(
                type=str(thinking_raw.get("type", "enabled")).strip().lower(),
                budget_tokens=(
                    int(thinking_raw["budget_tokens"])
                    if thinking_raw.get("budget_tokens") is not None
                    else None
                ),
            )
        beta_raw = claude_raw.get("anthropic_beta") or ()
        if isinstance(beta_raw, str):
            beta_tuple = (beta_raw.strip(),) if beta_raw.strip() else ()
        else:
            beta_tuple = tuple(
                str(b).strip() for b in beta_raw if str(b).strip()
            )
        claude_cfg = ClaudeConfig(
            api_key_env=claude_raw.get("api_key_env", "ANTHROPIC_API_KEY"),
            base_url=claude_raw.get("base_url", "https://api.anthropic.com"),
            anthropic_version=str(
                claude_raw.get("anthropic_version", "2023-06-01")
            ),
            anthropic_beta=beta_tuple,
            model=claude_raw.get("model", "claude-opus-4-7"),
            max_output_tokens=int(claude_raw.get("max_output_tokens", 32_000)),
            temperature=float(claude_raw.get("temperature", 1.0)),
            parallel_tool_calls=bool(claude_raw.get("parallel_tool_calls", True)),
            thinking=thinking_cfg,
            cache_strategy=str(
                claude_raw.get("cache_strategy", "anthropic_breakpoint")
            ),
            stream_idle_timeout_seconds=float(
                claude_raw.get("stream_idle_timeout_seconds", 90.0)
            ),
            request_timeout_seconds=float(
                claude_raw.get("request_timeout_seconds", 120.0)
            ),
            max_attempts=int(claude_raw.get("max_attempts", 3)),
            supports_multimodal=bool(claude_raw.get("supports_multimodal", True)),
            max_stream_wall_seconds=float(
                claude_raw.get("max_stream_wall_seconds", 600.0)
            ),
        )
    return AppConfig(
        database=DatabaseConfig(path=raw["database"]["path"]),
        storage=StorageConfig(temp_directory=raw["storage"]["temp_directory"]),
        logging=LoggingConfig(
            path=raw["logging"]["path"],
            level=raw["logging"].get("level", "INFO"),
        ),
        compaction=CompactionConfig(
            enabled=bool(compaction_raw.get("enabled", True)),
            trigger_ratio=float(compaction_raw.get("trigger_ratio", 0.8)),
            context_window_tokens=int(compaction_raw.get("context_window_tokens", 400000)),
            model=compaction_raw.get("model"),
            max_output_tokens=int(compaction_raw.get("max_output_tokens", 2000)),
            temperature=float(compaction_raw.get("temperature", 0.2)),
        ),
        skills=SkillsConfig(directory=raw["skills"]["directory"]),
        scheduler=SchedulerConfig(
            enabled=bool(scheduler_raw.get("enabled", True)),
            poll_interval_seconds=float(scheduler_raw.get("poll_interval_seconds", 5)),
            failure_retry_seconds=float(scheduler_raw.get("failure_retry_seconds", 60)),
            max_due_jobs_per_tick=int(scheduler_raw.get("max_due_jobs_per_tick", 10)),
        ),
        self_repair=SelfRepairConfig(
            enabled=bool(self_repair_raw.get("enabled", False)),
        ),
        openai=OpenAIConfig(
            api_key_env=raw["openai"]["api_key_env"],
            model=raw["openai"]["model"],
            max_output_tokens=int(raw["openai"]["max_output_tokens"]),
            temperature=float(raw["openai"]["temperature"]),
            parallel_tool_calls=bool(raw["openai"]["parallel_tool_calls"]),
            prompt_cache_key=raw["openai"].get("prompt_cache_key"),
            prompt_cache_retention=raw["openai"].get("prompt_cache_retention"),
            store=bool(raw["openai"].get("store", True)),
            reasoning=(
                ReasoningConfig(
                    effort=reasoning_raw.get("effort"),
                    summary=reasoning_raw.get("summary"),
                )
                if reasoning_raw is not None
                else None
            ),
            stream_idle_timeout_seconds=float(
                raw["openai"].get("stream_idle_timeout_seconds", 90.0)
            ),
        ),
        parallel=parallel_config,
        memory=_parse_memory_config(raw.get("memory") or {}),
        mcp=_parse_mcp_config(raw.get("mcp") or {}),
        active_provider=active_provider,
        openrouter=openrouter_cfg,
        claude=claude_cfg,
        default_lead=str(raw.get("default_lead") or "lead").strip() or "lead",
    )


def _parse_mcp_config(raw: dict[str, Any]) -> MCPConfig:
    """Parse the top-level ``mcp:`` server registry.

    Args:
        raw: Parsed YAML mapping under ``mcp:``.

    Returns:
        Parsed MCP configuration. Disabled servers are omitted.

    Raises:
        ValueError: If an enabled server lacks a URL or a valid mapping shape.
    """

    enabled = bool(raw.get("enabled", False))
    servers_raw = raw.get("servers") or {}
    if not servers_raw:
        return MCPConfig(enabled=enabled, servers=())

    server_items: list[tuple[str | None, dict[str, Any]]] = []
    if isinstance(servers_raw, dict):
        for label, server_raw in servers_raw.items():
            if not isinstance(server_raw, dict):
                raise ValueError(f"mcp.servers.{label} must be a mapping.")
            server_items.append((str(label), server_raw))
    elif isinstance(servers_raw, list):
        for idx, server_raw in enumerate(servers_raw):
            if not isinstance(server_raw, dict):
                raise ValueError(f"mcp.servers[{idx}] must be a mapping.")
            server_items.append((None, server_raw))
    else:
        raise ValueError("mcp.servers must be a mapping or list.")

    servers: list[MCPServerConfig] = []
    for fallback_label, server_raw in server_items:
        if server_raw.get("enabled", True) is False:
            continue
        label = str(server_raw.get("label") or fallback_label or "").strip()
        if not label:
            raise ValueError("mcp.servers entry must define `label`.")
        server_url = str(
            server_raw.get("url") or server_raw.get("server_url") or ""
        ).strip()
        command = str(server_raw.get("command") or "").strip()
        if server_url and command:
            raise ValueError(
                f"mcp.servers.{label} must define either `url` or `command`, not both."
            )
        if enabled and not server_url and not command:
            raise ValueError(f"mcp.servers.{label} must define `url` or `command`.")
        if not server_url and not command:
            continue
        transport = "http" if server_url else "stdio"
        headers = server_raw.get("headers") or {}
        header_envs = server_raw.get("header_envs") or {}
        env = server_raw.get("env") or {}
        if not isinstance(headers, dict):
            raise ValueError(f"mcp.servers.{label}.headers must be a mapping.")
        if not isinstance(header_envs, dict):
            raise ValueError(f"mcp.servers.{label}.header_envs must be a mapping.")
        if not isinstance(env, dict):
            raise ValueError(f"mcp.servers.{label}.env must be a mapping.")
        servers.append(
            MCPServerConfig(
                label=label,
                server_url=server_url or None,
                server_description=server_raw.get("description")
                or server_raw.get("server_description"),
                transport=transport,
                command=command or None,
                args=_tuple_of_strings(server_raw.get("args")),
                env={str(key): str(value) for key, value in env.items()},
                cwd=server_raw.get("cwd"),
                allowed_tools=_tuple_of_strings(server_raw.get("allowed_tools")),
                require_approval=server_raw.get("require_approval", "never"),
                providers=tuple(
                    provider.lower()
                    for provider in _tuple_of_strings(server_raw.get("providers"))
                ),
                agents=_tuple_of_strings(server_raw.get("agents")),
                headers={str(key): str(value) for key, value in headers.items()},
                header_envs={
                    str(key): str(value) for key, value in header_envs.items()
                },
                request_timeout_seconds=float(
                    server_raw.get("request_timeout_seconds", 30.0)
                ),
            )
        )
    return MCPConfig(enabled=enabled, servers=tuple(servers))


def _tuple_of_strings(raw: Any) -> tuple[str, ...]:
    """Normalize a scalar/list YAML value into a tuple of non-empty strings."""

    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list | tuple):
        values = list(raw)
    else:
        raise ValueError(f"Expected string or list of strings, got {type(raw).__name__}.")
    return tuple(str(value).strip() for value in values if str(value).strip())


def _parse_memory_config(raw: dict[str, Any]) -> MemoryConfig:
    """Parse the top-level `memory:` YAML block into a :class:`MemoryConfig`.

    Sub-blocks that are absent fall back to their dataclass defaults. A partial
    sub-block fills in only the fields it lists; everything else uses defaults.
    The ``QDRANT_URL`` environment variable overrides ``memory.qdrant.url``
    after YAML parse; empty or unset env vars do not clobber a YAML value.

    Args:
        raw: Parsed YAML mapping under ``memory:`` (possibly empty).

    Returns:
        A fully-populated ``MemoryConfig``.
    """

    qdrant_raw: dict[str, Any] = raw.get("qdrant") or {}
    embedding_raw: dict[str, Any] = raw.get("embedding") or {}
    chunking_raw: dict[str, Any] = raw.get("chunking") or {}
    retrieval_raw: dict[str, Any] = raw.get("retrieval") or {}
    trigger_raw: dict[str, Any] = raw.get("trigger") or {}

    url = qdrant_raw.get("url")
    env_url = os.environ.get("QDRANT_URL")
    if env_url is not None and env_url.strip():
        url = env_url

    qdrant = MemoryQdrantConfig(
        url=url,
        api_key_env=qdrant_raw.get("api_key_env", "QDRANT_API_KEY"),
        collection_name=qdrant_raw.get("collection_name", "feather_memory"),
        embedding_dims=int(qdrant_raw.get("embedding_dims", 3072)),
        hnsw_m=int(qdrant_raw.get("hnsw_m", 32)),
        hnsw_ef_construct=int(qdrant_raw.get("hnsw_ef_construct", 256)),
        hnsw_ef_search=int(qdrant_raw.get("hnsw_ef_search", 128)),
        hnsw_full_scan_threshold=int(qdrant_raw.get("hnsw_full_scan_threshold", 10_000)),
        indexing_threshold=int(qdrant_raw.get("indexing_threshold", 20_000)),
        default_segment_number=int(qdrant_raw.get("default_segment_number", 2)),
        on_disk_vectors=bool(qdrant_raw.get("on_disk_vectors", False)),
        on_disk_payload=bool(qdrant_raw.get("on_disk_payload", False)),
        prefer_grpc=bool(qdrant_raw.get("prefer_grpc", False)),
        request_timeout_s=float(qdrant_raw.get("request_timeout_s", 15.0)),
        quantization=qdrant_raw.get("quantization"),
    )
    embedding = MemoryEmbeddingConfig(
        provider=embedding_raw.get("provider", "gemini"),
        model=embedding_raw.get("model", "gemini-embedding-2-preview"),
        output_dimensionality=int(embedding_raw.get("output_dimensionality", 3072)),
        task_type_document=embedding_raw.get("task_type_document", "RETRIEVAL_DOCUMENT"),
        task_type_query=embedding_raw.get("task_type_query", "RETRIEVAL_QUERY"),
        normalize_reduced_dims=bool(embedding_raw.get("normalize_reduced_dims", True)),
        request_timeout_s=float(embedding_raw.get("request_timeout_s", 30.0)),
        max_retries=int(embedding_raw.get("max_retries", 3)),
        retry_backoff_s=float(embedding_raw.get("retry_backoff_s", 1.5)),
    )
    chunking = MemoryChunkingConfig(
        chunk_size_tokens=int(chunking_raw.get("chunk_size_tokens", 1000)),
        chunk_overlap_tokens=int(chunking_raw.get("chunk_overlap_tokens", 100)),
        tokenizer=chunking_raw.get("tokenizer", "tiktoken"),
        tokenizer_encoding=chunking_raw.get("tokenizer_encoding", "o200k_base"),
    )
    retrieval = MemoryRetrievalConfig(
        enabled=bool(retrieval_raw.get("enabled", True)),
        top_k_prompt_injection=int(retrieval_raw.get("top_k_prompt_injection", 5)),
        top_k_tool=int(retrieval_raw.get("top_k_tool", 10)),
        score_threshold=float(retrieval_raw.get("score_threshold", 0.5)),
        classifier_top_k=int(retrieval_raw.get("classifier_top_k", 3)),
        classifier_score_threshold=float(
            retrieval_raw.get("classifier_score_threshold", 0.75)
        ),
        retrieval_timeout_s=float(retrieval_raw.get("retrieval_timeout_s", 2.0)),
        query_builder_enabled=bool(retrieval_raw.get("query_builder_enabled", True)),
        query_builder_recent_messages=int(
            retrieval_raw.get("query_builder_recent_messages", 8)
        ),
    )
    trigger = MemoryTriggerConfig(
        enabled=bool(trigger_raw.get("enabled", True)),
        trigger_turns=int(trigger_raw.get("trigger_turns", 10)),
        skip_compact_messages=bool(trigger_raw.get("skip_compact_messages", True)),
        background=bool(trigger_raw.get("background", True)),
        shutdown_timeout_s=float(trigger_raw.get("shutdown_timeout_s", 30.0)),
        max_concurrent_extractions_per_session=int(
            trigger_raw.get("max_concurrent_extractions_per_session", 1)
        ),
    )
    operations_raw = raw.get("operations") or {}
    extraction_raw = operations_raw.get("extraction") or raw.get("extraction") or {}
    classification_raw = (
        operations_raw.get("classification") or raw.get("classification") or {}
    )
    query_builder_raw = (
        operations_raw.get("query_builder") or raw.get("query_builder") or {}
    )
    global _FLAT_MEMORY_OPS_WARNED
    if not operations_raw and any(
        raw.get(k) for k in ("extraction", "classification", "query_builder")
    ):
        if not _FLAT_MEMORY_OPS_WARNED:
            import logging

            logging.getLogger(__name__).warning(
                "memory.{extraction,classification,query_builder} flat shape is "
                "deprecated; move under memory.operations.{...} (loader still "
                "accepts both for now)."
            )
            _FLAT_MEMORY_OPS_WARNED = True
    extraction = _parse_operation_model(
        extraction_raw, default_max=2000, default_temp=0.1
    )
    classification = _parse_operation_model(
        classification_raw, default_max=2000, default_temp=0.1
    )
    query_builder = _parse_operation_model(
        query_builder_raw, default_max=2000, default_temp=0.1
    )
    return MemoryConfig(
        enabled=bool(raw.get("enabled", False)),
        qdrant=qdrant,
        embedding=embedding,
        chunking=chunking,
        retrieval=retrieval,
        trigger=trigger,
        extraction=extraction,
        classification=classification,
        query_builder=query_builder,
    )


def _parse_operation_model(
    raw: dict[str, Any], *, default_max: int, default_temp: float
) -> MemoryOperationModelConfig:
    """Parse a per-operation model-override block (extraction/classification/query_builder).

    Args:
        raw: Parsed YAML mapping for the block (possibly empty).
        default_max: Fallback ``max_output_tokens`` when absent.
        default_temp: Fallback ``temperature`` when absent.

    Returns:
        A ``MemoryOperationModelConfig``; ``model=None`` means inherit the
        agent's current conversation model at call-time.
    """

    raw_provider = raw.get("provider")
    return MemoryOperationModelConfig(
        model=raw.get("model"),
        max_output_tokens=int(raw.get("max_output_tokens", default_max)),
        temperature=float(raw.get("temperature", default_temp)),
        provider=(str(raw_provider).strip().lower() if raw_provider else None),
    )


def load_agent_config(
    root: Path,
    agent_name: str,
    paths: "FeatherPaths | None" = None,
) -> AgentConfig:
    """Load one agent config with layered overrides.

    Resolution order, first hit wins:

    1. Project-staged ``<root>/config/agents/<name>.yaml`` if present.
    2. User-global override at ``paths.global_agents_dir/<name>.yaml``
       when ``paths`` is provided.
    3. Packaged default bundled in the wheel.

    Args:
        root: Working directory whose ``config/agents/`` (if present) is
            checked first.
        agent_name: Agent config name without extension.
        paths: Optional :class:`feather.paths.FeatherPaths` for the
            global override layer.

    Returns:
        Parsed agent config.

    Raises:
        FileNotFoundError: If no source provides this agent name.
    """

    raw = _resolve_agent_yaml(root, agent_name, paths=paths)
    prompt_modules = raw.get("prompt_modules")
    if prompt_modules is None:
        legacy_prompt_module = raw.get("system_prompt_module")
        if legacy_prompt_module is None:
            raise ValueError(
                f"Agent config `{agent_name}` must define `prompt_modules` or `system_prompt_module`."
            )
        prompt_modules = [legacy_prompt_module]
    provider_raw = raw.get("provider")
    reasoning_raw = raw.get("reasoning")
    reasoning_cfg: ReasoningConfig | None = None
    if reasoning_raw is not None:
        reasoning_cfg = ReasoningConfig(
            effort=reasoning_raw.get("effort"),
            summary=reasoning_raw.get("summary"),
        )
    temperature_raw = raw.get("temperature")
    max_output_tokens_raw = raw.get("max_output_tokens")
    return AgentConfig(
        name=raw["name"],
        role=raw["role"],
        personality=raw["personality"],
        prompt_modules=list(prompt_modules),
        registered_tools=list(raw["registered_tools"]),
        memory_enabled=bool(raw.get("memory_enabled", False)),
        description=str(raw.get("description") or "").strip(),
        inline_prompt=str(raw.get("inline_prompt") or "").strip(),
        provider=(str(provider_raw).strip().lower() if provider_raw else None),
        model=raw.get("model") or None,
        temperature=(float(temperature_raw) if temperature_raw is not None else None),
        max_output_tokens=(
            int(max_output_tokens_raw) if max_output_tokens_raw is not None else None
        ),
        reasoning=reasoning_cfg,
        soul=str(raw.get("soul") or "").strip(),
        color=(str(raw["color"]).strip() if raw.get("color") else None),
        emoji=(str(raw["emoji"]).strip() if raw.get("emoji") else None),
        capabilities={
            str(key): bool(value)
            for key, value in (raw.get("capabilities") or {}).items()
        },
    )


def _resolve_agent_yaml(
    root: Path,
    agent_name: str,
    *,
    paths: "FeatherPaths | None",
) -> dict[str, Any]:
    """Find an agent YAML across project, global, and packaged sources.

    Resolution order:

    1. Project-staged ``<root>/config/agents/<name>.yaml`` if present —
       returned as the **full replacement** (team-shared explicit config,
       same semantics as ``<root>/config/app.yaml``).
    2. Otherwise, start from the packaged default and **deep-merge** any
       ``<global>/config/agents/<name>.yaml`` overlay on top. This lets
       ``/config set agents.Lead.provider openai`` write a sparse global
       file with just one key and still load correctly — without the
       deep-merge, the partial overlay would shadow the packaged default
       entirely and fail validation for missing required fields like
       ``prompt_modules``.
    """

    project_path = root / "config" / "agents" / f"{agent_name}.yaml"
    if project_path.exists():
        return _read_yaml(project_path)
    base: dict[str, Any] = {}
    if has_packaged_agent(agent_name):
        base = yaml.safe_load(packaged_agent_yaml_text(agent_name)) or {}
    if paths is not None:
        global_path = paths.global_agents_dir / f"{agent_name}.yaml"
        if global_path.exists():
            overlay = _read_yaml(global_path)
            base = _deep_merge(base, overlay)
    if base:
        return base
    raise FileNotFoundError(
        f"Agent config '{agent_name}' not found in project ({project_path}), "
        "global, or packaged sources"
    )


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` over ``base`` without mutating either.

    Dict-typed leaves are merged element-wise; every other type (lists,
    scalars) is replaced wholesale by the overlay. The overlay always
    wins, so the per-user ``app.yaml`` can clear or override values from
    the packaged default.
    """

    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file from disk.

    Args:
        path: File to load.

    Returns:
        Parsed mapping.
    """

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
