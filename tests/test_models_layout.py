"""Pin the public surface of the split ``feather.models`` package.

``models.py`` became a package (config_models / records / runtime_models) behind
a full re-export ``__init__``. Every symbol importers relied on must remain
importable from ``feather.models`` unchanged — this is the canary for the split.
"""

from __future__ import annotations

import importlib

# The complete pre-split public surface (real model classes + EventHandler).
# Stdlib leaks the old module exposed (datetime/Enum/field/...) are intentionally
# NOT part of the contract — nothing imported them from feather.models.
_EXPECTED = frozenset(
    {
        # config_models
        "AgentConfig",
        "AppConfig",
        "ClaudeConfig",
        "ClaudeThinkingConfig",
        "CompactionConfig",
        "DatabaseConfig",
        "LoggingConfig",
        "MCPConfig",
        "MCPServerConfig",
        "OpenAIConfig",
        "OpenRouterConfig",
        "OpenRouterTracingConfig",
        "ParallelConfig",
        "ReasoningConfig",
        "SchedulerConfig",
        "SelfRepairConfig",
        "SkillsConfig",
        "StorageConfig",
        # records
        "AgentMessage",
        "AgentMessageStatus",
        "AttachmentKind",
        "AttachmentRecord",
        "CronJobRecord",
        "CronJobStatus",
        "CronScheduleType",
        "LoadedSkill",
        "MessageRole",
        "PendingAttachment",
        "PlanRecord",
        "PlanStatus",
        "SessionMessage",
        "SessionRecord",
        "SessionStatus",
        "SkillMetadata",
        "TaskEventRecord",
        "TaskOutputKind",
        "TaskOutputRecord",
        "TaskRecord",
        "TaskRunRecord",
        "TaskRunStatus",
        "TaskStatus",
        "WorkerHeartbeat",
        "WorkerStatus",
        # runtime_models
        "AgentOutcome",
        "AgentRunResult",
        "EventHandler",
        "EventKind",
        "ModelTurn",
        "ProviderRequestConfig",
        "RuntimeEvent",
        "ToolCall",
        "ToolExecutionContext",
        "ToolExecutionResult",
        "ToolOutputArtifact",
        "TraceContext",
    }
)


def test_every_pre_split_symbol_is_importable() -> None:
    models = importlib.import_module("feather.models")
    for name in _EXPECTED:
        assert hasattr(models, name), f"feather.models lost symbol: {name}"


def test_init_all_matches_expected_surface() -> None:
    models = importlib.import_module("feather.models")
    assert set(models.__all__) == _EXPECTED
    assert len(models.__all__) == len(set(models.__all__))  # no duplicates


def test_models_is_a_package_split_into_three_modules() -> None:
    models = importlib.import_module("feather.models")
    assert models.__file__.endswith("models/__init__.py")
    for sub in ("config_models", "records", "runtime_models"):
        mod = importlib.import_module(f"feather.models.{sub}")
        assert mod.__all__, f"{sub} must declare __all__"
        # Each submodule's surface is a subset of the aggregate.
        assert set(mod.__all__) <= _EXPECTED


def test_split_modules_partition_the_surface() -> None:
    """The three modules partition the surface with no overlap and no gaps."""

    from feather.models import config_models, records, runtime_models

    parts = [set(config_models.__all__), set(records.__all__), set(runtime_models.__all__)]
    union: set[str] = set()
    for part in parts:
        assert not (union & part), f"overlapping symbols: {union & part}"
        union |= part
    assert union == _EXPECTED
