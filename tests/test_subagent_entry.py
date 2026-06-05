"""Tests for the sub-agent subprocess entry point (exercised in-process)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feather.models import ModelTurn, ProviderRequestConfig
from feather.providers.base import BaseLLMProvider
from feather.subagent_entry import main, run_subagent_async
from feather.core.subagents.protocol import RESULT_BEGIN, RESULT_END


class FakeOneShotProvider(BaseLLMProvider):
    """Fake LLM provider that returns one canned turn without tool calls."""

    def __init__(self, final_text: str = "done") -> None:
        self._final_text = final_text
        self.call_count = 0

    async def complete(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        event_handler=None,
        request_config: ProviderRequestConfig | None = None,
    ) -> ModelTurn:
        self.call_count += 1
        return ModelTurn(
            response_id=f"resp-{self.call_count}",
            output_text=self._final_text,
            tool_calls=[],
            usage={"input_tokens": 100},
        )


_EXPLORE_YAML_NO_WEB = """name: Explore
role: explore
personality: Direct
prompt_modules:
  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT
  - feather.core.prompts.explore_agent_prompt:EXPLORE_AGENT_PROMPT
memory_enabled: false
registered_tools:
  - read_file
  - grep
"""


def _stage_repo(tmp_path: Path, *, roles: tuple[str, ...] = ("explore",)) -> None:
    """Stage the minimum config needed to boot a FeatherRuntime in tmp_path.

    The entry-point tests do not need web tools, so we write trimmed agent
    YAMLs that omit `web_search` / `web_fetch`. That sidesteps the Parallel
    client dependency without changing the behavior under test.
    """

    (tmp_path / "config" / "agents").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".feather" / "skills").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "app.yaml").write_text(
        """database:
  path: .feather/db/feather.db
storage:
  temp_directory: .feather/tmp
logging:
  path: .feather/logs/feather.log
  level: INFO
compaction:
  enabled: true
  trigger_ratio: 0.8
  context_window_tokens: 400000
  model:
  max_output_tokens: 2000
  temperature: 0.2
skills:
  directory: .feather/skills
scheduler:
  enabled: false
  poll_interval_seconds: 2
  failure_retry_seconds: 30
  max_due_jobs_per_tick: 10
openai:
  api_key_env: OPENAI_API_KEY
  model: gpt-5-mini
  max_output_tokens: 4000
  temperature: 1.0
  parallel_tool_calls: true
  store: true
memory:
  enabled: false
""",
        encoding="utf-8",
    )
    for role in roles:
        if role == "explore":
            (tmp_path / "config" / "agents" / "explore.yaml").write_text(
                _EXPLORE_YAML_NO_WEB, encoding="utf-8"
            )
        else:
            raise ValueError(f"Unsupported role fixture: {role}")


async def test_run_subagent_async_completes_with_fake_provider(tmp_path: Path) -> None:
    _stage_repo(tmp_path)

    envelope = await run_subagent_async(
        agent_name="explore",
        task_text="find usages of BaseAgent",
        parent_session_id="parent-xyz",
        root=tmp_path,
        provider_factory=lambda _cfg: FakeOneShotProvider("explore-done"),
    )

    # FakeOneShotProvider returns text with zero tool calls — exactly
    # the "wasted spawn" pattern the envelope builder now flags for
    # work roles (explore/research/validate). The smoke assertion is
    # that the envelope structure is right and the detection fired.
    assert envelope["status"] == "failed"
    assert envelope["agent_name"] == "explore"
    assert envelope["role"] == "explore"
    assert envelope["parent_session_id"] == "parent-xyz"
    assert envelope["assistant_text"] == "explore-done"
    assert envelope["total_tool_calls"] == 0
    assert envelope["error"] is not None and "wasted spawn" in envelope["error"]
    assert isinstance(envelope["session_id"], str) and envelope["session_id"]


async def test_run_subagent_async_rejects_lead_agent_name(tmp_path: Path) -> None:
    _stage_repo(tmp_path)

    try:
        await run_subagent_async(
            agent_name="lead",
            task_text="noop",
            parent_session_id="parent-xyz",
            root=tmp_path,
        )
    except ValueError as exc:
        assert "not dispatchable" in str(exc)
        return
    raise AssertionError("Expected ValueError for lead agent_name.")


async def test_run_subagent_async_rejects_invalid_name(tmp_path: Path) -> None:
    _stage_repo(tmp_path)

    try:
        await run_subagent_async(
            agent_name="../etc/passwd",
            task_text="noop",
            parent_session_id="parent-xyz",
            root=tmp_path,
        )
    except ValueError as exc:
        assert "Invalid sub-agent name" in str(exc)
        return
    raise AssertionError("Expected ValueError for path-traversal agent_name.")


def test_main_emits_envelope_and_exits_success(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _stage_repo(tmp_path)

    # Stage task file on disk so the CLI path exercises the real file loader.
    task_file = tmp_path / "task.txt"
    task_file.write_text("hello explorer", encoding="utf-8")

    # Force main() to use FakeOneShotProvider by patching the runtime helper.
    from feather.core.subagents import entry as subagent_entry_module

    real_run = subagent_entry_module.run_subagent_async

    async def patched_run(**kwargs: Any) -> dict[str, object]:
        return await real_run(
            **kwargs,
            provider_factory=lambda _cfg: FakeOneShotProvider("cli-done"),
        )

    monkeypatch.setattr(subagent_entry_module, "run_subagent_async", patched_run)

    exit_code = main(
        [
            "--agent-name",
            "explore",
            "--task-file",
            str(task_file),
            "--parent-session",
            "lead-123",
            "--root",
            str(tmp_path),
            "--keep-task-file",
        ]
    )
    captured = capsys.readouterr()
    # FakeOneShotProvider returns no tool calls on a work-role agent →
    # wasted-spawn detection forces status=failed and exit_code=1. The
    # envelope round-trip is still exercised correctly.
    assert exit_code == 1
    assert RESULT_BEGIN in captured.out
    assert RESULT_END in captured.out
    start = captured.out.index(RESULT_BEGIN) + len(RESULT_BEGIN)
    end = captured.out.index(RESULT_END)
    payload = json.loads(captured.out[start:end].strip())
    assert payload["status"] == "failed"
    assert payload["assistant_text"] == "cli-done"
    assert payload["total_tool_calls"] == 0
    assert "wasted spawn" in (payload["error"] or "")


def test_main_emits_envelope_when_task_file_missing(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "--agent-name",
            "explore",
            "--task-file",
            str(tmp_path / "does-not-exist.txt"),
            "--parent-session",
            "lead-123",
            "--root",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "failed to load task file" in captured.out


def test_main_emits_envelope_when_task_file_empty(tmp_path: Path, capsys) -> None:
    task_file = tmp_path / "empty.txt"
    task_file.write_text("   ", encoding="utf-8")
    exit_code = main(
        [
            "--agent-name",
            "explore",
            "--task-file",
            str(task_file),
            "--parent-session",
            "lead-123",
            "--root",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "failed to load task file" in captured.out
