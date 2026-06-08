"""Pre-completion guard: don't claim completion while sub-agent tasks
are still in `running` status.

The Lead's run loop currently treats "no tool_calls in this turn" as a
completion signal and returns AgentRunResult(status=COMPLETED). This is
wrong when sub-agent tasks the Lead spawned are still active — the Lead
silently claims success and the user sees a stale `LIVE` row that never
transitions.

The guard intercepts the completion exit, looks up tasks where the
agent's session is the `lead_session_id` and status is `RUNNING`, and
if any are sub-agent-attributed (responsible_session_id != self), it
injects a synthetic system message into the next iteration so the agent
has a chance to address them. The guard fires at most once per `.run()`
to avoid an infinite loop on a genuinely-hung sub-agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feather.core.agent.base import BaseAgent
from feather.core.agent.prompt_builder import PromptBuilder
from feather.models import (
    AgentConfig,
    AgentOutcome,
    ModelTurn,
    ProviderRequestConfig,
    RuntimeEvent,
    TaskStatus,
    ToolCall,
)
from feather.providers.base import BaseLLMProvider
from feather.skills.catalog import SkillCatalog
from feather.storage.session_store import SessionStore
from feather.storage.task_store import TaskStore
from feather.storage.tool_output_store import ToolOutputStore
from feather.tools.registry import ToolRegistry


class _RecordingProvider(BaseLLMProvider):
    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({"input_items": input_items})
        return self._turns.pop(0)


class _LeadAgent(BaseAgent):
    pass


async def _build(
    tmp_path: Path, *, turns: list[ModelTurn]
) -> tuple[_LeadAgent, _RecordingProvider, TaskStore, SessionStore]:
    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    task_store = TaskStore(tmp_path / "feather.db")
    await task_store.initialize()
    provider = _RecordingProvider(turns)
    prompt_builder = PromptBuilder(
        SkillCatalog(tmp_path / ".feather" / "skills"), ToolRegistry([])
    )
    agent = _LeadAgent(
        agent_config=AgentConfig(
            name="Lead",
            role="lead",
            personality="x",
            prompt_modules=[
                "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
            ],
            registered_tools=[],
        ),
        prompt_builder=prompt_builder,
        provider=provider,
        session_store=session_store,
        tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
        tool_registry=ToolRegistry([]),
        task_store=task_store,
    )
    return agent, provider, task_store, session_store


# ---------------------------------------------------------------------------


async def test_guard_injects_when_running_subagent_task_outstanding(
    tmp_path: Path,
) -> None:
    """If a sub-agent task is `running`, the Lead must not finalise on the
    first text-only turn — it should see one synthetic system message and
    get one more chance to act before the run completes."""

    # Two model turns:
    #   1. text only — would normally finalise → guard kicks in
    #   2. text only — second attempt, guard already fired, allow completion
    turns = [
        ModelTurn(response_id="r1", output_text="all done!", tool_calls=[]),
        ModelTurn(response_id="r2", output_text="acknowledged.", tool_calls=[]),
    ]
    agent, provider, task_store, session_store = await _build(tmp_path, turns=turns)
    try:
        session_id = await agent.create_session()
        # Outstanding sub-agent task owned by this Lead session.
        await task_store.create_task(
            lead_session_id=session_id,
            title="Research X",
            responsible_agent_name="research",
            responsible_session_id="sub-agent-session-id-xyz",
            status=TaskStatus.RUNNING,
        )

        events: list[RuntimeEvent] = []

        def on_event(e: RuntimeEvent) -> None:
            events.append(e)

        result = await agent.run(session_id, "go", on_event)

        # Should have made TWO provider calls — guard injected a second turn.
        assert len(provider.calls) == 2
        # Second call's input_items must include a hint mentioning the running task.
        second_input_text = str(provider.calls[1]["input_items"])
        assert "Research X" in second_input_text or "running" in second_input_text
        # Final outcome is COMPLETED with the SECOND turn's text — the guard
        # didn't block completion forever.
        assert result.status == AgentOutcome.COMPLETED
        assert result.assistant_text == "acknowledged."
        # An observable event fired so the CLI/TUI can flag this to the user.
        kinds = [e.kind for e in events]
        assert "completion_guard_injected" in kinds
    finally:
        await task_store.close()
        await session_store.close()


async def test_guard_does_not_fire_when_no_outstanding_tasks(tmp_path: Path) -> None:
    """Clean completion path is untouched."""

    turns = [ModelTurn(response_id="r1", output_text="done", tool_calls=[])]
    agent, provider, task_store, session_store = await _build(tmp_path, turns=turns)
    try:
        session_id = await agent.create_session()
        # No outstanding tasks created.
        result = await agent.run(session_id, "go")
        assert len(provider.calls) == 1
        assert result.status == AgentOutcome.COMPLETED
        assert result.assistant_text == "done"
    finally:
        await task_store.close()
        await session_store.close()


async def test_guard_ignores_self_attributed_running_tasks(tmp_path: Path) -> None:
    """Tasks where the agent itself is the `responsible_session_id` are
    work the agent is already executing inline — flagging them would
    cause a self-loop, so they are ignored."""

    turns = [ModelTurn(response_id="r1", output_text="done", tool_calls=[])]
    agent, provider, task_store, session_store = await _build(tmp_path, turns=turns)
    try:
        session_id = await agent.create_session()
        await task_store.create_task(
            lead_session_id=session_id,
            title="self task",
            responsible_agent_name="Lead",
            responsible_session_id=session_id,    # responsible == self
            status=TaskStatus.RUNNING,
        )
        result = await agent.run(session_id, "go")
        assert len(provider.calls) == 1
        assert result.status == AgentOutcome.COMPLETED
    finally:
        await task_store.close()
        await session_store.close()


async def test_guard_ignores_terminated_tasks(tmp_path: Path) -> None:
    """`failed`, `completed_with_*`, and `stopped` tasks are not flagged."""

    turns = [ModelTurn(response_id="r1", output_text="done", tool_calls=[])]
    agent, provider, task_store, session_store = await _build(tmp_path, turns=turns)
    try:
        session_id = await agent.create_session()
        await task_store.create_task(
            lead_session_id=session_id,
            title="finished",
            responsible_session_id="sub-x",
            status=TaskStatus.COMPLETED_WITH_ARTIFACTS,
        )
        await task_store.create_task(
            lead_session_id=session_id,
            title="failed",
            responsible_session_id="sub-y",
            status=TaskStatus.FAILED,
        )
        result = await agent.run(session_id, "go")
        assert len(provider.calls) == 1
        assert result.status == AgentOutcome.COMPLETED
    finally:
        await task_store.close()
        await session_store.close()


async def test_guard_fires_at_most_once_per_run(tmp_path: Path) -> None:
    """If the running task is still running on the second turn (e.g. it's
    genuinely hung), the guard must NOT fire a second time — otherwise the
    run loops forever."""

    turns = [
        ModelTurn(response_id="r1", output_text="all done!", tool_calls=[]),
        ModelTurn(response_id="r2", output_text="still done!", tool_calls=[]),
    ]
    agent, provider, task_store, session_store = await _build(tmp_path, turns=turns)
    try:
        session_id = await agent.create_session()
        await task_store.create_task(
            lead_session_id=session_id,
            title="hung sub-agent",
            responsible_session_id="hung-sess",
            status=TaskStatus.RUNNING,
        )
        # Note: the task stays `running` for the entire run — the guard
        # would loop forever if it didn't track injection per-run.
        result = await agent.run(session_id, "go")
        assert len(provider.calls) == 2
        assert result.status == AgentOutcome.COMPLETED
    finally:
        await task_store.close()
        await session_store.close()


async def test_guard_skipped_when_task_store_not_provided(tmp_path: Path) -> None:
    """Sub-agents (no task_store wired) must complete normally."""

    (tmp_path / ".feather" / "skills").mkdir(parents=True)
    session_store = SessionStore(tmp_path / "feather.db")
    await session_store.initialize()
    try:
        provider = _RecordingProvider(
            [ModelTurn(response_id="r1", output_text="done", tool_calls=[])]
        )
        agent = _LeadAgent(
            agent_config=AgentConfig(
                name="Sub",
                role="research",
                personality="x",
                prompt_modules=[
                    "feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT",
                ],
                registered_tools=[],
            ),
            prompt_builder=PromptBuilder(
                SkillCatalog(tmp_path / ".feather" / "skills"),
                ToolRegistry([]),
            ),
            provider=provider,
            session_store=session_store,
            tool_output_store=ToolOutputStore(tmp_path, ".feather/tmp"),
            tool_registry=ToolRegistry([]),
            # task_store omitted — sub-agent
        )
        session_id = await agent.create_session()
        result = await agent.run(session_id, "go")
        assert len(provider.calls) == 1
        assert result.status == AgentOutcome.COMPLETED
    finally:
        await session_store.close()
