"""Tests for the ``manage_memory`` proactive CRUD tool.

The tool is a thin façade over :class:`MemoryService.proactive_*`. These
tests substitute a fake service so we can verify dispatch + validation +
error rendering without spinning up the full memory pipeline.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from feather.memory.enums import MemoryOp, MemoryOwner
from feather.memory.models import AppliedOp
from feather.models import ToolExecutionContext
from feather.tools.manage_memory_tool import ManageMemoryTool


_CTX = ToolExecutionContext(session_id="sess-test", agent_name="Lead")


class _FakeService:
    """Drop-in test double for ``MemoryService`` exposing only proactive_*.

    Each method records its kwargs and returns whatever the test stages.
    """

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.create_response: AppliedOp | None = None
        self.update_response: AppliedOp | None = None
        self.delete_response: AppliedOp | None = None
        self.create_raises: BaseException | None = None
        self.update_raises: BaseException | None = None
        self.delete_raises: BaseException | None = None

    async def proactive_create(
        self,
        *,
        content: str,
        purpose: str,
        session_id: str,
        owner: MemoryOwner = MemoryOwner.USER,
    ) -> AppliedOp:
        self.create_calls.append(
            {
                "content": content,
                "purpose": purpose,
                "session_id": session_id,
                "owner": owner,
            }
        )
        if self.create_raises is not None:
            raise self.create_raises
        assert self.create_response is not None
        return self.create_response

    async def proactive_update(
        self,
        *,
        target_query: str,
        content: str,
        purpose: str,
        session_id: str,
        owner: MemoryOwner = MemoryOwner.USER,
        score_threshold: float = 0.4,
    ) -> AppliedOp:
        self.update_calls.append(
            {
                "target_query": target_query,
                "content": content,
                "purpose": purpose,
                "session_id": session_id,
                "owner": owner,
                "score_threshold": score_threshold,
            }
        )
        if self.update_raises is not None:
            raise self.update_raises
        assert self.update_response is not None
        return self.update_response

    async def proactive_delete(
        self,
        *,
        target_query: str,
        session_id: str,
        owner: MemoryOwner = MemoryOwner.USER,
        score_threshold: float = 0.4,
    ) -> AppliedOp:
        self.delete_calls.append(
            {
                "target_query": target_query,
                "session_id": session_id,
                "owner": owner,
                "score_threshold": score_threshold,
            }
        )
        if self.delete_raises is not None:
            raise self.delete_raises
        assert self.delete_response is not None
        return self.delete_response


def _tool(
    service: _FakeService, *, session_id: str | None = "sess-x"
) -> ManageMemoryTool:
    return ManageMemoryTool(
        service=service,  # type: ignore[arg-type]
        session_id_resolver=lambda: session_id,
    )


# -- Schema / metadata --------------------------------------------------------


def test_tool_metadata() -> None:
    tool = _tool(_FakeService())
    assert tool.name == "manage_memory"
    desc_lower = tool.description.lower()
    # The description must steer the lead toward direct user requests so
    # the tool isn't used for ambient observations (those are auto-extracted).
    assert "remember" in desc_lower or "create" in desc_lower
    # OpenAI strict mode: every property in `required`.
    assert set(tool.parameters_schema["required"]) == set(
        tool.parameters_schema["properties"].keys()
    )
    assert tool.parameters_schema["additionalProperties"] is False
    assert "operation" in tool.parameters_schema["properties"]


def test_operation_enum_lists_all_three_ops() -> None:
    tool = _tool(_FakeService())
    op_schema = tool.parameters_schema["properties"]["operation"]
    assert set(op_schema["enum"]) == {"CREATE", "UPDATE", "DELETE"}


# -- CREATE -------------------------------------------------------------------


async def test_create_dispatches_to_service_and_renders_group_id() -> None:
    service = _FakeService()
    new_id = uuid4()
    service.create_response = AppliedOp(
        op=MemoryOp.CREATE, group_id=new_id, chunk_count=1
    )
    tool = _tool(service, session_id="sess-42")

    result = await tool.execute(
        {
            "operation": "CREATE",
            "content": "The user prefers Python.",
            "purpose": "route language-specific suggestions to Python.",
            "target_query": None,
        },
        _CTX,
    )

    assert len(service.create_calls) == 1
    call = service.create_calls[0]
    assert call["session_id"] == "sess-42"
    assert call["content"] == "The user prefers Python."
    assert call["purpose"] == "route language-specific suggestions to Python."
    assert "created" in result.output.lower()
    assert str(new_id) in result.output


async def test_create_rejects_missing_content() -> None:
    tool = _tool(_FakeService())
    result = await tool.execute(
        {
            "operation": "CREATE",
            "content": None,
            "purpose": "p",
            "target_query": None,
        },
        _CTX,
    )
    assert "content" in result.output.lower()
    assert any(
        word in result.output.lower()
        for word in ("required", "requires", "missing")
    )


async def test_create_rejects_missing_purpose() -> None:
    tool = _tool(_FakeService())
    result = await tool.execute(
        {
            "operation": "CREATE",
            "content": "x",
            "purpose": None,
            "target_query": None,
        },
        _CTX,
    )
    assert "purpose" in result.output.lower()


async def test_create_surfaces_service_validation_errors() -> None:
    service = _FakeService()
    service.create_raises = ValueError("session has no user message")
    tool = _tool(service)
    result = await tool.execute(
        {
            "operation": "CREATE",
            "content": "x",
            "purpose": "y",
            "target_query": None,
        },
        _CTX,
    )
    assert "session has no user message" in result.output.lower() or "error" in result.output.lower()


# -- UPDATE -------------------------------------------------------------------


async def test_update_dispatches_to_service() -> None:
    service = _FakeService()
    matched = uuid4()
    service.update_response = AppliedOp(
        op=MemoryOp.UPDATE, group_id=matched, chunk_count=2
    )
    tool = _tool(service)

    result = await tool.execute(
        {
            "operation": "UPDATE",
            "content": "user now uses Rust",
            "purpose": "language routing",
            "target_query": "the user's programming language",
        },
        _CTX,
    )
    assert len(service.update_calls) == 1
    call = service.update_calls[0]
    assert call["target_query"] == "the user's programming language"
    assert call["content"] == "user now uses Rust"
    assert "updated" in result.output.lower()
    assert str(matched) in result.output


async def test_update_returns_failed_message_when_service_no_match() -> None:
    service = _FakeService()
    service.update_response = AppliedOp.failed(
        MemoryOp.UPDATE, "no match for target_query='x' above 0.4"
    )
    tool = _tool(service)
    result = await tool.execute(
        {
            "operation": "UPDATE",
            "content": "c",
            "purpose": "p",
            "target_query": "x",
        },
        _CTX,
    )
    assert "no match" in result.output.lower()
    # Hint the agent toward CREATE so it doesn't loop on UPDATE.
    assert "create" in result.output.lower()


async def test_update_rejects_missing_target_query() -> None:
    tool = _tool(_FakeService())
    result = await tool.execute(
        {
            "operation": "UPDATE",
            "content": "c",
            "purpose": "p",
            "target_query": None,
        },
        _CTX,
    )
    assert "target_query" in result.output.lower()


# -- DELETE -------------------------------------------------------------------


async def test_delete_dispatches_to_service() -> None:
    service = _FakeService()
    matched = uuid4()
    service.delete_response = AppliedOp(
        op=MemoryOp.DELETE, group_id=matched, chunk_count=0
    )
    tool = _tool(service)
    result = await tool.execute(
        {
            "operation": "DELETE",
            "content": None,
            "purpose": None,
            "target_query": "the user's cafe preference",
        },
        _CTX,
    )
    assert len(service.delete_calls) == 1
    call = service.delete_calls[0]
    assert call["target_query"] == "the user's cafe preference"
    assert "deleted" in result.output.lower() or "forgot" in result.output.lower()
    assert str(matched) in result.output


async def test_delete_returns_failed_message_when_service_no_match() -> None:
    service = _FakeService()
    service.delete_response = AppliedOp.failed(
        MemoryOp.DELETE, "no match for target_query='x' above 0.4"
    )
    tool = _tool(service)
    result = await tool.execute(
        {
            "operation": "DELETE",
            "content": None,
            "purpose": None,
            "target_query": "x",
        },
        _CTX,
    )
    assert "no match" in result.output.lower()


async def test_delete_rejects_missing_target_query() -> None:
    tool = _tool(_FakeService())
    result = await tool.execute(
        {
            "operation": "DELETE",
            "content": None,
            "purpose": None,
            "target_query": "  ",
        },
        _CTX,
    )
    assert "target_query" in result.output.lower()


# -- Cross-cutting ------------------------------------------------------------


async def test_invalid_operation_rejected() -> None:
    tool = _tool(_FakeService())
    result = await tool.execute(
        {
            "operation": "PURGE",
            "content": None,
            "purpose": None,
            "target_query": "x",
        },
        _CTX,
    )
    assert "invalid" in result.output.lower() or "expected" in result.output.lower()


async def test_returns_error_when_no_active_session() -> None:
    service = _FakeService()
    tool = _tool(service, session_id=None)
    result = await tool.execute(
        {
            "operation": "CREATE",
            "content": "x",
            "purpose": "y",
            "target_query": None,
        },
        _CTX,
    )
    assert "session" in result.output.lower()
    assert service.create_calls == []  # never dispatched
