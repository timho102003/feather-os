"""Tests for the bash tool."""

from pathlib import Path

from feather.models import ToolExecutionContext
from feather.tools.bash_tool import BashTool


async def test_bash_tool_runs_command_in_workspace(tmp_path: Path) -> None:
    """The bash tool should run a simple command and report the exit code."""

    tool = BashTool(tmp_path)
    result = await tool.execute(
        {"command": "printf 'hello'", "cwd": None, "timeout_ms": None, "max_output_chars": None},
        ToolExecutionContext(session_id="session-1", agent_name="Lead"),
    )

    assert "exit_code: 0" in result.output
    assert "stdout:\nhello" in result.output
    assert "cwd: ." in result.output


async def test_bash_tool_rejects_cwd_outside_workspace(tmp_path: Path) -> None:
    """The bash tool should not allow commands outside the workspace root."""

    tool = BashTool(tmp_path)

    try:
        await tool.execute(
            {"command": "pwd", "cwd": "../..", "timeout_ms": None, "max_output_chars": None},
            ToolExecutionContext(session_id="session-1", agent_name="Lead"),
        )
    except ValueError as exc:
        assert "inside the workspace" in str(exc)
    else:
        raise AssertionError("Expected bash tool to reject cwd outside the workspace.")


def test_bash_tool_schema_is_openai_strict_compatible(tmp_path: Path) -> None:
    """All bash tool properties should be required for strict OpenAI tool calling."""

    tool = BashTool(tmp_path)
    properties = tool.parameters_schema["properties"]

    assert sorted(tool.parameters_schema["required"]) == sorted(properties.keys())
    assert properties["cwd"]["type"] == ["string", "null"]
    assert properties["timeout_ms"]["type"] == ["integer", "null"]
    assert properties["max_output_chars"]["type"] == ["integer", "null"]
