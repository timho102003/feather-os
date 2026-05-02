"""Tests for the read-file tool."""

from pathlib import Path

from feather.models import ToolExecutionContext
from feather.tools.read_file_tool import ReadFileTool


async def test_read_file_tool_reads_selected_line_ranges(tmp_path: Path) -> None:
    """The read-file tool should support line slicing."""

    file_path = tmp_path / ".feather" / "tmp" / "bash" / "demo.output"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    tool = ReadFileTool(tmp_path)
    result = await tool.execute(
        {
            "path": ".feather/tmp/bash/demo.output",
            "start_line": 2,
            "end_line": 3,
            "max_chars": None,
        },
        ToolExecutionContext(session_id="session-1", agent_name="Lead"),
    )

    assert "file: .feather/tmp/bash/demo.output" in result.output
    assert "2: two" in result.output
    assert "3: three" in result.output
    assert "1: one" not in result.output


def test_read_file_tool_schema_is_openai_strict_compatible(tmp_path: Path) -> None:
    """All read-file tool properties should be required for strict tool calling."""

    tool = ReadFileTool(tmp_path)
    properties = tool.parameters_schema["properties"]

    assert sorted(tool.parameters_schema["required"]) == sorted(properties.keys())
    assert properties["start_line"]["type"] == ["integer", "null"]
    assert properties["end_line"]["type"] == ["integer", "null"]
    assert properties["max_chars"]["type"] == ["integer", "null"]
