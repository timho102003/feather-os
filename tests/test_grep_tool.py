"""Tests for the grep tool."""

from pathlib import Path

from feather.models import ToolExecutionContext
from feather.tools.grep_tool import GrepTool


async def test_grep_tool_finds_matches_in_workspace_files(tmp_path: Path) -> None:
    """The grep tool should return matching lines with file and line numbers."""

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "def main():\n    return 'needle'\n",
        encoding="utf-8",
    )
    (tmp_path / ".feather").mkdir()
    (tmp_path / ".feather" / "ignored.txt").write_text("needle", encoding="utf-8")

    tool = GrepTool(tmp_path)
    result = await tool.execute(
        {"pattern": "needle"},
        ToolExecutionContext(session_id="session-1", agent_name="Lead"),
    )

    assert "src/app.py:2:" in result.output
    assert "ignored.txt" not in result.output


def test_grep_tool_schema_is_openai_strict_compatible(tmp_path: Path) -> None:
    """All defined properties should be required for strict OpenAI tool calling."""

    tool = GrepTool(tmp_path)
    properties = tool.parameters_schema["properties"]

    assert sorted(tool.parameters_schema["required"]) == sorted(properties.keys())
    assert properties["path"]["type"] == ["string", "null"]
    assert properties["case_sensitive"]["type"] == ["boolean", "null"]
    assert properties["max_results"]["type"] == ["integer", "null"]
