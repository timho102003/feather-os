"""Tests for file-backed tool output storage."""

from pathlib import Path

from feather.storage.tool_output_store import ToolOutputStore


async def test_tool_output_store_writes_tool_output_files_under_temp_directory(tmp_path: Path) -> None:
    """Tool outputs should be written under `.feather/tmp/<tool_name>/`."""

    store = ToolOutputStore(tmp_path, ".feather/tmp")
    artifact = await store.write("web_search", "hello world")

    assert artifact.file_ref.startswith(".feather/tmp/web_search/")
    assert artifact.file_ref.endswith(".output")
    assert artifact.reference_text == f"web_search tool call output content file: {artifact.file_ref}"
    assert (tmp_path / artifact.file_ref).read_text(encoding="utf-8") == "hello world"
