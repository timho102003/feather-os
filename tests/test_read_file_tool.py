"""Tests for the read-file tool."""

from pathlib import Path

import pytest

from feather.models import ToolExecutionContext
from feather.paths import FeatherPaths
from feather.tools.read_file_tool import ReadFileTool


def _ctx() -> ToolExecutionContext:
    return ToolExecutionContext(session_id="session-1", agent_name="Lead")


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
        _ctx(),
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


async def test_read_file_rejects_paths_outside_workspace_without_paths(
    tmp_path: Path,
) -> None:
    """Without a FeatherPaths, the tool stays scoped to the workspace —
    `..` escapes and unrelated absolute paths must be rejected."""

    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("x", encoding="utf-8")

    tool = ReadFileTool(workspace)
    with pytest.raises(ValueError, match="must stay inside"):
        await tool.execute(
            {
                "path": "../outside/secret.txt",
                "start_line": None,
                "end_line": None,
                "max_chars": None,
            },
            _ctx(),
        )


async def test_read_file_allows_global_root_when_paths_provided(
    tmp_path: Path,
) -> None:
    """When a FeatherPaths is supplied, files anywhere under
    ``paths.global_root`` (typically ``~/.feather``) are readable —
    user.md, global agent YAMLs, global skills, the memory marker, etc."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"  # sibling of workspace, not a child
    paths = FeatherPaths(project_root=workspace, home=home)
    paths.ensure_global_dirs()

    user_md = paths.global_user_md
    user_md.write_text("# user persona\nrole: principal\n", encoding="utf-8")

    tool = ReadFileTool(workspace, paths=paths)
    result = await tool.execute(
        {
            "path": str(user_md),
            "start_line": None,
            "end_line": None,
            "max_chars": None,
        },
        _ctx(),
    )
    assert "role: principal" in result.output
    # Output should still surface a readable file label even when the
    # file lives outside workspace_root.
    assert "file:" in result.output


async def test_read_file_expands_tilde_into_global_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``~/.feather/...`` style paths are expanded so the model can use
    the natural form without knowing the exact home path. Layout mirrors
    a real install: ``$HOME/.feather`` is the global root, ``$HOME``
    is what tilde expands to."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    feather_global = fake_home / ".feather"
    paths = FeatherPaths(project_root=workspace, home=feather_global)
    paths.ensure_global_dirs()
    (feather_global / "user.md").write_text("hello", encoding="utf-8")

    monkeypatch.setenv("HOME", str(fake_home))

    tool = ReadFileTool(workspace, paths=paths)
    result = await tool.execute(
        {
            "path": "~/.feather/user.md",
            "start_line": None,
            "end_line": None,
            "max_chars": None,
        },
        _ctx(),
    )
    assert "hello" in result.output


async def test_read_file_rejects_dotenv_under_global_root(tmp_path: Path) -> None:
    """`.env` files under the global root contain API keys; the read
    tool must refuse them defensively even though the directory tree is
    otherwise readable. The user can still inspect via the bash tool."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"
    paths = FeatherPaths(project_root=workspace, home=home)
    paths.ensure_global_dirs()
    paths.env_file.write_text("OPENAI_API_KEY=sk-secret\n", encoding="utf-8")

    tool = ReadFileTool(workspace, paths=paths)
    with pytest.raises(ValueError, match=r"\.env"):
        await tool.execute(
            {
                "path": str(paths.env_file),
                "start_line": None,
                "end_line": None,
                "max_chars": None,
            },
            _ctx(),
        )


async def test_read_file_rejects_dotenv_inside_workspace(tmp_path: Path) -> None:
    """Project-local `.env` files are also blocked — same secrets risk."""

    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-secret\n", encoding="utf-8")
    tool = ReadFileTool(tmp_path)
    with pytest.raises(ValueError, match=r"\.env"):
        await tool.execute(
            {
                "path": ".env",
                "start_line": None,
                "end_line": None,
                "max_chars": None,
            },
            _ctx(),
        )


async def test_read_file_rejects_leaf_symlink_pointing_outside_workspace(
    tmp_path: Path,
) -> None:
    """A leaf symlink in the workspace whose target is outside must be
    rejected — `Path.resolve()` follows the symlink so the resolved
    path falls outside `_readable_roots` and the membership check
    rejects it. Locks in the docstring claim that symlink-escapes are
    rejected."""

    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("LEAK", encoding="utf-8")
    (workspace / "evil").symlink_to(outside / "secret.txt")

    tool = ReadFileTool(workspace)
    with pytest.raises(ValueError, match="must stay inside"):
        await tool.execute(
            {
                "path": "evil",
                "start_line": None,
                "end_line": None,
                "max_chars": None,
            },
            _ctx(),
        )


async def test_read_file_rejects_parent_symlink_pointing_outside_workspace(
    tmp_path: Path,
) -> None:
    """A *parent* component that is a symlink to a directory outside
    the workspace must also be rejected — `Path.resolve()` walks
    symlinks in every component."""

    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("LEAK", encoding="utf-8")
    (workspace / "sub").symlink_to(outside)

    tool = ReadFileTool(workspace)
    with pytest.raises(ValueError, match="must stay inside"):
        await tool.execute(
            {
                "path": "sub/secret.txt",
                "start_line": None,
                "end_line": None,
                "max_chars": None,
            },
            _ctx(),
        )


async def test_read_file_rejects_dotenv_via_symlink_with_innocent_name(
    tmp_path: Path,
) -> None:
    """A symlink with a non-`.env` name pointing at a `.env` file must
    still be rejected — `Path.resolve()` lands at `.env`, and the
    deny check runs against the resolved name."""

    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk\n", encoding="utf-8")
    (tmp_path / "innocent.txt").symlink_to(tmp_path / ".env")

    tool = ReadFileTool(tmp_path)
    with pytest.raises(ValueError, match=r"\.env"):
        await tool.execute(
            {
                "path": "innocent.txt",
                "start_line": None,
                "end_line": None,
                "max_chars": None,
            },
            _ctx(),
        )


@pytest.mark.parametrize(
    "filename",
    [".env", ".env.local", ".env.production", ".env_backup", ".envrc"],
)
async def test_read_file_rejects_dotenv_family(
    tmp_path: Path, filename: str
) -> None:
    """The deny is broader than just `.env` and `.env.*` — it also
    covers `.env_backup`, `.envrc` (direnv), and similar dotfiles that
    commonly hold secrets in real installs."""

    (tmp_path / filename).write_text("SECRET=x\n", encoding="utf-8")
    tool = ReadFileTool(tmp_path)
    with pytest.raises(ValueError, match=r"\.env"):
        await tool.execute(
            {
                "path": filename,
                "start_line": None,
                "end_line": None,
                "max_chars": None,
            },
            _ctx(),
        )


async def test_read_file_wraps_expanduser_runtime_error(tmp_path: Path) -> None:
    """`Path.expanduser()` raises `RuntimeError` on `~nosuchuser/...`
    when the user does not exist. The tool's contract is that bad
    input raises `ValueError`; verify the RuntimeError gets wrapped."""

    tool = ReadFileTool(tmp_path)
    with pytest.raises(ValueError, match="expand path"):
        await tool.execute(
            {
                "path": "~nosuchuser_xyz_98765/foo.txt",
                "start_line": None,
                "end_line": None,
                "max_chars": None,
            },
            _ctx(),
        )


async def test_read_file_renders_global_reads_with_tilde_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reads outside the workspace but inside `$HOME` render as
    `~/...` so the user's real `$HOME` is not echoed into chat or the
    prompt cache."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    feather_global = fake_home / ".feather"
    paths = FeatherPaths(project_root=workspace, home=feather_global)
    paths.ensure_global_dirs()
    (feather_global / "user.md").write_text("hi", encoding="utf-8")

    monkeypatch.setenv("HOME", str(fake_home))

    tool = ReadFileTool(workspace, paths=paths)
    result = await tool.execute(
        {
            "path": str(feather_global / "user.md"),
            "start_line": None,
            "end_line": None,
            "max_chars": None,
        },
        _ctx(),
    )
    # The display path should hide the real $HOME with a `~/` prefix.
    assert "file: ~/.feather/user.md" in result.output
    assert str(fake_home) not in result.output


async def test_read_file_rejects_paths_outside_all_roots_with_paths(
    tmp_path: Path,
) -> None:
    """A file that is neither in the workspace nor under the global
    root must still be rejected even when paths are wired up."""

    workspace = tmp_path / "workspace"
    home = tmp_path / "home"
    elsewhere = tmp_path / "elsewhere"
    workspace.mkdir()
    elsewhere.mkdir()
    (elsewhere / "x.txt").write_text("nope", encoding="utf-8")
    paths = FeatherPaths(project_root=workspace, home=home)
    paths.ensure_global_dirs()

    tool = ReadFileTool(workspace, paths=paths)
    with pytest.raises(ValueError, match="must stay inside"):
        await tool.execute(
            {
                "path": str(elsewhere / "x.txt"),
                "start_line": None,
                "end_line": None,
                "max_chars": None,
            },
            _ctx(),
        )


async def test_read_file_does_not_block_event_loop(tmp_path: Path, monkeypatch) -> None:
    """A slow file read must run off-loop so concurrent tasks keep running."""

    import asyncio
    import time

    (tmp_path / "file.txt").write_text("hello", encoding="utf-8")
    tool = ReadFileTool(tmp_path)
    original_read_text = Path.read_text

    def slow_read_text(self: Path, *args: object, **kwargs: object) -> str:
        time.sleep(0.2)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", slow_read_text)
    order: list[str] = []

    async def run_tool() -> None:
        await tool.execute(
            {"path": "file.txt", "start_line": None, "end_line": None, "max_chars": None},
            ToolExecutionContext(session_id="session-1", agent_name="Lead"),
        )
        order.append("tool")

    async def heartbeat() -> None:
        await asyncio.sleep(0.05)
        order.append("heartbeat")

    await asyncio.gather(run_tool(), heartbeat())
    assert order == ["heartbeat", "tool"]
