"""Tests for MCP configuration parsing."""

from __future__ import annotations

from pathlib import Path

from feather.config import load_app_config


_MIN_APP_YAML = """database: {path: .feather/db/feather.db}
storage: {temp_directory: .feather/tmp}
logging: {path: .feather/logs/feather.log, level: INFO}
compaction: {}
skills: {directory: .feather/skills}
scheduler: {}
openai:
  api_key_env: OPENAI_API_KEY
  model: gpt-5-mini
  max_output_tokens: 4000
  temperature: 1.0
  parallel_tool_calls: true
"""


def test_load_app_config_defaults_mcp_to_disabled_empty_config(tmp_path: Path) -> None:
    """Omitting ``mcp:`` keeps the integration inert."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "app.yaml").write_text(_MIN_APP_YAML, encoding="utf-8")

    cfg = load_app_config(tmp_path)

    assert cfg.mcp.enabled is False
    assert cfg.mcp.servers == ()


def test_load_app_config_parses_mcp_servers_mapping(tmp_path: Path) -> None:
    """The top-level ``mcp.servers`` mapping registers remote MCP servers."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "app.yaml").write_text(
        _MIN_APP_YAML
        + """mcp:
  enabled: true
  servers:
    docs:
      url: https://developers.openai.com/mcp
      description: OpenAI docs MCP server
      allowed_tools: [search_openai_docs, fetch_openai_doc]
      require_approval: never
      providers: [openai, openrouter]
      agents: [Lead]
      headers:
        X-Static: static
      header_envs:
        Authorization: OPENAI_DOCS_MCP_TOKEN
      request_timeout_seconds: 12.5
""",
        encoding="utf-8",
    )

    cfg = load_app_config(tmp_path)

    assert cfg.mcp.enabled is True
    assert len(cfg.mcp.servers) == 1
    server = cfg.mcp.servers[0]
    assert server.label == "docs"
    assert server.server_url == "https://developers.openai.com/mcp"
    assert server.transport == "http"
    assert server.command is None
    assert server.server_description == "OpenAI docs MCP server"
    assert server.allowed_tools == ("search_openai_docs", "fetch_openai_doc")
    assert server.require_approval == "never"
    assert server.providers == ("openai", "openrouter")
    assert server.agents == ("Lead",)
    assert server.headers == {"X-Static": "static"}
    assert server.header_envs == {"Authorization": "OPENAI_DOCS_MCP_TOKEN"}
    assert server.request_timeout_seconds == 12.5


def test_load_app_config_parses_stdio_mcp_server(tmp_path: Path) -> None:
    """A top-level MCP server can be registered as a stdio launcher."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "app.yaml").write_text(
        _MIN_APP_YAML
        + """mcp:
  enabled: true
  servers:
    playwright:
      command: npx
      args: ["-y", "@playwright/mcp@latest"]
      env:
        DEBUG: pw:mcp
      cwd: .
      providers: [openrouter]
""",
        encoding="utf-8",
    )

    cfg = load_app_config(tmp_path)

    server = cfg.mcp.servers[0]
    assert server.label == "playwright"
    assert server.transport == "stdio"
    assert server.command == "npx"
    assert server.args == ("-y", "@playwright/mcp@latest")
    assert server.env == {"DEBUG": "pw:mcp"}
    assert server.cwd == "."
    assert server.server_url is None


def test_load_app_config_rejects_enabled_mcp_server_without_url_or_command(
    tmp_path: Path,
) -> None:
    """A registered MCP server must define an HTTP URL or stdio command."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "app.yaml").write_text(
        _MIN_APP_YAML
        + """mcp:
  enabled: true
  servers:
    broken: {}
""",
        encoding="utf-8",
    )

    try:
        load_app_config(tmp_path)
    except ValueError as exc:
        assert "mcp.servers.broken" in str(exc)
        assert "`url` or `command`" in str(exc)
    else:
        raise AssertionError("Expected missing MCP server transport to raise ValueError")
