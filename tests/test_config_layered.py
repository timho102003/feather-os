"""Tests for layered config + agent loading (packaged + project + global)."""

from pathlib import Path

import pytest

from feather.config import load_agent_config, load_app_config
from feather.paths import FeatherPaths


def _write_yaml(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# load_app_config layering
# ---------------------------------------------------------------------------


def test_app_config_falls_back_to_packaged_when_no_project_yaml(tmp_path):
    """A fresh CWD with no config/ should still produce a parseable config."""
    cfg = load_app_config(tmp_path)
    assert cfg.openai.model  # packaged default has an openai.model
    assert cfg.compaction.enabled in (True, False)


def test_app_config_project_yaml_wins_over_packaged(tmp_path):
    _write_yaml(
        tmp_path / "config" / "app.yaml",
        """database:
  path: project/db.sqlite
storage:
  temp_directory: project/tmp
logging:
  path: project/log.txt
  level: DEBUG
compaction:
  enabled: false
  trigger_ratio: 0.9
  context_window_tokens: 100
  model:
  max_output_tokens: 100
  temperature: 0.0
skills:
  directory: project/skills
scheduler:
  enabled: false
  poll_interval_seconds: 1
  failure_retry_seconds: 1
  max_due_jobs_per_tick: 1
openai:
  api_key_env: PROJECT_KEY_ENV
  model: project-model
  max_output_tokens: 1
  temperature: 0.0
  parallel_tool_calls: false
  store: false
memory:
  enabled: false
""",
    )
    cfg = load_app_config(tmp_path)
    assert cfg.openai.model == "project-model"
    assert cfg.openai.api_key_env == "PROJECT_KEY_ENV"
    assert cfg.compaction.enabled is False


def test_app_config_global_overlay_merges_over_packaged(tmp_path):
    """User overrides ~/.feather/config/app.yaml deep-merge over packaged base."""
    home = tmp_path / "home"
    paths = FeatherPaths(project_root=None, home=home)
    paths.ensure_global_dirs()
    _write_yaml(
        paths.global_config_dir / "app.yaml",
        """openai:
  model: my-custom-override-model
""",
    )
    cfg = load_app_config(tmp_path, paths=paths)
    assert cfg.openai.model == "my-custom-override-model"
    # Packaged-default keys not touched by the overlay still come through:
    assert cfg.compaction.enabled is True


def test_app_config_global_overlay_merges_over_project_yaml(tmp_path):
    """When project + global both exist, global wins on the keys it overrides."""
    _write_yaml(
        tmp_path / "config" / "app.yaml",
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
  context_window_tokens: 100000
  model:
  max_output_tokens: 100
  temperature: 0.0
skills:
  directory: .feather/skills
scheduler:
  enabled: false
  poll_interval_seconds: 1
  failure_retry_seconds: 1
  max_due_jobs_per_tick: 1
openai:
  api_key_env: OPENAI_API_KEY
  model: project-model
  max_output_tokens: 100
  temperature: 0.0
  parallel_tool_calls: false
  store: false
memory:
  enabled: false
""",
    )
    home = tmp_path / "home"
    paths = FeatherPaths(project_root=None, home=home)
    paths.ensure_global_dirs()
    _write_yaml(
        paths.global_config_dir / "app.yaml",
        """openai:
  model: global-override
""",
    )
    cfg = load_app_config(tmp_path, paths=paths)
    assert cfg.openai.model == "global-override"
    # project-staged base values that the overlay did NOT touch survive
    assert cfg.openai.api_key_env == "OPENAI_API_KEY"
    assert cfg.compaction.enabled is True
    assert cfg.compaction.context_window_tokens == 100000


# ---------------------------------------------------------------------------
# load_agent_config layering
# ---------------------------------------------------------------------------


def test_agent_config_falls_back_to_packaged(tmp_path):
    """No project/global agent yaml → packaged default loads."""
    cfg = load_agent_config(tmp_path, "lead")
    assert cfg.role == "lead"


def test_agent_config_project_yaml_wins_over_packaged(tmp_path):
    _write_yaml(
        tmp_path / "config" / "agents" / "lead.yaml",
        """name: ProjectLead
role: lead
personality: project personality
prompt_modules:
  - feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT
registered_tools:
  - read_file
description: project-only override
""",
    )
    cfg = load_agent_config(tmp_path, "lead")
    assert cfg.name == "ProjectLead"
    assert cfg.description == "project-only override"


def test_agent_config_global_yaml_wins_over_packaged(tmp_path):
    home = tmp_path / "home"
    paths = FeatherPaths(project_root=None, home=home)
    paths.ensure_global_dirs()
    _write_yaml(
        paths.global_agents_dir / "lead.yaml",
        """name: GlobalLead
role: lead
personality: global personality
prompt_modules:
  - feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT
registered_tools:
  - read_file
description: global override
""",
    )
    cfg = load_agent_config(tmp_path, "lead", paths=paths)
    assert cfg.name == "GlobalLead"


def test_agent_config_project_wins_over_global(tmp_path):
    """Project-staged agent yaml beats user-global override."""
    home = tmp_path / "home"
    paths = FeatherPaths(project_root=None, home=home)
    paths.ensure_global_dirs()
    _write_yaml(
        paths.global_agents_dir / "lead.yaml",
        """name: GlobalLead
role: lead
personality: global
prompt_modules: [feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT]
registered_tools: [read_file]
""",
    )
    _write_yaml(
        tmp_path / "config" / "agents" / "lead.yaml",
        """name: ProjectLead
role: lead
personality: project
prompt_modules: [feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT]
registered_tools: [read_file]
""",
    )
    cfg = load_agent_config(tmp_path, "lead", paths=paths)
    assert cfg.name == "ProjectLead"


def test_agent_config_unknown_name_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found in project"):
        load_agent_config(tmp_path, "does-not-exist")
