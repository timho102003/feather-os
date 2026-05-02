"""Tests for bundled package resources."""

from feather.resources import (
    has_packaged_agent,
    iter_packaged_agent_names,
    packaged_app_yaml_dict,
    packaged_app_yaml_text,
    packaged_root,
    packaged_skills_root,
)


def test_packaged_root_is_traversable():
    root = packaged_root()
    children = {child.name for child in root.iterdir()}
    assert "config" in children
    assert "skills" in children


def test_app_yaml_text_is_loadable():
    text = packaged_app_yaml_text()
    assert "openai" in text
    assert "compaction" in text


def test_app_yaml_dict_parses_to_known_keys():
    raw = packaged_app_yaml_dict()
    assert "openai" in raw
    assert "compaction" in raw
    assert "storage" in raw
    assert "skills" in raw


def test_iter_packaged_agent_names_includes_lead():
    names = set(iter_packaged_agent_names())
    assert "lead" in names
    assert "explore" in names
    assert "research" in names
    assert "validate" in names


def test_has_packaged_agent_distinguishes_real_and_fake():
    assert has_packaged_agent("lead") is True
    assert has_packaged_agent("does-not-exist") is False


def test_packaged_skills_root_lists_built_ins():
    root = packaged_skills_root()
    skill_dirs = {child.name for child in root.iterdir() if child.is_dir()}
    assert "agent-creator" in skill_dirs
    assert "mcp-config" in skill_dirs
    assert "pdf-reading" in skill_dirs
    assert "planning" in skill_dirs
    assert "repo-navigation" in skill_dirs


def test_packaged_skill_md_is_readable():
    root = packaged_skills_root()
    skill = root / "mcp-config" / "SKILL.md"
    assert skill.is_file()
    text = skill.read_text(encoding="utf-8")
    assert "register_mcp_server" in text or "mcp" in text.lower()


def test_packaged_skill_reference_files_resolvable():
    """repo-navigation has a reference.md sibling to SKILL.md."""
    root = packaged_skills_root()
    ref = root / "repo-navigation" / "reference.md"
    assert ref.is_file()
