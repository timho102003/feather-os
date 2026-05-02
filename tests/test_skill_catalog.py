"""Tests for skill discovery and loading."""

from pathlib import Path

from feather.resources import packaged_skills_root
from feather.skills.catalog import SkillCatalog


def test_skill_catalog_lists_metadata_and_loads_refs(tmp_path: Path) -> None:
    """Skill metadata and referenced docs should both load correctly."""

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "reference.md").write_text("# Reference\n\nDetails", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo-skill
description: Demo description
refs:
  - ./reference.md
---

# Demo Skill

Use this for testing.
""",
        encoding="utf-8",
    )

    catalog = SkillCatalog(tmp_path / "skills")
    metadata = catalog.list_metadata()

    assert len(metadata) == 1
    assert metadata[0].name == "demo-skill"

    loaded = catalog.load_skill("demo-skill")
    assert "Use this for testing." in loaded.content
    assert "Reference: reference.md" in loaded.content


def test_builtin_mcp_config_skill_is_discoverable() -> None:
    """The packaged skill catalog should teach agents how to add MCP config."""

    catalog = SkillCatalog([packaged_skills_root()])

    metadata = catalog.list_metadata()
    names = {meta.name for meta in metadata}
    loaded = catalog.load_skill("mcp-config")

    assert "mcp-config" in names
    assert "config/app.yaml" in loaded.content or "app.yaml" in loaded.content
    assert "register_mcp_server" in loaded.content


def test_builtin_pdf_reading_skill_is_discoverable() -> None:
    """The packaged skill catalog should teach agents how to read PDFs."""

    catalog = SkillCatalog([packaged_skills_root()])

    metadata = catalog.list_metadata()
    names = {meta.name for meta in metadata}
    loaded = catalog.load_skill("pdf-reading")

    assert "pdf-reading" in names
    assert "read_pdf" in loaded.content
    assert "OPENDATALOADER_PDF_COMMAND" in loaded.content


def test_project_source_overrides_packaged_by_name(tmp_path: Path) -> None:
    """A project-local skill with the same name shadows the packaged one."""

    project_skills = tmp_path / "project_skills" / "mcp-config"
    project_skills.mkdir(parents=True)
    (project_skills / "SKILL.md").write_text(
        """---
name: mcp-config
description: Project-local override
---

PROJECT MCP body.
""",
        encoding="utf-8",
    )

    catalog = SkillCatalog(
        [packaged_skills_root(), tmp_path / "project_skills"]
    )
    loaded = catalog.load_skill("mcp-config")
    assert "PROJECT MCP body." in loaded.content
    assert "register_mcp_server" not in loaded.content


def test_missing_source_paths_are_silently_skipped(tmp_path: Path) -> None:
    """A configured project skills dir that doesn't yet exist must not crash."""

    catalog = SkillCatalog(
        [packaged_skills_root(), tmp_path / "does-not-exist"]
    )
    names = {meta.name for meta in catalog.list_metadata()}
    assert "mcp-config" in names
