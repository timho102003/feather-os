"""Tests for the comment-preserving config writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.config_writer import write_yaml_value


def test_writer_preserves_inline_comment(tmp_path: Path) -> None:
    src = tmp_path / "app.yaml"
    src.write_text(
        "openai:\n"
        "  model: gpt-5-mini   # default model\n"
        "  temperature: 1.0\n",
        encoding="utf-8",
    )

    write_yaml_value(src, ["openai", "model"], "gpt-5")

    text = src.read_text(encoding="utf-8")
    assert "model: gpt-5" in text
    assert "# default model" in text
    assert "temperature: 1.0\n" in text


def test_writer_preserves_block_comments_above(tmp_path: Path) -> None:
    src = tmp_path / "app.yaml"
    src.write_text(
        "# top comment\n"
        "openai:\n"
        "  # provider-level note\n"
        "  model: gpt-5-mini\n",
        encoding="utf-8",
    )

    write_yaml_value(src, ["openai", "model"], "gpt-5")

    text = src.read_text(encoding="utf-8")
    assert "# top comment" in text
    assert "# provider-level note" in text


def test_writer_handles_boolean_lower_case(tmp_path: Path) -> None:
    src = tmp_path / "app.yaml"
    src.write_text("memory:\n  enabled: false\n", encoding="utf-8")

    write_yaml_value(src, ["memory", "enabled"], True)

    text = src.read_text(encoding="utf-8")
    assert "enabled: true" in text
    assert "enabled: false" not in text


def test_writer_handles_integers(tmp_path: Path) -> None:
    src = tmp_path / "app.yaml"
    src.write_text("memory:\n  retrieval:\n    top_k_tool: 10\n", encoding="utf-8")

    write_yaml_value(src, ["memory", "retrieval", "top_k_tool"], 25)

    assert "top_k_tool: 25" in src.read_text(encoding="utf-8")


def test_writer_handles_strings_with_quoting(tmp_path: Path) -> None:
    src = tmp_path / "app.yaml"
    src.write_text("logging:\n  level: INFO\n", encoding="utf-8")

    write_yaml_value(src, ["logging", "level"], "DEBUG")

    assert "level: DEBUG" in src.read_text(encoding="utf-8")


def test_writer_creates_nested_path_in_empty_file(tmp_path: Path) -> None:
    src = tmp_path / "fresh.yaml"

    write_yaml_value(src, ["openai", "reasoning", "effort"], "high")

    text = src.read_text(encoding="utf-8")
    assert "openai:" in text
    assert "effort: high" in text


def test_writer_creates_intermediate_node_in_existing_file(tmp_path: Path) -> None:
    src = tmp_path / "app.yaml"
    src.write_text("openai:\n  model: gpt-5-mini\n", encoding="utf-8")

    write_yaml_value(src, ["openai", "reasoning", "effort"], "high")

    text = src.read_text(encoding="utf-8")
    assert "model: gpt-5-mini" in text
    assert "effort: high" in text


def test_writer_atomic_no_partial_on_failure(tmp_path: Path, monkeypatch) -> None:
    """If the dump raises, the original file is untouched."""

    src = tmp_path / "app.yaml"
    src.write_text("openai:\n  model: original\n", encoding="utf-8")

    from feather import config_writer as cw

    def boom(*args, **kwargs):
        raise RuntimeError("simulated dump failure")

    monkeypatch.setattr(cw.YAML, "dump", boom)

    with pytest.raises(RuntimeError):
        write_yaml_value(src, ["openai", "model"], "gpt-5")

    assert "model: original\n" in src.read_text(encoding="utf-8")


def test_delete_removes_leaf(tmp_path: Path) -> None:
    from feather.config_writer import delete_yaml_value

    src = tmp_path / "app.yaml"
    src.write_text(
        "openai:\n  model: gpt-5-mini\n  temperature: 1.0\n", encoding="utf-8"
    )

    assert delete_yaml_value(src, ["openai", "model"]) is True
    text = src.read_text(encoding="utf-8")
    assert "model" not in text
    assert "temperature: 1.0" in text


def test_delete_missing_leaf_returns_false(tmp_path: Path) -> None:
    from feather.config_writer import delete_yaml_value

    src = tmp_path / "app.yaml"
    src.write_text("openai:\n  model: gpt-5-mini\n", encoding="utf-8")

    assert delete_yaml_value(src, ["openai", "missing"]) is False
