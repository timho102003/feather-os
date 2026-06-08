"""Tests for `.env` loading."""

from __future__ import annotations

import os
from pathlib import Path

from feather.config.env import load_dotenv


def test_load_dotenv_loads_simple_and_quoted_values(tmp_path: Path, monkeypatch) -> None:
    """The dotenv loader should parse simple assignments and quoted values."""

    env_path = tmp_path / ".env"
    env_path.write_text(
        """
# comment
OPENAI_API_KEY=test-key
API_BASE="https://example.com/v1"
EMPTY=
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_BASE", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)

    loaded = load_dotenv(env_path)

    assert loaded["OPENAI_API_KEY"] == "test-key"
    assert loaded["API_BASE"] == "https://example.com/v1"
    assert loaded["EMPTY"] == ""
    assert os.environ["OPENAI_API_KEY"] == "test-key"


def test_load_dotenv_does_not_override_existing_env_by_default(tmp_path: Path, monkeypatch) -> None:
    """Existing environment values should win unless override is enabled."""

    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")

    loaded = load_dotenv(env_path)

    assert loaded == {}
    assert os.environ["OPENAI_API_KEY"] == "existing-key"


def test_load_dotenv_can_override_existing_env(tmp_path: Path, monkeypatch) -> None:
    """Override mode should replace existing environment values."""

    env_path = tmp_path / ".env"
    env_path.write_text("export OPENAI_API_KEY=override-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")

    loaded = load_dotenv(env_path, override=True)

    assert loaded["OPENAI_API_KEY"] == "override-key"
    assert os.environ["OPENAI_API_KEY"] == "override-key"
