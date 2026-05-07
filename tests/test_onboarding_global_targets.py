"""Tests that the wizard writes to global paths when feather_paths is given."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feather.onboarding import OnboardingWizard
from feather.paths import FeatherPaths


class _Recorder:
    def __init__(self, answers):
        self._answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self._answers:
            raise AssertionError(f"Out of scripted answers; next prompt was {prompt!r}")
        return self._answers.pop(0)


@pytest.fixture
def paths(tmp_path):
    p = FeatherPaths(project_root=tmp_path / "proj", home=tmp_path / "home")
    p.ensure_global_dirs()
    return p


async def test_wizard_writes_env_to_global_path(tmp_path, paths):
    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=_Recorder(["Tester", "", "", "", "", "1", "n", "n"]),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=_Recorder(["sk-test"]),
        qdrant_launcher=lambda say: "http://localhost:6333",
        feather_paths=paths,
    )
    await wizard.run()

    assert paths.env_file.is_file()
    body = paths.env_file.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in body
    # Project-local .env must NOT have been created
    assert not (tmp_path / ".env").exists()


async def test_wizard_writes_user_md_to_global_path(tmp_path, paths):
    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=_Recorder(["Tester", "Pref", "engineer", "Python", "Loves agents", "1", "n", "n"]),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=_Recorder(["sk-test"]),
        qdrant_launcher=lambda say: "http://localhost:6333",
        feather_paths=paths,
    )
    await wizard.run()

    assert paths.global_user_md.is_file()
    body = paths.global_user_md.read_text(encoding="utf-8")
    assert "Tester" in body
    # Per-project user.md must NOT have been created
    assert not (tmp_path / ".feather" / "user.md").exists()


async def test_wizard_writes_onboarded_marker_to_global_state(tmp_path, paths):
    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=_Recorder(["Tester", "", "", "", "", "1", "n", "n"]),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=_Recorder(["sk-test"]),
        qdrant_launcher=lambda say: "http://localhost:6333",
        feather_paths=paths,
    )
    await wizard.run()

    assert paths.onboarded_marker.is_file()
    payload = json.loads(paths.onboarded_marker.read_text(encoding="utf-8"))
    assert payload["openai_key_configured"] is True


async def test_wizard_materializes_app_yaml_at_global_path(tmp_path, paths):
    """First run with no global app.yaml should copy the packaged default
    so the toggle regex has something to rewrite."""
    target = paths.global_config_dir / "app.yaml"
    assert not target.exists()
    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=_Recorder(["Tester", "", "", "", "", "1", "n", "n"]),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=_Recorder(["sk-test"]),
        qdrant_launcher=lambda say: "http://localhost:6333",
        feather_paths=paths,
    )
    await wizard.run()

    assert target.is_file()
    body = target.read_text(encoding="utf-8")
    assert "openai" in body
    # Toggle should have rewritten active_provider
    assert "active_provider: openai" in body


async def test_wizard_legacy_paths_when_no_feather_paths(tmp_path):
    """Without feather_paths the wizard preserves the original behavior."""
    wizard = OnboardingWizard(
        root=tmp_path,
        # Legacy flow: identity + provider + memory("n") + web("n") + self_repair("n").
        input_fn=_Recorder(["Tester", "", "", "", "", "1", "n", "n", "n"]),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=_Recorder(["sk-test"]),
        qdrant_launcher=lambda say: "http://localhost:6333",
    )
    await wizard.run()

    # Project-local writes happened
    assert (tmp_path / ".env").exists()
    assert (tmp_path / ".feather" / "user.md").exists()
    assert (tmp_path / ".feather" / "onboarded.json").exists()
