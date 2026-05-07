"""Tests for marker-driven memory question skipping in the onboarding wizard."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.onboarding import OnboardingWizard
from feather.paths import FeatherPaths


class _Recorder:
    """Replays scripted answers and records every prompt the wizard issued."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self._answers:
            raise AssertionError(
                f"Wizard ran out of scripted answers; next prompt was {prompt!r}"
            )
        return self._answers.pop(0)


def _wizard_for(tmp_path: Path, paths: FeatherPaths, answers, secrets):
    """Build a wizard with deterministic input/secret callables and no docker."""
    return OnboardingWizard(
        root=tmp_path,
        input_fn=_Recorder(answers),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=_Recorder(secrets),
        qdrant_launcher=lambda say: "http://localhost:6333",
        feather_paths=paths,
    )


@pytest.fixture
def paths(tmp_path):
    p = FeatherPaths(project_root=tmp_path, home=tmp_path / "home")
    p.ensure_global_dirs()
    return p


async def test_wizard_skips_memory_question_when_marker_absent(tmp_path, paths):
    """No marker → wizard records memory_enabled=False and never asks the question."""
    wizard = _wizard_for(
        tmp_path,
        paths,
        # identity (5 prompts) + 3 provider yes/no answers
        ["Tester", "", "", "", "", "y", "n", "n"],
        # OPENAI key
        ["sk-test"],
    )
    # The 'web search' yes/no comes after memory; reuse the input recorder
    wizard.input_fn._answers.append("n")  # type: ignore[attr-defined]
    answers = await wizard.run()

    assert answers.memory_enabled is False
    # Sanity: no prompt about memory was asked
    asked = " ".join(wizard.input_fn.prompts)  # type: ignore[attr-defined]
    assert "Enable long-term memory" not in asked


async def test_wizard_treats_marker_present_as_memory_on(tmp_path, paths):
    """Marker present → wizard records memory_enabled=True and skips deployment-choice."""
    paths.memory_marker.write_text(
        '{"url": "http://feather-qdrant:6333"}', encoding="utf-8"
    )

    wizard = _wizard_for(
        tmp_path,
        paths,
        # identity (5) + 3 provider yes/no answers + web-search (n)
        ["Tester", "", "", "", "", "y", "n", "n", "n"],
        # OPENAI + GEMINI
        ["sk-test", "gemini-key"],
    )
    answers = await wizard.run()

    assert answers.memory_enabled is True
    assert answers.qdrant_mode == "preconfigured"
    assert answers.qdrant_url == "http://feather-qdrant:6333"
    asked = " ".join(wizard.input_fn.prompts)  # type: ignore[attr-defined]
    assert "Enable long-term memory" not in asked
    assert "Qdrant deployment" not in asked


async def test_wizard_with_no_paths_asks_memory_question_legacy(tmp_path):
    """When feather_paths is None, the wizard preserves the legacy prompt."""
    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=_Recorder(
            # identity (5) + 3 provider yes/no answers + memory? + web search
            ["Tester", "", "", "", "", "y", "n", "n", "n", "n"]
        ),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=_Recorder(["sk-test"]),
        qdrant_launcher=lambda say: "http://localhost:6333",
    )
    answers = await wizard.run()

    assert answers.memory_enabled is False
    asked = " ".join(wizard.input_fn.prompts)  # type: ignore[attr-defined]
    assert "Enable long-term memory" in asked
