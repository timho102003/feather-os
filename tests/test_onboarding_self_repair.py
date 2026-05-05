"""Tests for the self-repair onboarding integration.

Covers the YAML toggle, the wizard's question flow, and the layered
env > yaml resolution in textual_tui's _should_use_lead_worker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.onboarding import apply_self_repair_toggle
from feather.textual_tui import _should_use_lead_worker, _LEAD_WORKER_ENV


# --------------------------------------------------------------------- #
# YAML toggle: apply_self_repair_toggle
# --------------------------------------------------------------------- #


def _stage_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "app.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_toggle_flips_self_repair_enabled_from_false_to_true(tmp_path: Path) -> None:
    yaml = _stage_yaml(
        tmp_path,
        "self_repair:\n  enabled: false  # comment preserved\n",
    )
    changed = apply_self_repair_toggle(yaml, enabled=True)
    assert changed is True
    assert "enabled: true  # comment preserved" in yaml.read_text()


def test_toggle_flips_self_repair_enabled_from_true_to_false(tmp_path: Path) -> None:
    yaml = _stage_yaml(tmp_path, "self_repair:\n  enabled: true\n")
    changed = apply_self_repair_toggle(yaml, enabled=False)
    assert changed is True
    assert "enabled: false" in yaml.read_text()


def test_toggle_no_op_when_block_absent(tmp_path: Path) -> None:
    """Files without a self_repair block (older YAMLs) must not crash —
    the wizard treats this as 'feature defaults stay default'."""

    yaml = _stage_yaml(
        tmp_path, "scheduler:\n  enabled: true\nactive_provider: openai\n"
    )
    changed = apply_self_repair_toggle(yaml, enabled=True)
    assert changed is False
    assert "scheduler" in yaml.read_text()  # untouched


def test_toggle_only_writes_inside_self_repair_block(tmp_path: Path) -> None:
    """An ``enabled:`` line under a different top-level block (e.g.
    scheduler) must NOT be rewritten by the self_repair toggle."""

    yaml = _stage_yaml(
        tmp_path,
        "scheduler:\n  enabled: true\n"
        "self_repair:\n  enabled: false\n",
    )
    changed = apply_self_repair_toggle(yaml, enabled=True)
    assert changed is True
    text = yaml.read_text()
    # scheduler.enabled stays true.
    assert "scheduler:\n  enabled: true" in text
    # self_repair.enabled flipped.
    assert "self_repair:\n  enabled: true" in text


def test_toggle_preserves_trailing_comments(tmp_path: Path) -> None:
    yaml = _stage_yaml(
        tmp_path,
        "self_repair:\n  enabled: false   # trade-off: cron paused\n",
    )
    apply_self_repair_toggle(yaml, enabled=True)
    assert "# trade-off: cron paused" in yaml.read_text()


# --------------------------------------------------------------------- #
# Layered resolution: _should_use_lead_worker
# --------------------------------------------------------------------- #


def test_yaml_disabled_no_env_returns_false(monkeypatch) -> None:
    """The default — both signals off — keeps the legacy in-process behavior."""

    monkeypatch.delenv(_LEAD_WORKER_ENV, raising=False)
    assert _should_use_lead_worker(yaml_enabled=False) is False


def test_yaml_enabled_no_env_returns_true(monkeypatch) -> None:
    """Onboarding wrote yaml=true; in absence of any env override, we honor it."""

    monkeypatch.delenv(_LEAD_WORKER_ENV, raising=False)
    assert _should_use_lead_worker(yaml_enabled=True) is True


def test_env_truthy_overrides_yaml_disabled(monkeypatch) -> None:
    """Power-user one-off: enable for this run without flipping the YAML."""

    monkeypatch.setenv(_LEAD_WORKER_ENV, "1")
    assert _should_use_lead_worker(yaml_enabled=False) is True


def test_env_falsy_overrides_yaml_enabled(monkeypatch) -> None:
    """Power-user one-off: disable for this run without flipping the YAML.

    Lets a user with self-repair enabled persistently still launch the
    legacy in-process path for one-off testing (e.g. while cron-driven
    workflows are critical for that session)."""

    monkeypatch.setenv(_LEAD_WORKER_ENV, "0")
    assert _should_use_lead_worker(yaml_enabled=True) is False


def test_env_garbage_falls_through_to_yaml(monkeypatch) -> None:
    """A typo in the env var must NOT silently disable the persistent setting."""

    monkeypatch.setenv(_LEAD_WORKER_ENV, "yesplease")
    assert _should_use_lead_worker(yaml_enabled=True) is True
    assert _should_use_lead_worker(yaml_enabled=False) is False


@pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "yes", "Yes", "on", "ON"])
def test_env_recognised_truthy_values(monkeypatch, truthy: str) -> None:
    monkeypatch.setenv(_LEAD_WORKER_ENV, truthy)
    assert _should_use_lead_worker(yaml_enabled=False) is True


@pytest.mark.parametrize("falsy", ["0", "false", "FALSE", "no", "No", "off", "OFF"])
def test_env_recognised_falsy_values(monkeypatch, falsy: str) -> None:
    monkeypatch.setenv(_LEAD_WORKER_ENV, falsy)
    assert _should_use_lead_worker(yaml_enabled=True) is False
