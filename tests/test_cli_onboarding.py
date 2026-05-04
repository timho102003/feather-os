"""Tests that the CLI honours --skip-onboarding and the onboard subcommand."""

from __future__ import annotations

import sys
from pathlib import Path


def test_cli_onboard_subcommand_calls_wizard(tmp_path: Path, monkeypatch) -> None:
    """`feather onboard` runs the wizard and exits without booting the runtime."""

    from feather import cli

    invocations = {"count": 0, "kwargs": None}

    async def fake_maybe_run_onboarding(*_args, **kwargs):
        invocations["count"] += 1
        invocations["kwargs"] = kwargs
        return None

    async def fake_run_cli(*_args, **_kwargs):
        raise AssertionError("run_cli should not be invoked under `feather onboard`.")

    monkeypatch.setattr(cli, "maybe_run_onboarding", fake_maybe_run_onboarding)
    monkeypatch.setattr(cli, "run_cli", fake_run_cli)
    monkeypatch.setattr(sys, "argv", ["feather", "onboard"])
    cli.main()
    assert invocations["count"] == 1


def test_cli_onboard_force_propagates_flag(tmp_path: Path, monkeypatch) -> None:
    """`feather onboard --force` passes ``force=True`` through."""

    from feather import cli

    captured = {"force": None}

    async def fake_maybe_run_onboarding(*_args, **kwargs):
        captured["force"] = kwargs.get("force")
        return None

    async def fake_run_cli(*_args, **_kwargs):
        raise AssertionError("run_cli should not be invoked under `feather onboard`.")

    monkeypatch.setattr(cli, "maybe_run_onboarding", fake_maybe_run_onboarding)
    monkeypatch.setattr(cli, "run_cli", fake_run_cli)
    monkeypatch.setattr(sys, "argv", ["feather", "onboard", "--force"])
    cli.main()
    assert captured["force"] is True


def test_cli_skip_onboarding_flag_does_not_call_wizard(tmp_path: Path, monkeypatch) -> None:
    """`--skip-onboarding` propagates ``skip=True`` to maybe_run_onboarding."""

    from feather import cli, textual_tui

    captured = {"skip": None}

    async def fake_maybe_run_onboarding(*_args, **kwargs):
        captured["skip"] = kwargs.get("skip")
        return None

    async def fake_run_textual_tui(*_args, **_kwargs):
        return None

    monkeypatch.setattr(cli, "maybe_run_onboarding", fake_maybe_run_onboarding)
    monkeypatch.setattr(textual_tui, "run_textual_tui", fake_run_textual_tui)
    monkeypatch.setattr(sys, "argv", ["feather", "--skip-onboarding"])
    cli.main()
    assert captured["skip"] is True


def test_cli_default_run_invokes_onboarding_then_runtime(tmp_path: Path, monkeypatch) -> None:
    """A bare ``feather`` invocation runs onboarding first, then launches the Textual TUI."""

    from feather import cli, textual_tui

    order: list[str] = []

    async def fake_maybe_run_onboarding(*_args, **_kwargs):
        order.append("onboard")
        return None

    async def fake_run_textual_tui(*_args, **_kwargs):
        order.append("run_textual_tui")
        return None

    monkeypatch.setattr(cli, "maybe_run_onboarding", fake_maybe_run_onboarding)
    monkeypatch.setattr(textual_tui, "run_textual_tui", fake_run_textual_tui)
    monkeypatch.setattr(sys, "argv", ["feather"])
    cli.main()
    assert order == ["onboard", "run_textual_tui"]


def test_cli_explicit_cli_subcommand_uses_rich_loop(tmp_path: Path, monkeypatch) -> None:
    """`feather cli` runs onboarding then the Rich streaming loop, not the TUI."""

    from feather import cli, textual_tui

    order: list[str] = []

    async def fake_maybe_run_onboarding(*_args, **_kwargs):
        order.append("onboard")
        return None

    async def fake_run_cli(*_args, **_kwargs):
        order.append("run_cli")
        return None

    async def fake_run_textual_tui(*_args, **_kwargs):
        raise AssertionError("run_textual_tui should not be invoked under `feather cli`.")

    monkeypatch.setattr(cli, "maybe_run_onboarding", fake_maybe_run_onboarding)
    monkeypatch.setattr(cli, "run_cli", fake_run_cli)
    monkeypatch.setattr(textual_tui, "run_textual_tui", fake_run_textual_tui)
    monkeypatch.setattr(sys, "argv", ["feather", "cli"])
    cli.main()
    assert order == ["onboard", "run_cli"]
