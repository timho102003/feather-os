"""Tests for the shared subprocess environment builder."""

from __future__ import annotations

import pytest

from feather.core.ipc.subprocess_env import subprocess_env_with_home


def test_home_present_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/tmp/somewhere")
    env = subprocess_env_with_home()
    assert env["HOME"] == "/tmp/somewhere"


def test_missing_home_is_rederived(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOME", raising=False)
    env = subprocess_env_with_home()
    assert env.get("HOME")  # pwd-derived on POSIX


def test_empty_home_is_rederived(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "")
    env = subprocess_env_with_home()
    assert env.get("HOME")
