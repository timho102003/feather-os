"""Tests for the FEATHER_USE_LEAD_WORKER env-flag detection in textual_tui."""

from __future__ import annotations

import pytest

from feather.tui.app import _LEAD_WORKER_ENV, _should_use_lead_worker


def test_default_is_false(monkeypatch) -> None:
    """With the env var unset, the lead always runs in-process."""

    monkeypatch.delenv(_LEAD_WORKER_ENV, raising=False)
    assert _should_use_lead_worker() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes", "on", "ON"])
def test_truthy_values_enable_worker(monkeypatch, value: str) -> None:
    monkeypatch.setenv(_LEAD_WORKER_ENV, value)
    assert _should_use_lead_worker() is True


@pytest.mark.parametrize(
    "value", ["", "0", "false", "no", "off", "  ", "yesplease", "ON1"]
)
def test_falsy_or_unrecognised_values_keep_in_process_default(
    monkeypatch, value: str
) -> None:
    """Anything that isn't a recognised truthy literal must NOT opt in.

    The conservative parse is deliberate: a typo like ``yesplease`` should
    not silently flip a runtime architecture, so we accept only the
    canonical positive forms and treat everything else as off.
    """

    monkeypatch.setenv(_LEAD_WORKER_ENV, value)
    assert _should_use_lead_worker() is False


def test_whitespace_around_truthy_value_is_tolerated(monkeypatch) -> None:
    monkeypatch.setenv(_LEAD_WORKER_ENV, "  1  ")
    assert _should_use_lead_worker() is True
