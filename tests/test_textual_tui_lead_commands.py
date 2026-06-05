"""/lead slash command: list, switch, and new-lead scaffolding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from feather.config import load_agent_config
from feather.core.agent.capabilities import CapabilityProfile
from feather.core.leads.soul_library import SoulLibrary
from feather.textual_tui import FeatherTextualApp, PerLeadState


def _state(name: str, display: str) -> PerLeadState:
    return PerLeadState(
        name=name,
        display_name=display,
        handle=None,
        session_id=f"{name}-sess",
        agent=SimpleNamespace(config=SimpleNamespace(name=display)),
        emoji="🧭",
    )


def _two_lead_app(monkeypatch):
    app = FeatherTextualApp(root=Path("."))
    app._leads = {"lead": _state("lead", "Tim"), "sophia": _state("sophia", "Sophia")}
    app._active_lead_name = "lead"
    markers: list[tuple[str, str]] = []
    convos: list[tuple[str, str]] = []
    renders: list[int] = []
    monkeypatch.setattr(
        app, "_write_marker", lambda title, text="", **k: markers.append((title, text))
    )
    monkeypatch.setattr(
        app, "_write_conversation", lambda title, body, **k: convos.append((title, body))
    )
    monkeypatch.setattr(app, "_render_active", lambda: renders.append(1))
    return app, markers, convos, renders


def test_lead_list_shows_active_leads(monkeypatch) -> None:
    app, _markers, convos, _r = _two_lead_app(monkeypatch)
    app._cmd_lead("list")
    assert convos and convos[-1][0] == "Leads"
    body = convos[-1][1]
    assert "Tim" in body and "Sophia" in body
    assert "active" in body


def test_lead_switch_changes_active(monkeypatch) -> None:
    app, _m, _c, renders = _two_lead_app(monkeypatch)
    app._cmd_lead("switch sophia")
    assert app._active_lead_name == "sophia"
    assert renders == [1]


def test_lead_switch_unknown_warns(monkeypatch) -> None:
    app, markers, _c, _r = _two_lead_app(monkeypatch)
    app._cmd_lead("switch nope")
    assert app._active_lead_name == "lead"
    assert markers and "not active" in markers[-1][1]


def test_lead_new_rejects_invalid_name(monkeypatch) -> None:
    app, markers, _c, _r = _two_lead_app(monkeypatch)
    app._cmd_lead("new my/agent")
    assert markers and "invalid lead name" in markers[-1][1]


def test_lead_usage_on_unknown_subcommand(monkeypatch) -> None:
    app, markers, _c, _r = _two_lead_app(monkeypatch)
    app._cmd_lead("frobnicate")
    assert markers and "usage:" in markers[-1][1]


def test_parse_lead_new_extracts_soul_flag() -> None:
    assert FeatherTextualApp._parse_lead_new("backend --soul atlas-architect") == (
        "backend", "atlas-architect", "",
    )
    assert FeatherTextualApp._parse_lead_new("sophia free text persona") == (
        "sophia", None, "free text persona",
    )
    assert FeatherTextualApp._parse_lead_new("x --soul foo extra words") == (
        "x", "foo", "extra words",
    )


def test_lead_souls_lists_library(monkeypatch) -> None:
    app, _m, convos, _r = _two_lead_app(monkeypatch)
    app._runtime = SimpleNamespace(soul_library=SoulLibrary(Path(".")))
    app._cmd_lead("souls")
    assert convos and convos[-1][0] == "Souls"
    body = convos[-1][1]
    assert "The Systems Thinker" in body and "systems-thinker" in body


def test_lead_new_unknown_soul_warns(monkeypatch) -> None:
    app, markers, _c, _r = _two_lead_app(monkeypatch)
    app._runtime = SimpleNamespace(soul_library=SoulLibrary(Path(".")))
    app._cmd_lead("new fresh --soul not-a-real-soul")
    assert markers and "unknown soul" in markers[-1][1]
    assert "fresh" not in app._leads  # never created


def test_scaffold_lead_yaml_is_loadable_lead(tmp_path: Path) -> None:
    app = FeatherTextualApp(root=tmp_path)
    app._scaffold_lead_yaml("tim", "You are Tim, a pragmatic operator.")
    path = tmp_path / "config" / "agents" / "tim.yaml"
    assert path.exists()

    cfg = load_agent_config(tmp_path, "tim")
    assert cfg.role == "lead"
    assert "pragmatic operator" in cfg.soul
    profile = CapabilityProfile.from_config(cfg)
    assert profile.is_lead and not profile.dispatchable
    # Scaffolded lead has spawn capability + no web tools (no Parallel dep).
    assert "spawn_agent" in cfg.registered_tools
    assert "web_search" not in cfg.registered_tools


def test_scaffold_is_idempotent_when_file_exists(tmp_path: Path) -> None:
    app = FeatherTextualApp(root=tmp_path)
    app._scaffold_lead_yaml("tim", "first")
    app._scaffold_lead_yaml("tim", "second")  # must not overwrite
    cfg = load_agent_config(tmp_path, "tim")
    assert "first" in cfg.personality or "first" in cfg.soul
