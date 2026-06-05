"""Multi-lead cockpit: per-lead state routing, switching, and the lead strip.

Unit-style (matching the other textual_tui tests): the app is constructed and
its internals are driven directly, with widget-touching methods monkeypatched,
so no full runtime / Textual mount is needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from feather.textual_tui import FeatherTextualApp, PerLeadState, build_lead_strip
from feather.models import RuntimeEvent


def _agent(name: str) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(name=name))


def _state(name: str, display: str) -> PerLeadState:
    return PerLeadState(
        name=name,
        display_name=display,
        handle=None,
        session_id=f"{name}-sess",
        agent=_agent(display),
        emoji="🧭",
    )


def _two_lead_app(monkeypatch) -> tuple[FeatherTextualApp, PerLeadState, PerLeadState]:
    app = FeatherTextualApp(root=Path("."))
    tim = _state("lead", "Tim")
    sophia = _state("sophia", "Sophia")
    app._leads = {"lead": tim, "sophia": sophia}
    app._active_lead_name = "lead"
    # Silence widget access.
    monkeypatch.setattr(app, "_render_conversation", lambda: None)
    monkeypatch.setattr(app, "_update_header", lambda: None)
    monkeypatch.setattr(app, "_update_work", lambda: None)
    return app, tim, sophia


def test_switch_lead_rotates_active(monkeypatch) -> None:
    app, _tim, _sophia = _two_lead_app(monkeypatch)
    rendered: list[str] = []
    monkeypatch.setattr(app, "_render_active", lambda: rendered.append(app._active_lead_name))

    assert app._active_lead_name == "lead"
    app.action_lead_next()
    assert app._active_lead_name == "sophia"
    app.action_lead_next()  # wraps
    assert app._active_lead_name == "lead"
    app.action_lead_prev()  # wraps back
    assert app._active_lead_name == "sophia"
    assert rendered == ["sophia", "lead", "sophia"]


def test_switch_is_noop_with_single_lead(monkeypatch) -> None:
    app = FeatherTextualApp(root=Path("."))
    app._leads = {"lead": _state("lead", "Tim")}
    app._active_lead_name = "lead"
    calls: list[int] = []
    monkeypatch.setattr(app, "_render_active", lambda: calls.append(1))
    app.action_lead_next()
    assert app._active_lead_name == "lead"
    assert calls == []


def test_event_routes_to_background_lead_without_touching_widgets(monkeypatch) -> None:
    app, tim, sophia = _two_lead_app(monkeypatch)
    renders: list[int] = []
    monkeypatch.setattr(app, "_render_conversation", lambda: renders.append(1))

    # Active = Tim. A streaming delta for the BACKGROUND lead (Sophia) must
    # accumulate on her state but NOT repaint the conversation.
    app._apply_event(sophia, RuntimeEvent(kind="assistant_text_delta", text="hi from sophia"))
    assert sophia.assistant_parts == ["hi from sophia"]
    assert tim.assistant_parts == []
    assert renders == []  # background → no repaint

    # A delta for the ACTIVE lead repaints.
    app._apply_event(tim, RuntimeEvent(kind="assistant_text_delta", text="hi from tim"))
    assert tim.assistant_parts == ["hi from tim"]
    assert renders == [1]


def test_tool_event_appends_block_to_owning_lead(monkeypatch) -> None:
    app, tim, sophia = _two_lead_app(monkeypatch)
    app._apply_event(
        sophia, RuntimeEvent(kind="tool_started", tool_name="grep", payload={})
    )
    # Sophia's conversation grew; Tim's did not.
    assert len(sophia.conversation_blocks) == 1
    assert sophia.conversation_blocks[0].title == "Sophia"
    assert tim.conversation_blocks == []


def test_lead_strip_marks_active_and_lists_all(monkeypatch) -> None:
    app, _tim, _sophia = _two_lead_app(monkeypatch)
    strip = app._build_lead_strip()
    assert strip is not None
    text = strip.plain
    assert "Tim" in text and "Sophia" in text
    assert "[🧭 Tim" in text  # active lead bracketed (with emoji)
    assert "ctrl+" in text


def test_lead_strip_is_none_for_single_lead() -> None:
    app = FeatherTextualApp(root=Path("."))
    app._leads = {"lead": _state("lead", "Tim")}
    app._active_lead_name = "lead"
    assert app._build_lead_strip() is None


def test_build_lead_strip_helper_formats_status_glyphs() -> None:
    strip = build_lead_strip(
        (("Tim", "🧭", "running"), ("Sophia", None, "idle")),
        active_name="Tim",
    )
    text = strip.plain
    assert "●" in text  # running glyph
    assert "○" in text  # idle glyph
    assert "[🧭 Tim" in text  # active, with emoji
