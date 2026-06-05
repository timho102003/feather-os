"""Pydantic request/response models for the Feather API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LeadOut(BaseModel):
    """One lead in the multi-lead cockpit."""

    name: str
    display_name: str
    personality: str
    soul: str
    color: str | None
    emoji: str | None
    session_id: str
    status: str


class SoulOut(BaseModel):
    """One selectable working-temperament preset from the soul library."""

    id: str
    title: str
    personality: str
    color: str
    emoji: str
    tags: list[str]


class MessageIn(BaseModel):
    """A user message sent to a lead."""

    text: str = Field(min_length=1)


class CreateLeadIn(BaseModel):
    """Create a new lead from the web UI.

    ``soul_id`` selects a packaged/custom soul preset (its persona + color +
    emoji are baked into the lead). ``soul`` is free-text persona used only
    when no preset is chosen.
    """

    name: str = Field(min_length=1, max_length=64)
    soul: str = ""
    soul_id: str | None = None


class SubagentOut(BaseModel):
    """One live sub-agent under a lead."""

    agent_name: str
    session_id: str
    task: str


class TranscriptMessageOut(BaseModel):
    """One persisted message in a session transcript."""

    role: str
    content: str
    sequence: int


class TranscriptOut(BaseModel):
    """A session's full transcript (for sub-agent drill-down + chat history)."""

    session_id: str
    messages: list[TranscriptMessageOut]


class ConfigOut(BaseModel):
    """Sanitized runtime configuration for display in the web UI."""

    active_provider: str
    default_lead: str
    model: str
    memory_enabled: bool
    self_repair: bool
    values: dict[str, Any]
