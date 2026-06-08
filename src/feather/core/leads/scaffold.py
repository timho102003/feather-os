"""Scaffold a project lead YAML for a runtime-created lead.

Shared by the TUI (``/lead new``) and the API (``POST /api/leads``) so both
create identical, immediately-loadable lead configs. Web tools are omitted on
purpose — they need a Parallel client, which isn't always configured, and an
unknown tool would make the factory refuse to build the agent.
"""

from __future__ import annotations

from pathlib import Path

from feather.core.agent.catalog import AgentCatalog
from feather.core.leads.soul import Soul

__all__ = ("scaffold_lead_yaml", "is_valid_lead_name")

_DEFAULT_TOOLS = (
    "read_file",
    "write_file",
    "grep",
    "bash",
    "ask_user",
    "load_skill",
    "spawn_agent",
    "terminate_agent",
    "send_message",
    "task_create",
    "task_list",
    "task_get",
    "task_update",
    "task_output",
    "task_stop",
    "task_resume",
)


def is_valid_lead_name(name: str) -> bool:
    """Allow alnum + ``_`` + ``-`` only (mirrors AgentCatalog.is_valid_name)."""

    return AgentCatalog.is_valid_name(name)


def scaffold_lead_yaml(
    root: Path,
    name: str,
    soul: str = "",
    *,
    soul_preset: Soul | None = None,
) -> Path:
    """Write ``config/agents/<name>.yaml`` under ``root`` (idempotent).

    Returns the path. Does not overwrite an existing file. ``name`` is assumed
    already validated by :func:`is_valid_lead_name` and is always the filename
    stem.

    The lead's ``name`` always comes from ``name`` (the user's choice) — a soul
    is a reusable temperament, not an identity, so it never renames the lead.
    When ``soul_preset`` is given, the preset's ``personality``, working-character
    ``prose`` (as the ``soul`` block), ``color``, and ``emoji`` are applied. When
    it is ``None`` the free-text ``soul`` argument is used exactly as before (its
    first line is the personality; no color/emoji).
    """

    agents_dir = Path(root) / "config" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / f"{name}.yaml"
    if path.exists():
        return path

    display_name = name.capitalize()
    if soul_preset is not None:
        personality = soul_preset.personality
        prose = soul_preset.prose.strip()
        # color starts with '#', which YAML reads as a comment unless quoted.
        identity_block = f'color: "{soul_preset.color}"\nemoji: "{soul_preset.emoji}"\n'
    else:
        prose = soul.strip()
        personality = prose.splitlines()[0] if prose else "A focused, helpful lead."
        identity_block = ""

    soul_block = ""
    if prose:
        indented = "\n".join(f"  {line}" for line in prose.splitlines())
        soul_block = f"soul: |\n{indented}\n"
    tools_block = "\n".join(f"  - {tool}" for tool in _DEFAULT_TOOLS)
    path.write_text(
        f"name: {display_name}\n"
        "role: lead\n"
        f"personality: {personality}\n"
        f"{identity_block}"
        f"{soul_block}"
        "memory_enabled: true\n"
        "prompt_modules:\n"
        "  - feather.core.prompts.base_agent_prompt:BASE_AGENT_PROMPT\n"
        "  - feather.core.prompts.agent_messaging_protocol:AGENT_MESSAGING_PROTOCOL_PROMPT\n"
        "  - feather.core.prompts.lead_agent_prompt:LEAD_AGENT_PROMPT\n"
        "registered_tools:\n"
        f"{tools_block}\n",
        encoding="utf-8",
    )
    return path
