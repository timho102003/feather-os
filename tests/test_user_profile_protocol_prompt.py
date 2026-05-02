"""Smoke tests: prompts teach agents about <user_profile> and user_info."""

from __future__ import annotations

from feather.core.prompts.base_agent_prompt import BASE_AGENT_PROMPT
from feather.core.prompts.lead_agent_prompt import LEAD_AGENT_PROMPT


def test_lead_prompt_documents_user_info_protocol() -> None:
    """The lead prompt must teach the lead when to use user_info."""

    assert "<user_profile_protocol>" in LEAD_AGENT_PROMPT
    assert "user_info" in LEAD_AGENT_PROMPT
    # Distinct from manage_memory (semantic) so the model doesn't conflate them.
    assert "manage_memory" in LEAD_AGENT_PROMPT


def test_base_prompt_explains_user_profile_block_is_data() -> None:
    """Sub-agents see <user_profile> too; they must treat it as read-only data."""

    assert "<user_profile>" in BASE_AGENT_PROMPT
    assert "do not edit" in BASE_AGENT_PROMPT.lower()
