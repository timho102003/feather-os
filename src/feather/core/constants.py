"""Shared canonical constants used across the Feather core.

Centralised so callers don't end up with case-divergent string literals
that look right at the call site but silently miss the recipient row
on a case-sensitive SQLite filter (the exact bug that stranded every
log_triage / restart_watcher / cron message until it was caught).
"""

from __future__ import annotations


# Canonical lead agent name. Must match
# ``src/feather/_resources/config/agents/lead.yaml``'s ``name:`` field
# byte-for-byte — the lead's ``BaseAgent`` filters its inbox via
# ``WHERE to_agent_name = self._agent_config.name`` and SQLite string
# equality is case-sensitive by default. Any sender that addresses the
# lead inbox MUST use this constant rather than a literal "lead" /
# "Lead" / etc.
LEAD_AGENT_NAME = "Lead"
