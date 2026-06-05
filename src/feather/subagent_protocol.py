"""Constants shared between the spawn_agent tool and the sub-agent subprocess entry.

Kept in its own module so importing the markers does not drag in
``feather.runtime`` (the sub-agent entry) or ``feather.core.agent.factory``
(the tool), which would form an import cycle.
"""

from __future__ import annotations

RESULT_BEGIN = "##FEATHER_SUBAGENT_RESULT_BEGIN##"
RESULT_END = "##FEATHER_SUBAGENT_RESULT_END##"
