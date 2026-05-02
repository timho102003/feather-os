"""Tool for loading full skill content on demand."""

from __future__ import annotations

from typing import Any

from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.skills.catalog import SkillCatalog
from feather.tools.base import BaseTool


class LoadSkillTool(BaseTool):
    """Load one skill from the catalog."""

    name = "load_skill"
    description = "Load the full contents of a skill by exact name after reviewing the available skill metadata."
    parameters_schema = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "The exact skill name to load.",
            }
        },
        "required": ["skill_name"],
        "additionalProperties": False,
    }

    def __init__(self, skill_catalog: SkillCatalog) -> None:
        self._skill_catalog = skill_catalog

    def get_prompt(self) -> str:
        """Describe the load-skill workflow for prompt assembly.

        Returns:
            Prompt text specific to skill loading.
        """

        return (
            "- `load_skill`: load the full instructions for one skill by exact name. "
            "Review the skill catalog in the prompt first, then load only the skills you need."
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        """Load a skill so it is available in the next prompt build.

        Args:
            arguments: Tool arguments from the model.
            context: Runtime context for the current tool invocation.

        Returns:
            Tool result confirming the loaded skill.
        """

        skill_name = arguments["skill_name"]
        loaded = self._skill_catalog.load_skill(skill_name)
        return ToolExecutionResult(
            output=f"Loaded skill `{loaded.metadata.name}`. It will be included in the next prompt.",
            loaded_skill_name=loaded.metadata.name,
        )
