"""Base tool abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from feather.models import ToolExecutionContext, ToolExecutionResult


class BaseTool(ABC):
    """Abstract interface shared by all Feather tools."""

    name: str
    description: str
    parameters_schema: dict[str, Any]

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert the tool to the OpenAI Responses function-tool schema.

        Returns:
            OpenAI-compatible tool schema.
        """

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
            "strict": True,
        }

    def get_prompt(self) -> str:
        """Describe the tool for prompt assembly.

        Returns:
            Prompt text describing usage and intent.
        """

        return f"- `{self.name}`: {self.description}"

    @abstractmethod
    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        """Execute the tool.

        Args:
            arguments: Validated tool arguments.
            context: Runtime context for the current tool invocation.

        Returns:
            Tool execution result.
        """
