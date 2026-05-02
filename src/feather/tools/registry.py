"""Tool registry for agent prompt and execution wiring."""

from __future__ import annotations

from typing import Iterable

from feather.tools.base import BaseTool


class ToolRegistry:
    """Registry for all available local tools."""

    def __init__(self, tools: Iterable[BaseTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def get(self, tool_name: str) -> BaseTool:
        """Look up one tool by name.

        Args:
            tool_name: Registered tool name.

        Returns:
            Tool instance.

        Raises:
            KeyError: If the tool is not registered.
        """

        return self._tools[tool_name]

    def prompts_for(self, tool_names: list[str]) -> list[str]:
        """Collect prompt descriptions for registered tools.

        Args:
            tool_names: Ordered tool names from agent config.

        Returns:
            Prompt snippets for those tools.
        """

        return [self.get(name).get_prompt() for name in tool_names]

    def openai_tools_for(self, tool_names: list[str]) -> list[dict]:
        """Collect OpenAI tool schemas for registered tools.

        Args:
            tool_names: Ordered tool names from agent config.

        Returns:
            OpenAI-compatible tool definitions.
        """

        return [self.get(name).to_openai_tool() for name in tool_names]
