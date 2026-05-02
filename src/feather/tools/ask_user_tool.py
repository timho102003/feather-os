"""Tool for handing control back to the user."""

from __future__ import annotations

from typing import Any

from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.tools.base import BaseTool


class AskUserTool(BaseTool):
    """Pause the agent and request more user input."""

    name = "ask_user"
    description = "Ask the user a focused question when you are blocked or a requirement is ambiguous."
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The exact question to ask the user.",
            }
        },
        "required": ["question"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        """Return a user-facing question.

        Args:
            arguments: Tool arguments from the model.
            context: Runtime context for the current tool invocation.

        Returns:
            Tool result carrying the question.
        """

        question = arguments["question"].strip()
        return ToolExecutionResult(
            output=f"User input required: {question}",
            await_user_question=question,
        )
