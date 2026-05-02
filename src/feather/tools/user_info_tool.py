"""Lead-only tool that keeps ``.feather/user.md`` current.

The tool mirrors :class:`feather.tools.manage_memory_tool.ManageMemoryTool`'s
shape but writes the deterministic profile file instead of semantic
memory. Sub-agents must not register it: the lead is the only role in
direct conversation with the user, so it is the only role allowed to
mutate the user's identity.
"""

from __future__ import annotations

import logging
from typing import Any

from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.profile import UserProfileStore
from feather.tools.base import BaseTool

logger = logging.getLogger(__name__)


class UserInfoTool(BaseTool):
    """Maintain ``.feather/user.md`` via deterministic CRUD / APPEND_NOTE."""

    name = "user_info"
    description = (
        "Maintain the persistent user profile (.feather/user.md). Use when "
        "the user shares NEW personal information (name, role, preferences, "
        "ongoing projects) so future turns and future sessions remember it. "
        "Operations: CREATE (new field), UPDATE (correct existing field), "
        "DELETE (forget a field), APPEND_NOTE (free-form bullet under "
        "Notes). Reserved fields `created_at` and `updated_at` are managed "
        "automatically."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["operation", "field", "value", "note"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["CREATE", "UPDATE", "DELETE", "APPEND_NOTE"],
                "description": (
                    "CREATE = persist a new structured field. UPDATE = "
                    "replace an existing structured field. DELETE = forget "
                    "a structured field. APPEND_NOTE = add a dated bullet "
                    "under the free-form Notes section."
                ),
            },
            "field": {
                "type": ["string", "null"],
                "description": (
                    "Snake_case structured-field key. REQUIRED for CREATE, "
                    "UPDATE, DELETE; null for APPEND_NOTE."
                ),
            },
            "value": {
                "type": ["string", "null"],
                "description": (
                    "Value for the structured field. REQUIRED for CREATE "
                    "and UPDATE; null for DELETE and APPEND_NOTE."
                ),
            },
            "note": {
                "type": ["string", "null"],
                "description": (
                    "Free-form note text. REQUIRED for APPEND_NOTE; null "
                    "for the other operations."
                ),
            },
        },
    }

    def __init__(self, store: UserProfileStore) -> None:
        self._store = store

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolExecutionResult:
        del context
        operation = (arguments.get("operation") or "").strip().upper()
        if operation not in {"CREATE", "UPDATE", "DELETE", "APPEND_NOTE"}:
            return _err(f"invalid operation {operation!r}.")
        field_name = _clean(arguments.get("field"))
        value = _clean(arguments.get("value"))
        note = _clean(arguments.get("note"))
        try:
            if operation == "CREATE":
                if field_name is None or value is None:
                    return _err("CREATE requires both `field` and `value`.")
                await self._store.create(field_name, value)
                return _ok(f"created field `{field_name}`.")
            if operation == "UPDATE":
                if field_name is None or value is None:
                    return _err("UPDATE requires both `field` and `value`.")
                await self._store.update(field_name, value)
                return _ok(f"updated field `{field_name}`.")
            if operation == "DELETE":
                if field_name is None:
                    return _err("DELETE requires `field`.")
                await self._store.delete(field_name)
                return _ok(f"deleted field `{field_name}`.")
            if note is None:
                return _err("APPEND_NOTE requires `note`.")
            await self._store.append_note(note)
            return _ok("appended note.")
        except ValueError as exc:
            return _err(str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("user_info.unexpected_error op=%s", operation)
            return _err(f"unexpected error: {exc}")


def _clean(value: object) -> str | None:
    """Return a stripped non-empty string or ``None``."""

    if value is None or not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _ok(message: str) -> ToolExecutionResult:
    return ToolExecutionResult(output=f"user_info: {message}")


def _err(message: str) -> ToolExecutionResult:
    return ToolExecutionResult(output=f"user_info: {message}")


__all__ = ["UserInfoTool"]
