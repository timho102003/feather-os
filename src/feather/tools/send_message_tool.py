"""Tool that sends one inter-agent message via the SQLite mailbox.

Available to every agent role (lead, sub-agents, custom). Unlike the
lead-only ``spawn_agent`` / cron tools, ``send_message`` is the shared
communication primitive: any agent can call it to push a message onto
another agent's inbox. Message delivery is asynchronous — the recipient
drains its inbox at the top of its next ``run_loop`` iteration.

Addressing is explicit: the caller must supply the recipient's
``agent_name`` and ``session_id``. Agents obtain these identifiers from
context they already have:

- Sub-agents know their ``parent_session_id`` + parent name ("lead") from
  their launch env (populated in the agent prompt by the runtime).
- The lead knows each child's ``session_id`` from the ``spawn_agent`` tool
  result.
- For sub-agent-to-sibling messaging, the lead must pass each sibling's
  session_id in the dispatched task (no auto-directory in this slice).

``expects_response=true`` returns a correlation_id that the caller should
include in ``in_reply_to`` on any subsequent reply, and should watch for
on subsequent inbox drains.
"""

from __future__ import annotations

from typing import Any

from feather.core.subagents.registry import SubagentRegistry
from feather.models import ToolExecutionContext, ToolExecutionResult
from feather.storage.agent_message_store import AgentMessageStore
from feather.storage.session_store import SessionStore
from feather.tools.base import BaseTool

_MAX_BODY_CHARS = 16_000


class SendMessageTool(BaseTool):
    """Send one message to another agent's inbox."""

    name = "send_message"
    description = (
        "Send a message to another agent's inbox. Delivery is asynchronous — "
        "the recipient reads it on the top of its next turn. Set "
        "`expects_response` to true when you need the recipient to reply; the "
        "tool returns a correlation_id you can watch for on subsequent turns. "
        "When you are replying to a prior message, set `in_reply_to` to that "
        "message's correlation_id so the conversation is correctly paired."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "to_agent_name": {
                "type": "string",
                "description": (
                    "Catalog name of the recipient agent (e.g. `lead`, `engineer-custom`)."
                ),
            },
            "to_session_id": {
                "type": "string",
                "description": (
                    "Session id of the recipient. Sub-agents messaging the lead pass "
                    "the parent_session_id they were given at launch. The lead gets "
                    "child session ids from the spawn_agent tool result. Sibling "
                    "targets must be passed in by whoever coordinates the agents."
                ),
            },
            "body": {
                "type": "string",
                "description": (
                    "Message body (plain text). Keep it compact and self-contained — "
                    "the recipient only sees this text, not your conversation history."
                ),
            },
            "expects_response": {
                "type": "boolean",
                "description": (
                    "Set true when you need the recipient to reply. The tool allocates "
                    "and returns a correlation_id; the recipient should pass it back "
                    "via `in_reply_to` when they reply."
                ),
            },
            "in_reply_to": {
                "type": ["string", "null"],
                "description": (
                    "Correlation id of the prior message you are replying to, if any. "
                    "Leave null for a fresh thread or a one-way status update."
                ),
            },
        },
        "required": [
            "to_agent_name",
            "to_session_id",
            "body",
            "expects_response",
            "in_reply_to",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        store: AgentMessageStore,
        *,
        from_agent_name: str,
        session_store: SessionStore | None = None,
        subagent_registry: SubagentRegistry | None = None,
    ) -> None:
        self._store = store
        self._from_agent_name = from_agent_name
        self._session_store = session_store
        self._subagent_registry = subagent_registry

    def get_prompt(self) -> str:
        return (
            "- `send_message`: push one message onto another agent's inbox. "
            "Delivery is asynchronous (read on the recipient's next turn). "
            "Always fill `to_agent_name` and `to_session_id`. Set "
            "`expects_response=true` when you need a reply; keep `in_reply_to` "
            "null unless you are replying to a prior correlation_id."
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        to_agent = _require_str(arguments, "to_agent_name")
        to_session = _require_str(arguments, "to_session_id")
        body = _require_str(arguments, "body")
        expects_response = bool(arguments.get("expects_response", False))
        raw_reply = arguments.get("in_reply_to")
        in_reply_to = (
            raw_reply.strip() if isinstance(raw_reply, str) and raw_reply.strip() else None
        )

        if to_agent == self._from_agent_name and to_session == context.session_id:
            raise ValueError(
                "send_message cannot target the sender's own inbox."
            )

        if len(body) > _MAX_BODY_CHARS:
            body = body[:_MAX_BODY_CHARS] + f"\n...(truncated to {_MAX_BODY_CHARS} chars)"

        # Validate the target session exists. If it doesn't, refuse — a
        # typo'd UUID otherwise rots in the recipient's mailbox forever,
        # counting against the cap for any future agent that gets assigned
        # the same id.
        if self._session_store is not None:
            try:
                await self._session_store.get_session(to_session)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"send_message target session_id `{to_session}` is not a "
                    f"known session: {exc}"
                ) from exc

        # Sub-agent liveness check: sub-agent subprocesses exit after
        # their single run_loop completes. Any send_message addressed to
        # a sub-agent whose process has already terminated lands in a
        # dead inbox that nothing will ever drain — leaving the sender
        # waiting for a reply that cannot arrive. Refuse early, with a
        # clear hint that the sub-agent's final report is already in
        # the sender's inbox.
        if self._subagent_registry is not None:
            live = await self._subagent_registry.get(to_session)
            if live is not None:
                proc_returncode = getattr(live.process, "returncode", None)
                if proc_returncode is not None:
                    raise ValueError(
                        f"send_message target `{to_agent}` "
                        f"(session_id={to_session}) has already exited "
                        f"(returncode={proc_returncode}). Its final report, if "
                        f"any, is already in your inbox — do not send further "
                        f"messages to it; spawn a new sub-agent if more work is "
                        f"needed."
                    )
            elif await self._subagent_registry.is_recently_exited(to_session):
                # The reaper has already delivered this sub-agent's
                # final report and removed it from the live registry.
                raise ValueError(
                    f"send_message target sub-agent `{to_agent}` "
                    f"(session_id={to_session}) has finished and exited. "
                    f"Its final report is in your inbox — do not send further "
                    f"messages to it; spawn a new sub-agent if more work is "
                    f"needed."
                )

        message = await self._store.send(
            from_session_id=context.session_id,
            from_agent_name=self._from_agent_name,
            to_session_id=to_session,
            to_agent_name=to_agent,
            body=body,
            expects_response=expects_response,
            in_reply_to=in_reply_to,
        )

        lines = [
            f"message_id: {message.id}",
            f"to: {to_agent} (session_id={to_session})",
            f"status: {message.status.value}",
        ]
        if message.correlation_id is not None:
            lines.append(f"correlation_id: {message.correlation_id}")
        if in_reply_to is not None:
            lines.append(f"in_reply_to: {in_reply_to}")
        return ToolExecutionResult(output="\n".join(lines))


def _require_str(arguments: dict[str, Any], key: str) -> str:
    raw = arguments.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"send_message `{key}` must be a non-empty string.")
    return raw.strip()
