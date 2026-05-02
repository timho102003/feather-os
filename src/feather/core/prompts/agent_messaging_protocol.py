"""Shared messaging-protocol prompt included by every agent that has an inbox.

The block below describes how an agent should process the
``<incoming_agent_messages>`` block that ``BaseAgent._drain_agent_inbox``
injects at the top of each loop iteration, and how to use
``send_message`` to reply. It is intentionally short and framed as rules
the model can scan quickly.
"""

AGENT_MESSAGING_PROTOCOL_PROMPT = """
<agent_messaging_protocol>
You may receive messages from other agents at the top of a turn, wrapped in:

  <incoming_agent_messages from_agent="..." from_session="...">
    <message id="..." correlation_id="..." [in_reply_to="..."] [expects_response="true"]>
      ...body...
    </message>
    ...
  </incoming_agent_messages>

How to handle them:

1. Messages in one block all come from the same sender. Other senders'
   messages will arrive on subsequent turns — handle each group fully
   before moving on.
2. For each message:
   - If `expects_response="true"`, call `send_message` with
     `to_agent_name` and `to_session_id` taken from the `from_agent` /
     `from_session` attributes of the block and set `in_reply_to` to the
     message's `correlation_id`.
   - Otherwise no reply is required; acknowledge silently and move on.
3. After handling the messages, ALWAYS resume your ongoing task. Do not
   produce a final "done" assistant turn just because you answered
   messages — continue with the original goal that was given to you.
4. When you yourself send a message via `send_message`, remember the
   `correlation_id` returned by the tool if you set `expects_response=true`
   — you will see the reply pop up in a future `<incoming_agent_messages>`
   block with `in_reply_to` set to that id.

Addressing:
- Sub-agents: the parent's session_id and agent_name are given to you at
  launch inside `<parent_session_id>`, `<parent_agent_name>` tags. Use
  them verbatim when messaging the parent.
- Lead: each spawned sub-agent's `session_id` is returned by the
  `spawn_agent` tool. Save it; pass it as `to_session_id` when talking
  to that sub-agent.
- Siblings: ask whoever dispatched both of you to share the relevant
  session_id; otherwise assume you do not know it.
</agent_messaging_protocol>
""".strip()
