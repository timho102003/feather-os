# Scheduling

You can ask the agent to do something later. Once. On a schedule. At a
specific time. The agent owns a small set of cron-style tools that let
it schedule prompts to fire back into the same chat session.

## The simplest version

In a chat:

> "Every weekday at 9 AM, look at my GitHub notifications and tell me
> what needs attention."

> "Tomorrow at 3 PM, remind me to review the PR queue."

> "In 30 minutes, run the test suite and post a summary."

The agent figures out the schedule, calls `create_cron`, and confirms.
When the time comes, the scheduled prompt is delivered as if you had
typed it yourself, and the agent processes it.

You don't have to know cron syntax. Plain English works.

## What you get

When a job fires:

1. The scheduler delivers the saved prompt into your session's inbox.
2. The agent picks it up at the start of the next turn.
3. You see a system event in the transcript: "scheduled task
   triggered: <name>".
4. The agent responds normally.

If you are not watching the chat at the time, the response is still
written to history. Next time you open the session, the conversation
already has the result.

If you connected Telegram, LINE, or WhatsApp (see
[messaging.md](messaging.md)), the response goes there too.

## Tools the agent uses

| Tool | What it does |
|---|---|
| `create_cron` | Create a recurring or one-time job. |
| `update_cron` | Change schedule, prompt, or status. |
| `delete_cron` | Remove a job permanently. |
| `list_crons` | List the session's jobs. |

You don't call these directly. You ask the agent and the agent calls
them.

## Schedule formats

Two kinds of schedule, both passed to `create_cron`:

* **`cron`**: a five-field cron expression, like `0 9 * * 1-5` for
  weekdays at 9 AM, plus an IANA timezone like `America/New_York`.
* **`once`**: a single ISO 8601 datetime, like
  `2026-05-15T15:00:00-04:00`.

The agent picks the right format based on what you asked for.

## Scope: jobs belong to the session that created them

A job is attached to the chat session it was created in. When you
resume that session, the job's results show up there. If you delete or
abandon the session, its jobs go with it.

If you want jobs to keep firing across many short-lived chats, work
inside one long-running session and resume it with
`feather --session-id <id>` each time.

## Inspecting jobs

Ask the agent: "list my scheduled jobs." The agent calls `list_crons`
and prints them.

Or talk about a specific one: "delete the daily PR-review job" or
"change the weekday standup reminder to 8:30 AM instead of 9."

## Server-side knobs

The scheduler runs as a background loop. Defaults in
`~/.feather/config/app.yaml`:

```yaml
scheduler:
  enabled: true
  poll_interval_seconds: 2          # how often the scheduler ticks
  failure_retry_seconds: 30         # backoff after a failed delivery
  max_due_jobs_per_tick: 10         # cap on how many jobs fire in one tick
```

Defaults are sensible for personal use. Bump `poll_interval_seconds`
if you have many sessions and want to spread load. Set `enabled:
false` to disable scheduling entirely (existing jobs stay in the
database but never fire).

## Things to avoid

* **Don't schedule destructive jobs without thinking through what
  happens if the agent fails.** A nightly "delete old branches" job
  that misclassifies "old" can destroy work. Prefer "prepare a list
  of stale branches and email it to me" so a human approves.
* **Don't expect millisecond accuracy.** The scheduler ticks every two
  seconds by default. Jobs may fire a second or two late.
* **Don't use cron jobs to poll external services that have their own
  push hooks.** If you can subscribe instead, do that.

## Next

* For "send me a Telegram every weekday morning" you need
  [messaging.md](messaging.md) connected first.
* For longer-running scheduled work that needs a sub-agent, combine
  scheduling with [agents.md](agents.md).
