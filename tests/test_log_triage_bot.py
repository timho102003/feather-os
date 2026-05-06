"""Tests for the supervisor-side error-triage bot.

The bot tails ``.feather/logs/feather.log``, filters for ERROR-level
entries since its last reported timestamp, and posts a single summary
message into the lead's mailbox so the lead can investigate. The bot
runs only in worker mode (the supervisor is the consumer of these
notifications); in default in-process mode the lead would just
re-trigger its own log lines, which would never converge.
"""

from __future__ import annotations

from pathlib import Path

from feather.core.log_triage_bot import LogTriageBot, parse_log_errors
from feather.storage.agent_message_store import AgentMessageStore


def _write_log(tmp_path: Path, body: str) -> Path:
    log = tmp_path / "feather.log"
    log.write_text(body, encoding="utf-8")
    return log


# --------------------------------------------------------------------- #
# parse_log_errors — pure function
# --------------------------------------------------------------------- #


def test_parse_log_errors_returns_only_error_lines() -> None:
    """Non-ERROR lines (INFO/WARNING/DEBUG) must be filtered out."""

    body = (
        "2026-05-05 09:10:11 | INFO | Lead | s1 | feather.module | hi\n"
        "2026-05-05 09:10:12 | ERROR | Lead | s1 | feather.module | boom\n"
        "2026-05-05 09:10:13 | WARNING | Lead | s1 | feather.module | meh\n"
        "2026-05-05 09:10:14 | ERROR | Lead | s1 | feather.module | crash\n"
    )
    errors = parse_log_errors(body)
    assert len(errors) == 2
    assert errors[0].message == "boom"
    assert errors[1].message == "crash"


def test_parse_log_errors_skips_malformed_lines() -> None:
    """Lines that don't match the expected pipe-delimited format are skipped."""

    body = (
        "2026-05-05 09:10:11 | ERROR | Lead | s1 | feather.module | real error\n"
        "this is not a structured log line\n"
        "\n"
        "  | malformed |\n"
        "2026-05-05 09:10:12 | ERROR | Lead | s1 | feather.module | another\n"
    )
    errors = parse_log_errors(body)
    assert len(errors) == 2
    assert {e.message for e in errors} == {"real error", "another"}


def test_parse_log_errors_handles_empty_body() -> None:
    assert parse_log_errors("") == []
    assert parse_log_errors("\n\n") == []


# --------------------------------------------------------------------- #
# LogTriageBot.run_once — integration with mailbox
# --------------------------------------------------------------------- #


async def _open_message_store(tmp_path: Path) -> AgentMessageStore:
    store = AgentMessageStore(tmp_path / "feather.db")
    await store.initialize()
    return store


async def test_run_once_sends_summary_when_errors_present(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        "2026-05-05 09:10:11 | ERROR | Lead | s1 | feather.tool | tool failed: timeout\n"
        "2026-05-05 09:10:12 | INFO | Lead | s1 | feather.module | normal\n"
        "2026-05-05 09:10:13 | ERROR | Lead | s1 | feather.tool | tool failed: oom\n",
    )
    store = await _open_message_store(tmp_path)
    try:
        bot = LogTriageBot(
            log_path=log,
            message_store=store,
            lead_session_id="s-lead",
        )
        sent = await bot.run_once()
        assert sent == 1
        inbox = await store.inbox(to_session_id="s-lead", to_agent_name="Lead")
        assert len(inbox) == 1
        msg = inbox[0]
        assert "tool failed: timeout" in msg.body
        assert "tool failed: oom" in msg.body
        assert msg.from_agent_name == "__system_log_triage"
    finally:
        await store.close()


async def test_run_once_does_not_send_when_no_errors(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        "2026-05-05 09:10:11 | INFO | Lead | s1 | feather.module | nothing wrong\n",
    )
    store = await _open_message_store(tmp_path)
    try:
        bot = LogTriageBot(
            log_path=log, message_store=store, lead_session_id="s-lead"
        )
        sent = await bot.run_once()
        assert sent == 0
        inbox = await store.inbox(to_session_id="s-lead", to_agent_name="Lead")
        assert inbox == []
    finally:
        await store.close()


async def test_run_once_deduplicates_already_reported_errors(
    tmp_path: Path,
) -> None:
    """A second tick over the same log must not re-report the same errors."""

    log = _write_log(
        tmp_path,
        "2026-05-05 09:10:11 | ERROR | Lead | s1 | feather.tool | repeated failure\n",
    )
    store = await _open_message_store(tmp_path)
    try:
        bot = LogTriageBot(
            log_path=log, message_store=store, lead_session_id="s-lead"
        )
        first = await bot.run_once()
        second = await bot.run_once()
        assert first == 1
        assert second == 0
        inbox = await store.inbox(to_session_id="s-lead", to_agent_name="Lead")
        assert len(inbox) == 1
    finally:
        await store.close()


async def test_run_once_picks_up_new_errors_after_first_run(tmp_path: Path) -> None:
    """A new ERROR appearing in the log between ticks must be reported."""

    log = _write_log(
        tmp_path,
        "2026-05-05 09:10:11 | ERROR | Lead | s1 | feather.tool | first failure\n",
    )
    store = await _open_message_store(tmp_path)
    try:
        bot = LogTriageBot(
            log_path=log, message_store=store, lead_session_id="s-lead"
        )
        await bot.run_once()
        # Append a new error.
        with log.open("a", encoding="utf-8") as fh:
            fh.write(
                "2026-05-05 09:11:00 | ERROR | Lead | s1 | feather.tool | second failure\n"
            )
        sent = await bot.run_once()
        assert sent == 1
        inbox = await store.inbox(to_session_id="s-lead", to_agent_name="Lead")
        # First message + second message; both still pending until lead drains.
        assert len(inbox) == 2
        assert "second failure" in inbox[1].body
    finally:
        await store.close()


async def test_run_once_caps_summary_at_max_errors(tmp_path: Path) -> None:
    """A flood of errors must be capped so the inbox message stays readable."""

    lines = [
        f"2026-05-05 09:10:{i:02d} | ERROR | Lead | s1 | feather.x | err {i}\n"
        for i in range(40)
    ]
    log = _write_log(tmp_path, "".join(lines))
    store = await _open_message_store(tmp_path)
    try:
        bot = LogTriageBot(
            log_path=log,
            message_store=store,
            lead_session_id="s-lead",
            max_errors_per_summary=10,
        )
        await bot.run_once()
        inbox = await store.inbox(to_session_id="s-lead", to_agent_name="Lead")
        assert len(inbox) == 1
        # Body should include the cap notice and at most 10 enumerated errors.
        body = inbox[0].body
        assert "showing 10 of 40" in body.lower() or "showing 10 / 40" in body.lower()
    finally:
        await store.close()


async def test_dedup_high_water_mark_blocks_old_re_reports(tmp_path: Path) -> None:
    """An old fingerprint that's been evicted from the recency set must
    still be filtered by the high-water mark — without it a chatty log
    with > dedup_cap distinct errors would re-report old ones forever.
    """

    log = _write_log(
        tmp_path,
        "2026-05-05 09:10:01 | ERROR | Lead | s1 | feather.x | err one\n"
        "2026-05-05 09:10:02 | ERROR | Lead | s1 | feather.x | err two\n",
    )
    store = await _open_message_store(tmp_path)
    try:
        # Tiny dedup cap so the second tick provably evicts the first.
        bot = LogTriageBot(
            log_path=log,
            message_store=store,
            lead_session_id="s-lead",
            dedup_cap=1,
        )
        first = await bot.run_once()
        # Second tick: same log content, no new entries, must be 0
        # even though the recency set evicted "err one" via the cap.
        second = await bot.run_once()
        assert first == 1
        assert second == 0
    finally:
        await store.close()


async def test_dedup_resets_on_log_rotation(tmp_path: Path) -> None:
    """If the inode changes (logrotate moved the file aside and a fresh
    log appeared), the dedup state must reset so the rotated log's
    first errors aren't suppressed."""

    log = _write_log(
        tmp_path,
        "2026-05-05 09:10:01 | ERROR | Lead | s1 | feather.x | repeated msg\n",
    )
    store = await _open_message_store(tmp_path)
    try:
        bot = LogTriageBot(
            log_path=log, message_store=store, lead_session_id="s-lead"
        )
        first = await bot.run_once()
        # Simulate logrotate: unlink the old file (changes inode on
        # the next create) and write a new one with the same content.
        log.unlink()
        log.write_text(
            "2026-05-05 09:11:01 | ERROR | Lead | s1 | feather.x | repeated msg\n",
            encoding="utf-8",
        )
        second = await bot.run_once()
        # NEW timestamp, new inode → dedup must reset and report it.
        assert first == 1
        assert second == 1
    finally:
        await store.close()


async def test_run_once_handles_missing_log_file(tmp_path: Path) -> None:
    """If the log file doesn't exist yet, the bot must no-op cleanly."""

    store = await _open_message_store(tmp_path)
    try:
        bot = LogTriageBot(
            log_path=tmp_path / "does-not-exist.log",
            message_store=store,
            lead_session_id="s-lead",
        )
        sent = await bot.run_once()
        assert sent == 0
    finally:
        await store.close()
