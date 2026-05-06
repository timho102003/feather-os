"""Supervisor-side bot that surfaces recent ERROR log lines to the lead.

When the lead runs as a worker subprocess (see
:class:`feather.core.lead_supervisor.LeadSupervisor`), the supervisor
process owns the log file. The lead has no in-process access to the
file's tail; without a bridge, an error inside a tool would scroll past
in the logs and the lead would stay oblivious. This bot scans the log
file periodically, extracts ERROR-level entries the lead has not seen
yet, and posts a single summary message into the lead's mailbox via
:class:`feather.storage.agent_message_store.AgentMessageStore`.

The lead picks the message up via its existing ``resume_on_inbox``
machinery — no new agent-side code is required. The bot is a
pure consumer of the log file plus a producer for the mailbox; it
does not itself drive the agent loop.

Design notes:

* In-memory dedup. The bot tracks a bounded set of "already reported"
  fingerprints (timestamp + message text) so a tick that overlaps a
  previous tick does not re-report. The set is intentionally bounded
  to keep the bot's memory footprint flat over a long-running session.
* No on-disk cursor. The bot starts from scratch on TUI relaunch; in
  the worst case the user sees a duplicate report shortly after
  restart, which is preferable to losing a real error to a stale
  cursor.
* Worker-mode only. In default in-process mode the lead would drive its
  own tool failures into its own conversation — a triage echo loop
  would be a footgun. The TUI gates the bot on the env flag.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from feather.core.constants import LEAD_AGENT_NAME
from feather.storage.agent_message_store import AgentMessageStore

logger = logging.getLogger(__name__)


# Matches the structured log line emitted by ``logging_utils.configure_logging``:
#   "<ISO timestamp> | <LEVEL> | <agent> | <session_id> | <logger> | <message>"
_LOG_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|"
    r"\s*(?P<level>\w+)\s*\|"
    r"\s*[^|]*\|"  # agent name (we don't surface this)
    r"\s*[^|]*\|"  # session_id
    r"\s*(?P<logger>[^|]+)\|"
    r"\s*(?P<message>.+)$"
)

_DEFAULT_MAX_ERRORS_PER_SUMMARY = 10
_DEFAULT_DEDUP_CAP = 256
_DEFAULT_SCAN_INTERVAL_SECONDS = 60.0
_DEFAULT_TAIL_BYTES = 256 * 1024  # 256 KB — bounded read so we don't load
# unbounded log history on first scan.


@dataclass(slots=True, frozen=True)
class LogError:
    """One ERROR-level log entry parsed off the structured log file."""

    timestamp: str
    logger_name: str
    message: str

    def fingerprint(self) -> tuple[str, str]:
        """Stable identity for dedup — timestamp + message text."""

        return (self.timestamp, self.message)


def parse_log_errors(body: str) -> list[LogError]:
    """Extract every ERROR-level entry from ``body``.

    Lines that don't match the structured format are silently skipped.
    """

    errors: list[LogError] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _LOG_LINE.match(line)
        if match is None:
            continue
        if match.group("level") != "ERROR":
            continue
        errors.append(
            LogError(
                timestamp=match.group("ts").strip(),
                logger_name=match.group("logger").strip(),
                message=match.group("message").strip(),
            )
        )
    return errors


class LogTriageBot:
    """Periodic scanner that posts ERROR-line summaries to the lead's mailbox."""

    def __init__(
        self,
        *,
        log_path: Path,
        message_store: AgentMessageStore,
        lead_session_id: str,
        scan_interval_seconds: float = _DEFAULT_SCAN_INTERVAL_SECONDS,
        max_errors_per_summary: int = _DEFAULT_MAX_ERRORS_PER_SUMMARY,
        tail_bytes: int = _DEFAULT_TAIL_BYTES,
        dedup_cap: int = _DEFAULT_DEDUP_CAP,
    ) -> None:
        if scan_interval_seconds <= 0:
            raise ValueError("scan_interval_seconds must be positive")
        if max_errors_per_summary <= 0:
            raise ValueError("max_errors_per_summary must be positive")
        self._log_path = log_path
        self._message_store = message_store
        self._lead_session_id = lead_session_id
        self._scan_interval = scan_interval_seconds
        self._max_per_summary = max_errors_per_summary
        self._tail_bytes = tail_bytes
        # Two-layer dedup so chatty sessions don't re-report old errors:
        #   1) high-water mark — drop any entry whose (ts, msg) sorts at
        #      or before the last reported one. Bounded memory, no
        #      eviction edge-cases.
        #   2) bounded recency set — fast O(1) membership check for the
        #      last N reported fingerprints. Belt-and-braces against
        #      out-of-order log writes (rare under the stdlib logger but
        #      not impossible under contention).
        self._high_water: tuple[str, str] | None = None
        self._reported: deque[tuple[str, str]] = deque(maxlen=dedup_cap)
        self._reported_set: set[tuple[str, str]] = set()
        # Track the file's inode so we can reset dedup state when
        # logrotate moves feather.log out from under us. Without this a
        # rotated log's first error could be silently suppressed if its
        # fingerprint matches one we'd already reported in the prior
        # generation.
        self._last_inode: int | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def run_once(self) -> int:
        """Scan the log file once, post a summary if new errors are present.

        Returns the number of mailbox messages sent (0 or 1).
        """

        # Off-thread file I/O: even a 256 KB read from a busy log can
        # stall the asyncio loop noticeably; the supervisor process is
        # also serving the TUI, so we owe it responsiveness.
        body = await asyncio.to_thread(self._read_tail)
        if not body:
            return 0
        errors = parse_log_errors(body)
        new = [e for e in errors if self._is_new(e.fingerprint())]
        if not new:
            return 0
        await self._post_summary(new, total_in_window=len(errors))
        self._remember(new)
        return 1

    def _is_new(self, fp: tuple[str, str]) -> bool:
        """Decide whether ``fp`` should be reported.

        Two filters run in parallel:

        1. **High-water mark** — drop anything that sorts at or before
           the last reported fingerprint. Catches the "log already grew
           past this entry" case without needing every old fingerprint
           in memory.
        2. **Recency set** — extra guard against out-of-order log
           writes so we don't re-report a fingerprint we just sent.
        """

        if fp in self._reported_set:
            return False
        if self._high_water is not None and fp <= self._high_water:
            return False
        return True

    async def start(self) -> None:
        """Launch the periodic scanner as a background task."""

        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(), name="log_triage_bot")

    async def stop(self) -> None:
        """Cancel the background scanner."""

        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                # Surface bugs in _loop instead of silently swallowing.
                logger.exception("log_triage_bot.stop_task_failed")
            self._task = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._scan_interval
                )
                return
            except asyncio.TimeoutError:
                pass
            try:
                await self.run_once()
            except Exception:  # noqa: BLE001
                logger.exception("log_triage_bot.tick_failed")

    def _read_tail(self) -> str:
        """Read at most the last ``tail_bytes`` of the log file.

        Detects log rotation by comparing the file's inode against the
        last seen value; on a change we reset the dedup state so the
        rotated log's first errors are treated as fresh.
        """

        try:
            stat = self._log_path.stat()
        except FileNotFoundError:
            return ""
        if self._last_inode is not None and stat.st_ino != self._last_inode:
            # Log was rotated (logrotate moved the old file aside and
            # created a new one with a different inode). Reset dedup so
            # we don't suppress an error that happens to share a
            # fingerprint with one from the rotated-out generation.
            self._high_water = None
            self._reported.clear()
            self._reported_set.clear()
        self._last_inode = stat.st_ino
        size = stat.st_size
        offset = max(0, size - self._tail_bytes)
        try:
            with self._log_path.open("rb") as fh:
                if offset:
                    fh.seek(offset)
                raw = fh.read()
        except OSError:
            logger.exception(
                "log_triage_bot.read_failed path=%s", self._log_path
            )
            return ""
        text = raw.decode("utf-8", errors="replace")
        # Drop a partial first line when we seeked into the file mid-line so
        # ``parse_log_errors`` doesn't see a truncated timestamp prefix.
        if offset and "\n" in text:
            text = text.split("\n", 1)[1]
        return text

    async def _post_summary(
        self, errors: list[LogError], *, total_in_window: int
    ) -> None:
        capped = errors[: self._max_per_summary]
        header = (
            f"Found {total_in_window} ERROR-level log entries "
            f"in the last scan window."
        )
        if total_in_window > self._max_per_summary:
            header += f" Showing {self._max_per_summary} / {total_in_window}."
        bullets = "\n".join(
            f"- [{e.timestamp}] {e.logger_name}: {e.message}" for e in capped
        )
        body = (
            f"{header}\n\n{bullets}\n\n"
            "These entries were captured by the supervisor's log triage "
            "bot. If any look like a real bug, investigate (read_file, "
            "git log, etc) and either patch + request_restart, or report "
            "them upstream via submit_github_report."
        )
        await self._message_store.send(
            from_session_id=self._lead_session_id,
            # Namespaced "from" name so a future user-defined agent
            # called "system" can't shadow these supervisor-side bot
            # messages.
            from_agent_name="__system_log_triage",
            to_session_id=self._lead_session_id,
            # The lead's BaseAgent filters its inbox by exact name match
            # (case-sensitive SQL). Use the canonical constant to avoid
            # the silent-strand bug where "lead" rows are never drained
            # because the agent's own name is "Lead".
            to_agent_name=LEAD_AGENT_NAME,
            body=body,
            expects_response=False,
        )
        logger.info(
            "log_triage_bot posted summary lead_session=%s errors=%s",
            self._lead_session_id,
            len(capped),
        )

    def _remember(self, errors: Iterable[LogError]) -> None:
        for err in errors:
            fp = err.fingerprint()
            # Advance the high-water mark monotonically so old entries
            # never come back into scope even if they get evicted from
            # the recency set.
            if self._high_water is None or fp > self._high_water:
                self._high_water = fp
            if fp in self._reported_set:
                continue
            if len(self._reported) == self._reported.maxlen:
                evicted = self._reported.popleft()
                self._reported_set.discard(evicted)
            self._reported.append(fp)
            self._reported_set.add(fp)
