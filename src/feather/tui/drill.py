"""Sub-agent drill-down modal screen for the Textual TUI."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, RichLog, Static
from textual.widgets.option_list import Option

from feather.runtime import FeatherRuntime
from feather.tui import preview_inline


class SubagentDrillScreen(ModalScreen[None]):
    """Read-only modal that shows a lead's sub-agents and one's transcript.

    Transcript-on-demand (the Phase A decision): the sub-agent ran detached and
    its messages are persisted under its own session id, so we just read them
    from the session store. ``r`` refreshes, ``Esc`` closes.
    """

    BINDINGS = [
        Binding("escape", "close", "Close", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
    ]

    DEFAULT_CSS = """
    SubagentDrillScreen {
        align: center middle;
    }
    #drill {
        width: 90%;
        height: 90%;
        border: solid #888888;
        background: #0b0d12;
        padding: 0 1;
    }
    #drill_title { height: 1; color: #cccccc; }
    #drill_list { height: 8; border: solid #444444; }
    #drill_log { height: 1fr; border: solid #444444; background: #090b10; }
    """

    def __init__(
        self,
        *,
        runtime: FeatherRuntime,
        parent_session_id: str,
        lead_display_name: str,
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self._parent_session_id = parent_session_id
        self._lead_display_name = lead_display_name
        self._subagents: list[Any] = []
        self._selected_session: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="drill"):
            yield Static(
                f"Sub-agents of {self._lead_display_name}  ·  Esc close · r refresh",
                id="drill_title",
            )
            yield OptionList(id="drill_list")
            yield RichLog(id="drill_log", wrap=True, markup=False, highlight=False)

    async def on_mount(self) -> None:
        await self._reload()

    async def _reload(self) -> None:
        live = await self._runtime.subagent_registry.snapshot()
        self._subagents = [
            entry for entry in live if entry.parent_session_id == self._parent_session_id
        ]
        option_list = self.query_one("#drill_list", OptionList)
        option_list.clear_options()
        for entry in self._subagents:
            option_list.add_option(
                Option(
                    f"{entry.agent_name} {entry.session_id[:8]}: "
                    f"{preview_inline(entry.task_text, limit=60)}",
                    id=entry.session_id,
                )
            )
        if not self._subagents:
            log = self.query_one("#drill_log", RichLog)
            log.clear()
            log.write(Text("No live sub-agents for this lead.", style="dim"))
            self._selected_session = None
            return
        if self._selected_session is None or self._selected_session not in {
            entry.session_id for entry in self._subagents
        }:
            self._selected_session = self._subagents[0].session_id
        await self._load_transcript(self._selected_session)

    async def _load_transcript(self, session_id: str) -> None:
        log = self.query_one("#drill_log", RichLog)
        log.clear()
        try:
            messages = await self._runtime.session_store.list_messages(session_id)
        except Exception:  # noqa: BLE001
            log.write(Text("(could not load transcript)", style="red"))
            return
        if not messages:
            log.write(Text("(no transcript yet — sub-agent just started)", style="dim"))
            return
        for message in messages:
            role = message.role.value if hasattr(message.role, "value") else str(message.role)
            log.write(Text(f"[{role}]", style="bold cyan"))
            if message.content:
                log.write(Text(message.content))
            log.write("")

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if event.option.id:
            self._selected_session = event.option.id
            await self._load_transcript(event.option.id)

    def action_close(self) -> None:
        self.dismiss(None)

    async def action_refresh(self) -> None:
        await self._reload()


