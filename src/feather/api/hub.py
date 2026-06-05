"""ApiHub: owns the runtime + lead manager + per-lead channels for the API.

This is the API's analogue of the TUI app object — the long-lived holder of
runtime services that the route handlers read from. It is created once at
server startup (FastAPI lifespan) and closed at shutdown.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from feather.api.channel import LeadChannel
from feather.api.models import (
    ConfigApplyOut,
    ConfigFieldOut,
    ConfigOut,
    LeadOut,
    SoulOut,
    SubagentOut,
    TranscriptMessageOut,
    TranscriptOut,
)
from feather.core.leads.scaffold import is_valid_lead_name, scaffold_lead_yaml
from feather.runtime import FeatherRuntime

logger = logging.getLogger(__name__)

__all__ = ("ApiHub",)


def _enum_str(value: object) -> str:
    """Render an enum (or anything) as its wire string."""
    return value.value if hasattr(value, "value") else str(value)


class ApiHub:
    """Hold runtime services and one streaming channel per lead."""

    def __init__(self, runtime: FeatherRuntime, root: Path) -> None:
        self._runtime = runtime
        self._root = root
        self._manager = runtime.lead_manager(worker_mode=False)
        self._soul_library = runtime.soul_library
        self._channels: dict[str, LeadChannel] = {}

    @classmethod
    async def create(
        cls,
        root: Path,
        *,
        provider_factory: Callable[[Any], Any] | None = None,
    ) -> "ApiHub":
        runtime = await FeatherRuntime.create(root, provider_factory=provider_factory)
        hub = cls(runtime, root)
        await hub._manager.start()
        await runtime.start_background_services()
        for name in hub._manager.active_names():
            hub._open_channel(name)
        return hub

    def _open_channel(self, name: str) -> LeadChannel:
        info = self._manager.info(name)
        channel = LeadChannel(
            name=name,
            display_name=info.display_name,
            handle=self._manager.handle(name),
            session_id=info.session_id,
            runtime=self._runtime,
        )
        channel.start()
        self._channels[name] = channel
        return channel

    # --- reads -----------------------------------------------------------

    def list_leads(self) -> list[LeadOut]:
        out: list[LeadOut] = []
        for info in self._manager.list_leads():
            channel = self._channels.get(info.name)
            out.append(
                LeadOut(
                    name=info.name,
                    display_name=info.display_name,
                    personality=info.personality,
                    soul=info.soul,
                    color=info.color,
                    emoji=info.emoji,
                    session_id=info.session_id,
                    status=channel.status if channel else "idle",
                )
            )
        return out

    def channel(self, name: str) -> LeadChannel | None:
        return self._channels.get(name)

    def list_souls(self) -> list[SoulOut]:
        """Return the soul library, sorted by title, for the picker."""
        souls = [
            SoulOut(
                id=soul.id,
                title=soul.title,
                personality=soul.personality,
                color=soul.color,
                emoji=soul.emoji,
                tags=list(soul.tags),
            )
            for soul in self._soul_library.list()
        ]
        souls.sort(key=lambda s: s.title.lower())
        return souls

    async def list_subagents(self, name: str) -> list[SubagentOut]:
        channel = self._channels.get(name)
        if channel is None:
            return []
        live = await self._runtime.subagent_registry.snapshot()
        return [
            SubagentOut(
                agent_name=entry.agent_name,
                session_id=entry.session_id,
                task=entry.task_text,
            )
            for entry in live
            if entry.parent_session_id == channel.session_id
        ]

    async def get_transcript(self, session_id: str) -> TranscriptOut:
        messages = await self._runtime.session_store.list_messages(session_id)
        return TranscriptOut(
            session_id=session_id,
            messages=[
                TranscriptMessageOut(
                    role=m.role.value if hasattr(m.role, "value") else str(m.role),
                    content=m.content,
                    sequence=m.sequence,
                )
                for m in messages
            ],
        )

    def list_config_fields(self) -> list[ConfigFieldOut]:
        """Every editable config field (path, value, source, reload class)."""
        from feather.config.service import ConfigService

        service = ConfigService(paths=self._runtime.paths, app_config=self._runtime.config)
        out: list[ConfigFieldOut] = []
        for row in service.list():
            field = row.field
            sensitive = bool(getattr(field, "sensitive", False))
            out.append(
                ConfigFieldOut(
                    path=field.path,
                    value=None if sensitive else row.current,
                    type=_enum_str(field.type),
                    widget=_enum_str(field.widget),
                    reload=_enum_str(field.reload),
                    scope=_enum_str(field.scope),
                    source=_enum_str(row.source),
                    description=field.description,
                )
            )
        return out

    def get_config(self) -> ConfigOut:
        config = self._runtime.config
        provider = config.active_provider
        if provider == "openrouter" and config.openrouter is not None:
            model = config.openrouter.model
        elif provider == "claude" and config.claude is not None:
            model = config.claude.model
        else:
            model = config.openai.model
        return ConfigOut(
            active_provider=provider,
            default_lead=config.default_lead,
            model=model,
            memory_enabled=config.memory.enabled,
            self_repair=config.self_repair.enabled,
            values={
                "active_provider": provider,
                "default_lead": config.default_lead,
                "model": model,
                "memory_enabled": config.memory.enabled,
                "self_repair": config.self_repair.enabled,
                "compaction_trigger_ratio": config.compaction.trigger_ratio,
            },
        )

    # --- writes ----------------------------------------------------------

    async def set_config(
        self, path: str, value: object, *, scope: str = "global", force: bool = False
    ) -> ConfigApplyOut:
        """Write one config field then apply its reload class (TUI /config set)."""
        from feather.config.resolver import PathScope
        from feather.config.service import ConfigService

        service = ConfigService(paths=self._runtime.paths, app_config=self._runtime.config)
        path_scope = PathScope.PROJECT if scope == "project" else PathScope.GLOBAL
        result = service.set(path, value, scope=path_scope, force=force)
        if not result.ok:
            return ConfigApplyOut(ok=False, path=path, error=result.error)
        outcome = await self._runtime.apply_config_change([path])
        return ConfigApplyOut(
            ok=True,
            path=path,
            applied=list(outcome.applied),
            needs_restart_lead=list(outcome.needs_restart_lead),
            needs_restart_app=list(outcome.needs_restart_app),
        )

    async def create_lead(
        self, name: str, soul: str = "", soul_id: str | None = None
    ) -> LeadOut:
        if not is_valid_lead_name(name):
            raise ValueError(f"invalid lead name: {name!r}")
        name = name.lower()
        if name in self._channels:
            raise ValueError(f"lead {name!r} already exists")
        preset = None
        if soul_id:
            preset = self._soul_library.get(soul_id)
            if preset is None:
                raise ValueError(f"unknown soul: {soul_id!r}")
        scaffold_lead_yaml(self._root, name, soul, soul_preset=preset)
        await self._manager.add_lead(name)
        self._open_channel(name)
        info = self._manager.info(name)
        return LeadOut(
            name=info.name,
            display_name=info.display_name,
            personality=info.personality,
            soul=info.soul,
            color=info.color,
            emoji=info.emoji,
            session_id=info.session_id,
            status="idle",
        )

    async def close(self) -> None:
        for channel in list(self._channels.values()):
            try:
                await channel.stop()
            except Exception:  # noqa: BLE001
                logger.exception("api.hub.channel_stop_failed", extra={"lead": channel.name})
        self._channels.clear()
        await self._runtime.close()
