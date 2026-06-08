"""HTTP + WebSocket routes for the Feather API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from feather.api.hub import ApiHub
from feather.api.models import (
    ConfigApplyOut,
    ConfigFieldOut,
    ConfigOut,
    ConfigSetIn,
    CreateLeadIn,
    InputIn,
    LeadOut,
    MessageIn,
    SoulOut,
    SubagentOut,
    TranscriptOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["feather"])


def get_hub(request: Request) -> ApiHub:
    hub = getattr(request.app.state, "hub", None)
    if hub is None:
        raise HTTPException(status_code=503, detail="server still starting")
    return hub


HubDep = Annotated[ApiHub, Depends(get_hub)]


@router.get("/leads")
async def list_leads(hub: HubDep) -> list[LeadOut]:
    return hub.list_leads()


@router.get("/souls")
async def list_souls(hub: HubDep) -> list[SoulOut]:
    return hub.list_souls()


@router.post("/leads", status_code=201)
async def create_lead(payload: CreateLeadIn, hub: HubDep) -> LeadOut:
    try:
        return await hub.create_lead(payload.name, payload.soul, payload.soul_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/leads/{name}/messages", status_code=202)
async def send_message(name: str, payload: MessageIn, hub: HubDep) -> dict[str, str]:
    channel = hub.channel(name)
    if channel is None:
        raise HTTPException(status_code=404, detail=f"unknown lead: {name}")
    await channel.send(payload.text)
    return {"status": "queued"}


@router.post("/leads/{name}/input", status_code=202)
async def inject_input(name: str, payload: InputIn, hub: HubDep) -> dict[str, str]:
    """Mid-turn input to steer the agent's current turn (TUI parity)."""
    channel = hub.channel(name)
    if channel is None:
        raise HTTPException(status_code=404, detail=f"unknown lead: {name}")
    accepted = await channel.enqueue_input(payload.text)
    return {"status": "injected" if accepted else "no_input_queue"}


@router.get("/leads/{name}/subagents")
async def list_subagents(name: str, hub: HubDep) -> list[SubagentOut]:
    if hub.channel(name) is None:
        raise HTTPException(status_code=404, detail=f"unknown lead: {name}")
    return await hub.list_subagents(name)


@router.get("/leads/{name}/transcript")
async def lead_transcript(name: str, hub: HubDep) -> TranscriptOut:
    channel = hub.channel(name)
    if channel is None:
        raise HTTPException(status_code=404, detail=f"unknown lead: {name}")
    return await hub.get_transcript(channel.session_id)


@router.get("/sessions/{session_id}/transcript")
async def session_transcript(session_id: str, hub: HubDep) -> TranscriptOut:
    return await hub.get_transcript(session_id)


@router.get("/config")
async def get_config(hub: HubDep) -> ConfigOut:
    return hub.get_config()


@router.get("/config/fields")
async def list_config_fields(hub: HubDep) -> list[ConfigFieldOut]:
    return hub.list_config_fields()


@router.post("/config")
async def set_config(payload: ConfigSetIn, hub: HubDep) -> ConfigApplyOut:
    try:
        return await hub.set_config(
            payload.path, payload.value, scope=payload.scope, force=payload.force
        )
    except Exception as exc:  # noqa: BLE001 - surface apply errors to the client
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@router.websocket("/leads/{name}/ws")
async def lead_events(websocket: WebSocket, name: str) -> None:
    """Stream one lead's runtime events; accept inbound {text} to send messages."""

    hub: ApiHub | None = getattr(websocket.app.state, "hub", None)
    channel = hub.channel(name) if hub is not None else None
    if channel is None:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    queue = channel.subscribe()
    await websocket.send_json({"kind": "connected", "payload": {"lead": name}})

    async def pump_events() -> None:
        while True:
            data = await queue.get()
            await websocket.send_json(data)

    async def pump_inbound() -> None:
        while True:
            message = await websocket.receive_json()
            text = (message or {}).get("text")
            if isinstance(text, str) and text.strip():
                await channel.send(text)

    events_task = asyncio.create_task(pump_events())
    inbound_task = asyncio.create_task(pump_inbound())
    try:
        await asyncio.wait(
            {events_task, inbound_task}, return_when=asyncio.FIRST_COMPLETED
        )
    except WebSocketDisconnect:
        pass
    finally:
        events_task.cancel()
        inbound_task.cancel()
        channel.unsubscribe(queue)
        # The peer may already be gone (browser navigated away / tab closed),
        # in which case a pump task hit WebSocketDisconnect and the socket is
        # already closing. Closing again raises "Unexpected ASGI message
        # 'websocket.close'" from uvicorn — guard the state AND suppress the race.
        if websocket.application_state != WebSocketState.DISCONNECTED:
            with contextlib.suppress(RuntimeError):
                await websocket.close()
