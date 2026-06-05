"""HTTP + WebSocket routes for the Feather API."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from feather.api.hub import ApiHub
from feather.api.models import (
    ConfigOut,
    CreateLeadIn,
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
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()
