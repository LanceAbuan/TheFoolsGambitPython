"""WebSocket hub: one connection per client, topic subscriptions, seq-numbered messages."""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from . import engine
from .api.train import metrics_snapshot

router = APIRouter()

_TOPIC = re.compile(r"^(lobby|train|game:[A-Za-z0-9]+)$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def lobby_state() -> list[dict[str, Any]]:
    return [
        {"game_id": s.id, "orientation": s.orientation, "status": s.status}
        for s in engine.all_sessions()
    ]


class Hub:
    """Topic → connections registry with per-topic monotonic sequence numbers."""

    def __init__(self) -> None:
        self._subs: dict[str, set[WebSocket]] = {}
        self._seq: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str, ws: WebSocket) -> None:
        async with self._lock:
            self._subs.setdefault(topic, set()).add(ws)

    async def unsubscribe(self, topic: str, ws: WebSocket) -> None:
        async with self._lock:
            connections = self._subs.get(topic)
            if connections is not None:
                connections.discard(ws)
                if not connections:
                    self._subs.pop(topic, None)
                    self._seq.pop(topic, None)

    async def send(self, topic: str, mtype: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            connections = list(self._subs.get(topic, ()))
            seq = self._seq.get(topic, 0) + 1
            self._seq[topic] = seq
        if not connections:
            return
        message = {
            "seq": seq,
            "topic": topic,
            "type": mtype,
            "payload": payload,
            "ts": _now(),
        }
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:  # dead connection — drop it
                await self.unsubscribe(topic, ws)


hub = Hub()


async def _reject(ws: WebSocket, message: str) -> None:
    await ws.send_json(
        {"seq": 0, "topic": "", "type": "error", "payload": {"message": message}, "ts": _now()}
    )


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Subscribe protocol: {"seq": n, "type": "subscribe", "topic": "lobby"|"train"|"game:{id}"}.

    Heartbeats are protocol-level pings (uvicorn default 20s interval).
    """
    await ws.accept()
    joined: set[str] = set()
    try:
        while True:
            raw = await ws.receive_json()
            if raw.get("type") != "subscribe":
                await _reject(ws, "unsupported message type")
                continue
            topic = str(raw.get("topic", ""))
            if not _TOPIC.match(topic):
                await _reject(ws, f"invalid topic {topic!r}")
                continue
            await hub.subscribe(topic, ws)
            joined.add(topic)
            if topic.startswith("game:"):
                try:
                    session = engine.get_session(topic.split(":", 1)[1])
                except engine.NotFound:
                    await _reject(ws, f"unknown game {topic}")
                    continue
                await ws.send_json(
                    {
                        "seq": 1,
                        "topic": topic,
                        "type": "state",
                        "payload": session.snapshot(),
                        "ts": _now(),
                    }
                )
            elif topic == "train":
                await ws.send_json(
                    {
                        "seq": 1,
                        "topic": topic,
                        "type": "metrics",
                        "payload": metrics_snapshot(),
                        "ts": _now(),
                    }
                )
            else:  # lobby
                await ws.send_json(
                    {
                        "seq": 1,
                        "topic": topic,
                        "type": "lobby",
                        "payload": {"active_games": lobby_state()},
                        "ts": _now(),
                    }
                )
    except WebSocketDisconnect:
        pass
    finally:
        for topic in joined:
            await hub.unsubscribe(topic, ws)
