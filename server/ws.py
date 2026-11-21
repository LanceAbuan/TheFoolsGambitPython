"""WebSocket hub: one connection per client, topic subscriptions, seq-numbered messages."""
from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from . import engine, journal
from .api.train import metrics_snapshot

router = APIRouter()

_TOPIC = re.compile(r"^(lobby|train|game:[A-Za-z0-9]+)$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def lobby_state() -> list[dict[str, Any]]:
    """Active games for the Watch tab. ``opponent``/``strength`` added in WP-A slice 2."""
    return [
        {
            "game_id": s.id,
            "orientation": s.orientation,
            "status": s.status,
            "opponent": s.opponent,
            "strength": s.strength,
        }
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

    async def reserve_seq(self, topic: str) -> int:
        """Allocate the next per-topic sequence number.

        Subscribe-time snapshots share the feed's counter so a client replaying
        its last seen seq sees one monotonic stream per topic.
        """
        async with self._lock:
            seq = self._seq.get(topic, 0) + 1
            self._seq[topic] = seq
            return seq

    def subscriber_count(self, topic: str) -> int:
        """Live connections on ``topic``. Event-loop-thread use only."""
        return len(self._subs.get(topic, ()))

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
            except Exception:  # noqa: BLE001 — a dying WS raises any transport error; drop it, keep fanning out
                await self.unsubscribe(topic, ws)


hub = Hub()


async def _reject(ws: WebSocket, message: str) -> None:
    await ws.send_json(
        {"seq": 0, "topic": "", "type": "error", "payload": {"message": message}, "ts": _now()}
    )


async def _snapshot(ws: WebSocket, topic: str, mtype: str, payload: dict[str, Any]) -> None:
    """Per-connection catch-up snapshot; consumes a seq from the topic's stream."""
    await ws.send_json(
        {"seq": await hub.reserve_seq(topic), "topic": topic, "type": mtype, "payload": payload, "ts": _now()}
    )


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Subscribe protocol: {"seq": n, "type": "subscribe", "topic": "lobby"|"train"|"game:{id}"}.

    Subscribe-time snapshots (state/metrics/lobby) are sent immediately, then
    the hub feeds live events. The train topic additionally receives a metrics
    push every 10s from :mod:`server.telemetry` while subscribed. Heartbeats
    are protocol-level pings (uvicorn default 20s interval).
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
                await _snapshot(ws, topic, "state", session.snapshot())
            elif topic == "train":
                await _snapshot(ws, topic, "metrics", metrics_snapshot())
                # FS-6: catch-up journal history (last 50 lines) right after the snapshot.
                for payload in journal.backlog():
                    await _snapshot(ws, topic, "log", payload)
            else:  # lobby
                await _snapshot(ws, topic, "lobby", {"active_games": lobby_state()})
    except WebSocketDisconnect:
        pass
    finally:
        for topic in joined:
            await hub.unsubscribe(topic, ws)
