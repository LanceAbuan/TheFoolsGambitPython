"""WebSocket hub: one connection per client, topic subscriptions, seq-numbered messages.

Fan-out architecture (slow-consumer isolation): every subscriber gets a
bounded outbound queue drained by its own pump task, so ``send()`` never
awaits a socket — one wedged/throttled connection (background tab, suspended
process, half-closed TCP) can never delay frames to the others (head-of-line
blocking). A consumer that overflows its queue is beyond interactive
catch-up: it is dropped with a warning, its transport is closed with code 1013
("Try Again Later"), and the client's reconnect+resubscribe path recovers it.
Closing the socket is what makes the drop observable: the endpoint's
``receive_json`` loop only ends when the transport closes, so without it a
dropped consumer would stay connected — permanently deaf rather than briefly
behind.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from . import engine, journal
from .api.train import metrics_snapshot

router = APIRouter()
log = logging.getLogger(__name__)

_TOPIC = re.compile(r"^(lobby|train|game:[A-Za-z0-9]+)$")

# Per-subscriber outbound queue depth. At the production shape (~10 live
# frames/s/topic after throttling + a 10s metrics tick) this holds ~a minute
# of catch-up; a consumer behind by more than that is dropped rather than
# allowed to stall the topic.
SUBSCRIBER_QUEUE_LIMIT = 512
# A single socket send taking longer than this logs a warning (evidence for
# transport-level wedges that survive until the drop).
SLOW_SEND_WARN_SECONDS = 1.0
# hub.send yields to the event loop every this-many enqueues so per-subscriber
# pumps can drain DURING a burst (a tail flush after a trainer idle gap can
# deliver hundreds of journal events back-to-back in one loop turn; without a
# yield no pump would run until the burst ended, overflowing even healthy
# consumers). Bounded: one yield per chunk, not per frame.
FANOUT_YIELD_EVERY = 64
# WebSocket close code for an overflow drop: RFC 6455 "Try Again Later". It is
# the explicit retry signal that ends the endpoint's receive loop, so the
# client's onclose reconnect path fires instead of leaving a deaf socket open.
OVERFLOW_CLOSE_CODE = 1013
# Upper bound on the overflow drop's close handshake: a wedged transport must
# not leak the `_drop` task forever.
OVERFLOW_CLOSE_TIMEOUT_SECONDS = 5.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def lobby_state() -> list[dict[str, Any]]:
    """Active games for the Watch tab. ``opponent``/``strength`` added in WP-A slice 2;
    ``id`` is an additive alias of ``game_id`` aligning with the REST archive shape."""
    return [
        {
            "game_id": s.id,
            "id": s.id,
            "orientation": s.orientation,
            "status": s.status,
            "opponent": s.opponent,
            "strength": s.strength,
        }
        for s in engine.all_sessions()
    ]


class _Subscriber:
    """One connection's outbound mailbox + its pump task."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_LIMIT)
        self.task: asyncio.Task[None] | None = None
        self.dropped = False  # overflow drop in flight; skip further enqueues silently


async def _pump(owner: Hub, topic: str, sub: _Subscriber) -> None:
    """Drain one subscriber's mailbox to its socket, forever.

    Blocking here (transport backpressure from a consumer that stopped
    reading) stalls ONLY this connection; the hub's queue puts are
    non-awaiting, so the topic keeps flowing to everyone else. A send error
    (disconnect mid-flight) unsubscribes the dead connection.
    """
    while True:
        message = await sub.queue.get()
        started = time.perf_counter()
        try:
            await sub.ws.send_json(message)
        except Exception:
            log.warning("hub: send failed on %s; dropping subscriber", topic, exc_info=True)
            await owner.unsubscribe(topic, sub.ws)
            return
        elapsed = time.perf_counter() - started
        if elapsed >= SLOW_SEND_WARN_SECONDS:
            log.warning(
                "hub: slow send on %s took %.1fs (seq=%s) — consumer not reading",
                topic, elapsed, message.get("seq"),
            )


class Hub:
    """Topic → subscribers registry with per-topic monotonic sequence numbers."""

    def __init__(self) -> None:
        self._subs: dict[str, dict[WebSocket, _Subscriber]] = {}
        self._seq: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str, ws: WebSocket) -> None:
        async with self._lock:
            sub = _Subscriber(ws)
            self._subs.setdefault(topic, {})[ws] = sub
        sub.task = asyncio.create_task(_pump(self, topic, sub), name=f"hub-pump-{topic}")

    async def unsubscribe(self, topic: str, ws: WebSocket) -> None:
        async with self._lock:
            connections = self._subs.get(topic)
            if connections is None:
                return
            sub = connections.pop(ws, None)
            if not connections:
                self._subs.pop(topic, None)
                self._seq.pop(topic, None)
        if sub is not None and sub.task is not None:
            sub.task.cancel()

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

    def _enqueue(self, sub: _Subscriber, message: dict[str, Any], topic: str) -> None:
        """Non-blocking put; overflow drops the slow consumer, never the frame's
        delivery to everyone else."""
        if sub.dropped:
            return
        try:
            sub.queue.put_nowait(message)
        except asyncio.QueueFull:
            sub.dropped = True
            log.warning(
                "hub: %s subscriber fell >%d frames behind; dropping it",
                topic, SUBSCRIBER_QUEUE_LIMIT,
            )
            task = sub.task
            # Cancel the pump first (below) and close AFTER the drop: the
            # endpoint's `receive_json` loop only finishes when the transport
            # closes, so without an explicit close a dropped consumer would
            # remain connected and permanently deaf (the client reconnects on
            # `onclose` only). 1013 = "Try Again Later": an unambiguous signal
            # to reconnect and resubscribe. Close even if the pump was already
            # cancelled; `wait_for` + `except Exception` keep a wedged
            # transport from leaking this task.
            async def _drop() -> None:
                await self.unsubscribe(topic, sub.ws)
                log.warning(
                    "hub: closing dropped %s transport with code %d",
                    topic, OVERFLOW_CLOSE_CODE,
                )
                try:
                    await asyncio.wait_for(
                        sub.ws.close(code=OVERFLOW_CLOSE_CODE),
                        timeout=OVERFLOW_CLOSE_TIMEOUT_SECONDS,
                    )
                except Exception:
                    log.warning(
                        "hub: close of dropped %s transport failed or timed out (code %d)",
                        topic, OVERFLOW_CLOSE_CODE, exc_info=True,
                    )

            if task is not None:
                task.cancel()
            asyncio.create_task(_drop(), name=f"hub-drop-{topic}")

    async def send(self, topic: str, mtype: str, payload: dict[str, Any]) -> None:
        started = time.perf_counter()
        async with self._lock:
            subscribers = list(self._subs.get(topic, {}).values())
            seq = self._seq.get(topic, 0) + 1
            self._seq[topic] = seq
        if not subscribers:
            return
        message = {
            "seq": seq,
            "topic": topic,
            "type": mtype,
            "payload": payload,
            "ts": _now(),
        }
        for index, sub in enumerate(subscribers):
            self._enqueue(sub, message, topic)
            if (index + 1) % FANOUT_YIELD_EVERY == 0:
                # Let per-subscriber pumps drain mid-burst (see FANOUT_YIELD_EVERY).
                await asyncio.sleep(0)
        # Every fan-out is a yield point: a caller emitting many sends in one
        # loop turn (the tail task flushing a large journal delta) must give
        # the pumps slices between frames, or even healthy queues overflow.
        await asyncio.sleep(0)
        elapsed = time.perf_counter() - started
        if elapsed >= SLOW_SEND_WARN_SECONDS:
            log.warning("hub: fan-out to %d %s subscriber(s) took %.1fs", len(subscribers), topic, elapsed)
        elif log.isEnabledFor(logging.DEBUG):
            log.debug("hub: fan-out %s seq=%d to %d subscriber(s) in %.2fms", topic, seq, len(subscribers), elapsed * 1000.0)

    async def send_to(self, ws: WebSocket, topic: str, mtype: str, payload: dict[str, Any]) -> None:
        """Targeted delivery (subscribe-time snapshots) through the same
        per-subscriber queue, preserving that connection's frame order."""
        sub = self._subs.get(topic, {}).get(ws)
        if sub is None:
            return
        seq = await self.reserve_seq(topic)
        self._enqueue(sub, {"seq": seq, "topic": topic, "type": mtype, "payload": payload, "ts": _now()}, topic)


hub = Hub()


async def _reject(ws: WebSocket, message: str) -> None:
    await ws.send_json(
        {"seq": 0, "topic": "", "type": "error", "payload": {"message": message}, "ts": _now()}
    )


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Subscribe protocol: {"seq": n, "type": "subscribe", "topic": "lobby"|"train"|"game:{id}"}.

    Subscribe-time snapshots (state/metrics/lobby) are sent immediately, then
    the hub feeds live events. The train topic additionally receives a metrics
    push every 10s from :mod:`server.telemetry` while subscribed, plus the
    FS-6 journal: an aggregate-log backlog (last ≤50 lines; ``live``/
    ``game_end`` lines excluded) at subscribe — each log payload carries the
    additive ``replayed: true`` marker — then tail-pushed ``live``/
    ``game_end``/``log`` frames every 2s (schemas in ``server.telemetry``'s
    docstring; live frames are never replayed pre-subscribe). Heartbeats are
    protocol-level pings (uvicorn default 20s interval).
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
            if topic == "train":
                # FS-6b: advance the journal cursor BEFORE this connection
                # counts as subscribed (sync — no await between seek and
                # subscribe), so live/game_end frames accumulated while
                # nobody watched are never replayed: a client joining
                # mid-stream gets live frames from subscription time onward.
                # Recent aggregate history still arrives via journal.backlog().
                # (seek_to_now stays on the loop deliberately: it is a stat +
                # cursor assignment, and moving it would open a race window
                # against the tail task's cursor advance.)
                journal.seek_to_now()
            await hub.subscribe(topic, ws)
            joined.add(topic)
            if topic.startswith("game:"):
                try:
                    session = engine.get_session(topic.split(":", 1)[1])
                except engine.NotFound:
                    await _reject(ws, f"unknown game {topic}")
                    continue
                await hub.send_to(ws, topic, "state", session.snapshot())
            elif topic == "train":
                # File scans (metrics file + ledger replay + journal tail
                # window) run in a thread: they are ms-scale on a healthy
                # disk but must never sit on the event loop where a stalled
                # read would freeze every WS fan-out.
                snapshot = await asyncio.to_thread(metrics_snapshot)
                await hub.send_to(ws, topic, "metrics", snapshot)
                for payload in await asyncio.to_thread(journal.backlog):
                    # Additive marker: these are replayed backlog lines, not
                    # the live tail — the client can drop its ts-sniffing
                    # heuristic and render them as history (or skip them).
                    await hub.send_to(ws, topic, "log", {**payload, "replayed": True})
            else:  # lobby
                await hub.send_to(ws, topic, "lobby", {"active_games": lobby_state()})
    except WebSocketDisconnect:
        pass
    finally:
        for topic in joined:
            await hub.unsubscribe(topic, ws)
