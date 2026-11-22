"""TTL reaper (FS-9): auto-resigns games idle for ``FSG_GAME_TTL_MIN`` minutes.

An active game whose last ply (or creation) is older than the TTL is finished
as an abandonment loss for the human, persisted, broadcast on its game topic
+ the lobby, and its Stockfish process (if any) is stopped. Runs as an
asyncio task started by the app lifespan; the scan interval is a module
constant so tests can tighten it.

Idle games only — the reaper never touches finished sessions, and the
expiry check is repeated under the session lock so a move racing the reaper
always wins.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress

import chess

from . import config, db, engine, stockfish
from .engine import Session
from .ws import hub, lobby_state

log = logging.getLogger(__name__)

SCAN_INTERVAL_SECONDS = 30.0

_task: asyncio.Task[None] | None = None


def expired_sessions(now: float | None = None) -> list[Session]:
    """Active sessions whose last activity is older than the TTL."""
    deadline = (now if now is not None else time.time()) - config.GAME_TTL_MIN * 60.0
    return [
        s for s in engine.all_sessions() if s.status == "active" and s.last_activity <= deadline
    ]


async def reap_once() -> int:
    """Auto-resign every expired session; returns the number reaped."""
    reaped = 0
    for session in expired_sessions():
        async with session.lock:
            if session.status != "active":
                continue  # finished while we waited for the lock
            if time.time() - session.last_activity < config.GAME_TTL_MIN * 60.0:
                continue  # a move landed while we waited — let it live
            session.status = "finished"
            session.winner = "black" if session.human_color is chess.WHITE else "white"
            session.reason = "abandoned"
        reaped += 1
        log.info("reaped idle game %s (auto-resign)", session.id)
        await asyncio.to_thread(db.persist_session, session)
        await hub.send(f"game:{session.id}", "state", session.snapshot())
        await hub.send("lobby", "lobby", {"active_games": lobby_state()})
        await stockfish.stop(session)
    return reaped


async def _loop() -> None:
    while True:
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)
        try:
            await reap_once()
        except Exception:
            log.exception("reaper scan failed")


def start() -> None:
    """Start the reaper loop (idempotent; no-op while already running)."""
    global _task
    if _task is not None and not _task.done():
        return
    _task = asyncio.create_task(_loop(), name="ttl-reaper")
    log.info("ttl reaper started (interval %.1fs, ttl %.1f min)", SCAN_INTERVAL_SECONDS, config.GAME_TTL_MIN)


async def stop() -> None:
    """Cancel the reaper loop (idempotent; safe before start)."""
    global _task
    if _task is None:
        return
    _task.cancel()
    with suppress(asyncio.CancelledError):
        await _task
    _task = None
    log.info("ttl reaper stopped")
