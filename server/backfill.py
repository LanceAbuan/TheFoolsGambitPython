"""Periodic arena-ledger backfill: keep the SQLite mirror current while up.

The trainer appends to ``data/arena.jsonl`` continuously; until this loop
existed the mirror was populated ONLY by the startup backfill
(``server/main.py`` lifespan), so every row appended after the server's last
restart was invisible to SQLite readers — the stale-leaderboard defect. The
loop re-runs :func:`server.db.backfill_arena` (idempotent: natural-key
INSERT OR IGNORE + null-safe rating reconcile) on a background tick.

Interval: 60s. The ledger grows one row per arena gate (~every 50 min on
tonight's run), so 60s bounds mirror staleness to ~1% of the trainer's own
cadence while costing one small-file read + a handful of no-op statements
per minute — negligible next to the 2s journal tail and 10s metrics push.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from . import db

log = logging.getLogger(__name__)

BACKFILL_INTERVAL_SECONDS = 60.0

_task: asyncio.Task[None] | None = None


async def _loop() -> None:
    """Re-run the idempotent ledger import once per interval tick.

    The interval is re-read each tick (telemetry's pattern, so tests — or
    ops — can tighten it mid-flight).
    """
    while True:
        await asyncio.sleep(BACKFILL_INTERVAL_SECONDS)
        try:
            added = await asyncio.to_thread(db.backfill_arena)
        except Exception:
            # The mirror must never take the server down over a bad tick;
            # the next interval retries and the reconcile repairs anything
            # the failed tick missed.
            log.exception("periodic arena backfill failed; retrying next tick")
            continue
        if added:
            log.info("periodic arena backfill: %d new row(s)", added)


def start() -> None:
    """Start the loop (idempotent; no-op while already running)."""
    global _task
    if _task is None or _task.done():
        _task = asyncio.create_task(_loop(), name="arena-backfill")
        log.info("periodic arena backfill started (interval %.1fs)", BACKFILL_INTERVAL_SECONDS)


async def stop() -> None:
    """Cancel the loop (idempotent; safe before start)."""
    global _task
    task, _task = _task, None
    if task is None:
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    log.info("periodic arena backfill stopped")
