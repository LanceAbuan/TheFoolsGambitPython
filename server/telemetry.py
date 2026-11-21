"""Background telemetry feed: pushes the ``GET /api/metrics`` snapshot onto the
WS ``train`` topic every 10 seconds (§14.0), for as long as subscribers watch.

The loop idles when nobody is subscribed to ``train``: one timer wake per
interval and no metrics-file reads, no hub traffic. Subscribe-time snapshots
are sent separately by the WS endpoint, so a new watcher gets data instantly
and then every ``INTERVAL_SECONDS``.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from .api.train import metrics_snapshot
from .ws import hub

log = logging.getLogger(__name__)

INTERVAL_SECONDS = 10.0

_task: asyncio.Task[None] | None = None


async def _loop(interval: float) -> None:
    """Push a metrics snapshot every ``interval`` seconds while ``train`` has subscribers."""
    while True:
        await asyncio.sleep(interval)
        if hub.subscriber_count("train") == 0:
            continue  # idle: no file read, no broadcast
        try:
            payload: dict[str, Any] = metrics_snapshot()
        except Exception:
            log.exception("metrics snapshot failed; skipping push")
            continue
        await hub.send("train", "metrics", payload)


def start() -> None:
    """Start the background feed (idempotent; no-op while already running)."""
    global _task
    if _task is not None and not _task.done():
        return
    _task = asyncio.create_task(_loop(INTERVAL_SECONDS), name="telemetry-feed")
    log.info("telemetry feed started (interval %.1fs)", INTERVAL_SECONDS)


async def stop() -> None:
    """Cancel the background feed (idempotent; safe before start)."""
    global _task
    if _task is None:
        return
    _task.cancel()
    with suppress(asyncio.CancelledError):
        await _task
    _task = None
    log.info("telemetry feed stopped")
