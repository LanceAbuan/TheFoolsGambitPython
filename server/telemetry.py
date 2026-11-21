"""Background telemetry feeds onto the WS ``train`` topic.

Two loops, both idle when nobody is subscribed to ``train`` (one timer wake
per interval, no file I/O, no hub traffic):

- metrics (§14.0): pushes the ``GET /api/metrics`` snapshot every 10 seconds.
- journal (FS-6): tails ``data/train_log.jsonl`` and pushes every new
  complete line as a ``log`` frame every 2 seconds, from the file position at
  loop start (subscribe-time history comes from :func:`server.journal.backlog`).

Subscribe-time snapshots are sent separately by the WS endpoint, so a new
watcher gets data instantly and then every interval.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from . import config, journal
from .api.train import metrics_snapshot
from .ws import hub

log = logging.getLogger(__name__)

INTERVAL_SECONDS = 10.0
JOURNAL_INTERVAL_SECONDS = 2.0

_task: asyncio.Task[None] | None = None
_journal_task: asyncio.Task[None] | None = None


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


async def _journal_loop(interval: float) -> None:
    """Push new journal lines as ``log`` frames every ``interval`` seconds while watched.

    The tail starts at the file's size when the loop starts — history is the
    subscribe-time backlog's job. A subscriber that joins between ticks may
    see a line twice (backlog covers it, then the next tick pushes it again);
    bounded by one interval, accepted for simplicity.
    """
    offset = journal.size()
    while True:
        await asyncio.sleep(interval)
        if hub.subscriber_count("train") == 0:
            continue  # idle: no file reads, no broadcast
        offset, payloads = journal.delta(config.TRAIN_LOG_PATH, offset)
        for payload in payloads:
            await hub.send("train", "log", payload)


def start() -> None:
    """Start both feeds (idempotent; no-op while already running)."""
    global _task, _journal_task
    if _task is None or _task.done():
        _task = asyncio.create_task(_loop(INTERVAL_SECONDS), name="telemetry-feed")
        log.info("telemetry feed started (interval %.1fs)", INTERVAL_SECONDS)
    if _journal_task is None or _journal_task.done():
        _journal_task = asyncio.create_task(
            _journal_loop(JOURNAL_INTERVAL_SECONDS), name="journal-feed"
        )
        log.info("journal feed started (interval %.1fs)", JOURNAL_INTERVAL_SECONDS)


async def stop() -> None:
    """Cancel both feeds (idempotent; safe before start)."""
    global _task, _journal_task
    tasks = (_task, _journal_task)
    _task = None
    _journal_task = None
    for task in tasks:
        if task is None:
            continue
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    log.info("telemetry + journal feeds stopped")
