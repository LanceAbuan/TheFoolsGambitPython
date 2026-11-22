"""Background telemetry feeds onto the WS ``train`` topic.

Two loops, both idle (no broadcast) when nobody is subscribed to ``train``:

- metrics (§14.0): pushes the ``GET /api/metrics`` snapshot every 10 seconds
  (no file reads while idle, as before).
- journal (FS-6/FS-6b): tails ``data/train_log.jsonl`` and pushes every new
  complete line, classified (see :mod:`server.journal`), every
  ``JOURNAL_INTERVAL_SECONDS`` — flushing multiple queued frames per wake is
  expected (~2-4 frames/sec peak at 16 games). The tail cursor advances only
  on flushes; a fresh subscriber is protected from stale replay by
  :func:`server.journal.seek_to_now` (called from the WS subscribe path).

Frame schemas on the ``train`` topic (§14.0 + FS-6b, additive — client
workers code against THIS docstring; every frame also carries the standard
``seq``/``topic``/``ts`` envelope):

- ``metrics``: payload mirrors ``GET /api/metrics`` (elo, loss, win_rate,
  games_played, elo_by_regime, plus the additive gate-visibility keys
  last_arena_ts / last_journal_ts / gate_in_progress — see
  :mod:`server.api.train`). Unchanged shape otherwise.
- ``log``: ``{"level": <str>, "message": <str>}`` — journal lines that are
  not live/game_end events (iteration/arena/aggregates), plus any
  unparseable/unknown line. Unchanged on the live tail; backlog-replayed
  lines (subscribe time only, see :mod:`server.ws`) additionally carry
  ``replayed: true``.
- ``live``: ``{"game": <int>, "ply": <int>, "move": <uci str>, "fen": <str>,
  "run_id": <str>, "ts": <num>}`` — one frame per flushed training-game
  position event. NOT replayed on subscribe.
- ``game_end``: ``{"game": <int>, "result": <str>, "plies": <int>,
  "ts": <num>}`` — terminal event for a game slot. NOT replayed on subscribe.

Subscribe-time delivery: metrics snapshot first, then the aggregate-log
backlog (last ≤50 journal lines, live/game_end excluded), then live frames
from subscription time onward.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from . import journal
from .api.train import metrics_snapshot
from .ws import hub

log = logging.getLogger(__name__)

INTERVAL_SECONDS = 10.0
JOURNAL_INTERVAL_SECONDS = 2.0

# FS-6b frame-rate safety valve: ~10 live frames/sec per connection. The
# train topic is a single broadcast stream, so capping the stream caps every
# connection. Per flush the budget is int(cap * interval) live frames; live
# events first coalesce per game (newest fen wins — dropping intermediate
# plies is CORRECT, the client always holds the newest position per game
# slot), then the oldest are dropped if still over budget. game_end frames
# are exempt: they are terminal and low-rate, and dropping one would strand
# the client's game slot.
LIVE_FRAME_CAP_PER_SECOND = 10.0

_task: asyncio.Task[None] | None = None
_journal_task: asyncio.Task[None] | None = None


def throttle_frames(
    events: list[tuple[str, dict[str, Any]]], interval: float
) -> list[tuple[str, dict[str, Any]]]:
    """Coalesce + cap one flush's journal events (see the cap comment above).

    Log frames pass through untouched, ahead of the live/game_end batch;
    kept live frames are ordered by event ts; game_end frames always come
    last (after their game's final position).
    """
    budget = max(1, int(LIVE_FRAME_CAP_PER_SECOND * max(interval, 0.001)))
    newest_by_game: dict[int, tuple[str, dict[str, Any]]] = {}
    ends: list[tuple[str, dict[str, Any]]] = []
    logs: list[tuple[str, dict[str, Any]]] = []
    for frame_type, payload in events:
        if frame_type == journal.LIVE_FRAME:
            newest_by_game[int(payload["game"])] = (frame_type, payload)  # newest wins
        elif frame_type == journal.GAME_END_FRAME:
            ends.append((frame_type, payload))
        else:
            logs.append((frame_type, payload))
    live = sorted(newest_by_game.values(), key=lambda item: float(item[1]["ts"]))
    return logs + live[-budget:] + ends


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


async def _journal_loop() -> None:
    """Push classified journal events every ``JOURNAL_INTERVAL_SECONDS`` while watched.

    The interval is re-read each tick (tests tighten it mid-flight). The
    cursor starts at EOF (history is the subscribe-time backlog's job)
    and advances only on actual flushes — idle periods accumulate lines that
    :func:`server.journal.seek_to_now` then skips at the next subscribe, so
    no client ever receives pre-subscribe position frames.
    """
    journal.reset_cursor()
    while True:
        interval = JOURNAL_INTERVAL_SECONDS  # re-read each tick (tests tighten it mid-flight)
        await asyncio.sleep(interval)
        if hub.subscriber_count("train") == 0:
            continue  # idle: no broadcast (cursor handled at subscribe time)
        events = journal.delta_from_cursor()
        if not events:
            continue
        for frame_type, payload in throttle_frames(events, interval):
            await hub.send("train", frame_type, payload)


def start() -> None:
    """Start both feeds (idempotent; no-op while already running)."""
    global _task, _journal_task
    if _task is None or _task.done():
        _task = asyncio.create_task(_loop(INTERVAL_SECONDS), name="telemetry-feed")
        log.info("telemetry feed started (interval %.1fs)", INTERVAL_SECONDS)
    if _journal_task is None or _journal_task.done():
        _journal_task = asyncio.create_task(_journal_loop(), name="journal-feed")
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
