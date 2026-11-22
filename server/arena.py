"""Per-regime Elo map derived from the trainer's own arena-ledger replay.

Source-of-truth choice (stale-leaderboard fix): ``elo_by_regime`` is computed
by replaying ``data/arena.jsonl`` with the trainer's own
:func:`training.ledger.recompute_elos` — the exact function the trainer runs
over the exact file the trainer writes — so the server's per-checkpoint map
agrees with the trainer by construction. The SQLite mirror
(:func:`server.db.elo_by_regime`) keeps the *recorded rating claims* (nulls
included) for the arena listing; it is no longer the regime-map source
because it was populated only at startup, which is precisely how the
leaderboard went stale while the headline number stayed live.

Regime attribution: a checkpoint belongs to the regime of its latest
candidate row (the last ledger line whose ``checkpoint`` field is the
checkpoint) — the run the checkpoint last belonged to on the ladder. A
checkpoint appears under exactly one regime; regimes whose checkpoints all
moved on legitimately disappear from the map.

Scale guarantee (the b36d82e access pattern, applied to this file): steady
state is O(1) — every call stats the ledger, and the file is only re-read +
re-replayed when its ``(st_mtime_ns, st_size)`` identity changes. The ledger
grows one row per arena gate, so a rebuild is rare and cheap.

Failure posture: a vanished/unreadable ledger is never fatal — the last
successfully computed map is served (staleness over emptiness); before any
successful compute, an empty map. The same map flows to the WS ``train``
topic via the metrics snapshot, so both surfaces stay identical.

The cache also carries the newest ledger row's ``ts`` — served additively as
``last_arena_ts`` by ``/api/metrics`` — one of the facts behind the feed's
``gate_in_progress`` hint (see :mod:`server.api.train`).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from training.ledger import ArenaLedger, recompute_elos

from . import config

log = logging.getLogger(__name__)

# Shared replay cache (event loop + threadpool callers): guarded by a lock,
# keyed on the ledger's (mtime_ns, size) identity. Holds the regime map plus
# the newest ledger row's ts (the ``last_arena_ts`` fact for the metrics feed).
_lock = threading.Lock()
_cache_key: tuple[int, int] | None = None
_cache_map: dict[str, dict[str, float]] = {}
_cache_last_ts: float | None = None


def _replay(path: Path) -> tuple[dict[str, dict[str, float]], float | None]:
    """One full ledger replay → (per-regime map, newest row ts). Caller holds the lock."""
    entries: list[dict[str, Any]] = ArenaLedger(path).read()
    elos = recompute_elos(entries)
    # Regime attribution: latest candidate row per checkpoint (chronological
    # sweep, last mention wins — see module docstring).
    regime_of: dict[str, str] = {}
    newest_ts: float | None = None
    for entry in entries:
        ckpt = str(entry.get("checkpoint", ""))
        if ckpt:
            regime_of[ckpt] = str(entry.get("regime", ""))
        ts = entry.get("ts")
        if isinstance(ts, (int, float)) and (newest_ts is None or float(ts) > newest_ts):
            newest_ts = float(ts)
    out: dict[str, dict[str, float]] = {}
    for ckpt, elo in elos.items():
        out.setdefault(regime_of.get(ckpt, ""), {})[ckpt] = elo
    return out, newest_ts


def _rebuild_if_stale(target: Path) -> tuple[dict[str, dict[str, float]], float | None]:
    """(regime map, last_arena_ts) from cache or a fresh replay.

    Rebuilds only when the file's ``(mtime_ns, size)`` identity changed since
    the last call; any read or replay failure serves the last good cache
    (and logs) — callers degrade to staleness, never to an error or a
    silently emptied map.
    """
    global _cache_key, _cache_map, _cache_last_ts
    with _lock:
        try:
            st = target.stat()
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            if _cache_key is not None:
                log.warning("arena ledger %s unreadable; serving last known map", target)
            return _cache_map, _cache_last_ts
        if key == _cache_key:
            return _cache_map, _cache_last_ts
        try:
            fresh, newest_ts = _replay(target)
        except Exception:
            if _cache_key is not None:
                log.exception("arena ledger replay failed; serving last known map")
            return _cache_map, _cache_last_ts
        _cache_map, _cache_last_ts, _cache_key = fresh, newest_ts, key
        return _cache_map, _cache_last_ts


def elo_by_regime(path: Path | None = None) -> dict[str, dict[str, float]]:
    """Per-regime checkpoint Elo map replayed from the trainer's ledger.

    Rebuilt only when the file's ``(mtime_ns, size)`` identity changed since
    the last call; otherwise the cached map returns in O(1). Any read or
    replay failure serves the last good map (and logs) — the endpoint and WS
    feed degrade to staleness, never to an error or a silently emptied map.
    """
    target = path or config.ARENA_LEDGER_PATH
    return _rebuild_if_stale(target)[0]


def last_arena_ts(path: Path | None = None) -> float | None:
    """Newest ledger row's ``ts`` (a gate result landed at that instant) —
    the fact behind the metrics feed's gate hint. Same cache/replay cadence
    as :func:`elo_by_regime`; None when the ledger is absent or unreadable."""
    target = path or config.ARENA_LEDGER_PATH
    return _rebuild_if_stale(target)[1]


def reset_cache() -> None:
    """Drop the replay cache (tests); the next call recomputes from disk."""
    global _cache_key, _cache_map, _cache_last_ts
    with _lock:
        _cache_key = None
        _cache_map = {}
        _cache_last_ts = None
