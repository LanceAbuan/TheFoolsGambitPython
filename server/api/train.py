"""Training telemetry + admin-gated training controls — §14.0 pinned contract.

Metrics/checkpoints read from files the WP-C pipeline writes (`data/metrics.json`,
`data/checkpoints/*.pt`); per-regime Elo is replayed from the trainer's own
arena ledger (`data/arena.jsonl`, :mod:`server.arena`) so it agrees with the
trainer by construction. Control endpoints stay 503 until the training
service attaches. ADMIN_TOKEN guards destructive calls.
"""
from __future__ import annotations

import json
import logging
import secrets
import time
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException

from .. import arena, config, journal

router = APIRouter()
log = logging.getLogger(__name__)

# gate_in_progress hint tuning (see _gate_hint): the trainer must have
# journaled something recently to count as alive, and no arena row may have
# landed in ARENA_QUIET_SECONDS — on the observed cycle (short training
# phase, 20-45 min gates) a longer quiet stretch while alive is a gate.
TRAINER_ALIVE_WINDOW_SECONDS = 120.0
ARENA_QUIET_SECONDS = 600.0


def _gate_hint(now: float, last_arena_ts: float | None, last_journal_ts: float | None) -> bool:
    """Approximate "a gate is probably running" — a hint, not a state.

    The trainer journals ``live`` frames through BOTH its training phase and
    its gate, and only marks the gate's END (the arena event/ledger row), so
    the server cannot see a gate start. The honest approximation: a trainer
    that is actively writing (journal fresh within
    TRAINER_ALIVE_WINDOW_SECONDS) with no arena result in
    ARENA_QUIET_SECONDS is most likely mid-gate — on the observed cycle the
    gate dominates wall-clock time. Known false positive: the tail of a long
    training phase reads the same. Label accordingly in the UI; the clean
    fix is a trainer-owned flag in metrics.json, which overrides this key
    the moment it exists (see :func:`metrics_snapshot`).
    """
    if last_journal_ts is None or now - last_journal_ts > TRAINER_ALIVE_WINDOW_SECONDS:
        return False  # trainer idle/gone: never claim a gate
    if last_arena_ts is None:
        return False  # no ledger yet: nothing to be quiet about
    return (now - last_arena_ts) > ARENA_QUIET_SECONDS


def metrics_snapshot() -> dict[str, Any]:
    """Metrics feed: file snapshot + ledger-replayed per-regime Elo map.

    Pinned keys (elo, loss, win_rate, games_played) come from the trainer's
    metrics.json; everything else in the file (checkpoints map, champion tag,
    U9's future champion_glicko/glicko keys) passes through additively, and
    absent keys are simply absent. `elo_by_regime` is the additive FS-2 key,
    now sourced from the ledger replay (see :mod:`server.arena`).

    Gate-visibility keys (additive, derived server-side when the trainer
    doesn't provide them — a trainer-owned key always wins via setdefault):
    ``last_arena_ts`` (newest ledger row: when the last gate result landed),
    ``last_journal_ts`` (newest train_log event: trainer last seen), and
    ``gate_in_progress`` (the :func:`_gate_hint` approximation). A page
    opened mid-gate can show "gate running — last result X min ago" instead
    of an honest-looking-but-wrong empty state.
    """
    defaults: dict[str, Any] = {
        "elo": None,
        "loss": None,
        "win_rate": None,
        "games_played": 0,
        "elo_by_regime": {},
    }
    try:
        data = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    snapshot = {**defaults, **data} if isinstance(data, dict) else dict(defaults)
    # Never raises: the arena module serves the last known map on failure.
    snapshot["elo_by_regime"] = arena.elo_by_regime()
    last_arena = arena.last_arena_ts()
    last_journal = journal.latest_event_ts()
    snapshot.setdefault("last_arena_ts", last_arena)
    snapshot.setdefault("last_journal_ts", last_journal)
    snapshot.setdefault("gate_in_progress", _gate_hint(time.time(), last_arena, last_journal))
    return snapshot


def _checkpoints() -> list[str]:
    if not config.CHECKPOINTS_DIR.is_dir():
        return []
    return sorted(p.name for p in config.CHECKPOINTS_DIR.glob("*.pt"))


def require_admin(x_admin_token: str = Header(default="")) -> None:
    if not config.ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="admin token not configured")
    if not secrets.compare_digest(x_admin_token, config.ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="invalid admin token")


@router.get("/metrics")
def metrics() -> dict[str, Any]:
    return metrics_snapshot()


@router.get("/checkpoints")
def checkpoints() -> dict[str, list[str]]:
    return {"checkpoints": _checkpoints()}


@router.post("/train/start", dependencies=[Depends(require_admin)])
async def train_start() -> dict:
    raise HTTPException(status_code=503, detail="training service not attached (WP-C)")


@router.post("/train/stop", dependencies=[Depends(require_admin)])
async def train_stop() -> dict:
    raise HTTPException(status_code=503, detail="training service not attached (WP-C)")


@router.post("/train/reset", dependencies=[Depends(require_admin)])
async def train_reset() -> dict:
    raise HTTPException(status_code=503, detail="training service not attached (WP-C)")
