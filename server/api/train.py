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

from .. import arena, config, journal, trainer

router = APIRouter()
log = logging.getLogger(__name__)

# gate_in_progress hint tuning (see _gate_hint). LIVE_QUIET_SECONDS is measured
# from the trainer's real journal cadence (~2.4 lines/s while self-play runs,
# ~2 s training phase): a live trainer that has written nothing for 3 minutes
# is not in a training phase. Arena gates write nothing at all for 40-80 min.
LIVE_QUIET_SECONDS = 180.0


def _gate_hint(
    now: float,
    last_arena_ts: float | None,
    last_journal_ts: float | None,
    *,
    alive: bool,
) -> bool:
    """Approximate "a gate is probably running" — a hint, not a state.

    Measured trainer behavior (production run ``second-light``, 2026-11-21):
    the journal streams ``live`` frames at ~2.4 lines/s through self-play and
    the short training phase, then writes NOTHING for the entire arena gate
    (gates are not broadcast move by move), and resumes with the iteration line
    when the gate result lands — a 40-80 min silence. The server therefore
    cannot see a gate start, and journal freshness cannot prove liveness during
    the one window this hint exists for; before this rule it made
    ``gate_in_progress`` structurally false during every gate.

    The honest rule: a trainer PROCESS is alive (:func:`server.trainer.trainer_alive`),
    the ledger already has a row (so past results are meaningful), and the
    journal has been silent longer than LIVE_QUIET_SECONDS. No arena-quiet
    requirement: a gate lasts 40-80 min, and demanding a 10 min arena silence
    on top of a 3 min journal silence only delays the hint by 10 min exactly
    when the page is opened right behind a fresh result.

    Known false positive: a trainer alive but hung/stalled past 3 min reads as
    mid-gate. The clean fix remains a trainer-owned flag in metrics.json, which
    overrides this key the moment it exists (see :func:`metrics_snapshot`).
    """
    if not alive:
        return False  # no trainer process: never claim a gate
    if last_arena_ts is None:
        return False  # no ledger yet: nothing to be quiet about
    if last_journal_ts is None:
        return False  # journal never written: no quiet stretch to measure
    return (now - last_journal_ts) > LIVE_QUIET_SECONDS


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
    ``last_journal_ts`` (newest train_log event: trainer last seen),
    ``trainer_alive`` (a trainer process is running — see
    :mod:`server.trainer`), and ``gate_in_progress`` (the
    :func:`_gate_hint` approximation). A page opened mid-gate can show
    "gate running — last result X min ago" instead of an
    honest-looking-but-wrong empty state.
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
    # Run id preference: a trainer-owned metrics key, else the journal's newest
    # event (the run actually writing). Used to scope the process match.
    run_id = snapshot.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        run_id = journal.latest_run_id()
    alive = trainer.trainer_alive(run_id)
    snapshot.setdefault("last_arena_ts", last_arena)
    snapshot.setdefault("last_journal_ts", last_journal)
    snapshot.setdefault("trainer_alive", alive)
    snapshot.setdefault("gate_in_progress", _gate_hint(
        time.time(), last_arena, last_journal, alive=alive,
    ))
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
