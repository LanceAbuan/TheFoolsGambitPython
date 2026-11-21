"""Training telemetry + admin-gated training controls — §14.0 pinned contract.

Metrics/checkpoints read from files the WP-C pipeline will write (`data/metrics.json`,
`data/checkpoints/*.pt`). Control endpoints stay 503 until WP-C lands. ADMIN_TOKEN
guards destructive calls.
"""
from __future__ import annotations

import json
import secrets
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException

from .. import config

router = APIRouter()


def metrics_snapshot() -> dict[str, Any]:
    defaults: dict[str, Any] = {"elo": None, "loss": None, "win_rate": None, "games_played": 0}
    try:
        data = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(defaults)
    return {**defaults, **data} if isinstance(data, dict) else defaults


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
