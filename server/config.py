"""Runtime configuration, read from the environment at import time."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DB_PATH: Path = Path(os.environ.get("FSG_DB", str(ROOT / "data" / "gambit.db")))
ADMIN_TOKEN: str = os.environ.get("FSG_ADMIN_TOKEN", "")
ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.environ.get(
        "FSG_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
METRICS_PATH: Path = Path(os.environ.get("FSG_METRICS", str(ROOT / "data" / "metrics.json")))
CHECKPOINTS_DIR: Path = Path(os.environ.get("FSG_CHECKPOINTS", str(ROOT / "data" / "checkpoints")))
RUNS_DIR: Path = Path(os.environ.get("FSG_RUNS", str(ROOT / "runs")))
TRAIN_LOG_PATH: Path = Path(os.environ.get("FSG_TRAIN_LOG", str(ROOT / "data" / "train_log.jsonl")))
ARENA_LEDGER_PATH: Path = Path(os.environ.get("FSG_ARENA_LEDGER", str(ROOT / "data" / "arena.jsonl")))
WEB_DIST: Path = Path(os.environ.get("FSG_WEB_DIST", str(ROOT / "web" / "dist")))
GAME_TTL_MIN: float = float(os.environ.get("FSG_GAME_TTL_MIN", "30"))
RATE_LIMIT_CAPACITY: int = int(os.environ.get("FSG_RATE_CAPACITY", "20"))
RATE_LIMIT_REFILL: float = float(os.environ.get("FSG_RATE_REFILL", "2.0"))
