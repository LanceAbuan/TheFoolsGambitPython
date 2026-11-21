"""Arena ledger in SQLite (FS-2): backfill idempotency, /api/games?kind=arena, Elo map."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import config, db, engine
from server.main import create_app


def _ledger(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "arena.db")
    monkeypatch.setattr(config, "ARENA_LEDGER_PATH", tmp_path / "arena.jsonl")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
    db.reset_for_tests()
    engine.reset_for_tests()
    with TestClient(create_app()) as c:
        yield c


def test_backfill_serves_arena_games(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ledger present at boot -> rows served by /api/games?kind=arena (newest first)."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "arena.db")
    monkeypatch.setattr(config, "ARENA_LEDGER_PATH", tmp_path / "arena.jsonl")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
    _ledger(
        tmp_path / "arena.jsonl",
        [
            {"run_id": "r1", "regime": "scratch", "checkpoint": "ckpt-a", "opponent": "none",
             "games": 0, "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
            {"run_id": "r2", "regime": "scratch", "checkpoint": "ckpt-b", "opponent": "ckpt-a",
             "games": 16, "wins": 9, "draws": 4, "elo": 812.5, "ts": 200.0},
            "not json at all",  # corrupt line must be skipped, not fatal
        ],
    )
    db.reset_for_tests()
    engine.reset_for_tests()
    with TestClient(create_app()) as client:
        games = client.get("/api/games", params={"kind": "arena"}).json()["games"]
    assert len(games) == 2
    newest = games[0]
    assert newest["ckpt"] == "ckpt-b"
    assert newest["kind"] == "arena"
    assert newest["elo"] == pytest.approx(812.5)
    assert newest["games"] == 16
    assert games[1]["ckpt"] == "ckpt-a"  # ordering: newest first


def test_double_boot_backfill_creates_no_dupes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "arena.db")
    monkeypatch.setattr(config, "ARENA_LEDGER_PATH", tmp_path / "arena.jsonl")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
    _ledger(
        tmp_path / "arena.jsonl",
        [
            {"regime": "scratch", "checkpoint": "ckpt-a", "opponent": "none", "games": 0,
             "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
            {"regime": "scratch", "checkpoint": "ckpt-b", "opponent": "ckpt-a", "games": 2,
             "wins": 1, "draws": 1, "elo": 808.0, "ts": 200.0},
        ],
    )
    db.reset_for_tests()
    with TestClient(create_app()):
        pass  # boot 1 (backfill runs in lifespan)
    with TestClient(create_app()):  # boot 2 on the same DB: must insert nothing
        pass
    rows = db.list_arena_games(100)
    assert len(rows) == 2  # (ckpt, ts) dedupe held across restarts


def test_play_games_listing_unaffected(client: TestClient):
    gid = client.post("/api/play/new", json={"color": "white", "opponent": "fool"}).json()["game_id"]
    play_games = client.get("/api/games", params={"kind": "play"}).json()["games"]
    assert [g["id"] for g in play_games] == [gid]
    assert client.get("/api/games", params={"kind": "arena"}).json()["games"] == []


def test_metrics_elo_by_regime_map(client: TestClient, tmp_path: Path):
    """Additive metrics key: {regime: {ckpt: latest recorded elo}}."""
    assert client.get("/api/metrics").json()["elo_by_regime"] == {}
    _ledger(
        tmp_path / "arena.jsonl",
        [
            {"regime": "scratch", "checkpoint": "ckpt-a", "opponent": "none", "games": 0,
             "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
            {"regime": "scratch", "checkpoint": "ckpt-a", "opponent": "ckpt-b", "games": 4,
             "wins": 3, "draws": 1, "elo": 824.0, "ts": 300.0},
            {"regime": "scratch", "checkpoint": "ckpt-b", "opponent": "ckpt-a", "games": 4,
             "wins": 0, "draws": 1, "elo": 776.0, "ts": 301.0},
            {"regime": "finetune", "checkpoint": "ckpt-f", "opponent": "none", "games": 0,
             "wins": 0, "draws": 0, "elo": 800.0, "ts": 400.0},
        ],
    )
    # Re-backfill into the running app's DB (a second boot would do this too).
    assert db.backfill_arena() == 4
    elo_map = client.get("/api/metrics").json()["elo_by_regime"]
    assert elo_map == {
        "scratch": {"ckpt-a": 824.0, "ckpt-b": 776.0},  # latest ts wins per checkpoint
        "finetune": {"ckpt-f": 800.0},
    }
