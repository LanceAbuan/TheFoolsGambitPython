"""Arena ledger in SQLite (FS-2): backfill idempotency, /api/games?kind=arena, Elo map."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import arena, config, db, engine
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
    arena.reset_cache()  # shared module-level replay cache: isolate per test
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
    """Additive metrics key: {regime: {ckpt: replayed elo}} — trainer agrees by
    construction (same recompute the trainer runs, same file)."""
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
    elo_map = client.get("/api/metrics").json()["elo_by_regime"]
    # Ledger replay (§5.1 per-checkpoint Elo computed from the ledger), NOT the
    # recorded elo claims: ckpt-a finishes 822.90 after ckpt-b's 0/1/3 reply.
    assert elo_map == {
        "scratch": {"ckpt-a": pytest.approx(822.8965137687414),
                    "ckpt-b": pytest.approx(777.1034862312586)},
        "finetune": {"ckpt-f": 800.0},  # seeded at base by its baseline row
    }
    # Agreement by construction: identical to the trainer's own recompute
    # over the same file, partitioned by latest candidate-row regime.
    from training.ledger import recompute_elos

    entries = [
        e for e in (json.loads(line) for line in
                    (tmp_path / "arena.jsonl").read_text(encoding="utf-8").splitlines()) if e
    ]
    replayed = recompute_elos(entries)
    assert {ckpt: val for regime in elo_map.values() for ckpt, val in regime.items()} == replayed


def test_metrics_elo_by_regime_updates_without_restart(client: TestClient, tmp_path: Path):
    """The replay cache follows ledger appends — no restart, no stale map
    (the live defect: rows appended after boot were invisible)."""
    _ledger(
        tmp_path / "arena.jsonl",
        [{"regime": "scratch", "checkpoint": "ckpt-a", "opponent": "none", "games": 0,
          "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0}],
    )
    assert client.get("/api/metrics").json()["elo_by_regime"] == {
        "scratch": {"ckpt-a": 800.0},
    }
    _ledger(
        tmp_path / "arena.jsonl",
        [
            {"regime": "scratch", "checkpoint": "ckpt-a", "opponent": "none", "games": 0,
             "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
            {"regime": "scratch", "checkpoint": "ckpt-b", "opponent": "ckpt-a", "games": 2,
             "wins": 2, "draws": 0, "elo": 816.0, "ts": 200.0},
        ],
    )
    elo_map = client.get("/api/metrics").json()["elo_by_regime"]
    assert set(elo_map["scratch"]) == {"ckpt-a", "ckpt-b"}
    assert elo_map["scratch"]["ckpt-a"] == pytest.approx(784.0)  # 800 - 32*(1.0-0.5)
    assert elo_map["scratch"]["ckpt-b"] == pytest.approx(816.0)


def test_db_mirror_keeps_recorded_claims(client: TestClient, tmp_path: Path):
    """The SQLite mirror keeps the trainer's *recorded claims* (824/776),
    distinct from the replayed regime map — and db.elo_by_regime reads it."""
    _ledger(
        tmp_path / "arena.jsonl",
        [
            {"regime": "scratch", "checkpoint": "ckpt-a", "opponent": "none", "games": 0,
             "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
            {"regime": "scratch", "checkpoint": "ckpt-a", "opponent": "ckpt-b", "games": 4,
             "wins": 3, "draws": 1, "elo": 824.0, "ts": 300.0},
            {"regime": "scratch", "checkpoint": "ckpt-b", "opponent": "ckpt-a", "games": 4,
             "wins": 0, "draws": 1, "elo": 776.0, "ts": 301.0},
        ],
    )
    assert db.backfill_arena() == 3
    assert db.elo_by_regime() == {
        "scratch": {"ckpt-a": 824.0, "ckpt-b": 776.0},  # latest ts wins per checkpoint
    }
