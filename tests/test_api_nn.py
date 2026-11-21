"""Neural opponent "nn" (FS-1): champion loading, 503 path, full turn, reload, shutdown.

Tiny random GambitNet checkpoints are generated at test time (TINY preset →
tmp runs dir) — no fixture files. Device-dependent assertions stay
device-agnostic (cuda when available, else cpu).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import chess
import pytest
from fastapi.testclient import TestClient

from server import config, db, engine, model_server
from server.main import create_app


def _write_champion(runs_dir: Path, tag: str = "tiny-0001", seed: int = 0, regime: str = "scratch") -> Path:
    """Save a tiny random GambitNet checkpoint + champion pointer; returns the .pt path."""
    import torch

    from training.model import MODEL_TINY, GambitNet

    regime_dir = runs_dir / regime
    ckpt_dir = regime_dir / "testrun" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = GambitNet(MODEL_TINY)
    ckpt = ckpt_dir / f"{tag}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": MODEL_TINY.to_dict(),
            "tag": tag,
            "regime": "scratch",
            "run_id": "testrun",
            "step": 1,
            "games": 0,
            "ts": time.time(),
        },
        ckpt,
    )
    regime_dir.joinpath("champion.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "checkpoint_path": str(ckpt),
                "elo": 800.0,
                "run_id": "testrun",
                "model_config": MODEL_TINY.to_dict(),
                "updated_at": time.time(),
            }
        ),
        encoding="utf-8",
    )
    return ckpt


def _future_mtime(pointer: Path) -> None:
    """Push a pointer's mtime past the current champion's (reload trigger)."""
    future = time.time() + 10
    os.utime(pointer, (future, future))


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "nn.db")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    db.reset_for_tests()
    engine.reset_for_tests()
    model_server.reset_for_tests()
    with TestClient(create_app()) as c:
        yield c
    model_server.reset_for_tests()


def test_sims_ladder_matches_spec():
    """wandering/curious/enlightened map onto the FS-1 sims (32/96/256), ascending."""
    assert set(model_server.NN_SIMS) == {"wandering", "curious", "enlightened"}
    sims = list(model_server.NN_SIMS.values())
    assert sims == sorted(sims) == [32, 96, 256]
    assert model_server.NN_SIMS["curious"] == 96


def test_no_champion_503(client: TestClient):
    r = client.post("/api/play/new", json={"color": "white", "opponent": "nn"})
    assert r.status_code == 503
    assert "no trained model yet" in r.json()["detail"]


def test_full_turn_returns_legal_nn_reply(client: TestClient, tmp_path: Path):
    _write_champion(tmp_path / "runs")
    body = client.post("/api/play/new", json={"color": "white", "opponent": "nn"}).json()
    assert model_server.loaded() is None  # lazy: pointer checked at create, model loads on move

    started = time.monotonic()
    r = client.post(f"/api/play/{body['game_id']}/move", json={"from": "e2", "to": "e4"})
    elapsed = time.monotonic() - started
    assert r.status_code == 200, r.text
    snap = r.json()
    assert snap["moves"][0] == "e4"
    assert len(snap["moves"]) == 2  # the NN replied to a legal position
    assert model_server.loaded() is not None  # lazy-loaded on first use

    # The reply must be a real legal chess move replaying to the reported fen.
    session = engine.get_session(body["game_id"])
    board = chess.Board()
    for uci in session.ucis:
        board.push_uci(uci)
    assert board.fen() == snap["fen"]
    assert elapsed < 2.0, f"NN reply took {elapsed:.2f}s (gate: < 2s at curious)"


def test_nn_opens_when_human_is_black(client: TestClient, tmp_path: Path):
    _write_champion(tmp_path / "runs")
    body = client.post("/api/play/new", json={"color": "black", "opponent": "nn"}).json()
    state = client.get(f"/api/play/{body['game_id']}").json()
    assert len(state["moves"]) == 1  # the NN opened as White


def test_reload_on_champion_mtime_change(client: TestClient, tmp_path: Path):
    runs = tmp_path / "runs"
    _write_champion(runs, tag="tiny-0001", seed=0)
    body = client.post("/api/play/new", json={"color": "white", "opponent": "nn"}).json()
    client.post(f"/api/play/{body['game_id']}/move", json={"from": "e2", "to": "e4"})
    assert model_server.loaded() is not None
    assert model_server.loaded().tag == "tiny-0001"

    # A new champion: fresh weights + newer pointer mtime (set explicitly to be
    # independent of filesystem timestamp granularity).
    _write_champion(runs, tag="tiny-0002", seed=1)
    _future_mtime(runs / "scratch" / "champion.json")

    r = client.post(f"/api/play/{body['game_id']}/move", json={"from": "d2", "to": "d4"})
    assert r.status_code == 200
    assert model_server.loaded() is not None
    assert model_server.loaded().tag == "tiny-0002"  # reloaded, not the cached champion


def test_device_cleanup_on_shutdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "nn-shutdown.db")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    db.reset_for_tests()
    engine.reset_for_tests()
    model_server.reset_for_tests()
    _write_champion(tmp_path / "runs")
    with TestClient(create_app()) as client:
        body = client.post("/api/play/new", json={"color": "white", "opponent": "nn"}).json()
        client.post(f"/api/play/{body['game_id']}/move", json={"from": "e2", "to": "e4"})
        assert model_server.loaded() is not None
        assert model_server.loaded().device.type in ("cuda", "cpu")
    assert model_server.loaded() is None  # lifespan shutdown released the model


def test_resolve_prefers_newest_pointer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Multiple regimes: the newest valid champion.json wins."""
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    _write_champion(tmp_path, tag="scr", seed=0, regime="scratch")
    older = model_server.resolve_champion()
    assert older is not None
    assert older[0].parent.name == "scratch"
    time.sleep(0.01)
    _write_champion(tmp_path, tag="ft", seed=1, regime="finetune")
    newest = model_server.resolve_champion()
    assert newest is not None
    assert newest[0].parent.name == "finetune"
    assert newest[1] > older[1]
