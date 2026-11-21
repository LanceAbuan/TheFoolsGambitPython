"""Stockfish opponent: §14.0 contract + WP-A slice 2 behaviors.

Binary-dependent tests skip when no stockfish binary resolves (fresh clones,
CI). The 503 path, the strength-preset table, and binary resolution are always
exercised.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chess
import pytest
from fastapi.testclient import TestClient

from server import config, db, engine, stockfish
from server.main import create_app

_SF_BIN = stockfish.resolve_binary()
requires_binary = pytest.mark.skipif(_SF_BIN is None, reason="no stockfish binary in env/tools/PATH")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "sf.db")
    db.reset_for_tests()
    engine.reset_for_tests()
    stockfish.reset_for_tests()
    with TestClient(create_app()) as c:
        yield c


def _new(client: TestClient, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"color": "white", "opponent": "stockfish", "strength": "wandering"}
    body.update(overrides)
    r = client.post("/api/play/new", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_strength_preset_ladder():
    """The documented mapping covers exactly the contract's strength strings, ascending."""
    assert set(stockfish.STRENGTH_PRESETS) == {"wandering", "curious", "enlightened"}
    skills = [p["skill"] for p in stockfish.STRENGTH_PRESETS.values()]
    assert skills == sorted(skills)
    assert stockfish.preset("nonsense") == stockfish.STRENGTH_PRESETS["curious"]


def test_resolve_env_overrides_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = tmp_path / "fake-stockfish.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("FSG_STOCKFISH", str(fake))
    assert stockfish.resolve_binary() == str(fake)


def test_resolve_env_missing_falls_through(monkeypatch: pytest.MonkeyPatch):
    bogus = "Z:/definitely/not/here/stockfish.exe"
    monkeypatch.setenv("FSG_STOCKFISH", bogus)
    assert stockfish.resolve_binary() != bogus


def test_create_without_binary_503(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(stockfish, "resolve_binary", lambda: None)
    r = client.post("/api/play/new", json={"color": "white", "opponent": "stockfish"})
    assert r.status_code == 503
    assert "stockfish binary not found" in r.json()["detail"]


def test_unknown_strength_422(client: TestClient):
    r = client.post("/api/play/new", json={"opponent": "fool", "strength": "grandmaster"})
    assert r.status_code == 422


@requires_binary
def test_stockfish_replies_and_records_both_plies(client: TestClient):
    gid = _new(client)["game_id"]
    snap = client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"}).json()
    assert snap["moves"][0] == "e4"
    assert len(snap["moves"]) == 2  # stockfish replied to a legal position
    session = engine.get_session(gid)
    board = chess.Board()
    for uci in session.ucis:
        board.push_uci(uci)
    assert board.fen() == snap["fen"]  # the reply is legal and consistent
    games = client.get("/api/games", params={"kind": "play"}).json()["games"]
    assert next(g for g in games if g["id"] == gid)["plies"] == 2  # both plies persisted
    assert stockfish.live_count() == 1  # one process per active game


@requires_binary
def test_stockfish_opens_when_human_is_black(client: TestClient):
    snap = _new(client, color="black")
    state = client.get(f"/api/play/{snap['game_id']}").json()
    assert len(state["moves"]) == 1  # stockfish opened as White


@requires_binary
def test_stockfish_cleanup_on_resign(client: TestClient):
    gid = _new(client)["game_id"]
    client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"})
    assert stockfish.live_count() == 1
    snap = client.post(f"/api/play/{gid}/resign").json()
    assert snap["status"] == "finished"
    assert stockfish.live_count() == 0


@requires_binary
def test_stockfish_cleanup_on_app_shutdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "sf-shutdown.db")
    db.reset_for_tests()
    engine.reset_for_tests()
    stockfish.reset_for_tests()
    with TestClient(create_app()) as client:
        gid = _new(client)["game_id"]
        client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"})
        assert stockfish.live_count() == 1
    assert stockfish.live_count() == 0  # lifespan shutdown stopped the process
