from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import config, db, engine
from server.main import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    db.reset_for_tests()
    engine.reset_for_tests()
    with TestClient(create_app()) as c:
        yield c


def _new_game(client, **overrides):
    body = {"color": "white", "opponent": "fool"}
    body.update(overrides)
    return client.post("/api/play/new", json=body)


def test_health(client):
    assert client.get("/api/health").json() == {"ok": True}


def test_new_game_shape(client):
    r = _new_game(client)
    assert r.status_code == 200
    body = r.json()
    assert body["orientation"] == "white"
    assert body["fen"].startswith("rnbqkbnr/pppppppp")


def test_move_then_fool_reply(client):
    gid = _new_game(client).json()["game_id"]
    r = client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"})
    assert r.status_code == 200
    snap = r.json()
    assert snap["moves"][0] == "e4"
    assert len(snap["moves"]) == 2  # the Fool replied


def test_illegal_move_422(client):
    gid = _new_game(client).json()["game_id"]
    r = client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e5"})
    assert r.status_code == 422


def test_undo_pops_both_plies(client):
    gid = _new_game(client).json()["game_id"]
    client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"})
    r = client.post(f"/api/play/{gid}/undo")
    assert r.status_code == 200
    assert r.json()["moves"] == []


def test_resign_then_move_conflicts(client):
    gid = _new_game(client).json()["game_id"]
    r = client.post(f"/api/play/{gid}/resign")
    assert r.status_code == 200
    snap = r.json()
    assert snap["status"] == "finished"
    assert snap["winner"] == "black"
    r2 = client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"})
    assert r2.status_code == 409


def test_unknown_game_404(client):
    assert client.get("/api/play/nope").status_code == 404


def test_games_archive_and_pgn(client):
    gid = _new_game(client).json()["game_id"]
    client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"})
    client.post(f"/api/play/{gid}/resign")
    games = client.get("/api/games", params={"kind": "play"}).json()["games"]
    assert len(games) == 1
    assert games[0]["id"] == gid
    pgn = client.get(f"/api/games/{gid}/pgn").text
    assert "1. e4" in pgn


def test_black_orientation_fool_opens(client):
    snap = _new_game(client, color="black").json()
    assert snap["orientation"] == "black"
    state = client.get(f"/api/play/{snap['game_id']}").json()
    assert len(state["moves"]) == 1  # the Fool opened as White


def test_metrics_shape(client):
    body = client.get("/api/metrics").json()
    assert set(body) >= {"elo", "loss", "win_rate", "games_played"}


def test_train_admin_gate(client, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_TOKEN", "sekrit")
    assert client.post("/api/train/reset").status_code == 403
    r = client.post("/api/train/reset", headers={"X-Admin-Token": "sekrit"})
    assert r.status_code == 503  # training service attaches with WP-C


def test_black_undo_before_first_move_is_safe_noop(client):
    """Regression: Black undoing before any human move must keep the opener, not strand the game."""
    snap = _new_game(client, color="black").json()
    gid = snap["game_id"]
    r = client.post(f"/api/play/{gid}/undo")
    assert r.status_code == 200
    state = client.get(f"/api/play/{gid}").json()
    assert state["status"] == "active"
    assert len(state["moves"]) == 1  # the Fool's opener is preserved


def test_white_undo_with_no_moves_is_noop(client):
    gid = _new_game(client).json()["game_id"]
    r = client.post(f"/api/play/{gid}/undo")
    assert r.status_code == 200
    assert client.get(f"/api/play/{gid}").json()["moves"] == []
