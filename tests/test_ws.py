from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import config, db, engine
from server.main import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "ws.db")
    db.reset_for_tests()
    engine.reset_for_tests()
    with TestClient(create_app()) as c:
        yield c


def test_subscribe_sends_state_then_position(client):
    gid = client.post("/api/play/new", json={"color": "white"}).json()["game_id"]
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": f"game:{gid}"})
        state = ws.receive_json()
        assert state["type"] == "state"
        assert state["payload"]["game_id"] == gid

        client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"})
        position = ws.receive_json()
        assert position["type"] == "position"
        # The Fool's reply is applied atomically with the player move, so the
        # broadcast reflects the final position of the turn: move 1 complete.
        assert position["payload"]["move_number"] == 1
        assert position["payload"]["last_move"] not in (None, "e2e4")
        assert position["seq"] >= 2


def test_invalid_topic_rejected(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "game:drop;table"})
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_train_topic_sends_metrics(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        msg = ws.receive_json()
        assert msg["type"] == "metrics"
        assert "games_played" in msg["payload"]


def test_lobby_topic_lists_active_games(client):
    client.post("/api/play/new", json={"color": "white"})
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "lobby"})
        msg = ws.receive_json()
        assert msg["type"] == "lobby"
        assert len(msg["payload"]["active_games"]) >= 1
