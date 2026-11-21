"""WS feeds: train-topic telemetry pushes + lobby opponent info (WP-A slice 2)."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import config, db, engine, stockfish, telemetry
from server.main import create_app

INTERVAL = 0.2  # test interval; well inside the ~2s push budget


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "feeds.db")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(telemetry, "INTERVAL_SECONDS", INTERVAL)
    db.reset_for_tests()
    engine.reset_for_tests()
    stockfish.reset_for_tests()
    with TestClient(create_app()) as c:
        yield c


def test_train_feed_pushes_metrics_within_interval(client: TestClient):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        first = ws.receive_json()
        assert first["type"] == "metrics"  # subscribe-time snapshot
        assert "games_played" in first["payload"]

        received: dict[str, Any] = {}

        def _recv() -> None:
            received["msg"] = ws.receive_json()

        watcher = threading.Thread(target=_recv, daemon=True)
        watcher.start()
        watcher.join(timeout=2.0)
        assert not watcher.is_alive(), "no metrics push within 2s"
        push = received["msg"]
        assert push["type"] == "metrics"
        assert push["topic"] == "train"
        assert push["seq"] > first["seq"]  # snapshots + feed share one monotonic stream


def test_train_feed_idles_without_subscribers(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str]] = []
    original = telemetry.hub.send

    async def _spy(topic: str, mtype: str, payload: dict[str, Any]) -> None:
        calls.append((topic, mtype))
        await original(topic, mtype, payload)

    monkeypatch.setattr(telemetry.hub, "send", _spy)
    time.sleep(INTERVAL * 4)  # several loop ticks with zero train subscribers
    assert calls == []  # the idle guard kept the loop from touching the hub


def test_lobby_payload_includes_opponent_info(client: TestClient):
    r = client.post("/api/play/new", json={"color": "white", "opponent": "fool", "strength": "enlightened"})
    assert r.status_code == 200
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "lobby"})
        msg = ws.receive_json()
        assert msg["type"] == "lobby"
        game = msg["payload"]["active_games"][0]
        assert game["game_id"] == r.json()["game_id"]
        assert game["opponent"] == "fool"
        assert game["strength"] == "enlightened"


def test_snapshot_then_feed_share_seq_counter(client: TestClient):
    """Subscribe snapshot consumes a real seq so the stream never repeats a number."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        first = ws.receive_json()
        assert first["seq"] == 1  # fresh topic starts at 1 (was hardcoded before)
        received: dict[str, Any] = {}

        def _recv() -> None:
            received["msg"] = ws.receive_json()

        watcher = threading.Thread(target=_recv, daemon=True)
        watcher.start()
        watcher.join(timeout=2.0)
        assert not watcher.is_alive()
        assert received["msg"]["seq"] == 2
