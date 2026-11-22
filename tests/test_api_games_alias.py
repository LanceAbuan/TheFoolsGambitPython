"""Archive id unification (review-loop polish): REST /api/games play rows and the
WS lobby payload carry BOTH `id` (the pinned §14.0 shape — never removed) and an
additive `game_id` alias so the web client's GameArchiveEntry can align."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import config, db, engine
from server.main import create_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "alias.db")
    monkeypatch.setattr(config, "ARENA_LEDGER_PATH", tmp_path / "no-arena.jsonl")  # isolate
    db.reset_for_tests()
    engine.reset_for_tests()
    with TestClient(create_app()) as c:
        yield c


def test_play_archive_rows_carry_both_ids(client: TestClient):
    """Finished + active rows alike: `id` (pinned) and `game_id` (alias), equal."""
    gid = client.post("/api/play/new", json={"color": "white", "opponent": "fool"}).json()["game_id"]
    client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"})
    client.post(f"/api/play/{gid}/resign")
    other = client.post("/api/play/new", json={"color": "white"}).json()["game_id"]

    rows = client.get("/api/games", params={"kind": "play"}).json()["games"]
    assert {r["id"] for r in rows} == {gid, other}
    for row in rows:
        assert "id" in row and "game_id" in row
        assert row["id"] == row["game_id"]
    assert next(r for r in rows if r["id"] == gid)["plies"] == 2  # row shape intact


def test_unfiltered_games_listing_carries_both_ids(client: TestClient):
    gid = client.post("/api/play/new", json={"color": "white"}).json()["game_id"]
    rows = client.get("/api/games").json()["games"]  # no kind filter: active rows too
    assert any(r["id"] == gid and r["game_id"] == gid for r in rows)


def test_lobby_entries_carry_both_ids(client: TestClient):
    first = client.post("/api/play/new", json={"color": "white"}).json()["game_id"]
    second = client.post("/api/play/new", json={"color": "black"}).json()["game_id"]
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "lobby"})
        lobby = ws.receive_json()
    entries = lobby["payload"]["active_games"]
    assert {e["game_id"] for e in entries} == {first, second}
    for entry in entries:
        assert entry["id"] == entry["game_id"]  # additive alias, equal to the WS name


def test_arena_rows_shape_untouched(client: TestClient):
    """Arena ledger rows are a different (documented) shape: no fabricated game id."""
    games = client.get("/api/games", params={"kind": "arena"}).json()["games"]
    assert games == []
    assert all("game_id" not in g for g in games)
