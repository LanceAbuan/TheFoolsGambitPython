"""Take-backs after game over (user story #10): undo on a finished game resumes it.

Board finishes (mate/stalemate/...) revert the finishing plies and clear the
result; resignations un-resign with the move list untouched; abandonments stay
final (409 — the TTL reaper's retirement is not undoable). The cleared state
must be persisted: the rehydrate layer trusts the stored result for finished
games, so an undone game must rehydrate as active, never resurrect the old
result.

Checkmates are driven by pushing moves directly onto the live session (the
same engine primitives the API uses) so the finishing position is
deterministic — the Fool's replies are random and cannot be scripted.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import chess
import pytest
from fastapi.testclient import TestClient

from server import config, db, engine, reaper
from server.main import create_app

REAP_SCAN = 0.05  # test reaper interval (real: 30s)
_SCHOLARS_MATE = ("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "takeback.db")
    monkeypatch.setattr(config, "ARENA_LEDGER_PATH", tmp_path / "no-arena.jsonl")
    monkeypatch.setattr(reaper, "SCAN_INTERVAL_SECONDS", REAP_SCAN)
    db.reset_for_tests()
    engine.reset_for_tests()
    with TestClient(create_app()) as c:
        yield c


def _boot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> TestClient:
    """A fresh app on a named temp FSG_DB; memory + pooled connection dropped first."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / name)
    monkeypatch.setattr(config, "ARENA_LEDGER_PATH", tmp_path / "no-arena.jsonl")
    monkeypatch.setattr(reaper, "SCAN_INTERVAL_SECONDS", REAP_SCAN)
    db.reset_for_tests()
    engine.reset_for_tests()
    return TestClient(create_app())


def _mate_session(gid: str) -> engine.Session:
    """Drive the live session to scholar's mate (the human, as White, delivers it)."""
    session = engine.get_session(gid)
    for raw in _SCHOLARS_MATE:
        engine._push(session, chess.Move.from_uci(raw))
        engine._finish_if_over(session)
    assert session.status == "finished"
    assert session.winner == "white"
    return session


def _row(gid: str) -> dict[str, Any]:
    """The persisted games row (plus a `plies` count) read through a second connection."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM games WHERE id = ?", (gid,)).fetchone()
        plies = conn.execute(
            "SELECT COUNT(*) AS n FROM moves WHERE game_id = ?", (gid,)
        ).fetchone()["n"]
    finally:
        conn.close()
    assert row is not None, f"game {gid} not persisted"
    return {**dict(row), "plies": plies}


def test_undo_after_checkmate_resumes_and_accepts_moves(client):
    """Mate -> undo: active game, mating ply reverted, human on turn, play continues."""
    gid = client.post("/api/play/new", json={"color": "white", "opponent": "fool"}).json()["game_id"]
    _mate_session(gid)

    r = client.post(f"/api/play/{gid}/undo")  # was a hard 409 before user story #10
    assert r.status_code == 200
    snap = r.json()
    assert snap["status"] == "active"
    assert snap["winner"] is None
    assert len(snap["moves"]) == 6  # parity rule: only the finishing ply is reverted
    assert snap["last_move"] == "g8f6"  # ...and the human is on turn
    assert snap["in_check"] is False

    r = client.post(f"/api/play/{gid}/move", json={"from": "a2", "to": "a3"})
    assert r.status_code == 200
    assert len(r.json()["moves"]) == 8  # the Fool replied; the game is live again


def test_undo_after_checkmate_persists_cleared_state(client):
    """The post-undo row is active with a truncated, board-consistent move list."""
    gid = client.post("/api/play/new", json={"color": "white", "opponent": "fool"}).json()["game_id"]
    _mate_session(gid)
    client.post(f"/api/play/{gid}/undo")

    row = _row(gid)
    assert row["status"] == "active"
    assert row["winner"] is None
    assert row["reason"] is None
    assert row["pgn"] is None  # active games carry no result PGN
    assert row["plies"] == 6  # reverted ply rows are gone, not stale


def test_rehydrate_after_undo_keeps_result_cleared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The trap: restore trusts the stored result — undo must rewrite what it trusts."""
    with _boot(tmp_path, monkeypatch, "takeback.db") as client:
        gid = client.post("/api/play/new", json={"color": "white", "opponent": "fool"}).json()["game_id"]
        _mate_session(gid)
        assert client.post(f"/api/play/{gid}/undo").status_code == 200
        before = client.get(f"/api/play/{gid}").json()
        row = _row(gid)
        assert row["status"] == "active"  # the stored row itself is cleared
        assert row["winner"] is None

    # -- full restart over the same DB: the old result must not come back --
    with _boot(tmp_path, monkeypatch, "takeback.db") as client:
        after = client.get(f"/api/play/{gid}").json()
        assert after == before
        assert after["status"] == "active"
        assert after["winner"] is None
        assert len(after["moves"]) == 6
        assert client.post(f"/api/play/{gid}/move", json={"from": "a2", "to": "a3"}).status_code == 200


def test_resign_then_undo_unresigns(client):
    """A resignation has no finishing position: only the result is cleared."""
    gid = client.post("/api/play/new", json={"color": "white", "opponent": "fool"}).json()["game_id"]
    snap = client.post(f"/api/play/{gid}/resign").json()
    assert snap["status"] == "finished"
    assert snap["winner"] == "black"

    r = client.post(f"/api/play/{gid}/undo")
    assert r.status_code == 200
    snap = r.json()
    assert snap["status"] == "active"
    assert snap["winner"] is None
    assert snap["moves"] == []  # move list untouched

    r = client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"})
    assert r.status_code == 200  # play continues
    assert client.post(f"/api/play/{gid}/resign").json()["status"] == "finished"  # no stranded state


def test_resign_midgame_then_undo_keeps_history(client):
    """Un-resign keeps every played move, and the rows stay board-consistent."""
    gid = client.post("/api/play/new", json={"color": "white", "opponent": "fool"}).json()["game_id"]
    client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"})
    assert client.post(f"/api/play/{gid}/resign").status_code == 200

    snap = client.post(f"/api/play/{gid}/undo").json()
    assert snap["status"] == "active"
    assert snap["winner"] is None
    assert len(snap["moves"]) == 2  # e4 + the Fool's reply, all still there

    row = _row(gid)
    assert row["status"] == "active"
    assert row["plies"] == 2


def test_pgn_after_undo_reflects_reverted_state(client):
    """PGN after take-back: finishing move gone, result back to the unfinished '*'."""
    gid = client.post("/api/play/new", json={"color": "white", "opponent": "fool"}).json()["game_id"]
    _mate_session(gid)
    assert "Qxf7" in client.get(f"/api/games/{gid}/pgn").text  # the mating SAN, pre-undo

    client.post(f"/api/play/{gid}/undo")
    pgn = client.get(f"/api/games/{gid}/pgn").text
    assert 'Result "*"' in pgn
    assert "1-0" not in pgn
    assert "Qxf7" not in pgn
    assert "Bc4" in pgn  # the retained moves are intact


def test_undo_after_black_checkmate_lands_on_human_turn(client):
    """Parity holds for the human-as-Black: mate take-back leaves Black to move."""
    session = engine.create_game("black", "fool")
    gid = session.id
    for raw in ("f2f4", "e7e6", "g2g4", "d8h4"):  # 1.f4 e6 2.g4?? Qh4#
        engine._push(session, chess.Move.from_uci(raw))
        engine._finish_if_over(session)
    assert session.status == "finished"
    assert session.winner == "black"  # the human (Black) delivered Qh4#

    snap = client.post(f"/api/play/{gid}/undo").json()
    assert snap["status"] == "active"
    assert snap["winner"] is None
    assert len(snap["moves"]) == 3  # the mating ply reverted; Black (human) to move

    r = client.post(f"/api/play/{gid}/move", json={"from": "a7", "to": "a6"})
    assert r.status_code == 200
    assert len(r.json()["moves"]) == 5  # the Fool replied as White


def test_undo_abandoned_game_conflicts(client, monkeypatch: pytest.MonkeyPatch):
    """Reaped games stay final: the reaper's retirement is exactly what undo must not resurrect."""
    monkeypatch.setattr(config, "GAME_TTL_MIN", 0.01)  # minutes -> 0.6s TTL
    gid = client.post("/api/play/new", json={"color": "white", "opponent": "fool"}).json()["game_id"]
    engine.get_session(gid).last_activity = time.time() - 60.0  # backdate past the TTL

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if client.get(f"/api/play/{gid}").json()["status"] == "finished":
            break
        time.sleep(REAP_SCAN)
    assert client.get(f"/api/play/{gid}").json()["status"] == "finished"
    assert engine.get_session(gid).reason == "abandoned"

    r = client.post(f"/api/play/{gid}/undo")
    assert r.status_code == 409
    assert "abandoned" in r.json()["detail"]
    snap = client.get(f"/api/play/{gid}").json()
    assert snap["status"] == "finished"  # no partial un-finish side effects
    row = _row(gid)
    assert row["status"] == "finished"
    assert row["winner"] == "black"
