"""Session rehydration across restarts (review-loop fix): boot replay, lazy miss,
legacy-DB migration, corrupt-row resilience, TTL semantics.

Every test drives two full app boots over the same FSG_DB file with the
in-memory session registry and the pooled connection dropped in between —
the in-process equivalent of a server restart (TestClient only, no sockets).
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
ANCIENT_TS = "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Legacy fixture: a pre-migration v1 database (no `strength` column, and no
# PRIMARY KEY on games — the shape duplicate-row dedup must defend against).
# ---------------------------------------------------------------------------

def _legacy_db(path: Path) -> None:
    """A pre-migration v1 database: the WP-A games/moves schema, no `strength`."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE games (
          id TEXT PRIMARY KEY,
          kind TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          orientation TEXT NOT NULL,
          opponent TEXT NOT NULL DEFAULT 'fool',
          status TEXT NOT NULL,
          winner TEXT,
          reason TEXT,
          pgn TEXT
        );
        CREATE TABLE moves (
          game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
          ply INTEGER NOT NULL,
          san TEXT NOT NULL,
          uci TEXT NOT NULL,
          fen_after TEXT NOT NULL,
          PRIMARY KEY (game_id, ply)
        );
        """
    )
    conn.commit()
    conn.close()


def _corrupt_db(path: Path) -> None:
    """A damaged DB: no PRIMARY KEY on games (duplicates possible), moves without
    the FK back-reference — the shape the duplicate-row dedup must defend against."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE games (
          id TEXT NOT NULL,
          kind TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          orientation TEXT NOT NULL,
          opponent TEXT NOT NULL DEFAULT 'fool',
          status TEXT NOT NULL,
          winner TEXT,
          reason TEXT,
          pgn TEXT
        );
        CREATE TABLE moves (
          game_id TEXT NOT NULL,
          ply INTEGER NOT NULL,
          san TEXT NOT NULL,
          uci TEXT NOT NULL,
          fen_after TEXT NOT NULL,
          PRIMARY KEY (game_id, ply)
        );
        """
    )
    conn.commit()
    conn.close()


def _insert_game(
    path: Path,
    gid: str,
    ucis: list[str],
    *,
    orientation: str = "white",
    opponent: str = "fool",
    status: str = "active",
    winner: str | None = None,
    reason: str | None = None,
    ts: str | None = None,
) -> None:
    """Append one games row (+ its move rows) to a legacy-shaped DB.

    ``ts`` defaults to now so fixture games sit well inside the TTL; pass
    ``ANCIENT_TS`` to make a game look idle forever (reaper test).
    """
    ts = ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO games (id, kind, created_at, updated_at, orientation, opponent,"
        " status, winner, reason, pgn) VALUES (?, 'play', ?, ?, ?, ?, ?, ?, ?, NULL)",
        (gid, ts, ts, orientation, opponent, status, winner, reason),
    )
    board = chess.Board()
    for ply, uci in enumerate(ucis):
        try:
            move = chess.Move.from_uci(uci)
            san = board.san(move)
            board.push(move)
            fen = board.fen()
        except ValueError:  # corrupt row: store it raw, sans/fens are not read back
            san, fen = "?", "?"
        conn.execute(
            "INSERT INTO moves (game_id, ply, san, uci, fen_after) VALUES (?,?,?,?,?)",
            (gid, ply, san, uci, fen),
        )
    conn.commit()
    conn.close()


def _boot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *,
    ttl_min: float | None = None,
) -> TestClient:
    """A fresh app on a named temp FSG_DB; memory + pooled connection dropped first."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / name)
    monkeypatch.setattr(config, "ARENA_LEDGER_PATH", tmp_path / "no-arena.jsonl")  # isolate
    if ttl_min is not None:
        monkeypatch.setattr(config, "GAME_TTL_MIN", ttl_min)
    db.reset_for_tests()
    engine.reset_for_tests()
    return TestClient(create_app())


def _legal_moves(fen: str) -> list[str]:
    return sorted(move.uci() for move in chess.Board(fen).legal_moves)


# ---------------------------------------------------------------------------
# Restart cycle: the defect from the review loop, end to end.
# ---------------------------------------------------------------------------

def test_restart_cycle_restores_state_lobby_ws_and_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Moves persist -> full restart -> identical state, lobby entry, WS resubscribe,
    persisted strength, and the game accepts further moves."""
    with _boot(tmp_path, monkeypatch, "cycle.db") as client:
        gid = client.post(
            "/api/play/new", json={"color": "white", "opponent": "fool", "strength": "enlightened"}
        ).json()["game_id"]
        empty_gid = client.post("/api/play/new", json={"color": "white"}).json()["game_id"]
        for frm, to in (("e2", "e4"), ("d2", "d4")):
            assert client.post(f"/api/play/{gid}/move", json={"from": frm, "to": to}).status_code == 200
        before = client.get(f"/api/play/{gid}").json()
        empty_before = client.get(f"/api/play/{empty_gid}").json()
        assert empty_before["moves"] == []  # zero-move game also hits the DB

    # -- full restart: same DB file, memory + connection dropped by _boot --
    with _boot(tmp_path, monkeypatch, "cycle.db") as client:
        after = client.get(f"/api/play/{gid}").json()
        assert after == before  # identical snapshot: fen, history, status, winner, last_move
        assert _legal_moves(after["fen"]) == _legal_moves(before["fen"])
        empty_after = client.get(f"/api/play/{empty_gid}").json()
        assert empty_after == empty_before

        # the lobby lists both restored games, with the persisted strength
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"seq": 1, "type": "subscribe", "topic": "lobby"})
            lobby = ws.receive_json()
        entries = {g["game_id"]: g for g in lobby["payload"]["active_games"]}
        assert entries[gid]["strength"] == "enlightened"
        assert entries[gid]["status"] == "active"
        assert entries[gid]["id"] == gid  # additive id alias rides along

        # WS game:{gid} resubscribes with the restored state snapshot
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"seq": 1, "type": "subscribe", "topic": f"game:{gid}"})
            snap = ws.receive_json()
        assert snap["type"] == "state"
        assert snap["payload"]["game_id"] == gid
        assert snap["payload"]["fen"] == before["fen"]

        # play continues on the rehydrated session
        assert client.post(f"/api/play/{gid}/move", json={"from": "a2", "to": "a3"}).status_code == 200


def test_reaper_still_reaps_rehydrated_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """last_activity is derived from the persisted updated_at: a game idle before
    the restart is reaped after it (TTL semantics unchanged)."""
    with _boot(tmp_path, monkeypatch, "reap.db") as client:
        gid = client.post("/api/play/new", json={"color": "white"}).json()["game_id"]
        assert client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"}).status_code == 200

    # Backdate the persisted game so rehydration must map updated_at -> last_activity.
    conn = sqlite3.connect(tmp_path / "reap.db")
    conn.execute("UPDATE games SET updated_at = ?", (ANCIENT_TS,))
    conn.commit()
    conn.close()
    monkeypatch.setattr(reaper, "SCAN_INTERVAL_SECONDS", REAP_SCAN)
    with _boot(tmp_path, monkeypatch, "reap.db", ttl_min=0.01) as client:  # 0.6 s TTL
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if client.get(f"/api/play/{gid}").json()["status"] == "finished":
                break
            time.sleep(REAP_SCAN)
        snap = client.get(f"/api/play/{gid}").json()
        assert snap["status"] == "finished"
        assert snap["winner"] == "black"  # the human (White) abandoned -> loss
        assert engine.get_session(gid).reason == "abandoned"
        row = next(
            g for g in client.get("/api/games", params={"kind": "play"}).json()["games"] if g["id"] == gid
        )
        assert row["status"] == "finished"  # reaped state persisted


# ---------------------------------------------------------------------------
# Migration: pre-migration fixture DB gains games.strength, rows backfill.
# ---------------------------------------------------------------------------

def test_legacy_db_migration_adds_strength_and_restores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "legacy.db"
    _legacy_db(dbfile)
    _insert_game(dbfile, "legacy001", ["e2e4", "e7e5"])  # human e4 + Fool e5

    with _boot(tmp_path, monkeypatch, "legacy.db") as client:
        # the legacy file was opened in place (not replaced by a fresh schema)
        with sqlite3.connect(dbfile) as conn:
            cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(games)")}
        assert "strength" in cols

        snap = client.get("/api/play/legacy001").json()
        assert snap["game_id"] == "legacy001"
        assert snap["moves"] == ["e4", "e5"]
        assert snap["status"] == "active"
        assert engine.get_session("legacy001").strength == "curious"  # backfilled default

        # writes work against the migrated schema (persist names the new column);
        # the rehydrated legacy session also gets a live Fool reply (2 + 1 + 1 plies)
        assert client.post("/api/play/legacy001/move", json={"from": "g1", "to": "f3"}).status_code == 200
        games = client.get("/api/games", params={"kind": "play"}).json()["games"]
        assert next(g for g in games if g["id"] == "legacy001")["plies"] == 4


def test_migration_is_idempotent_across_boots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "legacy2.db"
    _legacy_db(dbfile)
    _insert_game(dbfile, "legacy002", [])
    with _boot(tmp_path, monkeypatch, "legacy2.db"):
        pass
    with _boot(tmp_path, monkeypatch, "legacy2.db") as client:  # second boot: no duplicate ALTER
        assert client.get("/api/play/legacy002").json()["status"] == "active"
        with sqlite3.connect(dbfile) as conn:
            cols = [str(row[1]) for row in conn.execute("PRAGMA table_info(games)")]
        assert cols.count("strength") == 1


# ---------------------------------------------------------------------------
# Edge cases: malformed/illegal stored UCIs, duplicates, empty move lists,
# finished games.
# ---------------------------------------------------------------------------

def test_malformed_stored_uci_skips_game_without_crashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "corrupt.db"
    _legacy_db(dbfile)
    _insert_game(dbfile, "good0001", ["e2e4", "e7e5"])
    _insert_game(dbfile, "bad00001", ["e2e4", "zzzz"])  # unparseable UCI
    _insert_game(dbfile, "bad00002", ["a1a3"])  # parseable but illegal in position

    with _boot(tmp_path, monkeypatch, "corrupt.db") as client:
        assert client.get("/api/health").json() == {"ok": True}  # boot survived
        assert client.get("/api/play/good0001").json()["moves"] == ["e4", "e5"]
        # corrupt games are absent from play (404 on both the boot + lazy paths)...
        assert client.get("/api/play/bad00001").status_code == 404
        assert client.get("/api/play/bad00002").status_code == 404
        # ...but the archive listing still shows every stored row untouched
        rows = client.get("/api/games", params={"kind": "play"}).json()["games"]
        assert {g["id"] for g in rows} == {"good0001", "bad00001", "bad00002"}


def test_duplicate_game_rows_first_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "dupes.db"
    _corrupt_db(dbfile)  # no PK on games: duplicate rows are representable
    _insert_game(dbfile, "dup00001", ["e2e4"], orientation="white")
    _insert_game(dbfile, "dup00001", [], orientation="black")  # duplicate row

    with _boot(tmp_path, monkeypatch, "dupes.db") as client:
        assert client.get("/api/play/dup00001").json()["moves"] == ["e4"]
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"seq": 1, "type": "subscribe", "topic": "lobby"})
            lobby = ws.receive_json()
        entries = [g for g in lobby["payload"]["active_games"] if g["game_id"] == "dup00001"]
        assert len(entries) == 1
        assert entries[0]["orientation"] == "white"  # the FIRST row won


def test_finished_game_restores_lazily_with_result_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Finished games are not boot-rehydrated (memory stays bounded by real play)
    but rebuild on first access with their stored result state."""
    with _boot(tmp_path, monkeypatch, "finished.db") as client:
        gid = client.post("/api/play/new", json={"color": "white"}).json()["game_id"]
        client.post(f"/api/play/{gid}/move", json={"from": "e2", "to": "e4"})
        client.post(f"/api/play/{gid}/resign")

    with _boot(tmp_path, monkeypatch, "finished.db") as client:
        assert engine.all_sessions() == []  # not restored at boot
        snap = client.get(f"/api/play/{gid}").json()  # lazy miss -> rebuilt from the DB
        assert len(engine.all_sessions()) == 1
        assert snap["status"] == "finished"
        assert snap["winner"] == "black"
        assert client.post(f"/api/play/{gid}/move", json={"from": "a2", "to": "a3"}).status_code == 409
        assert client.post(f"/api/play/{gid}/resign").status_code == 409
        # PGN now round-trips the real result post-restart (memory fallback used '*')
        pgn = client.get(f"/api/games/{gid}/pgn").text
        assert "1. e4" in pgn
        assert "0-1" in pgn


def test_unknown_game_still_404s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with _boot(tmp_path, monkeypatch, "unknown.db") as client:
        assert client.get("/api/play/nope").status_code == 404


def test_rehydrated_session_reports_oriented_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A Black game restores the opener so the human is back on turn."""
    with _boot(tmp_path, monkeypatch, "black.db") as client:
        gid = client.post("/api/play/new", json={"color": "black"}).json()["game_id"]
        before = client.get(f"/api/play/{gid}").json()
        assert len(before["moves"]) == 1
    with _boot(tmp_path, monkeypatch, "black.db") as client:
        after = client.get(f"/api/play/{gid}").json()
        assert after == before
        # the human's first move is accepted (it is Black's turn, not a stranded position)
        assert client.post(f"/api/play/{gid}/move", json={"from": "e7", "to": "e5"}).status_code == 200


def test_restore_helpers_reject_bad_payloads() -> None:
    """The row validator is the corruption firewall: documented rejects stay rejects."""
    good: dict[str, Any] = {
        "id": "unit00001", "kind": "play", "orientation": "white", "opponent": "fool",
        "status": "active", "winner": None, "reason": None, "created_at": ANCIENT_TS,
        "updated_at": ANCIENT_TS, "strength": "curious",
    }
    assert engine._session_from_row(good, []) is not None
    for field, value in (
        ("orientation", "sideways"),
        ("opponent", "hal9000"),
        ("status", "paused"),
    ):
        bad = {**good, field: value}
        assert engine._session_from_row(bad, []) is None
    finished_bad = {**good, "status": "finished", "winner": "whoever"}
    assert engine._session_from_row(finished_bad, []) is None
    assert engine._parse_iso_ts(ANCIENT_TS) < time.time() - 1000.0
    assert engine._parse_iso_ts("garbage") == pytest.approx(time.time(), abs=5.0)


def test_restore_layer_self_migrates_legacy_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """db.restore_active on a raw legacy DB migrates it in place and backfills
    the missing strength default — order-safe even without the lifespan."""
    dbfile = tmp_path / "nomig.db"
    _legacy_db(dbfile)
    _insert_game(dbfile, "raw000001", ["d2d4", "d7d5"])
    db.reset_for_tests()
    engine.reset_for_tests()
    monkeypatch.setattr(config, "DB_PATH", dbfile)
    db.connect()  # creates missing tables but does NOT migrate (no strength yet)
    with sqlite3.connect(dbfile) as conn:
        assert "strength" not in {str(r[1]) for r in conn.execute("PRAGMA table_info(games)")}
    sources = db.restore_active()
    assert set(sources) == {"raw000001"}
    row, ucis = sources["raw000001"]
    assert ucis == ["d2d4", "d7d5"]
    assert row["strength"] == "curious"  # ALTER backfilled the default
    with sqlite3.connect(dbfile) as conn:
        assert "strength" in {str(r[1]) for r in conn.execute("PRAGMA table_info(games)")}
    db.reset_for_tests()
    engine.reset_for_tests()
