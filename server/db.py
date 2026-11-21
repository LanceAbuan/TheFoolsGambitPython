"""SQLite persistence (WAL mode). Games archive + full move history.

Sync access guarded by a lock; async endpoints call in via `asyncio.to_thread`.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
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
CREATE TABLE IF NOT EXISTS moves (
  game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
  ply INTEGER NOT NULL,
  san TEXT NOT NULL,
  uci TEXT NOT NULL,
  fen_after TEXT NOT NULL,
  PRIMARY KEY (game_id, ply)
);
"""

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def connect(path: Path | None = None) -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    target = path or config.DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    _conn = conn
    return _conn


def tx(operation: Callable[[sqlite3.Connection], Any]) -> Any:
    """Run operation(conn) under the lock; commit on success, rollback on error."""
    with _lock:
        conn = connect()
        try:
            result = operation(conn)
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise


def persist_session(session: Any) -> None:
    """Idempotent write-through of a live game session (header + all moves)."""

    def _op(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO games"
            " (id, kind, created_at, updated_at, orientation, opponent, status, winner, reason, pgn)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                session.id,
                session.kind,
                _iso(session.created_at),
                _iso(time.time()),
                session.orientation,
                session.opponent,
                session.status,
                session.winner,
                session.reason,
                session.pgn if session.status == "finished" else None,
            ),
        )
        conn.execute("DELETE FROM moves WHERE game_id = ?", (session.id,))
        conn.executemany(
            "INSERT INTO moves (game_id, ply, san, uci, fen_after) VALUES (?,?,?,?,?)",
            [
                (session.id, ply, san, uci, fen)
                for ply, (san, uci, fen) in enumerate(session.record())
            ],
        )

    tx(_op)


def list_games(kind: str | None, limit: int) -> list[dict[str, Any]]:
    def _op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        sql = (
            "SELECT g.id, g.kind, g.created_at, g.updated_at, g.orientation, g.opponent,"
            " g.status, g.winner, g.reason,"
            " (SELECT COUNT(*) FROM moves m WHERE m.game_id = g.id) AS plies"
            " FROM games g"
        )
        params: list[Any] = []
        if kind:
            sql += " WHERE g.kind = ?"
            params.append(kind)
        sql += " ORDER BY g.updated_at DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in conn.execute(sql, params)]

    return tx(_op)


def moves_for(gid: str) -> tuple[list[str], str] | None:
    """Replay data for archived games: (sans, orientation) or None if unknown."""

    def _op(conn: sqlite3.Connection) -> tuple[list[str], str] | None:
        game = conn.execute("SELECT orientation FROM games WHERE id = ?", (gid,)).fetchone()
        if game is None:
            return None
        rows = conn.execute(
            "SELECT san FROM moves WHERE game_id = ? ORDER BY ply", (gid,)
        ).fetchall()
        return [row["san"] for row in rows], game["orientation"]

    return tx(_op)


def reset_for_tests() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
