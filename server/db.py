"""SQLite persistence (WAL mode, busy_timeout 5s, synchronous=NORMAL).

Games archive + full move history + arena ledger. Sync access guarded by a
lock; async endpoints call in via `asyncio.to_thread`.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import config

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  orientation TEXT NOT NULL,
  opponent TEXT NOT NULL DEFAULT 'fool',
  strength TEXT NOT NULL DEFAULT 'curious',
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
CREATE TABLE IF NOT EXISTS arena_games (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL DEFAULT '',
  regime TEXT NOT NULL DEFAULT '',
  ckpt TEXT NOT NULL,
  opponent TEXT NOT NULL DEFAULT '',
  games INTEGER NOT NULL DEFAULT 0,
  wins INTEGER NOT NULL DEFAULT 0,
  draws INTEGER NOT NULL DEFAULT 0,
  elo REAL NOT NULL DEFAULT 800.0,
  ts REAL NOT NULL
);
-- FS-2 dedupe key: one ledger row per (checkpoint, ts), making backfill idempotent.
CREATE UNIQUE INDEX IF NOT EXISTS idx_arena_games_dedupe ON arena_games(ckpt, ts);
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
    # Another process (training worker, arena job) may hold SQLite's write
    # lock; without a busy timeout our writes would fail instantly with
    # "database is locked" instead of waiting their turn.
    conn.execute("PRAGMA busy_timeout=5000")
    # WAL's documented pairing: corruption-safe, and it drops the per-commit
    # fsync that would otherwise stall every writer queued behind the timeout.
    conn.execute("PRAGMA synchronous=NORMAL")
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
            " (id, kind, created_at, updated_at, orientation, opponent, strength,"
            " status, winner, reason, pgn)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                session.id,
                session.kind,
                _iso(session.created_at),
                _iso(time.time()),
                session.orientation,
                session.opponent,
                session.strength,
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
        # `game_id` is an additive alias of the pinned §14.0 `id` field so the
        # web client's GameArchiveEntry (which speaks `game_id`) can align.
        sql = (
            "SELECT g.id, g.id AS game_id, g.kind, g.created_at, g.updated_at,"
            " g.orientation, g.opponent, g.status, g.winner, g.reason,"
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


# ---------------------------------------------------------------------------
# Schema migration (additive only, run from the app lifespan — never from
# connect(), so merely importing this package, e.g. during test collection
# against the live DB, cannot mutate anything).
# ---------------------------------------------------------------------------

def migrate() -> None:
    """Apply additive schema migrations to a pre-existing database (idempotent).

    v1 → v2: ``games.strength`` records the opponent's skill preset so sessions
    rehydrate faithfully after a restart. Existing rows backfill with the
    default ``'curious'``; fresh databases already create the column.
    """

    def _op(conn: sqlite3.Connection) -> None:
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(games)")}
        if "strength" not in columns:
            conn.execute("ALTER TABLE games ADD COLUMN strength TEXT NOT NULL DEFAULT 'curious'")
            log.info("db migration: games.strength column added (existing rows -> 'curious')")

    tx(_op)


# ---------------------------------------------------------------------------
# Session rehydration (restart resilience): raw restore sources for engine.py,
# which owns the chess-level replay/validation.
# ---------------------------------------------------------------------------

RestoreSource = tuple[dict[str, Any], list[str]]  # (games row, ordered UCIs)

_GAME_COLUMNS = (
    "id, kind, created_at, updated_at, orientation, opponent, strength, status, winner, reason"
)


def _restore_source(conn: sqlite3.Connection, row: sqlite3.Row) -> RestoreSource:
    """(games row dict, ordered UCIs) for a single game row."""
    ucis = [
        str(m["uci"])
        for m in conn.execute("SELECT uci FROM moves WHERE game_id = ? ORDER BY ply", (row["id"],))
    ]
    return dict(row), ucis


def restore_active() -> dict[str, RestoreSource]:
    """Rehydration sources for every still-``active`` game row, oldest first.

    Duplicate ``game_id`` rows (possible only in legacy tables created before
    the primary key existed) are deduplicated first-wins with a warning.
    Self-migrating so a restore never reads a schema it did not ensure.
    """
    migrate()

    def _op(conn: sqlite3.Connection) -> dict[str, RestoreSource]:
        out: dict[str, RestoreSource] = {}
        rows = conn.execute(
            f"SELECT {_GAME_COLUMNS} FROM games WHERE status = 'active' ORDER BY rowid"
        ).fetchall()
        for row in rows:
            gid = str(row["id"])
            if gid in out:
                log.warning("db restore: duplicate games row for %s; keeping the first", gid)
                continue
            out[gid] = _restore_source(conn, row)
        return out

    return tx(_op)


def restore_one(gid: str) -> RestoreSource | None:
    """Rehydration source for one game (any status), or None when unknown.

    Self-migrating (idempotent, one PRAGMA check) so a lazy rehydrate on a
    legacy database works even if the lifespan has not run in this process.
    """
    migrate()

    def _op(conn: sqlite3.Connection) -> RestoreSource | None:
        row = conn.execute(
            f"SELECT {_GAME_COLUMNS} FROM games WHERE id = ?", (gid,)
        ).fetchone()
        if row is None:
            return None
        return _restore_source(conn, row)

    return tx(_op)


def reset_for_tests() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


# ---------------------------------------------------------------------------
# Arena ledger (FS-2): SQLite mirror of data/arena.jsonl + per-regime Elo.
# ---------------------------------------------------------------------------

def backfill_arena(path: Path | None = None) -> int:
    """Idempotent startup import of the training JSONL ledger into ``arena_games``.

    Every readable line is inserted with ``INSERT OR IGNORE`` against the
    UNIQUE ``(ckpt, ts)`` index, so re-running after a restart (double boot)
    inserts nothing. Returns the number of rows actually added.
    """
    source = path or config.ARENA_LEDGER_PATH
    if not source.is_file():
        return 0
    rows: list[tuple[str, str, str, str, int, int, int, float, float]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            log.warning("arena backfill: skipping corrupt line in %s", source)
            continue
        if not isinstance(entry, dict) or "checkpoint" not in entry or "ts" not in entry:
            log.warning("arena backfill: skipping malformed entry in %s", source)
            continue
        rows.append(
            (
                str(entry.get("run_id", "")),
                str(entry.get("regime", "")),
                str(entry.get("checkpoint", "")),
                str(entry.get("opponent", "")),
                int(entry.get("games", 0) or 0),
                int(entry.get("wins", 0) or 0),
                int(entry.get("draws", 0) or 0),
                float(entry.get("elo", 800.0) or 800.0),
                float(entry.get("ts", 0.0) or 0.0),
            )
        )

    def _op(conn: sqlite3.Connection) -> int:
        cursor = conn.executemany(
            "INSERT OR IGNORE INTO arena_games"
            " (run_id, regime, ckpt, opponent, games, wins, draws, elo, ts)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        return max(cursor.rowcount, 0)

    added = tx(_op)
    if added:
        log.info("arena backfill: %d new row(s) from %s", added, source)
    return added


def list_arena_games(limit: int) -> list[dict[str, Any]]:
    """Arena ledger rows (newest first) served by ``GET /api/games?kind=arena``."""
    rows: list[dict[str, Any]] = []

    def _op(conn: sqlite3.Connection) -> None:
        cursor = conn.execute(
            "SELECT run_id, regime, ckpt, opponent, games, wins, draws, elo, ts"
            " FROM arena_games ORDER BY ts DESC, id DESC LIMIT ?",
            (limit,),
        )
        rows.extend(dict(row) | {"kind": "arena"} for row in cursor)

    tx(_op)
    return rows


def elo_by_regime() -> dict[str, dict[str, float]]:
    """Per-regime checkpoint Elo map (latest recorded elo per checkpoint)."""

    def _op(conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for row in conn.execute(
            "SELECT regime, ckpt, elo, ts, id FROM arena_games ORDER BY ts, id"
        ):
            out.setdefault(row["regime"], {})[row["ckpt"]] = float(row["elo"])
        return out

    return tx(_op)
