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
  -- Nullable since the v3 migration (audit F-5): NULL = the trainer never
  -- claimed a rating for this row (non-promoted U9 candidate). The pre-U9
  -- schema coerced those to the 800.0 base, asserting unclaimed ratings.
  elo REAL,
  -- U9 additive fields: the candidate's own Glicko-2 record after the gate
  -- (NULL on pre-U9 rows, which carry no glicko keys at all).
  glicko_r REAL,
  glicko_rd REAL,
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
# Schema migration (run from the app lifespan — never from connect(), so
# merely importing this package, e.g. during test collection against the
# live DB, cannot mutate anything).
#
# v1 → v2 is a pure ADD COLUMN; v2 → v3 rebuilds ``arena_games`` in place.
# The rebuild is still startup-only, transactional (DDL rolls back with the
# tx), row- and id-preserving, and idempotent (the PRAGMA check makes a
# second run a no-op).
# ---------------------------------------------------------------------------

# v3 shape of arena_games (mirrors _SCHEMA; kept as a template so the rebuild
# and the fresh-database path cannot drift apart).
_ARENA_V3_DDL = """
CREATE TABLE {name} (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL DEFAULT '',
  regime TEXT NOT NULL DEFAULT '',
  ckpt TEXT NOT NULL,
  opponent TEXT NOT NULL DEFAULT '',
  games INTEGER NOT NULL DEFAULT 0,
  wins INTEGER NOT NULL DEFAULT 0,
  draws INTEGER NOT NULL DEFAULT 0,
  elo REAL,
  glicko_r REAL,
  glicko_rd REAL,
  ts REAL NOT NULL
)
"""

_ARENA_RATING_COLUMNS = ("elo", "glicko_r", "glicko_rd")


def migrate() -> None:
    """Apply schema migrations to a pre-existing database (idempotent).

    v1 → v2: ``games.strength`` records the opponent's skill preset so sessions
    rehydrate faithfully after a restart. Existing rows backfill with the
    default ``'curious'``; fresh databases already create the column.

    v2 → v3 (audit F-5 + U9): ``arena_games.elo`` becomes nullable — NULL
    means the trainer never claimed a rating for that row (non-promoted
    candidate) — and additive ``glicko_r``/``glicko_rd`` columns store the
    candidate's Glicko-2 record. SQLite cannot drop a NOT NULL constraint in
    place, so the table is rebuilt (copy with ids → drop → rename → reindex);
    values are preserved verbatim, including previously coerced 800.0s, which
    the next :func:`backfill_arena` reconcile re-stamps from the ledger file.
    """

    def _op(conn: sqlite3.Connection) -> None:
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(games)")}
        if "strength" not in columns:
            conn.execute("ALTER TABLE games ADD COLUMN strength TEXT NOT NULL DEFAULT 'curious'")
            log.info("db migration: games.strength column added (existing rows -> 'curious')")
        _migrate_arena_nullability(conn)

    tx(_op)


def _migrate_arena_nullability(conn: sqlite3.Connection) -> None:
    """v2 → v3 rebuild of ``arena_games`` when its schema predates U9/F-5.

    No-op when the table already has the v3 shape (nullable ``elo`` + glicko
    columns), which covers fresh databases and every run after the first.
    """
    info = {str(row["name"]): row for row in conn.execute("PRAGMA table_info(arena_games)")}
    if not info:
        return  # no table (connect() always creates it first; defensive only)
    elo_nullable = "elo" in info and not info["elo"]["notnull"]
    has_glicko = "glicko_r" in info and "glicko_rd" in info
    if elo_nullable and has_glicko:
        return
    conn.execute(_ARENA_V3_DDL.format(name="arena_games_v3"))
    carried = "id, run_id, regime, ckpt, opponent, games, wins, draws, elo, ts"
    if has_glicko:  # legacy rebuild never hits this; keeps the copy total anyway
        carried = "id, run_id, regime, ckpt, opponent, games, wins, draws, elo, glicko_r, glicko_rd, ts"
    conn.execute(f"INSERT INTO arena_games_v3 ({carried}) SELECT {carried} FROM arena_games")
    conn.execute("DROP TABLE arena_games")
    conn.execute("ALTER TABLE arena_games_v3 RENAME TO arena_games")
    # DROP TABLE took the old dedupe index with it; restore it under its
    # canonical name (connect()'s IF NOT EXISTS then stays a no-op).
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_arena_games_dedupe ON arena_games(ckpt, ts)")
    log.info(
        "db migration: arena_games rebuilt with nullable elo + glicko_r/glicko_rd"
        " (%d row(s) preserved)", conn.execute("SELECT COUNT(*) FROM arena_games").fetchone()[0],
    )


def downgrade_arena() -> None:
    """Rollback path for the v2 → v3 arena migration (ops runbook / tests).

    Rebuilds the pre-U9 schema (``elo REAL NOT NULL DEFAULT 800.0``, no glicko
    columns), coercing NULL ratings back to the 800.0 base — the old, dishonest
    behavior this migration removed. Rows, ids and the dedupe index survive.
    """
    old_ddl = (
        "CREATE TABLE {name} ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " run_id TEXT NOT NULL DEFAULT '',"
        " regime TEXT NOT NULL DEFAULT '',"
        " ckpt TEXT NOT NULL,"
        " opponent TEXT NOT NULL DEFAULT '',"
        " games INTEGER NOT NULL DEFAULT 0,"
        " wins INTEGER NOT NULL DEFAULT 0,"
        " draws INTEGER NOT NULL DEFAULT 0,"
        " elo REAL NOT NULL DEFAULT 800.0,"
        " ts REAL NOT NULL)"
    )

    def _op(conn: sqlite3.Connection) -> None:
        info = {str(row["name"]) for row in conn.execute("PRAGMA table_info(arena_games)")}
        if not info:
            return
        conn.execute(old_ddl.format(name="arena_games_v2down"))
        conn.execute(
            "INSERT INTO arena_games_v2down (id, run_id, regime, ckpt, opponent,"
            " games, wins, draws, elo, ts)"
            " SELECT id, run_id, regime, ckpt, opponent, games, wins, draws,"
            " COALESCE(elo, 800.0), ts FROM arena_games"
        )
        conn.execute("DROP TABLE arena_games")
        conn.execute("ALTER TABLE arena_games_v2down RENAME TO arena_games")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_arena_games_dedupe ON arena_games(ckpt, ts)"
        )
        log.info("db downgrade: arena_games restored to NOT NULL elo (NULLs -> 800.0 base)")

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
    """Idempotent import of the training JSONL ledger into ``arena_games``.

    Safe to run at every startup AND on the periodic backfill tick while the
    server is up (the trainer appends ledger rows continuously; a
    startup-only import was the stale-leaderboard defect). Two halves:

    1. **Insert** every readable line with ``INSERT OR IGNORE`` against the
       UNIQUE ``(ckpt, ts)`` index — a natural row key, so re-running after a
       restart (double boot) or on the periodic tick inserts nothing twice.
    2. **Reconcile ratings**: the ledger file is the source of truth, so
       drifted rating fields (``elo``, ``glicko_r``, ``glicko_rd``) are
       re-stamped from the file. This repairs rows mirrored by the pre-U9
       importer, which coerced honest ``elo: null`` rows to the 800.0 base
       (audit F-5). All other columns are append-only provenance the trainer
       never rewrites and cannot drift. The null-safe ``IS NOT`` guard makes
       the update a no-op on already-honest rows.

    Row-shape tolerance (the live file mixes formats): pre-U9 rows (numeric
    ``elo``, no glicko keys — what the RUNNING trainer still writes), U9 rows
    (``elo: null`` for non-promoted candidates + additive ``glicko_r`` /
    ``glicko_rd``), corrupt/malformed lines (skipped, never fatal). Returns
    the number of rows actually added.
    """
    source = path or config.ARENA_LEDGER_PATH
    if not source.is_file():
        return 0
    rows: list[tuple[str, str, str, str, int, int, int, float | None, float | None, float | None, float]] = []
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
        try:
            # Honest parse (audit F-5): missing or null `elo` stays None — the
            # trainer claims no rating for non-promoted candidates. The old
            # `float(entry.get("elo", 800.0) or 800.0)` coerced both to base.
            raw_elo = entry.get("elo")
            raw_r = entry.get("glicko_r")
            raw_rd = entry.get("glicko_rd")
            rows.append(
                (
                    str(entry.get("run_id", "")),
                    str(entry.get("regime", "")),
                    str(entry.get("checkpoint", "")),
                    str(entry.get("opponent", "")),
                    int(entry.get("games", 0) or 0),
                    int(entry.get("wins", 0) or 0),
                    int(entry.get("draws", 0) or 0),
                    None if raw_elo is None else float(raw_elo),
                    None if raw_r is None else float(raw_r),
                    None if raw_rd is None else float(raw_rd),
                    float(entry.get("ts", 0.0) or 0.0),
                )
            )
        except (TypeError, ValueError):
            log.warning("arena backfill: skipping non-numeric field in %s", source)
            continue

    def _op(conn: sqlite3.Connection) -> int:
        inserted = conn.executemany(
            "INSERT OR IGNORE INTO arena_games"
            " (run_id, regime, ckpt, opponent, games, wins, draws, elo, glicko_r, glicko_rd, ts)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        added = max(inserted.rowcount, 0)
        # Rating reconcile: re-stamp drifted rating fields from the file
        # (tuple indices: ckpt=2, elo=7, glicko_r=8, glicko_rd=9, ts=10).
        reconciled = conn.executemany(
            "UPDATE arena_games SET elo = ?, glicko_r = ?, glicko_rd = ?"
            " WHERE ckpt = ? AND ts = ?"
            " AND (elo IS NOT ? OR glicko_r IS NOT ? OR glicko_rd IS NOT ?)",
            [(row[7], row[8], row[9], row[2], row[10], row[7], row[8], row[9]) for row in rows],
        )
        drifted = max(reconciled.rowcount, 0)
        if drifted:
            log.info("arena backfill: reconciled %d drifted rating row(s) from %s", drifted, source)
        return added

    added = tx(_op)
    if added:
        log.info("arena backfill: %d new row(s) from %s", added, source)
    return added


def list_arena_games(limit: int) -> list[dict[str, Any]]:
    """Arena ledger rows (newest first) served by ``GET /api/games?kind=arena``.

    ``elo`` is the trainer's own rating claim (None when it claimed none);
    ``glicko_r``/``glicko_rd`` are the U9 additive Glicko fields (None on
    pre-U9 rows).
    """
    rows: list[dict[str, Any]] = []

    def _op(conn: sqlite3.Connection) -> None:
        cursor = conn.execute(
            "SELECT run_id, regime, ckpt, opponent, games, wins, draws, elo,"
            " glicko_r, glicko_rd, ts"
            " FROM arena_games ORDER BY ts DESC, id DESC LIMIT ?",
            (limit,),
        )
        rows.extend(dict(row) | {"kind": "arena"} for row in cursor)

    tx(_op)
    return rows


def elo_by_regime() -> dict[str, dict[str, float | None]]:
    """Per-regime checkpoint rating map from the mirrored ledger rows.

    Latest recorded rating claim per checkpoint (rows replayed oldest →
    newest, last write wins). ``None`` = the trainer never claimed a rating
    for that checkpoint's latest row (U9 non-promoted candidate) — served
    as null, never coerced to base. Note: this is the *recorded-claims*
    view of the mirror; ``/api/metrics`` derives ``elo_by_regime`` from the
    ledger replay instead (see :mod:`server.arena` for that choice).
    """

    def _op(conn: sqlite3.Connection) -> dict[str, dict[str, float | None]]:
        out: dict[str, dict[str, float | None]] = {}
        for row in conn.execute(
            "SELECT regime, ckpt, elo, ts, id FROM arena_games ORDER BY ts, id"
        ):
            rating: float | None = None if row["elo"] is None else float(row["elo"])
            out.setdefault(row["regime"], {})[row["ckpt"]] = rating
        return out

    return tx(_op)
