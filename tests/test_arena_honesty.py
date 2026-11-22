"""Honest arena ratings (audit F-5 + U9) + live mirror maintenance.

Covers the stale-leaderboard fix end to end:

- the v2 → v3 ``arena_games`` migration (nullable ``elo`` + additive
  ``glicko_r``/``glicko_rd``), its idempotency, and its rollback;
- ledger backfill honesty across the three live row shapes (pre-U9 numeric
  ``elo`` — what the RUNNING trainer still writes, U9 ``elo: null`` +
  glicko fields, garbage lines) and idempotency (run twice → identical DB);
- the rating reconcile that repairs rows mirrored by the old coercing
  importer (the live database's pre-fix state);
- additive ``/api/metrics`` passthrough of the U9 rating keys with and
  without them present;
- the periodic backfill loop actually importing rows appended while the
  server is up (the startup-only-import defect).

Every test runs against tmp copies — never the live ``data/gambit.db``.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import arena, backfill, config, db, engine
from server.main import create_app

# --- fixtures ---------------------------------------------------------------

# Pre-U9 (v2) shape of arena_games: elo NOT NULL DEFAULT 800.0, no glicko.
_LEGACY_ARENA_DDL = """
CREATE TABLE arena_games (
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
)
"""

# v1 shape of games: no strength column yet (migrate adds it).
_LEGACY_GAMES_DDL = """
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
)
"""


def _ledger(path: Path, entries: list[dict[str, Any] | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(e if isinstance(e, str) else json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


def _legacy_importer(entry: dict[str, Any]) -> tuple[Any, ...]:
    """The pre-fix backfill row tuple, coercion included (elo null → 800.0)."""
    return (
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


def _seed_legacy_db(path: Path, entries: list[dict[str, Any]]) -> None:
    """A pre-U9 database: v1 games + NOT NULL arena_games, old-importer rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_LEGACY_GAMES_DDL + ";" + _LEGACY_ARENA_DDL)
        conn.executemany(
            "INSERT INTO arena_games"
            " (run_id, regime, ckpt, opponent, games, wins, draws, elo, ts)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            [_legacy_importer(e) for e in entries],
        )
        conn.commit()
    finally:
        conn.close()


def _boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, legacy_rows: list[dict[str, Any]] | None = None
) -> TestClient:
    """Fresh tmp stack (DB + ledger + metrics), optionally pre-seeded legacy DB."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "arena.db")
    monkeypatch.setattr(config, "ARENA_LEDGER_PATH", tmp_path / "arena.jsonl")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
    if legacy_rows is not None:
        _seed_legacy_db(tmp_path / "arena.db", legacy_rows)
    arena.reset_cache()
    db.reset_for_tests()
    engine.reset_for_tests()
    return TestClient(create_app())


def _dump_arena() -> list[dict[str, Any]]:
    """Full mirror state (all columns, insertion order) for equality checks."""
    return list(db.tx(lambda conn: [dict(r) for r in conn.execute("SELECT * FROM arena_games ORDER BY id")]))


def _arena_columns() -> dict[str, sqlite3.Row]:
    rows = db.tx(lambda conn: conn.execute("PRAGMA table_info(arena_games)").fetchall())
    return {str(r["name"]): r for r in rows}


# --- migration v2 → v3 -------------------------------------------------------


def test_migration_makes_elo_nullable_and_adds_glicko(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    legacy = [
        {"run_id": "r1", "regime": "scratch", "checkpoint": "ck-a", "opponent": "none",
         "games": 0, "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
        {"run_id": "r1", "regime": "scratch", "checkpoint": "ck-b", "opponent": "ck-a",
         "games": 2, "wins": 1, "draws": 1, "elo": 812.5, "ts": 200.0},
    ]
    with _boot(tmp_path, monkeypatch, legacy_rows=legacy):
        columns = _arena_columns()
        assert columns["elo"]["notnull"] == 0  # honest nulls storable now
        assert {"glicko_r", "glicko_rd"} <= set(columns)
        rows = _dump_arena()
        assert [(r["ckpt"], r["elo"]) for r in rows] == [("ck-a", 800.0), ("ck-b", 812.5)]
        assert [r["id"] for r in rows] == [1, 2]  # ids preserved through the rebuild
        # dedupe index survived the rebuild under its canonical name
        with pytest.raises(sqlite3.IntegrityError):
            db.tx(lambda conn: conn.execute(
                "INSERT INTO arena_games (run_id, regime, ckpt, opponent, games, ts)"
                " VALUES ('x', 'scratch', 'ck-a', 'none', 0, 100.0)"
            ))


def test_migration_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with _boot(tmp_path, monkeypatch, legacy_rows=[
        {"regime": "scratch", "checkpoint": "ck-a", "opponent": "none", "games": 0,
         "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
    ]):
        before = _dump_arena()
        db.migrate()  # second run: no-op, no dup tables, data intact
        db.migrate()
        assert _dump_arena() == before
        assert _arena_columns()["elo"]["notnull"] == 0
        tables = {r[0] for r in db.tx(lambda conn: conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"))}
        assert "arena_games_v3" not in tables and "arena_games_v2down" not in tables


def test_migration_down_up_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ledger = [
        {"regime": "scratch", "checkpoint": "ck-old", "opponent": "none", "games": 0,
         "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
        {"regime": "scratch", "checkpoint": "ck-held", "opponent": "ck-old", "games": 4,
         "wins": 0, "draws": 4, "elo": None, "promoted": False,
         "glicko_r": 1461.3, "glicko_rd": 181.2, "ts": 200.0},
    ]
    _ledger(tmp_path / "arena.jsonl", ledger)
    with _boot(tmp_path, monkeypatch):  # lifespan already migrated + backfilled
        rows = {r["ckpt"]: r for r in _dump_arena()}
        assert rows["ck-held"]["elo"] is None  # honest null stored
        db.downgrade_arena()
        columns = _arena_columns()
        assert columns["elo"]["notnull"] == 1  # rollback restores the old lie
        assert "glicko_r" not in columns
        down_rows = {r["ckpt"]: r for r in _dump_arena()}
        assert down_rows["ck-held"]["elo"] == 800.0  # NULLs coerced, old behavior
        assert len(down_rows) == 2
        db.migrate()  # and back up again
        assert _arena_columns()["elo"]["notnull"] == 0
        assert "glicko_r" in _arena_columns()


# --- backfill honesty across row shapes --------------------------------------


def test_backfill_old_format_rows_stay_floats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Old-only ledger (what the RUNNING trainer still writes): no nulls to
    store, no glicko keys — every elo a float, glicko columns NULL."""
    _ledger(tmp_path / "arena.jsonl", [
        {"run_id": "old-run", "regime": "scratch", "checkpoint": "ck-a", "opponent": "none",
         "games": 0, "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
        {"run_id": "old-run", "regime": "scratch", "checkpoint": "ck-a", "opponent": "none",
         "games": 2, "wins": 1, "draws": 1, "elo": 808.0, "ts": 150.0, "kind": "arena"},
    ])
    with _boot(tmp_path, monkeypatch):
        rows = _dump_arena()
        assert [r["elo"] for r in rows] == [800.0, 808.0]
        assert all(r["glicko_r"] is None and r["glicko_rd"] is None for r in rows)


def test_backfill_mixed_formats_preserves_nulls_and_glicko(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One file, all three shapes: old numeric rows, U9 null rows, U9 glicko
    rows — plus a corrupt line and a non-numeric field, skipped not fatal."""
    _ledger(tmp_path / "arena.jsonl", [
        {"run_id": "old-run", "regime": "scratch", "checkpoint": "ck-old", "opponent": "none",
         "games": 0, "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
        "not json at all",
        {"run_id": "second-light", "regime": "scratch", "checkpoint": "ck-new",
         "opponent": "ck-old", "games": 16, "wins": 8, "draws": 8, "elo": 792.0,
         "promoted": True, "glicko_r": 1516.7, "glicko_rd": 191.9, "ts": 200.0},
        {"run_id": "second-light", "regime": "scratch", "checkpoint": "ck-held",
         "opponent": "ck-new", "games": 16, "wins": 0, "draws": 16, "elo": None,
         "promoted": False, "glicko_r": 1385.4, "glicko_rd": 174.6, "ts": 300.0},
        {"run_id": "second-light", "regime": "scratch", "checkpoint": "ck-bad",
         "opponent": "ck-new", "games": 16, "wins": 0, "draws": 16, "elo": "high",
         "ts": 400.0},
    ])
    with _boot(tmp_path, monkeypatch):
        rows = {r["ckpt"]: r for r in _dump_arena()}
        assert set(rows) == {"ck-old", "ck-new", "ck-held"}  # garbage + bad skipped
        assert rows["ck-old"]["elo"] == 800.0
        assert rows["ck-new"]["elo"] == 792.0
        assert rows["ck-held"]["elo"] is None  # F-5: no rating was claimed
        assert rows["ck-new"]["glicko_r"] == pytest.approx(1516.7)
        assert rows["ck-held"]["glicko_rd"] == pytest.approx(174.6)
        assert rows["ck-old"]["glicko_r"] is None  # pre-U9 row: no glicko keys

        # Served honestly end to end (additive glicko keys, null elo).
        games = db.list_arena_games(50)
        held = next(g for g in games if g["ckpt"] == "ck-held")
        assert held["elo"] is None and held["glicko_r"] == pytest.approx(1385.4)


def test_backfill_run_twice_identical_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Idempotency contract: second run inserts AND reconciles nothing."""
    _ledger(tmp_path / "arena.jsonl", [
        {"run_id": "old-run", "regime": "scratch", "checkpoint": "ck-old", "opponent": "none",
         "games": 0, "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
        {"run_id": "second-light", "regime": "scratch", "checkpoint": "ck-held",
         "opponent": "ck-old", "games": 16, "wins": 0, "draws": 16, "elo": None,
         "glicko_r": 1385.4, "glicko_rd": 174.6, "ts": 300.0},
    ])
    with _boot(tmp_path, monkeypatch):  # run 1 = the lifespan's startup import
        first = _dump_arena()
        assert len(first) == 2
        assert db.backfill_arena() == 0  # run 2: nothing new...
        assert db.backfill_arena() == 0  # run 3: still nothing (no rewrite)
        assert _dump_arena() == first    # state byte-identical across runs


def test_backfill_repairs_coerced_800_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The live-DB repair path: rows mirrored by the OLD importer carry the
    coerced 800.0; after migrate + backfill the mirror matches the file's
    truth (null stays null, glicko stamped) — no restart beyond the fix."""
    file_rows = [
        {"run_id": "old-run", "regime": "scratch", "checkpoint": "ck-old", "opponent": "none",
         "games": 0, "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
        {"run_id": "second-light", "regime": "scratch", "checkpoint": "ck-held",
         "opponent": "ck-old", "games": 16, "wins": 0, "draws": 16, "elo": None,
         "glicko_r": 1385.4, "glicko_rd": 174.6, "ts": 300.0},
    ]
    _ledger(tmp_path / "arena.jsonl", file_rows)
    with _boot(tmp_path, monkeypatch, legacy_rows=file_rows):  # legacy import coerced null→800
        rows = {r["ckpt"]: r for r in _dump_arena()}
        assert rows["ck-held"]["elo"] is None       # repaired by the reconcile
        assert rows["ck-held"]["glicko_r"] == pytest.approx(1385.4)  # stamped
        assert rows["ck-old"]["elo"] == 800.0       # honest claim untouched


def test_regime_map_attribution_and_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Regime attribution: a checkpoint lives where its LATEST candidate row
    lives (opponent rows don't re-home it); values come from the global
    ledger replay exactly like the trainer's own recompute."""
    _ledger(tmp_path / "arena.jsonl", [
        {"regime": "smoke", "checkpoint": "ck-x", "opponent": "none", "games": 0,
         "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
        {"regime": "scratch", "checkpoint": "ck-x", "opponent": "none", "games": 0,
         "wins": 0, "draws": 0, "elo": 800.0, "ts": 200.0},
        {"regime": "smoke", "checkpoint": "ck-y", "opponent": "ck-x", "games": 2,
         "wins": 2, "draws": 0, "elo": 816.0, "ts": 300.0},
    ])
    with _boot(tmp_path, monkeypatch) as client:
        # Replay map (served): ck-x is re-homed to its latest candidate regime
        # (scratch) and carries the REPLAYED value 784.0 (it lost the ck-y
        # match as opponent); ck-y stays in smoke at 816.0.
        assert client.get("/api/metrics").json()["elo_by_regime"] == {
            "scratch": {"ck-x": 784.0},
            "smoke": {"ck-y": 816.0},
        }
        # Mirror map (db utility): the old last-claim-per-(regime, ckpt) view,
        # where ck-x legitimately appears under every regime it candidated in.
        assert db.elo_by_regime() == {
            "scratch": {"ck-x": 800.0},
            "smoke": {"ck-x": 800.0, "ck-y": 816.0},
        }


# --- /api/metrics additive rating passthrough --------------------------------


def test_metrics_without_glicko_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Absence tolerated: no champion_glicko/glicko keys invented; headline
    elo passes through untouched; elo_by_regime is the replay."""
    (tmp_path / "metrics.json").write_text(json.dumps({
        "elo": 793.3747499688052, "loss": 3.3, "win_rate": 0.5, "games_played": 288,
        "champion": "ck-new", "regime": "scratch",
        "checkpoints": {"ck-new": 806.6},
    }), encoding="utf-8")
    _ledger(tmp_path / "arena.jsonl", [
        {"regime": "scratch", "checkpoint": "ck-new", "opponent": "none", "games": 0,
         "wins": 0, "draws": 0, "elo": None, "ts": 100.0},
    ])
    with _boot(tmp_path, monkeypatch) as client:
        snap = client.get("/api/metrics").json()
        assert snap["elo"] == pytest.approx(793.3747499688052)  # canonical headline
        assert "champion_glicko" not in snap and "glicko" not in snap
        assert snap["elo_by_regime"] == {"scratch": {"ck-new": 800.0}}


def test_metrics_with_glicko_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """U9 additive passthrough: when the trainer starts writing champion_glicko
    + the glicko map, /api/metrics carries them verbatim (next launch)."""
    (tmp_path / "metrics.json").write_text(json.dumps({
        "elo": 793.3747499688052, "loss": 3.3, "win_rate": 0.5, "games_played": 288,
        "champion_glicko": {"r": 1524.9, "rd": 88.1},
        "glicko": {"ck-new": {"r": 1516.7, "rd": 191.9}},
        "checkpoints": {"ck-new": 806.6},
    }), encoding="utf-8")
    _ledger(tmp_path / "arena.jsonl", [
        {"regime": "scratch", "checkpoint": "ck-new", "opponent": "none", "games": 0,
         "wins": 0, "draws": 0, "elo": None, "ts": 100.0},
    ])
    with _boot(tmp_path, monkeypatch) as client:
        snap = client.get("/api/metrics").json()
        assert snap["champion_glicko"] == {"r": 1524.9, "rd": 88.1}
        assert snap["glicko"] == {"ck-new": {"r": 1516.7, "rd": 191.9}}
        assert snap["checkpoints"] == {"ck-new": 806.6}  # secondary diagnostics
        assert snap["elo_by_regime"] == {"scratch": {"ck-new": 800.0}}


# --- gate-visibility keys (addendum) ------------------------------------------


def test_gate_hint_truth_table():
    """Unit: hint fires only for an actively-journaling trainer with a quiet
    ledger — never for an idle/gone trainer, never with no ledger."""
    from server.api.train import _gate_hint

    now = 1_000_000.0
    # Trainer gone (journal stale): never a gate, however quiet the ledger.
    assert _gate_hint(now, now - 3600.0, now - 300.0) is False
    # No journal at all: same.
    assert _gate_hint(now, now - 3600.0, None) is False
    # Trainer alive but no ledger yet: no claim (nothing to be quiet about).
    assert _gate_hint(now, None, now - 5.0) is False
    # Trainer alive + fresh arena result: training phase, not a gate.
    assert _gate_hint(now, now - 60.0, now - 5.0) is False
    # Trainer alive + quiet ledger: the mid-gate case.
    assert _gate_hint(now, now - 3600.0, now - 5.0) is True


def test_metrics_gate_visibility_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Additive gate keys: facts always served; the hint approximates; a
    trainer-owned gate_in_progress (the clean future fix) wins over ours."""
    (tmp_path / "metrics.json").write_text(json.dumps(
        {"elo": 793.0, "loss": 3.3, "win_rate": 0.5, "games_played": 288}),
        encoding="utf-8")
    _ledger(tmp_path / "arena.jsonl", [
        {"regime": "scratch", "checkpoint": "ck-a", "opponent": "none", "games": 0,
         "wins": 0, "draws": 0, "elo": 800.0, "ts": 1234.5},
    ])
    with _boot(tmp_path, monkeypatch) as client:
        snap = client.get("/api/metrics").json()
        assert snap["last_arena_ts"] == pytest.approx(1234.5)
        assert isinstance(snap["gate_in_progress"], bool)
        assert isinstance(snap["last_journal_ts"], (float, type(None)))
    # Trainer-owned override: when the pipeline starts writing the flag
    # itself, the server must not clobber it.
    (tmp_path / "metrics.json").write_text(json.dumps(
        {"elo": 793.0, "gate_in_progress": True}), encoding="utf-8")
    (tmp_path / "train_log.jsonl").write_text("", encoding="utf-8")
    with _boot(tmp_path, monkeypatch) as client:
        snap = client.get("/api/metrics").json()
        assert snap["gate_in_progress"] is True  # trainer's word, kept


def test_journal_latest_event_ts(tmp_path: Path):
    """Bounded backward scan: newest ts found through the live-frame noise;
    honest None for a missing/empty/ts-less journal."""
    from server import journal

    path = tmp_path / "train_log.jsonl"
    assert journal.latest_event_ts(path) is None  # absent
    path.write_text("", encoding="utf-8")
    assert journal.latest_event_ts(path) is None  # empty
    lines = "\n".join(
        json.dumps({"kind": "live", "game": 1, "ply": p, "ts": 1000.0 + p})
        for p in range(500)
    )
    path.write_text(lines + "\n", encoding="utf-8")
    assert journal.latest_event_ts(path) == pytest.approx(1499.0)
    # Trailing partial line (writer mid-write) is skipped, not fatal.
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "live", "game": 1, "ply": 500, "ts": 15')
    assert journal.latest_event_ts(path) == pytest.approx(1499.0)
    # ts-less lines: walk back to the newest line that carries one.
    path.write_text('{"kind": "log", "message": "x"}\n{"kind": "live", "ts": 42.0}\n',
                    encoding="utf-8")
    assert journal.latest_event_ts(path) == pytest.approx(42.0)
    path.write_text('{"kind": "log", "message": "x"}\n', encoding="utf-8")
    assert journal.latest_event_ts(path) is None


def test_metrics_survives_missing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """No ledger + no metrics: pinned keys null/default, empty regime map —
    never a 500, never a coerced 800."""
    with _boot(tmp_path, monkeypatch) as client:
        snap = client.get("/api/metrics").json()
        assert snap["elo"] is None and snap["elo_by_regime"] == {}
        assert snap["games_played"] == 0


# --- periodic backfill loop ---------------------------------------------------


def test_periodic_backfill_imports_rows_appended_while_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The defect, reproduced live: a row appended AFTER boot reaches the
    mirror within one tick of the periodic loop (interval patched small)."""
    monkeypatch.setattr(backfill, "BACKFILL_INTERVAL_SECONDS", 0.05)
    _ledger(tmp_path / "arena.jsonl", [
        {"regime": "scratch", "checkpoint": "ck-a", "opponent": "none", "games": 0,
         "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
    ])
    with _boot(tmp_path, monkeypatch) as client:
        assert client.get("/api/metrics").json()["elo_by_regime"] == {"scratch": {"ck-a": 800.0}}
        # The trainer appends mid-flight (old format, as the running process does).
        _ledger(tmp_path / "arena.jsonl", [
            {"regime": "scratch", "checkpoint": "ck-a", "opponent": "none", "games": 0,
             "wins": 0, "draws": 0, "elo": 800.0, "ts": 100.0},
            {"regime": "scratch", "checkpoint": "ck-late", "opponent": "ck-a", "games": 4,
             "wins": 4, "draws": 0, "elo": None, "ts": 200.0},
        ])
        deadline = time.time() + 5.0
        while time.time() < deadline:
            ckpts = {g["ckpt"] for g in db.list_arena_games(100)}
            if "ck-late" in ckpts:
                break
            time.sleep(0.05)
        else:
            pytest.fail("periodic backfill did not import the appended row in time")
        late = next(g for g in db.list_arena_games(100) if g["ckpt"] == "ck-late")
        assert late["elo"] is None  # honest through the periodic path too
