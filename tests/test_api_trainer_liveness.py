"""Gate-visibility honesty: ``gate_in_progress`` from trainer PROCESS liveness.

Regression for the live defect measured on production (2026-11-21): the hint
required a fresh journal, but the trainer writes nothing for a whole arena gate
(40-80 min), so ``gate_in_progress`` was structurally false during every gate
and ``/watch`` showed IDLE while a title match ran.

These tests never touch real local processes: the scanner seam
(``server.trainer._cmdlines`` / ``_scan_processes``) and the monotonic clock
(``server.trainer._now``) are patched, and the cache is reset per test.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psutil
import pytest
from fastapi.testclient import TestClient

from server import arena, config, db, engine, journal, trainer
from server.api.train import _gate_hint
from server.main import create_app

# Fixed reference instant for the pure unit checks; the REST checks anchor on
# the real clock because the endpoint stamps `now` itself.
NOW = 1_900_000_000.0


@pytest.fixture(autouse=True)
def clean_trainer_cache() -> Iterator[None]:
    """No cached liveness answer leaks between tests."""
    trainer.reset_cache()
    yield
    trainer.reset_cache()


def _boot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, alive: bool) -> TestClient:
    """Tmp stack with the trainer probe stubbed to a fixed liveness answer."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "gate.db")
    monkeypatch.setattr(config, "ARENA_LEDGER_PATH", tmp_path / "arena.jsonl")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(config, "TRAIN_LOG_PATH", tmp_path / "train_log.jsonl")
    monkeypatch.setattr(trainer, "_scan_processes", lambda run_id: alive)
    trainer.reset_cache()
    arena.reset_cache()
    db.reset_for_tests()
    engine.reset_for_tests()
    return TestClient(create_app())


def _ledger(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def _journal(path: Path, *, age_seconds: float) -> None:
    """One journal event ``age_seconds`` before the real clock (the REST path
    stamps its own ``now``; the unit checks use :data:`NOW` directly)."""
    path.write_text(
        json.dumps({"kind": "iteration", "run_id": "second-light", "ts": time.time() - age_seconds})
        + "\n",
        encoding="utf-8",
    )


# --- the rule ----------------------------------------------------------------


def test_alive_with_journal_quiet_200s_is_a_gate():
    """The live defect: trainer alive, journal silent 200 s (mid-gate), ledger
    has a row -> the hint must fire."""
    assert _gate_hint(NOW, NOW - 3600.0, NOW - 200.0, alive=True) is True


def test_alive_while_self_play_streams_is_not_a_gate():
    """Journal fresh 10 s ago = the ~2.4 lines/s self-play stream: not a gate."""
    assert _gate_hint(NOW, NOW - 3600.0, NOW - 10.0, alive=True) is False


def test_dead_trainer_is_never_a_gate_even_after_an_hour_of_silence():
    assert _gate_hint(NOW, NOW - 3600.0, NOW - 3600.0, alive=False) is False


def test_no_arena_row_yet_is_never_a_gate():
    assert _gate_hint(NOW, None, NOW - 200.0, alive=True) is False


# --- the process probe -------------------------------------------------------


def test_trainer_alive_caches_the_scan(monkeypatch: pytest.MonkeyPatch):
    """Two calls inside the TTL enumerate once; an expiry rescans once."""
    calls: list[str | None] = []
    clock = {"t": 1_000.0}

    def fake_scan(run_id: str | None) -> bool:
        calls.append(run_id)
        return True

    monkeypatch.setattr(trainer, "_scan_processes", fake_scan)
    monkeypatch.setattr(trainer, "_now", lambda: clock["t"])

    assert trainer.trainer_alive() is True
    assert trainer.trainer_alive() is True
    assert calls == [None]  # second call served from cache

    clock["t"] += trainer.CACHE_TTL_SECONDS + 0.01
    assert trainer.trainer_alive() is True
    assert len(calls) == 2  # cache expired: scanned again


def test_scan_matches_scripts_train_py_on_either_slash_style(monkeypatch: pytest.MonkeyPatch):
    """Windows launches appear as ``scripts/train.py`` and ``scripts\\train.py``."""
    monkeypatch.setattr(trainer, "_cmdlines", list)
    assert trainer._scan_processes(None) is False

    monkeypatch.setattr(trainer, "_cmdlines", lambda: [
        '"e:/repo/.venv/scripts/python.exe" scripts/train.py --run-id abc',
        '"e:/repo/.venv/scripts/python.exe" -m uvicorn server.main:app',
    ])
    assert trainer._scan_processes(None) is True
    assert trainer._scan_processes("abc") is True  # run id present

    monkeypatch.setattr(trainer, "_cmdlines", lambda: [
        r'"e:\repo\.venv\Scripts\python.exe" scripts\train.py --run-id=other',
    ])
    assert trainer._scan_processes("other") is True


def test_scan_falls_back_to_plain_match_when_run_id_is_absent(monkeypatch: pytest.MonkeyPatch):
    """A trainer launched without ``--run-id`` generates its own id, so the
    journal id cannot appear in any cmdline: liveness must not regress."""
    monkeypatch.setattr(trainer, "_cmdlines", lambda: ['python scripts/train.py --profile night'])
    assert trainer._scan_processes("journal-generated-id") is True


def test_probe_never_raises_on_scan_failure(monkeypatch: pytest.MonkeyPatch):
    """A denied/exhausted process enumeration is False, never an exception."""
    def boom() -> list[str]:
        raise psutil.AccessDenied(pid=1)

    monkeypatch.setattr(trainer, "_cmdlines", boom)
    assert trainer.trainer_alive() is False


# --- the metrics feed --------------------------------------------------------


def test_metrics_snapshot_carries_trainer_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Additive key + the gate hint on the same facts; trainer-owned keys win."""
    (tmp_path / "metrics.json").write_text(json.dumps({"elo": 801.9}), encoding="utf-8")
    arena_ts = time.time() - 3600.0
    journal_ts = time.time() - 200.0
    _ledger(tmp_path / "arena.jsonl", [
        {"regime": "scratch", "checkpoint": "ck-a", "opponent": "none", "games": 0,
         "wins": 0, "draws": 0, "elo": 800.0, "ts": arena_ts},
    ])
    _journal(tmp_path / "train_log.jsonl", age_seconds=200.0)

    with _boot(tmp_path, monkeypatch, alive=True) as client:
        snap = client.get("/api/metrics").json()
        assert snap["trainer_alive"] is True
        assert snap["gate_in_progress"] is True  # alive + journal quiet 200 s
        assert snap["last_arena_ts"] == pytest.approx(arena_ts)
        assert snap["last_journal_ts"] == pytest.approx(journal_ts)

    trainer.reset_cache()
    with _boot(tmp_path, monkeypatch, alive=False) as client:
        snap = client.get("/api/metrics").json()
        assert snap["trainer_alive"] is False
        assert snap["gate_in_progress"] is False  # dead trainer: never a gate


def test_metrics_snapshot_keeps_trainer_owned_liveness_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A trainer-published ``trainer_alive`` is authoritative (setdefault)."""
    (tmp_path / "metrics.json").write_text(
        json.dumps({"elo": 801.9, "trainer_alive": False}), encoding="utf-8"
    )
    with _boot(tmp_path, monkeypatch, alive=True) as client:
        assert client.get("/api/metrics").json()["trainer_alive"] is False


def test_latest_run_id_reads_the_newest_run(tmp_path: Path):
    """The metrics feed scopes the process match with this run id."""
    path = tmp_path / "train_log.jsonl"
    assert journal.latest_run_id(path) is None  # absent
    path.write_text(
        json.dumps({"kind": "live", "game": 1, "run_id": "old", "ts": 10.0}) + "\n"
        + json.dumps({"kind": "iteration", "run_id": "second-light", "ts": 20.0}) + "\n"
        + '{"kind": "live", "game": 1, "ts"',  # partial write: skipped
        encoding="utf-8",
    )
    assert journal.latest_run_id(path) == "second-light"
