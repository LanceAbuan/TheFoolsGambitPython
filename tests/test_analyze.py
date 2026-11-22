"""Position analysis endpoint (POST /api/analyze): shape, math, degradation, cache.

The Fool tests inject a deterministic fake champion/evaluator into
``server.model_server`` — the real CUDA champion is never loaded here because
the training run owns the GPU. Stockfish tests use the real binary when one
resolves and skip otherwise.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import config, db, engine, model_server, stockfish
from server.api import analyze
from server.main import create_app

START_FEN = chess.STARTING_FEN
MATE_WHITE_TO_MOVE = "6k1/8/8/8/8/8/5PPP/r5K1 w - - 0 1"  # White is checkmated
STALEMATE_BLACK_TO_MOVE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"

_SF_BIN = stockfish.resolve_binary()
requires_binary = pytest.mark.skipif(_SF_BIN is None, reason="no stockfish binary in env/tools/PATH")


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """App with an isolated DB + empty runs dir (never loads the real champion)."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "analyze.db")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    db.reset_for_tests()
    engine.reset_for_tests()
    stockfish.reset_for_tests()
    model_server.reset_for_tests()
    analyze.reset_for_tests()
    with TestClient(create_app()) as c:
        yield c
    analyze.reset_for_tests()
    model_server.reset_for_tests()


class _FakeEvaluator:
    """Deterministic evaluator: every position is +0.5 for the side to move.

    In chess terms that is incoherent, but it makes every MCTS value (and every
    child q) exactly signed: the mover after any White move sees +0.5.
    """

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        self.calls += len(boards)
        logits = np.zeros((len(boards), 4672), dtype=np.float32)
        values = np.array(
            [0.5 if board.turn == chess.WHITE else -0.5 for board in boards], dtype=np.float32
        )
        return logits, values


def _install_fake_champion(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tag: str = "ckpt-000128-0064",
    elo: float = 812.0,
) -> _FakeEvaluator:
    """Patch the model_server singleton with a fake champion; return its evaluator."""
    evaluator = _FakeEvaluator()
    champion = SimpleNamespace(
        tag=tag,
        elo=elo,
        evaluator=evaluator,
        regime="scratch",
        mtime=0.0,
        model=None,
        device=None,
    )
    monkeypatch.setattr(model_server, "current", lambda: champion)
    return evaluator


def _stockfish_payload(depth: int, multipv: int) -> dict[str, Any]:
    """A minimal successful Stockfish report (for engine-call counting tests)."""
    return {
        "available": True,
        "label": "Stockfish",
        "depth": depth,
        "score_cp": 24,
        "mate": None,
        "expected_white": 0.53,
        "best_move": "e2e4",
        "best_san": "e4",
        "pv": ["e4", "e5", "Nf3"],
        "top": [
            {"uci": "e2e4", "san": "e4", "score_cp": 24, "mate": None, "pv": ["e4", "e5"]},
        ],
        "elapsed_ms": 1,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_invalid_fen_422(client: TestClient):
    r = client.post("/api/analyze", json={"fen": "this is not a fen"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, str)
    assert detail.startswith("invalid fen:")


def test_missing_fen_422(client: TestClient):
    r = client.post("/api/analyze", json={"engines": ["stockfish"]})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Real Stockfish
# ---------------------------------------------------------------------------


@requires_binary
def test_stockfish_only_real_analysis(client: TestClient):
    r = client.post(
        "/api/analyze",
        json={"fen": START_FEN, "engines": ["stockfish"], "depth": 6, "multipv": 3},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["turn"] == "white"
    assert body["terminal"] is None
    assert set(body["engines"]) == {"stockfish"}  # unrequested engine is omitted, not null
    sf = body["engines"]["stockfish"]
    assert sf["available"] is True
    assert sf["error"] is None
    assert sf["depth"] == 6
    assert sf["elapsed_ms"] >= 0

    board = chess.Board(START_FEN)
    best = chess.Move.from_uci(sf["best_move"])
    assert best in board.legal_moves
    assert sf["best_san"] == board.san(best)
    assert sf["pv"][0] == sf["best_san"]
    assert 0 < len(sf["top"]) <= 3
    for entry in sf["top"]:
        probe = chess.Board(START_FEN)
        assert chess.Move.from_uci(entry["uci"]) in probe.legal_moves
        for san in entry["pv"]:  # every SAN in a PV must replay legally
            probe.push(probe.parse_san(san))

    # White-POV coherence: expected_white is the logistic of score_cp.
    if sf["mate"] is None:
        cp = sf["score_cp"]
        assert isinstance(cp, int)
        assert sf["expected_white"] == pytest.approx(1.0 / (1.0 + 10.0 ** (-cp / 400.0)), abs=1e-4)


# ---------------------------------------------------------------------------
# Fool with a fake champion
# ---------------------------------------------------------------------------


def test_fool_shape_math_and_q(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    evaluator = _install_fake_champion(monkeypatch)
    r = client.post(
        "/api/analyze",
        json={"fen": START_FEN, "engines": ["fool"], "visits": 8, "multipv": 3},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["engines"]) == {"fool"}
    fool = body["engines"]["fool"]
    assert fool["available"] is True
    assert fool["error"] is None
    assert fool["label"] == analyze.FOOL_LABEL
    assert fool["checkpoint"] == "ckpt-000128-0064"
    assert fool["elo"] == pytest.approx(812.0)
    assert fool["visits"] == 8
    assert fool["mate"] is None
    assert evaluator.calls >= 8  # root + sims + child q evaluations

    board = chess.Board(START_FEN)
    best = chess.Move.from_uci(fool["best_move"])
    assert best in board.legal_moves
    assert fool["best_san"] == board.san(best)
    assert fool["pv"] == [fool["best_san"]]  # fool PV = best move only

    # expected_white math: root_value is side-to-move POV, White to move here.
    expected = (fool["root_value"] + 1.0) / 2.0
    assert fool["expected_white"] == pytest.approx(expected, abs=1e-3)
    assert 0.0 <= fool["expected_white"] <= 1.0
    assert fool["score_cp"] == analyze._cp_from_expected(fool["expected_white"])
    assert fool["root_value"] > 0  # fake evaluator: White is always better at the root

    # Top moves: <= multipv, ordered by visits, q is MOVER POV (positive when
    # the side that made the move is better). Child boards are Black to move in
    # the fake evaluator (-0.5), so q = -(-0.5) = +0.5.
    top = fool["top"]
    assert 0 < len(top) <= 3
    visit_counts = [entry["visits"] for entry in top]
    assert visit_counts == sorted(visit_counts, reverse=True)
    for entry in top:
        assert isinstance(entry["visits"], int)
        assert entry["q"] == pytest.approx(0.5)
        assert entry["q"] > 0


def test_fool_no_champion_degrades(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(model_server, "current", lambda: None)
    r = client.post("/api/analyze", json={"fen": START_FEN, "engines": ["fool"]})
    assert r.status_code == 200
    fool = r.json()["engines"]["fool"]
    assert fool["available"] is False
    assert fool["error"] == "no trained champion yet"
    assert fool["best_move"] is None


# ---------------------------------------------------------------------------
# Missing Stockfish
# ---------------------------------------------------------------------------


def test_missing_stockfish_binary_degrades(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(stockfish, "resolve_binary", lambda: None)
    r = client.post("/api/analyze", json={"fen": START_FEN, "engines": ["stockfish"]})
    assert r.status_code == 200, r.text
    sf = r.json()["engines"]["stockfish"]
    assert sf["available"] is False
    assert sf["error"] == "stockfish binary not found"


def test_both_engines_missing_never_500s(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(model_server, "current", lambda: None)
    monkeypatch.setattr(stockfish, "resolve_binary", lambda: None)
    r = client.post("/api/analyze", json={"fen": START_FEN})
    assert r.status_code == 200, r.text
    engines = r.json()["engines"]
    assert set(engines) == {"fool", "stockfish"}
    assert engines["fool"]["available"] is False
    assert engines["stockfish"]["available"] is False


# ---------------------------------------------------------------------------
# Terminal positions
# ---------------------------------------------------------------------------


def test_terminal_positions_skip_search(client: TestClient):
    r = client.post("/api/analyze", json={"fen": MATE_WHITE_TO_MOVE})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["terminal"] == "checkmate"
    assert body["turn"] == "white"
    for name in ("fool", "stockfish"):
        entry = body["engines"][name]
        assert entry["available"] is True
        assert entry["best_move"] is None
        assert entry["best_san"] is None
        assert entry["pv"] == []
        assert entry["top"] == []
        assert entry["score_cp"] is None
        assert entry["mate"] == 0
        assert entry["expected_white"] == 0.0  # White is the mated side here
        assert entry["elapsed_ms"] == 0
    assert body["engines"]["fool"]["root_value"] == -1.0

    draw = client.post("/api/analyze", json={"fen": STALEMATE_BLACK_TO_MOVE}).json()
    assert draw["terminal"] == "stalemate"
    for name in ("fool", "stockfish"):
        entry = draw["engines"][name]
        assert entry["expected_white"] == 0.5
        assert entry["mate"] is None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_short_circuits_identical_calls(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}

    async def counting_entry(board: chess.Board, depth: int, multipv: int) -> dict[str, Any]:
        calls["n"] += 1
        return _stockfish_payload(depth, multipv)

    monkeypatch.setattr(analyze, "_stockfish_entry", counting_entry)
    body = {"fen": START_FEN, "engines": ["stockfish"], "depth": 6, "multipv": 1}

    first = client.post("/api/analyze", json=body)
    second = client.post("/api/analyze", json=body)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert calls["n"] == 1  # second call served from cache

    # A different key (multipv) must run the engine again.
    third = client.post("/api/analyze", json={**body, "multipv": 2})
    assert third.status_code == 200
    assert calls["n"] == 2


def test_cache_never_stores_errors(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}

    async def failing_entry(board: chess.Board, depth: int, multipv: int) -> dict[str, Any]:
        calls["n"] += 1
        return analyze._unavailable_stockfish("stockfish exploded", depth, 1)

    monkeypatch.setattr(analyze, "_stockfish_entry", failing_entry)
    body = {"fen": START_FEN, "engines": ["stockfish"], "depth": 6, "multipv": 1}

    first = client.post("/api/analyze", json=body)
    second = client.post("/api/analyze", json=body)
    assert first.status_code == second.status_code == 200
    assert first.json()["engines"]["stockfish"]["available"] is False
    assert calls["n"] == 2  # errors are never cached


# ---------------------------------------------------------------------------
# Clamping helpers
# ---------------------------------------------------------------------------


def test_clamping_and_key_helpers():
    """The pinned clamps and cp conversion hold at their boundaries."""
    assert analyze._clamp_int(1, 8, 400) == 8
    assert analyze._clamp_int(999, 8, 400) == 400
    assert analyze._cp_from_expected(0.5) == 0
    assert analyze._cp_from_expected(0.55) == round(-400 * math.log10(1 / 0.55 - 1))
    assert analyze._cp_from_expected(0.0) < -1000  # clamped, finite (Black wins)
    assert analyze._cp_from_expected(1.0) > 1000  # clamped, finite (White wins)
