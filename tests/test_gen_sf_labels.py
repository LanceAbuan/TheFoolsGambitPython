"""Stockfish-labelled spool generator tests (no binary required)."""
import importlib.util
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest

from training.rust_workers import read_game_record, validate_game_record


def _load_script():
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "gen_sf_labels.py"
    spec = importlib.util.spec_from_file_location("gen_sf_labels_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubEngine:
    """Deterministic 'engine': first legal move, evaluation from material."""

    def get_move(self, board: chess.Board) -> chess.Move | None:
        moves = list(board.legal_moves)
        return moves[0] if moves else None

    def get_evaluation_normalized(self, board: chess.Board) -> float:
        material = 0.0
        for piece in board.piece_map().values():
            value = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                     chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}[piece.piece_type]
            material += value if piece.color == chess.WHITE else -value
        return float(np.tanh(material / 10.0))


def test_build_record_is_schema_valid(tmp_path):
    module = _load_script()
    record = module.build_record(
        _StubEngine(), seed=1, weight_version=42, max_plies=12, opening_plies=4
    )
    validate_game_record(record)  # the trainer's own validator
    assert record["worker_id"] == "sf"
    assert record["weight_version"] == 42
    assert record["ply_count"] == len(record["ucis"]) == len(record["examples"])
    assert record["termination"] in ("game-over", "max-plies")


def test_policy_targets_are_single_move_one_hot():
    module = _load_script()
    record = module.build_record(
        _StubEngine(), seed=2, weight_version=1, max_plies=6, opening_plies=2
    )
    for ex in record["examples"]:
        policy = np.frombuffer(
            __import__("base64").b64decode(ex["policy_b64"]), dtype="<f4"
        )
        assert policy.shape == (4672,)
        assert policy.sum() == pytest.approx(1.0)
        assert (policy > 0).sum() == 1, "distillation target must be a single move"


def test_record_roundtrips_through_the_trainer_reader(tmp_path):
    module = _load_script()
    from training.rust_workers import write_game_record

    record = module.build_record(
        _StubEngine(), seed=3, weight_version=7, max_plies=8, opening_plies=2
    )
    spool = tmp_path / "spool"
    path = write_game_record(spool, record)
    reread = read_game_record(path)
    assert reread["ply_count"] == record["ply_count"]
    assert reread["examples"][0]["board_tensor"].shape == (104, 8, 8)
    assert reread["examples"][0]["policy"].shape == (4672,)
