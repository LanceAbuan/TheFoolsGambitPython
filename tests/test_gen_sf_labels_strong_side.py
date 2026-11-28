"""Stockfish-labelled spool generator: strong-side-only distillation tests."""
import base64
import importlib.util
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest

from training.rust_workers import validate_game_record


def _load_script():
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "gen_sf_labels.py"
    spec = importlib.util.spec_from_file_location("gen_sf_labels_strong", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubEngine:
    """Deterministic 'engine': first legal move, evaluation from material."""

    def get_move(self, board):
        moves = list(board.legal_moves)
        return moves[0] if moves else None

    def get_evaluation_normalized(self, board):
        return 0.0


class _WeakBlackEngine(_StubEngine):
    """Plays the opposite first legal move, so its plies are identifiable."""

    def get_move(self, board):
        moves = list(board.legal_moves)
        return moves[-1] if moves else None


def test_only_the_strong_side_is_teachable():
    """Weak-side plies must be MASKED, never taught (2026-11-27 probe-game bug)."""
    module = _load_script()
    record = module.build_record(
        _StubEngine(),
        seed=11,
        weight_version=3,
        max_plies=8,
        opening_plies=0,
        black_engine=_WeakBlackEngine(),
        strong_side=chess.WHITE,
    )
    validate_game_record(record)
    import base64

    white_plies = [ex for i, ex in enumerate(record["examples"]) if i % 2 == 0]
    black_plies = [ex for i, ex in enumerate(record["examples"]) if i % 2 == 1]
    assert white_plies, "white must have teachable plies"
    for ex in white_plies:
        assert ex["policy_masked"] is False
        policy = np.frombuffer(base64.b64decode(ex["policy_b64"]), dtype="<f4")
        assert policy.sum() == pytest.approx(1.0)
    for ex in black_plies:
        assert ex["policy_masked"] is True
        policy = np.frombuffer(base64.b64decode(ex["policy_b64"]), dtype="<f4")
        assert policy.sum() == 0.0, "masked rows carry no policy target"


def test_equal_strength_records_both_sides():
    module = _load_script()
    record = module.build_record(
        _StubEngine(), seed=12, weight_version=3, max_plies=6, opening_plies=0
    )
    assert all(ex["policy_masked"] is False for ex in record["examples"])
