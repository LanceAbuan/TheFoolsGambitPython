"""Encoding-contract tests for the Stockfish label generator.

Two bugs were found by independent review on 2026-11-28 and confirmed by decoding
generated records:
  * history was oldest-first AND included the current position, so the v2
    encoder's prev-plane channels duplicated the current board (planes[0:12] ==
    planes[12:24]) and the deep-history planes were shifted;
  * `value` was written from WHITE's perspective while the spool contract is the
    MOVER's, inverting every Black-to-move row.
"""
import base64
import importlib.util
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest

from training.encoder import encode_state_v2


def _load_script():
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "gen_sf_labels.py"
    spec = importlib.util.spec_from_file_location("gen_sf_labels_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ConstantEvalEngine:
    """Always plays the first legal move and reports +0.6 (White POV)."""

    def get_move(self, board):
        moves = list(board.legal_moves)
        return moves[0] if moves else None

    def get_evaluation_normalized(self, board):
        return 0.6


class _ScriptedEngine:
    """Follows a fixed UCI move list so the position sequence is known."""

    def __init__(self, ucis):
        self._ucis = list(ucis)
        self._i = 0

    def get_move(self, board):
        if self._i >= len(self._ucis):
            moves = list(board.legal_moves)
            return moves[0] if moves else None
        move = chess.Move.from_uci(self._ucis[self._i])
        self._i += 1
        return move

    def get_evaluation_normalized(self, board):
        return 0.0


def _planes(record, index):
    return np.frombuffer(
        base64.b64decode(record["examples"][index]["board_tensor_b64"]), dtype="<f4"
    ).reshape(104, 8, 8)


def _value(record, index):
    return record["examples"][index]["value"]


def test_history_planes_are_not_duplicated():
    """planes[0:12] must be the PREVIOUS position, never a copy of the current."""
    module = _load_script()
    engine = _ScriptedEngine(["e2e4", "e7e5", "g1f3", "b8c6"])
    record = module.build_record(
        engine, seed=1, weight_version=1, max_plies=4, opening_plies=0
    )
    for index in range(1, min(4, record["ply_count"])):
        planes = _planes(record, index)
        assert not np.array_equal(planes[0:12], planes[12:24]), (
            f"ply {index}: prev-plane channels duplicate the current position"
        )


def test_first_ply_is_compared_against_a_real_previous_position():
    """For ply 1 the planes must equal the encoder's own v2 encoding of the board
    (whose move stack holds the played move)."""
    module = _load_script()
    engine = _ScriptedEngine(["e2e4", "e7e5"])
    record = module.build_record(
        engine, seed=2, weight_version=1, max_plies=2, opening_plies=0
    )
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    expected = encode_state_v2(board)
    got = _planes(record, 1)
    assert np.allclose(got, expected, atol=1e-6), "ply 1 planes != encode_state_v2(board)"


def test_value_is_from_the_movers_perspective():
    """Constant +0.6 (White POV): White-to-move rows -> +0.6, Black -> -0.6."""
    module = _load_script()
    record = module.build_record(
        _ConstantEvalEngine(), seed=3, weight_version=1, max_plies=4, opening_plies=0
    )
    assert record["ply_count"] >= 2
    assert _value(record, 0) == pytest.approx(0.6), "White to move must keep the sign"
    assert _value(record, 1) == pytest.approx(-0.6), "Black to move must be negated"
