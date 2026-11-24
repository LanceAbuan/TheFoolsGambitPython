"""WP8: record conversion applies tablebase value overrides when enabled."""
import chess
import numpy as np

from training.rust_workers import (
    game_record_from_rust,
    read_game_record,
    record_to_selfplay_result,
    write_game_record,
)


class _FakeTB:
    enabled = True

    def __init__(self, value):
        self.value = value

    def value_for(self, board):
        return self.value


def _game_dict():
    from training.encoder import encode_move

    board = chess.Board()
    ucis = ["e2e4", "e7e5"]
    policies = []
    for uci in ucis:
        pol = np.zeros(4672, dtype=np.float32)
        pol[encode_move(board, chess.Move.from_uci(uci))] = 1.0
        policies.append(pol)
        board.push(chess.Move.from_uci(uci))
    return {
        "examples": [
            {"board_tensor": np.zeros((104, 8, 8), dtype=np.float32), "policy": policies[0],
             "value": -1.0, "policy_masked": False, "root_value": 0.0},
            {"board_tensor": np.zeros((104, 8, 8), dtype=np.float32), "policy": policies[1],
             "value": 1.0, "policy_masked": False, "root_value": 0.0},
        ],
        "moves": ["e4", "e5"],
        "ply_count": 2,
        "ucis": ucis,
        "fens": [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        ],
        "opening_fen": chess.STARTING_FEN,
        "result": "0-1",
        "termination": "game-over",
    }


def _record(tmp_path):
    record = game_record_from_rust(
        _game_dict(),
        weight_version=0,
        worker_id="w0",
        seed=1,
        opening_seed=None,
        sims=64,
        max_plies=200,
        temperature=1.0,
        temperature_plies=30,
    )
    return read_game_record(write_game_record(tmp_path, record))


def test_tb_overrides_values_when_enabled(tmp_path):
    result = record_to_selfplay_result(_record(tmp_path), tb=_FakeTB(0.5))
    assert [ex.value for ex in result.examples] == [0.5, 0.5]


def test_without_tb_keeps_outcome_values(tmp_path):
    result = record_to_selfplay_result(_record(tmp_path))
    assert [ex.value for ex in result.examples] == [-1.0, 1.0]
