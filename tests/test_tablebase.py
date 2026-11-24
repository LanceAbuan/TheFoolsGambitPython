"""WP8: Syzygy corrector — graceful disable + WDL mapping (mock tablebase)."""
from types import SimpleNamespace

import chess

from training.tablebase import MAX_TB_PIECES, TablebaseCorrector, chess_piece_count


class _FakeTB:
    def __init__(self, wdl):
        self.wdl = wdl
        self.calls = 0

    def probe_wdl(self, board):
        self.calls += 1
        return self.wdl

    def close(self):
        pass


def _corrector_with(wdl):
    corrector = TablebaseCorrector.__new__(TablebaseCorrector)
    corrector.tablebase = _FakeTB(wdl)
    corrector.enabled = True
    return corrector


def test_disabled_without_path_or_dir():
    assert TablebaseCorrector(None).enabled is False
    assert TablebaseCorrector("definitely-not-a-dir").enabled is False


def test_wdl_mapping():
    board = chess.Board("8/8/8/8/8/8/5KP1/6k1 w - - 0 1")  # 3 pieces (K+P vs K)
    for wdl, expected in ((2, 1.0), (1, 0.5), (0, 0.0), (-1, -0.5), (-2, -1.0)):
        assert _corrector_with(wdl).value_for(board) == expected


def test_skips_out_of_range_positions():
    corrector = _corrector_with(2)
    board = chess.Board()  # 32 pieces
    assert chess_piece_count(board) > MAX_TB_PIECES
    assert corrector.value_for(board) is None
    assert corrector.tablebase.calls == 0


def test_skips_finished_positions():
    corrector = _corrector_with(2)
    mated = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert mated.is_game_over()
    assert corrector.value_for(mated) is None
