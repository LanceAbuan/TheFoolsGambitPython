"""Tests for training/tensorize.py — pure numpy/chess, no GPU needed."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
from training.tensorize import (
    board_to_tensor,
    legal_moves_mask,
    move_to_idx,
    idx_to_move,
    NUM_CHANNELS,
    NUM_PIECE_CHANNELS,
    NUM_AUX_CHANNELS,
    NUM_POSSIBLE_MOVES,
)


class TestBoardToTensor:
    def test_shape(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.shape == (8, 8, NUM_CHANNELS)

    def test_dtype(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.dtype == np.float32

    def test_start_position_white_pawns(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        # White pawns on rank 1 (files 0-7), channel 0 (white pawn)
        assert np.all(tensor[1, :, 0] == 1.0)
        # No other piece-type channels on rank 1 (channels 1-11)
        assert np.all(tensor[1, :, 1:12] == 0.0)

    def test_start_position_black_pawns(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        # Black pawns on rank 6, channel 6+0=6 (black pawn)
        assert np.all(tensor[6, :, 6] == 1.0)

    def test_start_position_white_pieces(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        # PIECE_ORDER = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]
        # White piece channels: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
        # Rank 0 (a1-h1): R N B Q K B N R
        assert tensor[0, 0, 3] == 1.0  # White Rook at a1 (ROOK channel=3)
        assert tensor[0, 1, 1] == 1.0  # White Knight at b1 (KNIGHT channel=1)
        assert tensor[0, 3, 4] == 1.0  # White Queen at d1 (QUEEN channel=4)
        assert tensor[0, 4, 5] == 1.0  # White King at e1 (KING channel=5)

    def test_side_to_move_white(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor[0, 0, 12] == 1.0  # White to move

    def test_side_to_move_black(self):
        board = chess.Board()
        board.push_uci("e2e4")
        tensor = board_to_tensor(board)
        assert tensor[0, 0, 12] == -1.0  # Black to move

    def test_castling_rights_full(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        # All 4 castling rights: 0.25 * 4 = 1.0
        assert tensor[0, 0, 13] == 1.0

    def test_castling_rights_none(self):
        board = chess.Board("K7/8/8/8/8/8/8/8 w - - 0 1")
        # No castling rights encoded in FEN
        tensor = board_to_tensor(board)
        assert tensor[0, 0, 13] == 0.0

    def test_en_passant_square(self):
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        tensor = board_to_tensor(board)
        # EP square e3 → rank 2, file 4
        assert tensor[2, 4, 14] == 1.0

    def test_halfmove_clock_zero(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor[0, 0, 15] == 0.0

    def test_halfmove_clock_normalized(self):
        board = chess.Board("K7/8/8/8/8/8/8/8 w - - 50 1")
        tensor = board_to_tensor(board)
        assert tensor[0, 0, 15] == 0.5

    def test_halfmove_clock_clamped(self):
        board = chess.Board("K7/8/8/8/8/8/8/8 w - - 200 1")
        tensor = board_to_tensor(board)
        assert tensor[0, 0, 15] == 1.0

    def test_piece_movement(self):
        board = chess.Board()
        board.push_uci("e2e4")
        tensor = board_to_tensor(board)
        # Pawn moved from e2 (rank 1, file 4) to e4 (rank 3, file 4)
        assert tensor[3, 4, 0] == 1.0  # White pawn at e4
        assert tensor[1, 4, 0] == 0.0  # No pawn at e2


class TestLegalMovesMask:
    def test_shape(self):
        board = chess.Board()
        mask = legal_moves_mask(board)
        assert mask.shape == (1, 8, 8)

    def test_start_position_legal_squares(self):
        board = chess.Board()
        mask = legal_moves_mask(board)
        # Pawns can go to rank 2 and 3; knights can reach rank 1 and 2
        # There should be some legal target squares
        assert mask.sum() > 0

    def test_checkmate_mask(self):
        board = chess.Board("8/6rk/6p1/8/3b1p2/4B3/6PP/6K1 b - - 1 45")
        # Force a position with very few legal moves
        mask = legal_moves_mask(board)
        # Just check it returns a valid mask
        assert mask.shape == (1, 8, 8)
        assert mask.sum() >= 0

    def test_empty_board_no_legal(self):
        board = chess.Board("8/8/8/8/8/8/7k/K7 b - - 0 1")
        mask = legal_moves_mask(board)
        # Lone king has legal moves on empty board
        assert mask.sum() > 0


class TestMoveIndexing:
    def test_move_to_idx_basic(self):
        move = chess.Move.from_uci("e2e4")
        idx = move_to_idx(move)
        assert idx == move.from_square * 64 + move.to_square

    def test_idx_to_move_roundtrip(self):
        move = chess.Move.from_uci("e2e4")
        idx = move_to_idx(move)
        recovered = idx_to_move(idx)
        assert recovered.from_square == move.from_square
        assert recovered.to_square == move.to_square

    def test_all_squares_roundtrip(self):
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                if from_sq == to_sq:
                    continue
                move = chess.Move(from_sq, to_sq)
                idx = move_to_idx(move)
                recovered = idx_to_move(idx)
                assert recovered.from_square == from_sq
                assert recovered.to_square == to_sq

    def test_num_possible_moves(self):
        assert NUM_POSSIBLE_MOVES == 4096

    def test_idx_bounds(self):
        # Corner case: a1→h8 (0→63)
        idx_min = move_to_idx(chess.Move(0, 63))
        assert idx_min == 63
        # h8→a1 (63→0)
        idx_max = move_to_idx(chess.Move(63, 0))
        assert idx_max == 63 * 64  # 4032
