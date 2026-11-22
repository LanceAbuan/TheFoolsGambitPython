"""Encoder v2 tests: 29-plane correctness + 73x64 policy round-trips (plan §2)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest

from training.encoder import (
    NUM_PLANES,
    POLICY_MOVES,
    decode_index,
    encode_batch,
    encode_move,
    encode_state,
    legal_mask,
    legal_move_map,
)

STARTPOS_OUR_PLANES = {  # white to move: our = white
    0: [(1, f) for f in range(8)],                      # pawns rank 2
    1: [(0, 1), (0, 6)],                                # knights b1 g1
    2: [(0, 2), (0, 5)],                                # bishops
    3: [(0, 0), (0, 7)],                                # rooks
    4: [(0, 3)],                                        # queen
    5: [(0, 4)],                                        # king
}


class TestInputPlanes:
    def test_shape_and_dtype(self):
        tensor = encode_state(chess.Board())
        assert tensor.shape == (NUM_PLANES, 8, 8)  # (plane, rank, file), channels-first
        assert NUM_PLANES == 29
        assert tensor.dtype == np.float32

    def test_startpos_plane_counts(self):
        """Startpos: each our/their piece plane has the right squares set."""
        tensor = encode_state(chess.Board())
        for channel, squares in STARTPOS_OUR_PLANES.items():
            for rank, file in squares:
                assert tensor[channel, rank, file] == 1.0
        # their pawns on rank 7 (plane 6)
        assert np.all(tensor[6, 6, :] == 1.0)
        # everything else empty
        assert tensor[12:24].sum() == 0.0  # no history yet
        assert tensor[28].sum() == 0.0  # no ep square

    def test_total_ones_matches_pieces(self):
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        tensor = encode_state(board)
        piece_count = len(board.piece_map())
        assert tensor[0:12].sum() == piece_count

    def test_perspective_flips_for_black(self):
        """When black is to move, 'our' planes are black's pieces, mirrored."""
        board = chess.Board()
        board.push_uci("e2e4")  # black to move
        tensor = encode_state(board)
        # black pawns are 'ours', intact on absolute rank 6 -> perspective rank 1
        assert np.all(tensor[0, 1, :] == 1.0)
        # white pawns are 'theirs': 7 left on absolute rank 1 -> perspective rank 6
        assert tensor[6, 6, :].sum() == 7.0
        assert tensor[6, 6, 4] == 0.0  # e-pawn is gone from rank 2
        # white's e4 pawn: absolute (3,4) -> perspective (4,4), their pawn plane
        assert tensor[6, 4, 4] == 1.0

    def test_history_planes_from_previous_position(self):
        board = chess.Board()
        prev = board.copy()
        board.push_uci("e2e4")
        tensor = encode_state(board, prev)
        # black to move: OUR previous pawns are black's, absolute rank 6 -> persp rank 1
        assert np.all(tensor[12, 1, :] == 1.0)
        # white's e2 pawn in the previous position is THEIR prev pawn:
        # absolute (1,4) -> perspective (6,4)
        assert tensor[18, 6, 4] == 1.0
        # and that pawn is NOT in the current 'their' row (it moved to e4)
        assert tensor[6, 6, 4] == 0.0

    def test_castling_planes_four_distinct(self):
        tensor = encode_state(chess.Board())  # all four rights
        assert tensor[24, 0, 0] == 1.0  # our kingside
        assert tensor[25, 0, 0] == 1.0  # our queenside
        assert tensor[26, 0, 0] == 1.0  # their kingside
        assert tensor[27, 0, 0] == 1.0  # their queenside

    def test_castling_rights_cleared(self):
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5")
        board.castling_rights = 0
        tensor = encode_state(board)
        assert tensor[24:28, 0, 0].sum() == 0.0

    def test_castling_rights_partial_black_only(self):
        board = chess.Board("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1")
        tensor = encode_state(board)
        assert tensor[24, 0, 0] == 1.0  # our (black) kingside
        assert tensor[25, 0, 0] == 1.0  # our (black) queenside
        assert tensor[26, 0, 0] == 0.0
        assert tensor[27, 0, 0] == 0.0

    def test_en_passant_plane(self):
        board = chess.Board()
        board.push_uci("e2e4")  # ep target e3, black to move
        tensor = encode_state(board)
        # e3 absolute (2,4) -> perspective (5,4)
        assert tensor[28, 5, 4] == 1.0

    def test_encode_batch_shapes(self):
        boards = [chess.Board(), chess.Board(), chess.Board()]
        batch = encode_batch(boards)
        assert batch.shape == (3, NUM_PLANES, 8, 8)
        assert np.allclose(batch[0], batch[1])


class TestPolicyEncoding:
    def test_policy_space_size(self):
        assert POLICY_MOVES == 73 * 64 == 4672

    def test_startpos_roundtrip_all_legal_moves(self):
        board = chess.Board()
        for move in board.legal_moves:
            decoded = decode_index(board, encode_move(board, move))
            assert decoded == move

    def test_kiwi_petete_roundtrip(self):
        """Rich position: promotions, captures, ep, castling all present."""
        board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        checked = 0
        for move in board.legal_moves:
            decoded = decode_index(board, encode_move(board, move))
            assert decoded == move, move
            checked += 1
        assert checked > 40

    def test_promotion_roundtrip_all_four_pieces(self):
        board = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
        promos = [m for m in board.legal_moves if m.promotion is not None]
        assert {m.promotion for m in promos} == {
            chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT
        }
        for move in promos:
            assert decode_index(board, encode_move(board, move)) == move

    def test_underpromotions_get_distinct_planes(self):
        """§10 promotion-policy fix: N/B/R promotions do NOT share one logit."""
        board = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
        indices = {
            move.promotion: encode_move(board, move)
            for move in board.legal_moves
            if move.promotion is not None
        }
        assert len(set(indices.values())) == 4  # four distinct logits
        # underpromotion planes live in the 64-72 region; queen promo does not
        assert 64 <= indices[chess.KNIGHT] % 73 <= 72
        assert 64 <= indices[chess.BISHOP] % 73 <= 72
        assert 64 <= indices[chess.ROOK] % 73 <= 72
        assert indices[chess.QUEEN] % 73 < 56

    def test_black_underpromotions_roundtrip(self):
        board = chess.Board("8/7k/8/8/8/8/p6K/8 b - - 0 1")
        promos = [m for m in board.legal_moves if m.promotion is not None]
        assert len(promos) == 4
        for move in promos:
            assert decode_index(board, encode_move(board, move)) == move

    def test_castling_encoded_as_king_two_square(self):
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        e1g1 = chess.Move.from_uci("e1g1")
        index = encode_move(board, e1g1)
        from_p, plane = divmod(index, 73)
        assert from_p == chess.E1  # white to move: no flip
        assert plane == 2 * 7 + 1  # direction E, distance 2
        assert decode_index(board, index) == e1g1
        # black mirrors onto the same perspective squares
        board_black = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 1 1")
        e8g8 = chess.Move.from_uci("e8g8")
        index_b = encode_move(board_black, e8g8)
        from_pb, plane_b = divmod(index_b, 73)
        assert from_pb == chess.E1  # e8 -> perspective e1
        assert plane_b == plane
        assert decode_index(board_black, index_b) == e8g8

    def test_en_passant_capture_roundtrip(self):
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("a7a6")
        board.push_uci("e4e5")
        board.push_uci("d7d5")
        ep_capture = chess.Move.from_uci("e5d6")
        assert ep_capture in board.legal_moves
        assert decode_index(board, encode_move(board, ep_capture)) == ep_capture

    def test_legal_mask_bijection(self):
        """Mask count == legal move count, and every set bit decodes legally."""
        for fen in (
            chess.STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        ):
            board = chess.Board(fen)
            mask = legal_mask(board)
            mapping = legal_move_map(board)
            assert mask.shape == (POLICY_MOVES,)
            assert int(mask.sum()) == board.legal_moves.count() == len(mapping)
            for index in mapping:
                assert mask[index] == 1.0
                assert decode_index(board, index) == mapping[index]

    def test_decode_illegal_index_returns_none(self):
        board = chess.Board()
        assert decode_index(board, POLICY_MOVES + 5) is None
        # d4 (index 27) is empty in the start position
        assert decode_index(board, 27 * 73) is None

    def test_encode_move_requires_occupied_square(self):
        board = chess.Board()
        with pytest.raises(ValueError):
            encode_move(board, chess.Move.from_uci("d4d5"))
