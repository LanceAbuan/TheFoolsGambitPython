"""Encoder v2 tests: 29-plane correctness + 73x64 policy round-trips (plan §2)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest

from training.encoder import (
    ENCODER_VERSION,
    NUM_PLANES,
    NUM_PLANES_V2,
    POLICY_MOVES,
    V2_REP_CAP,
    V2_REP_OPP_PLANE,
    V2_REP_OWN_PLANE,
    V2_SIDE_TO_MOVE_PLANE,
    decode_index,
    encode_batch,
    encode_move,
    encode_state,
    encode_v2,
    legal_mask,
    legal_move_map,
    repetition_counts,
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


# --- WP-B: encoder v2 (104 channels) ------------------------------------------

# Varied, feature-rich start positions for the real-play random walks (same
# set philosophy as tests/test_gambit_core.py).
V2_WALK_STARTS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "2k5/4P3/8/8/8/8/8/2K5 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]

V2_WALK_GAMES = 14
V2_MAX_WALK_PLIES = 30


def walk_positions_with_history(start_fen: str, rng: random.Random):
    """Yield (board, history) per ply of a random legal game.

    `history` is most-recent-first: history[0] is t-1 (the position before
    `board`), exactly the `encode_v2` convention. The game ROOT position is
    included once the first move has been played.
    """
    board = chess.Board(start_fen)
    past: list[chess.Board] = []  # chronological
    yield board.copy(), []
    for _ in range(V2_MAX_WALK_PLIES):
        moves = list(board.legal_moves)
        if not moves:
            return
        board.push(rng.choice(moves))
        yield board.copy(), list(reversed(past + [chess.Board(start_fen)]))
        past.append(board.copy())


class TestEncoderV2Constants:
    def test_constants(self):
        assert NUM_PLANES_V2 == 104
        assert ENCODER_VERSION == 2

    def test_layout_offsets(self):
        # 8 x 12 piece planes (t0..t-7) + 2 rep + 4 castling + 1 ep + 1 stm.
        # t0/t-1/castling/ep live inside the v1 prefix; the deep history
        # t-2..t-7 lands on planes 29..100; 101/102 repetition; 103 stm.
        from training.encoder import (
            V2_DEEP_HISTORY_OFFSET,
            V2_DEEP_HISTORY_PLANES,
        )

        assert V2_DEEP_HISTORY_OFFSET == 29
        assert V2_DEEP_HISTORY_PLANES == 72
        assert V2_REP_OWN_PLANE == 101
        assert V2_REP_OPP_PLANE == 102
        assert V2_SIDE_TO_MOVE_PLANE == 103
        assert NUM_PLANES_V2 == 96 + 2 + 4 + 1 + 1


class TestEncoderV2:
    def test_shape_and_dtype(self):
        tensor = encode_v2(chess.Board())
        assert tensor.shape == (NUM_PLANES_V2, 8, 8)
        assert tensor.dtype == np.float32

    def test_v1_prefix_no_history(self):
        """encode_v2(b)[:29] == encode_state(b) bit-exactly (t-1 zeros both)."""
        for fen in V2_WALK_STARTS:
            board = chess.Board(fen)
            assert np.array_equal(encode_v2(board)[:NUM_PLANES], encode_state(board))

    def test_v1_prefix_random_real_play(self):
        """The plan's prefix gate: >= 50 random real-play positions, bit-exact."""
        rng = random.Random(20261122)
        checked = 0
        for game in range(V2_WALK_GAMES):
            start = V2_WALK_STARTS[game % len(V2_WALK_STARTS)]
            for board, history in walk_positions_with_history(start, rng):
                expected = encode_state(board, history[0] if history else None)
                assert np.array_equal(encode_v2(board, history)[:NUM_PLANES], expected), (
                    f"prefix divergence at ply {len(history)} in {board.fen()}"
                )
                checked += 1
        assert checked >= 50, f"walk produced only {checked} positions"

    def test_history_step_offsets_t_minus_2_through_t_minus_7(self):
        """history[1] -> planes 29-40, ..., history[6] -> planes 89-100."""
        from training.encoder import _piece_planes

        board = chess.Board()
        moves = ["g1f3", "g8f6", "f3g1", "f6g8", "b1c3", "b8c6", "c3b1", "c6b8"]
        boards = [board.copy()]
        for uci in moves:
            board.push_uci(uci)
            boards.append(board.copy())
        history = list(reversed(boards[:-1]))  # t-1 .. t-8 (8 entries)
        assert board.turn == chess.WHITE  # 8 plies of knight shuffles: white to move
        tensor = encode_v2(board, history)
        for step in range(6):
            offset = 29 + step * 12
            expected = np.zeros((12, 8, 8), dtype=np.float32)
            _piece_planes(history[step + 1], False, 0, expected, chess.WHITE)
            assert np.array_equal(tensor[offset : offset + 12], expected), f"t-{step + 2}"

    def test_short_history_pads_with_initial_position(self):
        """Missing deep slots (t-2..t-7) pad with the initial-position snapshot."""
        from training.encoder import _piece_planes

        board = chess.Board()
        board.push_uci("e2e4")  # black to move; history = [startpos]
        tensor = encode_v2(board, [chess.Board()])
        # t-1 (prefix, planes 12-23) from the provided startpos.
        assert np.array_equal(tensor[12:24], encode_state(board, chess.Board())[12:24])
        # t-2..t-7 = initial position from BLACK's perspective (mirrored).
        expected = np.zeros((12, 8, 8), dtype=np.float32)
        _piece_planes(chess.Board(), True, 0, expected, chess.BLACK)
        for step in range(6):
            offset = 29 + step * 12
            assert np.array_equal(tensor[offset : offset + 12], expected), f"pad slot t-{step + 2}"

    def test_empty_history_zeroes_all_history_planes(self):
        """Empty history: t-1 (prefix) AND deep planes all zero (v1 parity)."""
        board = chess.Board()
        board.push_uci("d2d4")
        tensor = encode_v2(board, None)
        assert tensor[12:24].sum() == 0.0  # t-1 inside the prefix
        assert tensor[29:101].sum() == 0.0  # deep history t-2..t-7 also zeroed
        assert tensor[V2_REP_OWN_PLANE, 0, 0] == 0.0
        assert tensor[V2_REP_OPP_PLANE, 0, 0] == 0.0
        assert np.array_equal(tensor[:NUM_PLANES], encode_state(board))

    def test_repetition_planes_increase_after_repeat(self):
        """Acceptance: the repetition plane rises once the position repeats."""
        board = chess.Board()
        history: list[chess.Board] = []
        first = encode_v2(board, history)
        # Authoritative layout: PRIOR occurrences only, so a fresh position is 0.
        assert first[V2_REP_OWN_PLANE, 0, 0] == 0.0
        # Nf3 Nf6 Ng1 Ng8 returns to the starting position.
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            history.insert(0, board.copy())
            board.push_uci(uci)
        repeated = encode_v2(board, history)
        # The game root (start position, white to move) is in the history once.
        assert repeated[V2_REP_OWN_PLANE, 0, 0] == pytest.approx(1.0 / V2_REP_CAP)
        assert repeated[V2_REP_OWN_PLANE, 0, 0] > first[V2_REP_OWN_PLANE, 0, 0]
        # The mirrored placement (start position, black to move) never occurred.
        assert repeated[V2_REP_OPP_PLANE, 0, 0] == 0.0

    def test_repetition_opp_plane_counts_flipped_mover(self):
        """`opp` counts the same placement with the opposite side to move."""
        board = chess.Board()
        board.push_uci("e2e4")  # current: pawn-e4 placement, black to move
        twin = board.copy(stack=False)
        twin.turn = not board.turn  # SAME placement, white to move (synthetic)
        own, opp = repetition_counts(board, [twin])
        assert own == 0  # mover differs: not an own match
        assert opp == 1
        tensor = encode_v2(board, [twin])
        assert tensor[V2_REP_OPP_PLANE, 0, 0] == pytest.approx(1.0 / V2_REP_CAP)
        # The current position itself contributes to neither count.
        assert repetition_counts(board, []) == (0, 0)

    def test_repetition_planes_cap_at_three(self):
        """Counts saturate at 3 -> plane value exactly 1.0."""
        board = chess.Board()
        history = [chess.Board() for _ in range(6)]  # startpos six more times
        own, _ = repetition_counts(board, history)
        assert own == 6
        tensor = encode_v2(board, history)
        assert tensor[V2_REP_OWN_PLANE, 0, 0] == 1.0  # min(6, 3) / 3

    def test_side_to_move_plane(self):
        white = encode_v2(chess.Board())
        assert white[V2_SIDE_TO_MOVE_PLANE, 0, 0] == 1.0
        black = chess.Board()
        black.push_uci("e2e4")
        black_tensor = encode_v2(black)
        assert black_tensor[V2_SIDE_TO_MOVE_PLANE, 0, 0] == 0.0

    def test_repetition_no_false_repeat_across_pawn_move(self):
        """A pawn move resets the repetition window: no false repeat."""
        board = chess.Board()
        history = [chess.Board()]  # the game root (same placement, white to move)
        board.push_uci("e2e4")
        # Current: startpos placement minus e2 pawn, black to move — key differs
        # from every history entry; opp (same placement, white to move) never
        # occurred either.
        own, opp = repetition_counts(board, history)
        assert own == 0
        assert opp == 0
