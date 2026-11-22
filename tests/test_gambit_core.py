"""gambit_core correctness gates (FS-7).

python-chess is the oracle: standard perft suite via the Rust binding, plus
randomized differential fuzzing of movegen / FEN state / encoder / legal masks
against ``training.encoder`` (the bit-exact ground truth) on random legal walks
from varied start positions (castling, en passant and promotion rich).
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest

import gambit_core
from training.encoder import encode_state, legal_mask, legal_move_map

PERFT_STARTPOS = [20, 400, 8902, 197_281, 4_865_609]
PERFT_KIWIPETE = [48, 2039, 97_862]
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

# CPW standard perft suite (positions 3-6) at depths that keep pytest fast.
PERFT_SUITE = [
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43_238, 674_624]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467, 422_333]),
    ("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62_379]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079, 89_890]),
]

# Varied, feature-rich start positions for the random walks.
WALK_STARTS = [
    chess.STARTING_FEN,
    KIWIPETE,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # castling-rich
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # ep-rich
    "2k5/4P3/8/8/8/8/8/2K5 w - - 0 1",  # promotion-rich
    "kn1r4/2P5/8/8/8/8/8/K7 w - - 0 1",  # capture-underpromotion-rich
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # pinned-pawn / ep edge cases
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
]

NUM_WALK_GAMES = 44
MAX_WALK_PLIES = 80


def raw_fen(board: chess.Board) -> str:
    """python-chess FEN with the RAW en-passant field.

    python-chess always drops ep squares no enemy pawn can capture (its
    ``"legal"``/``"fen"`` FEN modes), but ``board.ep_square`` — and therefore
    ``training.encoder`` plane 28 and ``gambit_core``'s raw ep state — is set
    unconditionally after any double push. Re-inject it so the FEN handed to
    Rust carries exactly the state the Python encoder sees.
    """
    parts = board.fen().split()
    if board.ep_square is not None:
        parts[3] = chess.SQUARE_NAMES[board.ep_square]
    return " ".join(parts)


def walk_random_game(start_fen: str, rng: random.Random):
    """Yield (python_board, uci_chosen) for each position of a random legal game.

    Promotions are over-sampled (50% of the time when available) so the
    underpromotion planes get real coverage.
    """
    board = chess.Board(start_fen)
    yield board, None
    for _ in range(MAX_WALK_PLIES):
        moves = list(board.legal_moves)
        if not moves:
            return
        promos = [m for m in moves if m.promotion]
        if promos and rng.random() < 0.5:
            move = rng.choice(promos)
        else:
            move = rng.choice(moves)
        board.push(move)
        yield board, move.uci()
        if board.is_game_over():
            return


class TestApiSurface:
    def test_version_and_module(self):
        assert gambit_core.__version__

    def test_default_board_is_startpos(self):
        assert gambit_core.Board().fen() == chess.STARTING_FEN
        assert gambit_core.Board.from_fen(chess.STARTING_FEN).fen() == chess.STARTING_FEN

    def test_bad_fen_raises(self):
        with pytest.raises(ValueError):
            gambit_core.Board("not a fen at all!!")
        with pytest.raises(ValueError):
            gambit_core.Board("8/8/8/8/8/8/8/8 w - - 0 1")  # no kings

    def test_illegal_move_raises(self):
        board = gambit_core.Board()
        with pytest.raises(ValueError):
            board.make_move("e2e5")
        with pytest.raises(ValueError):
            board.make_move("e1g1")  # castling blocked by pieces
        with pytest.raises(ValueError):
            board.make_move("nonsense")

    def test_move_counters(self):
        b = gambit_core.Board()
        assert b.turn is True and b.fullmove_number == 1
        b = b.make_move("e2e4")
        assert b.turn is False and b.halfmove_clock == 0
        b = b.make_move("g8f6")
        assert b.turn is True and b.halfmove_clock == 1 and b.fullmove_number == 2

    def test_copy_on_make(self):
        b = gambit_core.Board()
        b2 = b.make_move("e2e4")
        assert b.fen() == chess.STARTING_FEN
        assert b2.fen() != chess.STARTING_FEN


class TestPerft:
    @pytest.mark.parametrize("depth,expected", list(enumerate(PERFT_STARTPOS, start=1)))
    def test_startpos(self, depth, expected):
        assert gambit_core.perft(chess.STARTING_FEN, depth) == expected

    @pytest.mark.parametrize("depth,expected", list(enumerate(PERFT_KIWIPETE, start=1)))
    def test_kiwipete(self, depth, expected):
        assert gambit_core.perft(KIWIPETE, depth) == expected

    @pytest.mark.parametrize("fen,expected", PERFT_SUITE)
    def test_standard_suite(self, fen, expected):
        for depth, nodes in enumerate(expected, start=1):
            assert gambit_core.perft(fen, depth) == nodes, f"{fen} at depth {depth}"

    def test_perft_terminal_positions(self):
        # Checkmate (fool's mate): perft(1) == 0.
        mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        assert gambit_core.perft(mate, 1) == 0
        # Stalemate: perft(1) == 0.
        stale = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
        assert gambit_core.perft(stale, 1) == 0


class TestBoardState:
    def test_castling_rights_cleared(self):
        b = gambit_core.Board()
        b = b.make_move("a2a3").make_move("a7a6").make_move("a1a2")
        # White rook left a1: queenside right gone.
        assert "Q" not in b.fen().split()[2]
        b = b.make_move("a8a7")  # black rook leaves a8: black queenside gone
        assert "q" not in b.fen().split()[2]
        b = b.make_move("h2h3").make_move("h7h6").make_move("h1h2")
        # White rook left h1: kingside right gone; only black kingside left.
        assert b.fen().split()[2] == "k"

    def test_ep_square_raw(self):
        # python-chess keeps ep_square after ANY double push; so must we.
        py = chess.Board()
        py.push_san("e4")
        b = gambit_core.Board().make_move("e2e4")
        assert b.fen().split()[3] == "e3"
        # raw_fen() re-injects the ep field python-chess's FEN writer drops
        # (no black pawn can capture on e3 yet).
        assert raw_fen(py).split()[3] == "e3"
        assert raw_fen(py) == b.fen()

    def test_en_passant_capture(self):
        b = gambit_core.Board()
        b = b.make_move("e2e4").make_move("a7a6").make_move("e4e5").make_move("d7d5")
        assert "e5d6" in b.legal_moves()
        b = b.make_move("e5d6")
        assert b.fen().split()[0] == "rnbqkbnr/1pp1pppp/p2P4/8/8/8/PPPP1PPP/RNBQKBNR"

    def test_in_check(self):
        mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        assert gambit_core.Board(mate).in_check() is True
        assert gambit_core.Board().in_check() is False

    def test_terminal_checkmate_stalemate(self):
        py_mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        py_stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        assert py_mate.is_checkmate()
        b = gambit_core.Board(py_mate.fen())
        assert b.is_terminal() is True and b.legal_moves() == []
        assert gambit_core.outcome(py_mate.fen()) == "0-1"
        assert py_stale.is_stalemate()
        b = gambit_core.Board(py_stale.fen())
        assert b.is_terminal() is True
        assert gambit_core.outcome(py_stale.fen()) == "1/2-1/2"
        assert gambit_core.outcome(chess.STARTING_FEN) is None

    def test_terminal_insufficient_material(self):
        cases = {
            "8/8/8/8/8/8/8/K5k1 w - - 0 1": True,  # K vs K
            "8/8/8/8/8/8/8/KB4k1 w - - 0 1": True,  # K+B vs K
            "8/8/8/8/8/8/8/KN4k1 w - - 0 1": True,  # K+N vs K
            "8/8/8/8/8/8/8/KNN3k1 w - - 0 1": False,  # two knights can mate
            "8/8/8/8/8/8/8/K1B3k1 w - - 0 1": True,  # K+B vs K, offside bishop
        }
        for fen, expected in cases.items():
            assert gambit_core.Board(fen).is_terminal() is expected, fen
            assert gambit_core.Board(fen).is_terminal() is chess.Board(fen).is_insufficient_material()

    def test_terminal_fifty_move(self):
        # Halfmove clock 100 with legal moves: draw, matching python-chess
        # is_fifty_moves (mate/stalemate take precedence — excluded here).
        fen = "8/8/8/8/8/1k6/8/1K1R4 w - - 100 80"
        assert gambit_core.Board(fen).is_terminal() is True
        assert chess.Board(fen).is_fifty_moves() is True

    def test_step_batch(self):
        fens = [chess.STARTING_FEN] * 5
        ucis = ["e2e4", "d2d4", "g1f3", "c2c4", "a2a3"]
        out = gambit_core.step_batch(fens, ucis)
        assert len(out) == 5
        for fen, uci in zip(fens, ucis):
            assert out.pop(0) == gambit_core.make_move(fen, uci)
        with pytest.raises(ValueError):
            gambit_core.step_batch(fens[:1], ["e2e9"])

    def test_zobrist_stable(self):
        assert gambit_core.zobrist(chess.STARTING_FEN) == gambit_core.zobrist(chess.STARTING_FEN)
        assert gambit_core.zobrist(chess.STARTING_FEN) != gambit_core.zobrist(KIWIPETE)


class TestEncoder:
    def test_encode_shapes(self):
        tensor = gambit_core.Board().encode()
        assert tensor.shape == (29, 8, 8)
        assert tensor.dtype == np.float32
        assert tensor.flags["C_CONTIGUOUS"]
        mask = gambit_core.Board().legal_mask()
        assert mask.shape == (4672,)
        assert mask.dtype == np.float32
        batch = gambit_core.encode_batch([chess.STARTING_FEN, KIWIPETE])
        assert batch.shape == (2, 29, 8, 8)
        masks = gambit_core.legal_masks([chess.STARTING_FEN, KIWIPETE])
        assert masks.shape == (2, 4672)

    def test_encode_batch_matches_singles(self):
        fens = [chess.STARTING_FEN, KIWIPETE, PERFT_SUITE[1][0]]
        batched = gambit_core.encode_batch(fens)
        for i, fen in enumerate(fens):
            assert np.array_equal(batched[i], gambit_core.Board(fen).encode())

    def test_startpos_mask_matches_python_map(self):
        board = chess.Board()
        mask = gambit_core.Board().legal_mask()
        assert set(np.nonzero(mask)[0].tolist()) == set(legal_move_map(board).keys())
        assert mask.sum() == len(list(board.legal_moves))

    def test_kiwipete_mask_and_planes_match_python(self):
        board = chess.Board(KIWIPETE)
        fen = raw_fen(board)
        assert np.array_equal(gambit_core.Board(fen).encode(), encode_state(board))
        assert np.array_equal(gambit_core.legal_masks([fen])[0], legal_mask(board))


class TestDifferentialFuzz:
    """Random legal walks from varied starts: python-chess + training.encoder
    vs gambit_core — bit-exact on every position."""

    def test_random_walks_bit_exact(self):
        rng = random.Random(20261121)
        checked = 0
        fen_pairs: list[tuple[str, chess.Board]] = []
        for game in range(NUM_WALK_GAMES):
            start = WALK_STARTS[game % len(WALK_STARTS)]
            rust = gambit_core.Board(start)
            for board, uci in walk_random_game(start, rng):
                if uci is not None:
                    rust = rust.make_move(uci)
                fen = raw_fen(board)
                # 1. FEN state equality (placement, turn, castling, ep, clocks).
                assert rust.fen() == fen, f"game {game} ply {checked}: FEN divergence"
                # 2. Legal move sets identical.
                py_moves = {m.uci() for m in board.legal_moves}
                assert set(rust.legal_moves()) == py_moves, (
                    f"game {game} ply {checked}: movegen divergence in {fen}"
                )
                # 3. Check/terminal state identical. The Python side mirrors
                # the documented is_terminal semantics exactly: no legal
                # moves (mate/stalemate), fifty-move, or insufficient
                # material. Repetition draws are excluded by design (no move
                # stack; the MCTS layer owns repetition state).
                assert rust.in_check() == board.is_check()
                rust_terminal = rust.is_terminal()
                py_terminal = (
                    not any(board.legal_moves)
                    or board.is_fifty_moves()
                    or board.is_insufficient_material()
                )
                assert rust_terminal == py_terminal, (
                    f"game {game} ply {checked}: terminal divergence in {fen}"
                )
                fen_pairs.append((fen, board.copy()))  # copy: board mutates in place
                checked += 1
        assert checked >= 1000, f"walk produced only {checked} positions"

        # 4. Encoder + masks: bit-exact on the full collected batch.
        fens = [fen for fen, _ in fen_pairs]
        rust_planes = gambit_core.encode_batch(fens)
        rust_masks = gambit_core.legal_masks(fens)
        py_planes = np.stack([encode_state(board) for _, board in fen_pairs])
        py_masks = np.stack([legal_mask(board) for _, board in fen_pairs])
        assert rust_planes.dtype == np.float32 and rust_planes.shape == py_planes.shape
        assert np.array_equal(rust_planes, py_planes), "encoder planes diverged"
        assert np.array_equal(rust_masks, py_masks), "legal masks diverged"

        # 5. Mask <-> move bijection at a sample of positions.
        for fen, board in fen_pairs[:: max(1, checked // 50)]:
            mask = gambit_core.legal_masks([fen])[0]
            assert set(np.nonzero(mask)[0].tolist()) == set(legal_move_map(board).keys())

    def test_black_perspective_positions(self):
        """Dedicated check that mirroring is exact for black-to-move states."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        board = chess.Board(fen)
        assert np.array_equal(gambit_core.Board(fen).encode(), encode_state(board))
        assert np.array_equal(gambit_core.legal_masks([fen])[0], legal_mask(board))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
