"""WP9: puzzle scoring semantics (move-by-move, mate accepted)."""
import chess

from scripts.bench_puzzles import gives_mate, play_puzzle


SCHOLAR_MATE = [
    "e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7",
]


def test_play_puzzle_counts_solution_completion():
    # bot answers exactly the solution moves
    moves = ["e7e5", "g1f3"]
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    calls = iter(["g1f3"])

    def choose(b):
        return chess.Move.from_uci(next(calls))

    assert play_puzzle(board, moves, choose) is True


def test_wrong_move_fails():
    moves = ["e7e5", "g1f3"]
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")

    def choose(b):
        return chess.Move.from_uci("b1c3")

    assert play_puzzle(board, moves, choose) is False


def test_gives_mate_helper():
    # White to move: Rd8# in the classic back-rank position.
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    assert gives_mate(board, chess.Move.from_uci("d1d8"))
    # Board is restored after the probe.
    assert not board.is_checkmate()
