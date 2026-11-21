"""M3 fast-path tests: gambit_core wiring behind GAMBIT_CORE_FASTPATH.

Parity contract (docs/UPGRADE_PLAN.md 4.1 + docs/RESEARCH_STANDARD.md):

1. Gate semantics: default OFF, enable/disable, loud ImportError when the
   gambit_core extension is missing (a silent fallback would poison the
   Python-vs-Rust ablation).
2. Rust legal-move ORDER parity with python-chess ``board.legal_moves`` --
   MCTS child order and tie-breaks depend on it, so order is part of the
   bit-exactness contract. Differential fuzz over random games.
3. ``encode_batch`` FEN-bridge parity, including raw en-passant squares with
   no legal capture (the ``en_passant="fen"`` bridge exists for exactly that).
4. ``legal_move_map`` parity: same keys, same insertion order, same moves.
5. FastBoard claim-semantics differential fuzz vs plain python-chess Board
   (threefold / repetition / draw-claim outcomes must be IDENTICAL).
6. End-to-end bit-exactness: ``MCTSSearch.search`` and ``self_play_game``
   produce IDENTICAL results with the fast path off vs on (same seeds).
"""

from __future__ import annotations

import importlib
import os
import random
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest

from training import fastpath as fp
from training.encoder import encode_batch, encode_move, legal_move_map
from training.selfplay import (
    BatchNNEvaluator,
    FastBoard,
    MCTSSearch,
    SearchConfig,
    SelfPlayConfig,
    self_play_game,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _gate_default_off():
    """Every test starts and ends with the fast path OFF."""
    fp.disable_fastpath()
    yield
    fp.disable_fastpath()


class StubEvaluator:
    """Deterministic evaluator: uniform legal priors, configurable value."""

    def __init__(self, value_fn=None) -> None:
        self.value_fn = value_fn

    def evaluate(self, boards):
        n = len(boards)
        if self.value_fn is not None:
            values = np.array([self.value_fn(b) for b in boards], dtype=np.float32)
        else:
            values = np.full(n, 0.0, dtype=np.float32)
        return np.zeros((n, 4672), dtype=np.float32), values


def _side_value(board: chess.Board) -> float:
    """Board-dependent value so the MCTS tree is not perfectly flat."""
    return 0.12 if board.turn == chess.WHITE else -0.12


def _claim_snapshot(b: chess.Board) -> tuple:
    outcome = b.outcome(claim_draw=True)
    return (
        b.can_claim_threefold_repetition(),
        b.is_repetition(3),
        b.is_repetition(2),
        b.is_game_over(claim_draw=True),
        outcome.termination.name if outcome is not None else None,
        b.can_claim_draw(),
    )


START_FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
]

CRAFTED_FENS = [
    # en passant set but NO legal capture: the "legal"-vs-"fen" divergence case
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    # en passant set AND capture legal
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
    # ep target set but the capturing pawn is pinned along the 4th rank
    "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1",
    # full castling rights both sides
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    # bogus castling rights (king off e1): exercises rights cleaning
    "r3k2r/8/8/8/8/8/8/R3K1R1 w KQkq - 0 1",
    # check with castling rights still on the board
    "r3k2r/8/5Q2/8/8/8/8/R3K2R b KQkq - 0 1",
    # promotion-heavy
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    # fifty-move clock at the claim boundary
    "4k3/8/8/8/8/8/8/4K2R w K - 99 60",
]

MATE_IN_1_FEN = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"


def _walk_positions(fens, plies_per_start, rng):
    """Yield FENs from random walks (movegen-rich: checks, ep, castling)."""
    seen: list[str] = []
    for start in fens:
        board = chess.Board(start)
        seen.append(board.fen())
        for _ in range(plies_per_start):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                board = chess.Board(rng.choice(fens))
                continue
            board.push(rng.choice(moves))
            seen.append(board.fen())
    return seen


# ---------------------------------------------------------------------------
# 1. gate semantics
# ---------------------------------------------------------------------------


class TestGate:
    def test_default_is_off(self, monkeypatch):
        monkeypatch.delenv("GAMBIT_CORE_FASTPATH", raising=False)
        importlib.reload(fp)
        assert fp.fastpath_enabled() is False

    def test_env_var_turns_it_on(self, monkeypatch):
        monkeypatch.setenv("GAMBIT_CORE_FASTPATH", "1")
        importlib.reload(fp)
        assert fp.fastpath_enabled() is True

    def test_env_var_rejects_other_values(self, monkeypatch):
        monkeypatch.setenv("GAMBIT_CORE_FASTPATH", "0")
        importlib.reload(fp)
        assert fp.fastpath_enabled() is False

    def test_enable_disable_roundtrip(self):
        assert fp.fastpath_enabled() is False
        fp.enable_fastpath()  # raises if gambit_core is not importable
        assert fp.fastpath_enabled() is True
        fp.disable_fastpath()
        assert fp.fastpath_enabled() is False

    def test_enable_without_gambit_core_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(sys, "modules", {**sys.modules, "gambit_core": None})
        monkeypatch.setattr(fp, "_gambit_core", None)
        with pytest.raises(ImportError, match="maturin develop"):
            fp.enable_fastpath()
        assert fp.fastpath_enabled() is False  # gate stays OFF after failure


# ---------------------------------------------------------------------------
# 2. Rust legal-move order parity (differential fuzz)
# ---------------------------------------------------------------------------


class TestRustOrderParity:
    def test_order_matches_python_chess(self):
        gc = fp.import_gambit_core()
        rng = random.Random(20261121)
        positions = _walk_positions(START_FENS + CRAFTED_FENS, 450, rng)

        in_check = 0
        promotions = 0
        mismatches = 0
        for fen in positions:
            board = chess.Board(fen)
            if board.is_check():
                in_check += 1
            expected = [(encode_move(board, m), m.uci()) for m in board.legal_moves]
            actual = gc.legal_move_pairs(fen)
            if expected != actual:
                mismatches += 1
                if mismatches <= 3:
                    print(f"ORDER MISMATCH: {fen}\n  py  ={expected[:8]}\n  rust={actual[:8]}")
            promotions += sum(1 for _, uci in actual if len(uci) == 5)

        assert mismatches == 0, f"{mismatches} order mismatches over {len(positions)} positions"
        assert len(positions) >= 3000, "fuzz must cover thousands of positions"
        assert in_check > 50, "fuzz must cover checked positions"
        assert promotions > 0, "fuzz must cover promotions"


# ---------------------------------------------------------------------------
# 3. encode_batch FEN bridge
# ---------------------------------------------------------------------------


class TestEncodeBatchBridge:
    def test_bridge_bit_exact_including_raw_ep(self):
        boards = [chess.Board(f) for f in CRAFTED_FENS]
        fp.enable_fastpath()
        fast = encode_batch(boards)
        fp.disable_fastpath()
        slow = encode_batch(boards)
        assert fast.dtype == slow.dtype == np.float32
        assert fast.shape == slow.shape
        assert np.array_equal(fast, slow), "encode_batch fast path diverged (ep bridge?)"

    def test_bridge_skipped_when_history_present(self):
        boards = [chess.Board(), chess.Board()]
        prevs = [None, chess.Board()]
        fp.enable_fastpath()
        fast = encode_batch(boards, prev_boards=prevs)
        fp.disable_fastpath()
        slow = encode_batch(boards, prev_boards=prevs)
        assert np.array_equal(fast, slow)


# ---------------------------------------------------------------------------
# 4. legal_move_map parity
# ---------------------------------------------------------------------------


class TestLegalMoveMap:
    def test_keys_order_and_moves_match(self):
        rng = random.Random(777)
        positions = _walk_positions(START_FENS + CRAFTED_FENS, 60, rng)
        for fen in positions:
            board = chess.Board(fen)
            slow = legal_move_map(board)  # gate OFF (autouse fixture)
            fp.enable_fastpath()
            fast = legal_move_map(board)
            fp.disable_fastpath()
            assert list(fast.keys()) == list(slow.keys()), fen
            assert all(fast[k].uci() == m.uci() for k, m in slow.items()), fen


# ---------------------------------------------------------------------------
# 5. FastBoard claim semantics (differential fuzz vs plain Board)
# ---------------------------------------------------------------------------


class TestFastBoardClaims:
    def test_crafted_edge_positions(self):
        for fen in CRAFTED_FENS:
            assert _claim_snapshot(FastBoard(fen)) == _claim_snapshot(chess.Board(fen)), fen

    def test_copy_preserves_type_stack_and_claims(self):
        # MCTSSearch._simulate runs on board.copy(); the copy must stay a
        # FastBoard with the full stack so the overrides keep engaging.
        fast = FastBoard()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
            fast.push(chess.Move.from_uci(uci))
        clone = fast.copy()
        plain = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
            plain.push(chess.Move.from_uci(uci))
        assert isinstance(clone, FastBoard)
        assert len(clone.move_stack) == len(fast.move_stack) == 6
        assert len(clone._stack) == 6
        assert _claim_snapshot(clone) == _claim_snapshot(fast) == _claim_snapshot(plain)

    def test_knight_shuttle_repetition_sequence(self):
        # Pure reversible knight shuffling: startpos-wtm occurs 3x by ply 8.
        moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]
        fast, plain = FastBoard(), chess.Board()
        threefold_true = 0
        for uci in moves:
            mv = chess.Move.from_uci(uci)
            fast.push(mv)
            plain.push(mv)
            sf, sp = _claim_snapshot(fast), _claim_snapshot(plain)
            assert sf == sp, f"divergence after {uci}: {sf} vs {sp}"
            threefold_true += sf[0]
        assert fast.can_claim_threefold_repetition() is True
        assert threefold_true >= 1, "threefold claim branch not covered"

    def test_probe_only_threefold(self):
        # After 7 plies the current position occurred ONCE, but black's
        # g8f6 recreates a position seen twice -> claimable via the
        # probe branch, while is_repetition(3) stays False.
        moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]
        fast, plain = FastBoard(), chess.Board()
        for uci in moves:
            fast.push(chess.Move.from_uci(uci))
            plain.push(chess.Move.from_uci(uci))
        assert fast.is_repetition(3) is False
        assert fast.can_claim_threefold_repetition() is True
        assert _claim_snapshot(fast) == _claim_snapshot(plain)

    def test_random_walk_differential(self):
        rng = random.Random(11212026)
        mismatches: list[tuple] = []
        plies = 0
        threefold_true = 0
        for _ in range(18):
            start = rng.choice(START_FENS)
            fast, plain = FastBoard(start), chess.Board(start)
            for _ in range(120):
                moves = list(fast.generate_legal_moves())
                if not moves or fast.is_game_over(claim_draw=True):
                    break
                # Bias toward reversible piece moves so repetition cycles
                # (and therefore repetition CLAIMS) occur naturally.
                piece_moves = [m for m in moves if fast.piece_type_at(m.from_square) != chess.PAWN]
                mv = rng.choice(piece_moves if piece_moves and rng.random() < 0.85 else moves)
                fast.push(mv)
                plain.push(mv)
                plies += 1
                sf, sp = _claim_snapshot(fast), _claim_snapshot(plain)
                if sf != sp:
                    mismatches.append((fast.fen(), sf, sp))
                threefold_true += sf[0]
        assert not mismatches, mismatches[:3]
        assert plies > 800
        # Threefold-TRUE coverage lives in the crafted shuttle tests above;
        # this walk covers exactness over diverse midgame states.


# ---------------------------------------------------------------------------
# 6. end-to-end bit-exactness (off vs on, same seeds)
# ---------------------------------------------------------------------------


class TestEndToEndBitExact:
    POSITIONS: typing.ClassVar[list[str]] = [
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        MATE_IN_1_FEN,
    ]

    @staticmethod
    def _assert_results_equal(off, on, label: str) -> None:
        assert np.array_equal(off.visit_counts, on.visit_counts), label
        assert off.root_value == on.root_value, label
        assert off.chosen_move == on.chosen_move, label
        assert off.visited_root == on.visited_root, label

    def test_mcts_search_bit_exact_with_root_noise(self):
        rng_seed = 424242
        for fen in self.POSITIONS:
            fp.disable_fastpath()
            np.random.seed(rng_seed)
            off = MCTSSearch(StubEvaluator(_side_value), SearchConfig(num_simulations=24)).search(
                chess.Board(fen)
            )
            fp.enable_fastpath()
            np.random.seed(rng_seed)
            on = MCTSSearch(StubEvaluator(_side_value), SearchConfig(num_simulations=24)).search(
                FastBoard(fen)
            )
            fp.disable_fastpath()
            self._assert_results_equal(off, on, fen)

    def test_self_play_game_bit_exact(self, monkeypatch):
        torch = pytest.importorskip("torch")
        from training.model import MODEL_TINY, GambitNet

        torch.manual_seed(7)
        torch.set_num_threads(1)
        model = GambitNet(MODEL_TINY)
        model.eval()
        evaluator = BatchNNEvaluator(model)
        config = SelfPlayConfig(
            search=SearchConfig(num_simulations=12),
            max_plies=36,
            temperature=1.0,
            temperature_plies=12,
        )

        def run():
            np.random.seed(11)
            rng = np.random.default_rng(11)
            return self_play_game(evaluator, config, rng=rng)

        fp.disable_fastpath()
        off = run()

        # Prove the ON arm actually wires FastBoard into the game loop.
        created: list[FastBoard] = []

        class RecordingFastBoard(FastBoard):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                created.append(self)

        monkeypatch.setattr("training.selfplay.FastBoard", RecordingFastBoard)
        fp.enable_fastpath()
        on = run()
        fp.disable_fastpath()

        assert created, "FastBoard was not used by self_play_game with the gate ON"
        assert on.moves == off.moves
        assert on.result == off.result
        assert on.ply_count == off.ply_count
        assert on.resigned_side == off.resigned_side
        assert len(on.examples) == len(off.examples) > 0
        for ex_on, ex_off in zip(on.examples, off.examples):
            assert np.array_equal(ex_on.planes, ex_off.planes)
            assert np.array_equal(ex_on.policy, ex_off.policy)
            assert ex_on.value == ex_off.value
