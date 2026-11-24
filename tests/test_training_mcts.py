"""MCTS v2 tests: §10 sign bug regression (mate-in-1), PUCT, noise, temperature."""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest
import torch
from torch import nn

import gambit_core
from training.encoder import encode_move, legal_move_map
from training.model import ModelConfig
from training.rust_search import RUST_SEARCH_AVAILABLE, RustMCTSSearch
from training.selfplay import (
    MCTS,  # legacy class — §10 sign fix verified on it
    BatchNNEvaluator,
    MCTSSearch,
    SearchConfig,
    SelfPlayConfig,
    SelfPlayGame,  # legacy class — §10 resign fix verified on it
    _Node,
    build_search,
    play_match_game,
    select_move_from_visits,
    self_play_game,
    terminal_value,
)

MATE_IN_1_FEN = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"  # Rd8#
MATED_FEN = "3R2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"  # black to move, back-rank mated
SINGLE_MOVE_FEN = "7k/5K2/8/8/8/8/8/6R1 b - - 0 1"  # black has exactly one legal move: h8h7


class StubEvaluator:
    """Deterministic evaluator: uniform legal priors, configurable value."""

    def __init__(self, value: float = 0.0, value_fn=None) -> None:
        self.value = value
        self.value_fn = value_fn
        self.calls: list[chess.Board] = []

    def evaluate(self, boards):
        self.calls.extend(boards)
        n = len(boards)
        if self.value_fn is not None:
            values = np.array([self.value_fn(b) for b in boards], dtype=np.float32)
        else:
            values = np.full(n, self.value, dtype=np.float32)
        return np.zeros((n, 4672), dtype=np.float32), values


class TestTerminalValue:
    def test_mated_side_is_loss(self):
        """§10 sign bug: the mated side-to-move must get -1.0, never +1.0."""
        board = chess.Board(MATED_FEN)
        assert board.is_checkmate()
        assert terminal_value(board) == -1.0

    def test_white_mated_is_loss_for_white(self):
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assert board.is_checkmate()  # fool's mate: white is mated
        assert terminal_value(board) == -1.0

    def test_stalemate_is_draw(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        assert board.is_stalemate()
        assert terminal_value(board) == 0.0


class TestMCTSSearch:
    def test_mate_in_1_is_found(self):
        """§10 regression: from a mate-in-1 position, the search plays the mate."""
        board = chess.Board(MATE_IN_1_FEN)
        mate_move = chess.Move.from_uci("d1d8")
        assert mate_move in board.legal_moves

        evaluator = StubEvaluator(value=0.0)  # uniform priors, neutral values
        search = MCTSSearch(evaluator, SearchConfig(num_simulations=96, root_noise=True))
        result = search.search(board)

        mate_index = encode_move(board, mate_move)
        assert result.chosen_move == mate_move
        assert result.visit_counts[mate_index] == result.visit_counts.max()
        assert result.visit_counts.sum() == 96

    def test_mate_in_1_without_root_noise(self):
        board = chess.Board(MATE_IN_1_FEN)
        result = MCTSSearch(
            StubEvaluator(), SearchConfig(num_simulations=64, root_noise=False)
        ).search(board)
        assert result.chosen_move == chess.Move.from_uci("d1d8")

    def test_root_value_of_mated_child_is_win_for_root(self):
        """Backprop sign check: the mating child backs up as a WIN for root.

        The exact mean depends on how many early exploration sims visited
        other children first, so assert the mate dominates the root value.
        """
        board = chess.Board(MATE_IN_1_FEN)
        search = MCTSSearch(StubEvaluator(), SearchConfig(num_simulations=64, root_noise=False))
        result = search.search(board)
        assert 0.5 < result.root_value <= 1.0

    def test_visit_counts_respect_legality(self):
        board = chess.Board()
        result = MCTSSearch(StubEvaluator(), SearchConfig(num_simulations=16)).search(board)
        legal = legal_move_map(board)
        nonzero = set(np.nonzero(result.visit_counts)[0])
        assert nonzero <= set(legal)

    def test_forced_playouts_prune_target(self):
        """WP1: forced playouts visit more root moves, and the exported target
        subtracts that forced share (pruned sum < raw times)."""
        board = chess.Board()
        off = MCTSSearch(
            StubEvaluator(), SearchConfig(num_simulations=64, root_noise=False)
        ).search(board)
        on = MCTSSearch(
            StubEvaluator(),
            SearchConfig(num_simulations=64, root_noise=False, forced_playouts=True),
        ).search(board)
        assert float(off.visit_counts.sum()) == 64.0
        assert float(on.visit_counts.sum()) < 64.0
        assert on.chosen_move == off.chosen_move

    def test_root_noise_spreads_over_legal_moves_only(self):
        """Dirichlet priors: sum to ~1 across children, all children legal."""
        board = chess.Board()
        legal = legal_move_map(board)
        search = MCTSSearch(StubEvaluator(), SearchConfig(root_noise=True))
        root = _Node(move=None, prior=0.0)
        search._evaluate_and_expand(root, board.copy(), add_root_noise=True)
        assert len(root.children) == len(legal)
        assert {c.move for c in root.children} == set(legal.values())
        assert abs(sum(c.prior for c in root.children) - 1.0) < 1e-9
        # uniform priors + noise: no single child swallows the distribution
        assert max(c.prior for c in root.children) < 0.9

    def test_no_noise_leaks_when_disabled(self):
        """With root_noise=False the priors are exactly uniform."""
        board = chess.Board()
        search = MCTSSearch(StubEvaluator(), SearchConfig(root_noise=False))
        root = _Node(move=None, prior=0.0)
        search._evaluate_and_expand(root, board.copy(), add_root_noise=False)
        expected = 1.0 / board.legal_moves.count()
        assert all(abs(c.prior - expected) < 1e-9 for c in root.children)

    def test_terminal_root_needs_no_evaluation(self):
        board = chess.Board(MATED_FEN)
        evaluator = StubEvaluator()
        result = MCTSSearch(evaluator, SearchConfig(num_simulations=16)).search(board)
        assert result.chosen_move is None
        assert result.root_value == -1.0
        assert result.visited_root is False
        assert evaluator.calls == []  # no NN eval needed

    def test_search_picks_winning_value_with_biased_evaluator(self):
        """A value head that punishes one continuation steers visits there.

        Leaf values are from the leaf side-to-move's perspective: after white
        plays e2e4 the evaluator sees black to move, so a value of -0.9 means
        white loves the move and it must dominate visit counts.
        """
        board = chess.Board()
        target = chess.Move.from_uci("e2e4")

        def value_fn(b: chess.Board) -> float:
            return -0.9 if b.move_stack and b.move_stack[-1] == target else 0.0

        evaluator = StubEvaluator(value_fn=value_fn)
        result = MCTSSearch(
            evaluator, SearchConfig(num_simulations=48, root_noise=False)
        ).search(board)
        assert result.visit_counts[encode_move(board, target)] == result.visit_counts.max()

    def test_single_forced_move_shortcut(self):
        board = chess.Board(SINGLE_MOVE_FEN)
        legal = list(board.legal_moves)
        assert len(legal) == 1
        result = MCTSSearch(StubEvaluator(), SearchConfig(num_simulations=10)).search(board)
        assert result.chosen_move == legal[0]
        assert result.visit_counts[encode_move(board, legal[0])] == 10.0


class TestTemperatureSampling:
    def test_zero_temperature_is_argmax(self):
        board = chess.Board()
        legal = legal_move_map(board)
        visits = np.zeros(4672, dtype=np.float32)
        first = next(iter(legal))
        visits[first] = 5.0
        assert select_move_from_visits(visits, legal, 0.0) == legal[first]

    def test_temperature_one_roughly_proportional(self):
        rng = np.random.default_rng(7)
        board = chess.Board()
        legal = legal_move_map(board)
        keys = list(legal)
        visits = np.zeros(4672, dtype=np.float32)
        visits[keys[0]] = 900.0
        visits[keys[1]] = 100.0
        picks = [select_move_from_visits(visits, legal, 1.0, rng) for _ in range(2000)]
        share_first = picks.count(legal[keys[0]]) / len(picks)
        assert 0.80 < share_first < 0.98  # ~90% with sampling noise

    def test_all_zero_visits_falls_back_to_a_legal_move(self):
        board = chess.Board()
        legal = legal_move_map(board)
        move = select_move_from_visits(np.zeros(4672, dtype=np.float32), legal, 1.0)
        assert move in legal.values()


class TestLegacySignAndResignFixes:
    """§10 fixes applied to the legacy classes too (failing-then-passing)."""

    def test_legacy_mcts_evaluate_mate_is_loss(self):
        dummy_model = nn.Linear(1, 1)

        class FakeLegacyEvaluator:
            def get_eval(self, tensor):
                return np.zeros(4096, dtype=np.float32), 0.0

        legacy = MCTS(dummy_model, evaluator=FakeLegacyEvaluator())
        board = chess.Board(MATED_FEN)
        assert board.is_checkmate()
        # Pre-fix this returned +1.0 for the mated side (the §10 sign bug).
        assert legacy._evaluate(board) == -1.0

    def test_legacy_selfplay_resign_off_by_default(self):
        """A constantly losing value must NOT end the game (§10 resign fix)."""
        dummy_model = nn.Linear(1, 1)

        class LosingEvaluator:
            def get_eval(self, tensor):
                return np.zeros(4096, dtype=np.float32), -1.0

        game = SelfPlayGame(
            dummy_model, num_mcts_simulations=2, max_moves=4, evaluator=LosingEvaluator()
        )
        result = game.play()
        # Pre-fix: instant resign at move 0 -> 0 examples. Fixed: plays on.
        assert game.resigned_side is None
        assert len(result["examples"]) == 4
        assert len(result["moves"]) == 4

    def test_legacy_selfplay_resign_needs_three_strikes_past_ply_40(self):
        dummy_model = nn.Linear(1, 1)

        class LosingEvaluator:
            def get_eval(self, tensor):
                return np.zeros(4096, dtype=np.float32), -1.0

        game = SelfPlayGame(
            dummy_model, num_mcts_simulations=2, max_moves=60,
            evaluator=LosingEvaluator(), resign_enabled=True,
        )
        result = game.play()
        # First eligible ply is 40 (streak 1) -> strike 3 lands at ply 42 -> 41 moves.
        assert game.resigned_side is not None
        assert result["result"] in ("1-0", "0-1")
        assert 41 <= len(result["moves"]) <= 42


class TestSelfPlayDriver:
    def test_examples_labelled_from_mover_perspective(self):
        config = SelfPlayConfig(
            search=SearchConfig(num_simulations=4, root_noise=False),
            max_plies=6,
            temperature_plies=0,  # argmax throughout
        )
        result = self_play_game(StubEvaluator(value=0.0), config, rng=np.random.default_rng(3))
        assert len(result.examples) == len(result.moves) == result.ply_count
        assert 0 < result.ply_count <= 6
        for example in result.examples:
            assert example.planes.shape == (29, 8, 8)
            assert example.policy.shape == (4672,)
            assert example.value in (-1.0, 0.0, 1.0)
        assert result.result in ("1-0", "0-1", "1/2-1/2")
        assert "1." in result.pgn

    def test_policy_targets_are_distributions_over_4672(self):
        config = SelfPlayConfig(
            search=SearchConfig(num_simulations=8, root_noise=False),
            max_plies=4,
            temperature_plies=0,
        )
        result = self_play_game(StubEvaluator(), config, rng=np.random.default_rng(11))
        for example in result.examples:
            assert abs(float(example.policy.sum()) - 1.0) < 1e-5
            assert (example.policy >= 0).all()

    def test_play_match_game_both_sides_move(self):
        white_eval = StubEvaluator(value=0.0)
        black_eval = StubEvaluator(value=0.0)
        result, termination = play_match_game(white_eval, black_eval, num_simulations=2, max_plies=6)
        assert result in ("1-0", "0-1", "1/2-1/2")
        assert termination == "max-plies"  # 6 plies cannot finish a real game
        assert len(white_eval.calls) > 0 and len(black_eval.calls) > 0

    def test_match_game_respects_max_plies_and_finishes_games(self):
        """A position with a forced mate finishes with the right result."""
        # Start white up a rook with mate in 1 available on move 1: not possible
        # from startpos; instead verify a full game caps out cleanly at 200 plies.
        result, termination = play_match_game(
            StubEvaluator(), StubEvaluator(), num_simulations=1, max_plies=200
        )
        assert result in ("1-0", "0-1", "1/2-1/2")
        assert termination in ("game-over", "max-plies")


class TestWDLValueThroughSearch:
    """WP-C: a wdl value head feeds MCTS the derived p_win - p_loss scalar.

    The search interface is unchanged: `Evaluator.evaluate` still returns
    (logits (N, 4672), values (N,)) — the derivation lives in
    `BatchNNEvaluator` via `training.model.value_head_scalar`.
    """

    WDL_TINY = ModelConfig(num_blocks=1, filters=16, value_channels=8, value_fc=32, value_head="wdl")

    def test_batch_nn_evaluator_returns_scalar_for_wdl_model(self):
        wdl_net = self._wdl_net()
        evaluator = BatchNNEvaluator(wdl_net, torch.device("cpu"))
        logits, values = evaluator.evaluate([chess.Board(), chess.Board()])
        assert logits.shape == (2, 4672)
        assert values.shape == (2,)
        # p_win - p_loss is in [-1, 1] by construction.
        assert all(-1.0 <= float(v) <= 1.0 for v in values)

    def test_batch_nn_evaluator_wdl_scalar_matches_hand_derivation(self):
        """The evaluator's scalar equals softmax(W) - softmax(L) of the net."""
        wdl_net = self._wdl_net()
        wdl_net.eval()
        board = chess.Board()
        evaluator = BatchNNEvaluator(wdl_net, torch.device("cpu"))
        _logits, values = evaluator.evaluate([board])
        from training.encoder import encode_batch

        batch = torch.from_numpy(encode_batch([board]))
        with torch.inference_mode():
            _policy, raw = wdl_net(batch)
        probs = torch.softmax(raw, dim=-1)[0]
        expected = float(probs[0] - probs[2])
        assert float(values[0]) == pytest.approx(expected, abs=1e-6)

    def test_mcts_search_with_wdl_model_is_unchanged_interface(self):
        """A full search over a wdl model: legal move chosen, visits sum right."""
        evaluator = BatchNNEvaluator(self._wdl_net(), torch.device("cpu"))
        board = chess.Board()
        result = MCTSSearch(evaluator, SearchConfig(num_simulations=16, root_noise=False)).search(board)
        assert result.chosen_move in legal_move_map(board).values()
        assert result.visit_counts.sum() == 16
        legal = set(legal_move_map(board))
        assert set(np.nonzero(result.visit_counts)[0]) <= legal

    def test_mate_in_1_found_with_wdl_model(self):
        """§10 regression on the wdl path: the mate is still the top move."""
        board = chess.Board(MATE_IN_1_FEN)
        evaluator = BatchNNEvaluator(self._wdl_net(), torch.device("cpu"))
        result = MCTSSearch(evaluator, SearchConfig(num_simulations=48, root_noise=False)).search(board)
        mate_index = encode_move(board, chess.Move.from_uci("d1d8"))
        assert result.visit_counts[mate_index] == result.visit_counts.max()

    def test_play_match_game_with_wdl_models(self):
        white = BatchNNEvaluator(self._wdl_net(), torch.device("cpu"))
        black = BatchNNEvaluator(self._wdl_net(), torch.device("cpu"))
        result, termination = play_match_game(white, black, num_simulations=2, max_plies=6)
        assert result in ("1-0", "0-1", "1/2-1/2")
        assert termination == "max-plies"

    def _wdl_net(self) -> torch.nn.Module:
        from training.model import GambitNet

        torch.manual_seed(42)
        return GambitNet(self.WDL_TINY)

# ---------------------------------------------------------------------------
# WP-F C3/C5/C6: the Rust search backend (gambit_core.search).
#
# The keyed stub below is the parity oracle: per-position deterministic
# (logits, value) derived from a stable hash of position-identity fields
# (no move-stack fields), so the python path (chess.Board with stack) and
# the rust bridge path (FEN -> chess.Board) see BIT-IDENTICAL evaluator
# outputs. That makes visit-count differences pure search-engine signals.
# ---------------------------------------------------------------------------


def _rust_search_available() -> bool:
    return RUST_SEARCH_AVAILABLE and hasattr(gambit_core, "search")


requires_rust_search = pytest.mark.skipif(
    not _rust_search_available(),
    reason="gambit_core wheel lacks the C3 search pyfunction (build + install the wheel)",
)


def _canon_key(board: chess.Board) -> tuple[str, bool, int, int | None]:
    """Position identity both bridge paths agree on (no move-stack fields)."""
    return (board.board_fen(), board.turn, board.castling_rights, board.ep_square)


def _keyed_logits_values(key: tuple) -> tuple[np.ndarray, float]:
    """Deterministic pseudo-random logits/value for one canonical position."""
    digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little")
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(4672).astype(np.float32)
    value = float(rng.uniform(-1.0, 1.0))
    return logits, value


def _keyed_eval_many(boards) -> tuple[np.ndarray, np.ndarray]:
    n = len(boards)
    logits = np.zeros((n, 4672), dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)
    for i, board in enumerate(boards):
        row, value = _keyed_logits_values(_canon_key(board))
        logits[i] = row
        values[i] = value
    return logits, values


class KeyedStubEvaluator:
    """Deterministic per-position evaluator (python Evaluator protocol)."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, boards):
        self.calls += len(boards)
        return _keyed_eval_many(boards)


def keyed_rust_callback(fens):
    """Rust bridge callback: FENs -> keyed (logits, values) numpy arrays."""
    return _keyed_eval_many([chess.Board(fen) for fen in fens])


class _UniformEval:
    """All-zero logits, zero value: exact structural-parity oracle."""

    def evaluate(self, boards):
        n = len(boards)
        return np.zeros((n, 4672), dtype=np.float32), np.zeros(n, dtype=np.float32)


def uniform_rust_callback(fens):
    """Rust bridge callback: all-zero logits/values (uniform priors)."""
    n = len(fens)
    return np.zeros((n, 4672), dtype=np.float32), np.zeros(n, dtype=np.float32)


@requires_rust_search
class TestRustSearchParity:
    """C6 gate (a): rust vs python in deterministic single-thread mode."""

    def test_uniform_stub_exact_parity(self):
        board = chess.Board()
        cfg = gambit_core.SearchConfig(num_simulations=64, root_noise=False)
        rust_counts, rust_value = gambit_core.search(chess.STARTING_FEN, uniform_rust_callback, cfg)
        py = MCTSSearch(_UniformEval(), SearchConfig(num_simulations=64, root_noise=False)).search(
            board.copy()
        )
        assert np.array_equal(np.asarray(rust_counts), py.visit_counts)
        assert float(rust_value) == pytest.approx(py.root_value, abs=1e-12)

    def test_keyed_stub_parity_on_fixed_positions(self):
        fens = [
            chess.STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ]
        for fen in fens:
            board = chess.Board(fen)
            py = MCTSSearch(
                KeyedStubEvaluator(), SearchConfig(num_simulations=96, root_noise=False)
            ).search(board.copy())
            rust_counts, rust_value = gambit_core.search(
                fen,
                keyed_rust_callback,
                gambit_core.SearchConfig(num_simulations=96, root_noise=False),
            )
            rust_counts = np.asarray(rust_counts)
            # fp exp ulps can flip knife-edge selections between the exp
            # implementations; distributions must stay near-identical.
            assert np.abs(rust_counts - py.visit_counts).max() <= 3, fen
            tv = float(0.5 * np.abs(rust_counts - py.visit_counts).sum()) / 96.0
            assert tv <= 0.05, fen
            # A few differing leaves shift the value MEAN; the gate is structural
            # drift (sign/terminal bugs move this by O(0.1)), not fp noise.
            assert float(rust_value) == pytest.approx(py.root_value, abs=0.05), fen

    def test_repetition_claim_parity_with_game_history(self):
        """Two full knight shuffles: the root is claimable-terminal for BOTH engines.

        This is the regression for the C2 Seq seam bug the C6 gate caught:
        suffix_count stopped at the path root and never walked the game
        history, so rust kept searching a position python scores as a draw.
        """
        moves = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2
        board = chess.Board()
        for uci in moves:
            board.push_uci(uci)
        assert board.can_claim_threefold_repetition()
        py = MCTSSearch(
            KeyedStubEvaluator(), SearchConfig(num_simulations=8, root_noise=False)
        ).search(board.copy())
        assert py.visited_root is False
        assert py.chosen_move is None
        assert py.root_value == 0.0
        rust_counts, rust_value = gambit_core.search(
            (chess.STARTING_FEN, moves),
            keyed_rust_callback,
            gambit_core.SearchConfig(num_simulations=8, root_noise=False),
        )
        assert float(np.asarray(rust_counts).sum()) == 0.0
        assert float(rust_value) == 0.0

    def test_wave_mode_produces_valid_distribution(self):
        cfg = gambit_core.SearchConfig(num_simulations=64, batch_size=8, root_noise=False)
        counts, _value = gambit_core.search(chess.STARTING_FEN, keyed_rust_callback, cfg)
        counts = np.asarray(counts)
        assert counts.sum() == 64.0
        legal = set(legal_move_map(chess.Board()))
        assert set(np.nonzero(counts)[0].tolist()) <= legal

    def test_mated_root_and_forced_move(self):
        counts, value = gambit_core.search(
            MATED_FEN, keyed_rust_callback, gambit_core.SearchConfig(num_simulations=8)
        )
        assert np.asarray(counts).sum() == 0.0
        assert float(value) == -1.0
        counts, _value = gambit_core.search(
            SINGLE_MOVE_FEN, keyed_rust_callback, gambit_core.SearchConfig(num_simulations=10)
        )
        counts = np.asarray(counts)
        assert counts.sum() == 10.0
        legal = set(legal_move_map(chess.Board(SINGLE_MOVE_FEN)))
        assert set(np.nonzero(counts)[0].tolist()) <= legal

    def test_callback_exception_propagates(self):
        def boom(fens):
            raise RuntimeError("evaluator exploded")

        with pytest.raises(RuntimeError, match="evaluator exploded"):
            gambit_core.search(
                chess.STARTING_FEN, boom, gambit_core.SearchConfig(num_simulations=4)
            )

    def test_callback_shape_mismatch_raises(self):
        def bad_cb(fens):
            return np.zeros((len(fens), 10), dtype=np.float32), np.zeros(len(fens), dtype=np.float32)

        with pytest.raises(ValueError):
            gambit_core.search(
                chess.STARTING_FEN, bad_cb, gambit_core.SearchConfig(num_simulations=4)
            )

    def test_bad_board_argument_raises(self):
        with pytest.raises(ValueError):
            gambit_core.search(123, keyed_rust_callback, gambit_core.SearchConfig())  # type: ignore[arg-type]


@requires_rust_search
class TestRustBackendWrapper:
    """C3 wrapper + C5 routing through the python pipeline entry points."""

    def test_build_search_routes_by_flag(self):
        evaluator = KeyedStubEvaluator()
        assert isinstance(build_search(evaluator, SearchConfig()), MCTSSearch)
        rust = build_search(evaluator, SearchConfig(search_backend="rust"))
        assert isinstance(rust, RustMCTSSearch)
        seeded = build_search(
            evaluator, SearchConfig(search_backend="rust"), rng=np.random.default_rng(0)
        )
        assert isinstance(seeded, RustMCTSSearch)

    def test_search_config_validation(self):
        with pytest.raises(ValueError):
            SearchConfig(search_backend="gpu")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            SearchConfig(search_backend="rust", search_kind="gumbel")

    def test_rust_backend_unavailable_raises_runtime_error(self, monkeypatch):
        import training.rust_search as rust_search_module

        monkeypatch.setattr(rust_search_module, "RUST_SEARCH_AVAILABLE", False)
        with pytest.raises(RuntimeError):
            build_search(KeyedStubEvaluator(), SearchConfig(search_backend="rust"))

    def test_wrapper_parity_matches_python(self):
        fens = [
            chess.STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ]
        for fen in fens:
            board = chess.Board(fen)
            py = MCTSSearch(
                KeyedStubEvaluator(), SearchConfig(num_simulations=96, root_noise=False)
            ).search(board.copy())
            rust = RustMCTSSearch(
                KeyedStubEvaluator(), SearchConfig(num_simulations=96, root_noise=False)
            ).search(board.copy())
            assert np.abs(rust.visit_counts - py.visit_counts).max() <= 3, fen
            assert rust.chosen_move == py.chosen_move, fen
            assert rust.visited_root == py.visited_root, fen
            assert rust.root_value == pytest.approx(py.root_value, abs=0.05), fen

    def test_wrapper_noise_is_seeded_and_deterministic(self):
        board = chess.Board()
        one = RustMCTSSearch(
            KeyedStubEvaluator(),
            SearchConfig(num_simulations=32, root_noise=True),
            rng=np.random.default_rng(17),
        ).search(board.copy())
        two = RustMCTSSearch(
            KeyedStubEvaluator(),
            SearchConfig(num_simulations=32, root_noise=True),
            rng=np.random.default_rng(17),
        ).search(board.copy())
        assert np.array_equal(one.visit_counts, two.visit_counts)

    def test_wrapper_forced_move_and_terminal_root(self):
        rust = RustMCTSSearch(KeyedStubEvaluator(), SearchConfig(num_simulations=10))
        forced = rust.search(chess.Board(SINGLE_MOVE_FEN))
        assert forced.chosen_move == chess.Move.from_uci("h8h7")
        assert forced.visit_counts.sum() == 10.0
        assert forced.visited_root is True
        mated = rust.search(chess.Board(MATED_FEN))
        assert mated.chosen_move is None
        assert mated.visited_root is False
        assert mated.root_value == -1.0
        assert mated.visit_counts.sum() == 0.0

    def test_self_play_game_with_rust_backend(self):
        config = SelfPlayConfig(
            search=SearchConfig(
                num_simulations=4, root_noise=False, search_backend="rust"
            ),
            max_plies=6,
            temperature_plies=0,
        )
        result = self_play_game(KeyedStubEvaluator(), config, rng=np.random.default_rng(5))
        assert result.moves
        assert result.ply_count == len(result.moves)
        replay = chess.Board()
        for san in result.moves:
            replay.push_san(san)  # raises on any illegal move

    def test_play_match_game_with_rust_backend(self):
        white = SearchConfig(root_noise=False, num_simulations=2, search_backend="rust")
        black = SearchConfig(root_noise=False, num_simulations=2, search_backend="rust")
        result, termination = play_match_game(
            KeyedStubEvaluator(),
            KeyedStubEvaluator(),
            num_simulations=2,
            max_plies=6,
            white_config=white,
            black_config=black,
        )
        assert result in ("1-0", "0-1", "1/2-1/2")
        assert termination == "max-plies"

