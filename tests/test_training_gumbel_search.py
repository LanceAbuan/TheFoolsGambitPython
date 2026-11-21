"""U1 gumbel root-search tests (Danihelka et al., ICLR 2022).

Covers: mate finding (halving elimination respects sigma(Q-hat)), Eq. 11
improved-policy mass/mask, seeded determinism, NN-eval budget parity with
PUCT, candidate-set respect, degenerate budgets, arena integration, and the
config defaults (flag-off).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest

from training.encoder import encode_move, legal_move_map
from training.fastpath import disable_fastpath
from training.regimes import REGIMES, config_hash, get_regime
from training.selfplay import MCTSSearch, SearchConfig, SelfPlayConfig, play_match_game

MATE_IN_1_FEN = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"  # Rd8#
MATED_FEN = "3R2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"
SINGLE_MOVE_FEN = "7k/5K2/8/8/8/8/8/6R1 b - - 0 1"  # h8h7 only


class CountingStubEvaluator:
    """Uniform legal priors, constant (or fn-steered) value; counts NN calls."""

    def __init__(self, value: float = 0.0, value_fn=None) -> None:
        self.value = value
        self.value_fn = value_fn
        self.calls = 0

    def evaluate(self, boards):
        self.calls += 1
        n = len(boards)
        if self.value_fn is not None:
            values = np.array([self.value_fn(b) for b in boards], dtype=np.float32)
        else:
            values = np.full(n, self.value, dtype=np.float32)
        return np.zeros((n, 4672), dtype=np.float32), values


@pytest.fixture(autouse=True)
def _fastpath_off():
    disable_fastpath()
    yield


def gumbel_config(sims: int, candidates: int = 16, root_noise: bool = False) -> SearchConfig:
    return SearchConfig(
        num_simulations=sims,
        root_noise=root_noise,
        search_kind="gumbel",
        gumbel_candidates=candidates,
    )


class TestGumbelSearch:
    def test_finds_mate_in_1(self):
        """With all legal moves as candidates, gumbel must play the mate.

        This is also the sigma(Q-hat) elimination check: with uniform priors
        the gumbel+logits term is uninformative, so only the completed-Q part
        of the Eq. 8 score can keep the mating move alive through halving.
        """
        board = chess.Board(MATE_IN_1_FEN)
        result = MCTSSearch(
            CountingStubEvaluator(), gumbel_config(sims=64, candidates=32)
        ).search(board)
        assert result.chosen_move == chess.Move.from_uci("d1d8")
        mate_index = encode_move(board, chess.Move.from_uci("d1d8"))
        assert result.improved_policy is not None
        assert float(result.improved_policy[mate_index]) > 0.99

    def test_improved_policy_mass_and_mask(self):
        """Eq. 11 target: sums to 1, nonnegative, support == all legal moves."""
        board = chess.Board()
        result = MCTSSearch(CountingStubEvaluator(), gumbel_config(64)).search(board)
        assert result.improved_policy is not None
        policy = result.improved_policy
        assert policy.shape == (4672,)
        assert abs(float(policy.sum()) - 1.0) < 1e-5
        assert float(policy.min()) >= 0.0
        legal = set(legal_move_map(board).keys())
        nonzero = set(np.nonzero(policy)[0])
        assert nonzero <= legal
        assert len(nonzero) == len(legal)  # Eq. 11 covers ALL legal moves

    def test_seeded_determinism(self):
        board = chess.Board()
        first = MCTSSearch(
            CountingStubEvaluator(), gumbel_config(64), rng=np.random.default_rng(7)
        ).search(board)
        second = MCTSSearch(
            CountingStubEvaluator(), gumbel_config(64), rng=np.random.default_rng(7)
        ).search(board)
        assert first.chosen_move == second.chosen_move
        assert first.improved_policy is not None and second.improved_policy is not None
        assert np.array_equal(first.improved_policy, second.improved_policy)
        assert np.array_equal(first.visit_counts, second.visit_counts)

    def test_different_seed_changes_search(self):
        def value_fn(b: chess.Board) -> float:
            if not b.move_stack:
                return 0.0
            # Asymmetric leaf values: different seeds sample different gumbel
            # candidate sets, visit different leaves, and the completed-Q mix
            # (hence the Eq. 11 improved policy) differs across seeds.
            return 0.25 * ((b.peek().from_square % 5) - 2)

        board = chess.Board()
        a = MCTSSearch(
            CountingStubEvaluator(value_fn=value_fn), gumbel_config(64), rng=np.random.default_rng(1)
        ).search(board)
        b = MCTSSearch(
            CountingStubEvaluator(value_fn=value_fn), gumbel_config(64), rng=np.random.default_rng(2)
        ).search(board)
        assert a.improved_policy is not None and b.improved_policy is not None
        # The seed must actually steer the search (candidate sampling), not
        # just re-shuffle an identical result.
        assert not (
            np.array_equal(a.improved_policy, b.improved_policy)
            and np.array_equal(a.visit_counts, b.visit_counts)
        )

    def test_nn_budget_matches_puct(self):
        """1 root eval + `sims` leaf evals, exactly like the PUCT path."""
        board = chess.Board()
        gumbel_ev = CountingStubEvaluator()
        MCTSSearch(gumbel_ev, gumbel_config(64)).search(board)
        puct_ev = CountingStubEvaluator()
        MCTSSearch(puct_ev, SearchConfig(num_simulations=64, root_noise=False)).search(board)
        assert gumbel_ev.calls == 65
        assert puct_ev.calls == 65

    def test_visits_stay_within_candidates(self):
        """Sequential halving must only simulate gumbel-sampled candidates."""
        board = chess.Board()
        result = MCTSSearch(
            CountingStubEvaluator(), gumbel_config(64, candidates=4)
        ).search(board)
        assert result.visit_counts.sum() == 64
        legal = legal_move_map(board)
        visited = {idx for idx in legal if result.visit_counts[idx] > 0}
        assert len(visited) <= 4

    def test_single_legal_move(self):
        board = chess.Board(SINGLE_MOVE_FEN)
        result = MCTSSearch(CountingStubEvaluator(), gumbel_config(16)).search(board)
        assert result.chosen_move == chess.Move.from_uci("h8h7")
        assert result.improved_policy is not None
        assert abs(float(result.improved_policy.sum()) - 1.0) < 1e-6
        assert float(result.improved_policy[encode_move(board, chess.Move.from_uci("h8h7"))]) == 1.0

    def test_tiny_budget(self):
        """Budget far below m degrades gracefully (no crash, legal action)."""
        board = chess.Board()
        ev = CountingStubEvaluator()
        result = MCTSSearch(ev, gumbel_config(2)).search(board)
        assert result.chosen_move in legal_move_map(board).values()
        assert result.improved_policy is not None
        assert abs(float(result.improved_policy.sum()) - 1.0) < 1e-5
        assert ev.calls == 3  # root + 2 sims, budget fully spent

    def test_terminal_root(self):
        board = chess.Board(MATED_FEN)
        ev = CountingStubEvaluator()
        result = MCTSSearch(ev, gumbel_config(16)).search(board)
        assert result.chosen_move is None
        assert result.root_value == -1.0
        assert result.visited_root is False
        assert ev.calls == 0

    def test_agrees_with_puct_at_large_budget(self):
        """Both searches find the steered best move at a large budget (P4)."""
        target = chess.Move.from_uci("e2e4")

        def value_fn(b: chess.Board) -> float:
            # Perspective-correct steering: every eval inside the e2e4 subtree
            # must say "the side responding to e2e4 is losing" FROM THAT
            # SIDE'S OWN PERSPECTIVE, so negamax backups agree at every depth
            # and root-perspective Q(e2e4) stays +0.9 under dilution. (Keying
            # every leaf to a constant sign is wrong: a depth-2 leaf's -0.9
            # means white is losing, which penalizes exactly the steered
            # move.) The root board has an empty move stack -> neutral.
            if not b.move_stack:
                return 0.0
            if b.move_stack[0] != target:
                return 0.0
            return -0.9 if b.turn == chess.BLACK else 0.9

        board = chess.Board()
        gumbel = MCTSSearch(
            CountingStubEvaluator(value_fn=value_fn), gumbel_config(512, candidates=32)
        ).search(board)
        puct = MCTSSearch(
            CountingStubEvaluator(value_fn=value_fn),
            SearchConfig(num_simulations=512, root_noise=False),
        ).search(board)
        assert gumbel.chosen_move == target
        assert puct.chosen_move == target


class TestArenaIntegration:
    def test_play_match_game_with_gumbel_side(self):
        """play_match_game accepts explicit per-side configs (U1 ablations)."""
        result, termination = play_match_game(
            CountingStubEvaluator(),
            CountingStubEvaluator(),
            num_simulations=8,
            max_plies=6,
            white_config=gumbel_config(8),
        )
        assert result in ("1-0", "0-1", "1/2-1/2")
        assert termination in ("game-over", "max-plies")


class TestConfigDefaults:
    def test_search_kind_defaults_off(self):
        assert SearchConfig().search_kind == "puct"
        assert SearchConfig().gumbel_candidates == 16
        assert SearchConfig().gumbel_c_visit == 50.0
        assert SearchConfig().gumbel_c_scale == 1.0

    def test_regimes_default_to_puct_and_hash_sensitive(self):
        for name in REGIMES:
            regime = get_regime(name)
            assert regime.search_kind == "puct", name
            assert regime.playout_randomization is False, name
        base = get_regime("scratch")
        tweaked = base.__class__(**{**base.to_dict(), "search_kind": "gumbel"})
        assert config_hash(base) != config_hash(tweaked)

    def test_regime_rejects_gumbel_with_playout_cap(self):
        """Regime-level guard mirrors the SelfPlayConfig U1xU2 restriction."""
        base = get_regime("scratch")
        fields = base.to_dict()
        with pytest.raises(ValueError, match="playout_randomization requires search_kind='puct'"):
            base.__class__(**{**fields, "search_kind": "gumbel", "playout_randomization": True})

    def test_playout_cap_defaults_off(self):
        config = SelfPlayConfig()
        assert config.playout_randomization is False
        assert config.playout_cap_full_prob == 0.25
        assert config.playout_cap_fast_fraction == 0.25
        assert config.playout_cap_fast_min == 2
        assert config.playout_cap_start_ply == 10
        assert config.playout_cap_record_fast is False
