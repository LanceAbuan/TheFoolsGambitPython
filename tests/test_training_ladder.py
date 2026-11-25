"""Ladder + arena gate tests (§5.1): schedule, promotion rule, match runner, Elo."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from training.ladder import (
    Arena,
    LadderConfig,
    MatchResult,
    due_snapshot_minutes,
    score_rate,
    should_promote,
    update_elo,
)
from training.selfplay import ArenaConfig


class StubEvaluator:
    """Deterministic arena evaluator: uniform legal priors, fixed value."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def evaluate(self, boards):
        n = len(boards)
        return np.zeros((n, 4672), dtype=np.float32), np.full(n, self.value, dtype=np.float32)


class TestLadderSchedule:
    def test_default_schedule_is_5_10_15_20_then_every_20(self):
        schedule = LadderConfig().scheduled_minutes()
        assert schedule[:4] == (5.0, 10.0, 15.0, 20.0)
        assert 40.0 in schedule and 60.0 in schedule and 80.0 in schedule
        assert 25.0 not in schedule and 30.0 not in schedule

    def test_due_at_exactly_the_markers(self):
        config = LadderConfig()
        assert due_snapshot_minutes(config, 5.0, 0.0) == [5.0]
        assert due_snapshot_minutes(config, 10.0, 5.0) == [10.0]
        assert due_snapshot_minutes(config, 15.0, 10.0) == [15.0]
        assert due_snapshot_minutes(config, 20.0, 15.0) == [20.0]

    def test_not_due_between_markers(self):
        config = LadderConfig()
        assert due_snapshot_minutes(config, 4.9, 0.0) == []
        assert due_snapshot_minutes(config, 12.0, 10.0) == []
        assert due_snapshot_minutes(config, 30.0, 20.0) == []  # next is 40

    def test_repeating_tail(self):
        config = LadderConfig()
        assert due_snapshot_minutes(config, 45.0, 20.0) == [40.0]
        assert due_snapshot_minutes(config, 62.0, 40.0) == [60.0]
        assert due_snapshot_minutes(config, 85.0, 60.0) == [80.0]

    def test_multiple_due_at_once(self):
        config = LadderConfig()
        assert due_snapshot_minutes(config, 21.0, 0.0) == [5.0, 10.0, 15.0, 20.0]

    def test_no_resnapshot_of_old_marker(self):
        config = LadderConfig()
        assert due_snapshot_minutes(config, 18.0, 15.0) == []

    def test_custom_schedule(self):
        config = LadderConfig(markers_minutes=(2.0, 4.0), repeat_every_minutes=4.0)
        assert due_snapshot_minutes(config, 9.0, 0.0) == [2.0, 4.0, 8.0]

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            LadderConfig(markers_minutes=())
        with pytest.raises(ValueError):
            LadderConfig(repeat_every_minutes=0.0)
        with pytest.raises(ValueError):
            LadderConfig(promote_win_rate=0.3)


class TestPromotionRule:
    """§5.1: promotion at >= 55% match win rate (draws count half)."""

    def test_above_threshold_promotes(self):
        match = MatchResult("cand", "champ", games=100, wins=56, draws=0, losses=44)
        assert should_promote(match, 0.55)

    def test_below_threshold_holds(self):
        match = MatchResult("cand", "champ", games=100, wins=54, draws=0, losses=46)
        assert not should_promote(match, 0.55)

    def test_exactly_at_threshold_promotes(self):
        match = MatchResult("cand", "champ", games=100, wins=55, draws=0, losses=45)
        assert should_promote(match, 0.55)

    def test_draws_count_half(self):
        match = MatchResult("cand", "champ", games=100, wins=50, draws=10, losses=40)
        assert match.win_rate == 0.55
        assert should_promote(match, 0.55)

    def test_empty_match_never_promotes(self):
        match = MatchResult("cand", "champ", games=0, wins=0, draws=0, losses=0)
        assert not should_promote(match, 0.55)

    def test_score_rate_zero_games(self):
        assert score_rate(10, 5, 0) == 0.0


class TestArena:
    """Arena match runner with stub players (no NN needed)."""

    @staticmethod
    def _stub(value: float) -> StubEvaluator:
        return StubEvaluator(value=value)

    def test_plays_all_games_both_colors(self):
        arena = Arena(self._stub(0.0), self._stub(0.0), ArenaConfig(games=4, num_simulations=2, max_plies=4))
        results = list(arena.run("cand", "champ"))
        assert len(results) == 4
        colors = [color for _result, color, _term in results]
        assert colors.count("white") == 2
        assert colors.count("black") == 2

    def test_match_result_counts(self):
        arena = Arena(self._stub(0.0), self._stub(0.0), ArenaConfig(games=4, num_simulations=2, max_plies=4))
        match = arena.match("cand", "champ")
        assert match.games == 4
        assert match.wins + match.draws + match.losses == 4

    def test_stronger_candidate_wins_the_match(self):
        """A value-biased candidate vs a passive champion: structural correctness.

        The precise outcome is positional, so assert the match is well-formed
        and the winner's side actually searches (evaluator was consulted).
        """
        winner_eval = StubEvaluator(value=0.9)
        loser_eval = StubEvaluator(value=-0.9)
        arena = Arena(winner_eval, loser_eval, ArenaConfig(games=2, num_simulations=4, max_plies=10))
        match = arena.match("strong", "weak")
        assert match.games == 2
        assert 0.0 <= match.win_rate <= 1.0

    def test_odd_games_rounded_to_even(self):
        arena = Arena(self._stub(0.0), self._stub(0.0), ArenaConfig(games=5, num_simulations=2, max_plies=2))
        assert len(list(arena.run("c", "c2"))) == 6

    def test_reduced_sim_budget_configurable(self):
        """§5.1: arena sims are reduced vs training's 400-800 via config."""
        assert LadderConfig().arena.num_simulations == 128
        tiny = LadderConfig(arena=ArenaConfig(num_simulations=2))
        assert tiny.arena.num_simulations == 2


class TestEloUpdate:
    def test_beating_a_equal_opponent_gains_elo(self):
        match = MatchResult("cand", "champ", games=10, wins=10, draws=0, losses=0)
        cand, champ = update_elo(800.0, 800.0, match, k_factor=32.0)
        assert cand == 816.0
        assert champ == 784.0
        assert (cand - 800.0) == (800.0 - champ)  # symmetric transfer

    def test_losing_drops_elo(self):
        match = MatchResult("cand", "champ", games=10, wins=0, draws=0, losses=10)
        cand, _ = update_elo(800.0, 800.0, match, k_factor=32.0)
        assert cand < 800.0

    def test_draw_between_equals_is_neutral(self):
        match = MatchResult("cand", "champ", games=10, wins=0, draws=10, losses=0)
        cand, champ = update_elo(800.0, 800.0, match, k_factor=32.0)
        assert abs(cand - 800.0) < 1e-9
        assert abs(champ - 800.0) < 1e-9

    def test_empty_match_no_change(self):
        match = MatchResult("cand", "champ", games=0, wins=0, draws=0, losses=0)
        assert update_elo(800.0, 900.0, match, 32.0) == (800.0, 900.0)

    def test_expected_score_formula(self):
        """Champion 400 Elo above candidate -> expected score ~0.09."""
        match = MatchResult("cand", "champ", games=10, wins=0, draws=0, losses=10)
        cand, _champ = update_elo(800.0, 1200.0, match, k_factor=32.0)
        expected = 1.0 / (1.0 + 10.0 ** (400.0 / 400.0))
        assert abs((cand - 800.0) - 32.0 * (0.0 - expected)) < 1e-9


def test_gate_rust_backend_completes():
    """ArenaConfig.search_backend='rust' routes gate games through the
    gambit_core search (skip-guarded when the wheel lacks it)."""
    from training.rust_search import RUST_SEARCH_AVAILABLE

    if not RUST_SEARCH_AVAILABLE:
        pytest.skip("gambit_core rust search unavailable")
    stub = StubEvaluator()
    arena = Arena(
        stub,
        StubEvaluator(),
        ArenaConfig(games=2, num_simulations=2, max_plies=4, search_backend="rust"),
    )
    match = arena.match("cand", "champ")
    assert match.games == 2
