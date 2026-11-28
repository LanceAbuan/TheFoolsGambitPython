"""U9 tests: Glicko-2 math, SPRT boundaries, promotion decision, ledger replay.

Pins (RESEARCH_STANDARD principle 1 + 3):
  * the Glicko-2 update against Glickman's worked example, exact to the
    paper's two decimals (http://www.glicko.net/glicko/glicko2.pdf);
  * SPRT [0, 10] alpha=beta=0.05 boundary behavior (streak accepts H1,
    coin-flip never accepts H1, forced losses accept H0, all-draws are
    zero-information);
  * the U9 promotion decision table (training.ladder.decide_promotion);
  * sequential early-stop: a provable no-op at 16-game gates, active at
    tight bounds, even-game-aligned;
  * ledger replay tolerance for OLD rows (pre-U9, elo float/null, no kind)
    and NEW rows (glicko_r/glicko_rd additive) in one fixture, including
    tonight's real second-light gate history;
  * resume: state files written before U9 (no champion_glicko key) restore.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import pytest

from training.glicko import (
    DEFAULT_RATING,
    DEFAULT_RD,
    GlickoRating,
    expected_games_to_accept,
    expected_score,
    match_honesty,
    rate_inactive,
    rate_player,
)
from training.ladder import (
    Arena,
    LadderConfig,
    MatchResult,
    decide_promotion,
    update_match_pair,
)
from training.ledger import LedgerEntry, recompute_elos, recompute_glickos
from training.resume import restore_run_state
from training.selfplay import ArenaConfig
from training.sprt import sprt_bounds, sprt_state, trinomial_llr


class TestGlickmanWorkedExample:
    """Glicko-2 paper §5 example, exact numbers.

    Player r=1500 RD=200 sigma=0.06, tau=0.5; opponents (1400, RD 30, win),
    (1550, RD 100, loss), (1700, RD 300, loss). Paper results:
    r' = 1464.06, RD' = 151.52, sigma' = 0.05999.
    """

    def test_paper_example_exact(self):
        player = GlickoRating(r=1500.0, rd=200.0, sigma=0.06)
        results = [
            (GlickoRating(r=1400.0, rd=30.0), 1.0),
            (GlickoRating(r=1550.0, rd=100.0), 0.0),
            (GlickoRating(r=1700.0, rd=300.0), 0.0),
        ]
        updated = rate_player(player, results)
        assert updated.r == pytest.approx(1464.06, abs=0.01)
        assert updated.rd == pytest.approx(151.52, abs=0.01)
        assert updated.sigma == pytest.approx(0.05999, abs=0.0005)

    def test_initial_constants_are_the_paper_defaults(self):
        fresh = GlickoRating()
        assert fresh.r == 1500.0
        assert fresh.rd == 350.0
        assert fresh.sigma == 0.06


class TestGlickoBehavior:
    def test_rd_decays_after_inactivity_and_r_is_unchanged(self):
        before = GlickoRating(r=1550.0, rd=120.0)
        after = rate_inactive(before)
        assert after.r == before.r
        assert after.rd > before.rd
        ten_periods = before
        for _ in range(10):
            ten_periods = rate_inactive(ten_periods)
        assert ten_periods.rd > after.rd  # monotone widening
        assert ten_periods.rd < DEFAULT_RD  # self-limiting below the ceiling

    def test_rd_never_exceeds_the_ceiling(self):
        at_ceiling = GlickoRating(rd=DEFAULT_RD)
        assert rate_inactive(at_ceiling).rd == pytest.approx(DEFAULT_RD, abs=0.5)

    def test_beating_a_higher_rated_opponent_raises_r(self):
        player = GlickoRating(rd=200.0)
        updated = rate_player(player, [(GlickoRating(r=1700.0, rd=50.0), 1.0)])
        assert updated.r > player.r

    def test_losing_to_a_lower_rated_opponent_lowers_r(self):
        player = GlickoRating(r=1650.0, rd=120.0)
        updated = rate_player(player, [(GlickoRating(r=1450.0, rd=60.0), 0.0)])
        assert updated.r < player.r

    def test_draw_shrinkage_rd_falls_with_games(self):
        player = GlickoRating()
        one = rate_player(player, [(GlickoRating(), 0.5)])
        sixteen = rate_player(player, [(GlickoRating(), 0.5)] * 16)
        assert one.rd < player.rd
        assert sixteen.rd < one.rd  # more games -> tighter deviation
        assert sixteen.r == pytest.approx(DEFAULT_RATING, abs=1.0)  # equal opponent, all draws

    def test_empty_period_returns_rating_unchanged(self):
        rating = GlickoRating(r=1600.0, rd=90.0, sigma=0.05)
        assert rate_player(rating, []) == rating


class TestSPRT:
    def test_bounds_are_symmetric_at_alpha_equals_beta(self):
        lower, upper = sprt_bounds(0.05, 0.05)
        assert lower == pytest.approx(-2.9444, abs=1e-3)
        assert upper == pytest.approx(2.9444, abs=1e-3)
        assert lower == pytest.approx(-upper)

    def test_guaranteed_win_streak_accepts_h1(self):
        state = sprt_state(110, 0, 0)
        assert state.verdict == "accept_h1"

    def test_forced_losses_accept_h0(self):
        state = sprt_state(0, 0, 110)
        assert state.verdict == "accept_h0"

    def test_coin_flip_never_accepts_h1(self):
        # 10k games at exactly 50%: under H1 (elo1=10 -> 51.4% expected) this
        # is EVIDENCE AGAINST H1, never for it.
        state = sprt_state(5000, 0, 5000)
        assert state.verdict == "accept_h0"

    def test_all_draw_match_is_zero_information_and_undecided(self):
        state = sprt_state(0, 16, 0)
        assert state.llr == 0.0
        assert state.verdict == "undecided"

    def test_empty_sample_undecided(self):
        assert sprt_state(0, 0, 0).verdict == "undecided"

    def test_16_game_bounds_are_unreachable_no_matter_the_score(self):
        """Pre-registration arithmetic: at N <= 16 no W/D/L pattern can cross
        a [0,10] alpha=beta=0.05 bound (per-game |LLR| <= 0.0293), so the
        sequential stop is a no-op at tonight's gate size."""
        lower, upper = sprt_bounds(0.05, 0.05)
        for wins in range(17):
            for losses in range(17 - wins):
                llr = trinomial_llr(wins, 16 - wins - losses, losses)
                assert lower < llr < upper

    def test_tonights_promote_gate_llr(self):
        # 8 wins / 8 draws / 0 losses (the real promotion gate): positive but
        # far from the H1 bound — SPRT alone would NOT have promoted.
        state = sprt_state(8, 8, 0)
        assert state.llr == pytest.approx(0.4477, abs=0.01)
        assert state.verdict == "undecided"

    def test_expected_games_to_accept_known_points(self):
        # A perfect scorer needs ~104 games minimum (2.944 / ln(0.51439/0.5)).
        assert expected_games_to_accept(1.0, 0.0) == pytest.approx(103.8, abs=1.0)
        # Tonight's 0.75-at-50%-draws scorer: ~105 games (pre-registered).
        assert expected_games_to_accept(0.75, 0.5) == pytest.approx(105.2, abs=2.0)
        # A true +10 at no draws: ~7.2k games (Wald, KL ~ 0.00041/game) — the
        # honest reason SPRT cannot decide overnight gates.
        assert expected_games_to_accept(0.5144, 0.0) == pytest.approx(7180.0, rel=0.02)
        # Infeasible cells (draw rate incompatible with the score) -> None.
        assert expected_games_to_accept(0.75, 0.9) is None
        assert expected_games_to_accept(0.5, 1.0) is None


class TestSequentialEarlyStop:
    class StubEvaluator:
        def __init__(self, value: float) -> None:
            self.value = value

        def evaluate(self, boards):
            import numpy as np

            n = len(boards)
            return np.zeros((n, 4672), dtype=np.float32), np.full(n, self.value, dtype=np.float32)

    @staticmethod
    def _scripted_run(results: list[str]):
        """Deterministic Arena.run replacement.

        ``results`` are CANDIDATE-perspective outcomes ("1-0" = the candidate
        won). They are translated into actual game results by colour, because
        ``Arena.match`` scores from the candidate's side (2026-11-28 fix).
        """

        def run(self, candidate_tag: str, champion_tag: str):
            planned = self.config.games + (self.config.games % 2)
            for index in range(planned):
                outcome = results[index % len(results)]
                color = "white" if index % 2 == 0 else "black"
                if outcome == "1/2-1/2":
                    game = "1/2-1/2"
                else:
                    candidate_won = outcome == "1-0"
                    if color == "white":
                        game = "1-0" if candidate_won else "0-1"
                    else:
                        game = "0-1" if candidate_won else "1-0"
                yield game, color, "game-over"

        return run

    def test_stop_is_a_noop_at_default_gate_size(self, monkeypatch):
        """Default knobs: even an all-win 16-game match (max |LLR| = 16 x
        0.0284 = 0.454) cannot cross ln(19) = 2.944 — the stop is provably
        inert at tonight's gate size, so enabling it is behavior-preserving."""
        monkeypatch.setattr(Arena, "run", self._scripted_run(["1-0"]))
        arena = Arena(
            self.StubEvaluator(0.0),
            self.StubEvaluator(0.0),
            ArenaConfig(games=16, num_simulations=2, max_plies=4, sprt_early_stop=True),
        )
        assert arena.match("cand", "champ").games == 16

    def test_decisive_match_stops_early_on_even_games(self, monkeypatch):
        # Tight bounds (alpha=beta~0.4875 -> +-0.0502): two straight wins
        # (LLR 0.0567) cross the H1 bound and the match stops at the first
        # EVEN count, keeping the colors balanced at the stop point.
        monkeypatch.setattr(Arena, "run", self._scripted_run(["1-0"]))
        tight = ArenaConfig(
            games=16,
            num_simulations=2,
            max_plies=4,
            sprt_early_stop=True,
            sprt_alpha=0.48745,
            sprt_beta=0.48745,
        )
        match = Arena(self.StubEvaluator(0.0), self.StubEvaluator(0.0), tight).match("cand", "champ")
        assert match.games == 2
        assert match.wins == 2
        assert match.draws == 0

    def test_forced_losses_stop_at_h0_with_tight_bounds(self, monkeypatch):
        monkeypatch.setattr(Arena, "run", self._scripted_run(["0-1"]))
        tight = ArenaConfig(
            games=16,
            num_simulations=2,
            max_plies=4,
            sprt_early_stop=True,
            sprt_alpha=0.48745,
            sprt_beta=0.48745,
        )
        match = Arena(self.StubEvaluator(0.0), self.StubEvaluator(0.0), tight).match("cand", "champ")
        assert match.games == 2
        assert match.losses == 2

    def test_stop_disabled_by_default(self):
        arena = Arena(
            self.StubEvaluator(0.9),
            self.StubEvaluator(-0.9),
            ArenaConfig(games=4, num_simulations=2, max_plies=4),
        )
        assert arena.match("cand", "champ").games == 4


class TestPromotionDecision:
    CONFIG = LadderConfig()  # threshold 0.55, rd cap 150, SPRT [0,10] 0.05/0.05

    def test_decision_table_threshold_with_rd_ok_promotes(self):
        match = MatchResult("c", "h", games=16, wins=8, draws=8, losses=0)
        candidate = GlickoRating(rd=121.0)  # post-16-game RD
        decision = decide_promotion(match, candidate, self.CONFIG)
        assert decision.promote is True
        assert decision.reason == "threshold_with_rd_ok"

    def test_decision_table_rd_above_cap_holds_even_at_threshold(self):
        match = MatchResult("c", "h", games=4, wins=4, draws=0, losses=0)
        candidate = GlickoRating(rd=208.0)  # post-4-game RD
        decision = decide_promotion(match, candidate, self.CONFIG)
        assert decision.promote is False
        assert decision.reason == "hold_rd_above_cap"

    def test_decision_table_sprt_accept_h1_promotes(self):
        # 110 straight wins: LLR crosses the H1 bound even though this
        # synthetic candidate's RD is above the cap — SPRT path is an OR.
        match = MatchResult("c", "h", games=110, wins=110, draws=0, losses=0)
        decision = decide_promotion(match, GlickoRating(rd=200.0), self.CONFIG)
        assert decision.promote is True
        assert decision.reason == "sprt_accept_h1"

    def test_decision_table_sprt_accept_h0_holds(self):
        match = MatchResult("c", "h", games=110, wins=0, draws=0, losses=110)
        decision = decide_promotion(match, GlickoRating(rd=100.0), self.CONFIG)
        assert decision.promote is False
        assert decision.reason == "hold_sprt_accept_h0"

    def test_decision_table_below_threshold_holds(self):
        match = MatchResult("c", "h", games=16, wins=0, draws=16, losses=0)
        decision = decide_promotion(match, GlickoRating(rd=121.0), self.CONFIG)
        assert decision.promote is False
        assert decision.reason == "hold_below_threshold"
        assert decision.sprt_verdict == "undecided"

    def test_no_games_never_promotes(self):
        decision = decide_promotion(
            MatchResult("c", "h", games=0, wins=0, draws=0, losses=0),
            GlickoRating(),
            self.CONFIG,
        )
        assert decision.promote is False
        assert decision.reason == "no_games"

    def test_update_match_pair_moves_both_parties(self):
        candidate, champion = update_match_pair(
            GlickoRating(),
            GlickoRating(rd=100.0),
            MatchResult("c", "h", games=16, wins=12, draws=4, losses=0),
        )
        assert candidate.r > DEFAULT_RATING
        assert champion.r < DEFAULT_RATING
        assert candidate.rd < DEFAULT_RD
        assert champion.rd < 100.0


class TestMatchHonesty:
    def test_fresh_16_game_gate_cannot_separate_anything(self):
        """Pre-registered honesty: with both RDs ~122 after one 16-game gate,
        the 95% band on the true score spans the whole [0,1] range — the
        score threshold is a heuristic, not a test, at this match size."""
        honesty = match_honesty(GlickoRating(rd=121.6), GlickoRating(rd=121.6), 16)
        assert honesty.min_significant_score > 1.0
        assert honesty.expected_score == pytest.approx(0.5)
        # Components: sampling noise alone would be a ±0.245 score band at
        # 95%; the two fresh RDs add ~1.8x more — rating uncertainty, not
        # sample size, is the dominant term at one 16-game gate each.
        assert honesty.per_game_sd == pytest.approx(0.125, abs=0.001)
        assert honesty.rd_sd > honesty.per_game_sd

    def test_converged_champion_still_leaves_the_fresh_rd_dominant(self):
        """Even vs a converged champion (RD ~36, the analytic equilibrium),
        the fresh candidate's own post-gate RD keeps the 95% bar above 1.0
        at 16 games — the pre-registered reason the threshold fallback stays
        a stated heuristic and the RD cap is what actually gates promotion.
        The bar DOES ease as the champion converges (compare the fresh pair)."""
        fresh_pair = match_honesty(GlickoRating(rd=121.6), GlickoRating(rd=121.6), 16)
        converged = match_honesty(GlickoRating(rd=121.6), GlickoRating(rd=36.0), 16)
        assert converged.min_significant_score > 1.0
        assert converged.min_significant_score < fresh_pair.min_significant_score

    def test_more_games_shrink_the_bar(self):
        champion = GlickoRating(rd=36.0)
        small = match_honesty(GlickoRating(rd=121.6), champion, 16)
        large = match_honesty(GlickoRating(rd=121.6), champion, 64)
        assert large.min_significant_score < small.min_significant_score
        assert large.min_significant_score < 1.0  # 64 games clears the bar


# --- tonight's real ledger, verbatim (data/arena.jsonl, read-only) ----------
# Old-format rows: no "kind" on the earliest, elo float 800.0 or null, no
# glicko keys anywhere. Plus one hand-built NEW-format row carrying the U9
# additive fields, to prove both formats parse in one replay.
_TONIGHT_ROWS = [
    {"checkpoint": "ckpt-000004-0004", "draws": 0, "elo": 800.0, "games": 0, "losses": 0,
     "opponent": "none", "promoted": True, "regime": "scratch", "run_id": "20261121-064558",
     "ts": 1795243561.77},
    {"checkpoint": "ckpt-000002-0002", "draws": 2, "elo": 800.0, "games": 2, "losses": 0,
     "opponent": "ckpt-000004-0004", "promoted": False, "regime": "scratch",
     "run_id": "fs5-cli-resume", "ts": 1795277389.61},
    {"checkpoint": "ckpt-000004-0004", "draws": 2, "elo": 800.0, "games": 2, "kind": "arena",
     "losses": 0, "opponent": "ckpt-000004-0004", "promoted": False, "regime": "scratch",
     "run_id": "fs5-cli-resume", "ts": 1795277400.71},
    # --- second-light gates 1-6 (the real run) ---
    {"checkpoint": "ckpt-000064-0032", "draws": 16, "elo": None, "games": 16, "kind": "arena",
     "losses": 0, "opponent": "ckpt-000004-0004", "promoted": False, "regime": "scratch",
     "run_id": "second-light", "ts": 1795288485.26, "wins": 0},
    {"checkpoint": "ckpt-000128-0064", "draws": 8, "elo": 792.0, "games": 16, "kind": "arena",
     "losses": 0, "opponent": "ckpt-000004-0004", "promoted": True, "regime": "scratch",
     "run_id": "second-light", "ts": 1795290903.13, "wins": 8},
    {"checkpoint": "ckpt-000192-0096", "draws": 16, "elo": None, "games": 16, "kind": "arena",
     "losses": 0, "opponent": "ckpt-000128-0064", "promoted": False, "regime": "scratch",
     "run_id": "second-light", "ts": 1795293574.29, "wins": 0},
    {"checkpoint": "ckpt-000256-0128", "draws": 16, "elo": None, "games": 16, "kind": "arena",
     "losses": 0, "opponent": "ckpt-000128-0064", "promoted": False, "regime": "scratch",
     "run_id": "second-light", "ts": 1795297084.87, "wins": 0},
    {"checkpoint": "ckpt-000320-0160", "draws": 16, "elo": None, "games": 16, "kind": "arena",
     "losses": 0, "opponent": "ckpt-000128-0064", "promoted": False, "regime": "scratch",
     "run_id": "second-light", "ts": 1795299802.64, "wins": 0},
    {"checkpoint": "ckpt-000384-0192", "draws": 16, "elo": None, "games": 16, "kind": "arena",
     "losses": 0, "opponent": "ckpt-000128-0064", "promoted": False, "regime": "scratch",
     "run_id": "second-light", "ts": 1795302744.22, "wins": 0},
]
# The same ledger with one NEW-format row appended (U9): additive glicko
# fields alongside the old elo — both formats must replay in one pass.
_MIXED_ROWS = [*_TONIGHT_ROWS,
    {"checkpoint": "ckpt-000448-0224", "draws": 16, "elo": None, "games": 16, "kind": "arena",
     "losses": 0, "opponent": "ckpt-000128-0064", "promoted": False, "regime": "scratch",
     "run_id": "second-light", "ts": 1795306000.0, "wins": 0,
     "glicko_r": 1620.0, "glicko_rd": 88.0},
]


class TestLedgerGlickoReplay:
    def test_old_and_new_rows_replay_together(self):
        records = recompute_glickos(_MIXED_ROWS)
        # Every checkpoint that appeared — old or new format — has a record.
        for tag in (
            "ckpt-000004-0004", "ckpt-000002-0002", "ckpt-000064-0032",
            "ckpt-000128-0064", "ckpt-000192-0096", "ckpt-000256-0128",
            "ckpt-000320-0160", "ckpt-000384-0192", "ckpt-000448-0224",
        ):
            assert tag in records

    def test_tonights_gates_produce_sane_r_and_shrinking_rd(self):
        records = recompute_glickos(_TONIGHT_ROWS)
        for tag, record in records.items():
            assert 1200.0 < record.r < 1800.0, tag  # sane band around the lineage
            assert 30.0 <= record.rd <= DEFAULT_RD, tag
        champion = records["ckpt-000128-0064"]
        # Champion since gate 2, then four 16-draw gates: RD shrinks well
        # below the fresh-candidate post-gate RD and r stays sane.
        assert champion.rd < records["ckpt-000192-0096"].rd
        assert champion.rd < 100.0
        # Pre-registered series band for the champion after 5 periods of play.
        assert 40.0 <= champion.rd <= 95.0

    def test_held_candidates_get_records_while_elo_stays_none(self):
        rows = LedgerEntry(
            run_id="r", regime="scratch", checkpoint="held", opponent="champ",
            games=16, wins=0, draws=16, elo=None, glicko_r=1500.0, glicko_rd=121.6,
        ).to_dict()
        assert rows["elo"] is None  # old field, old meaning: no claim
        assert rows["glicko_r"] == 1500.0  # new fields, additive
        assert rows["glicko_rd"] == 121.6

    def test_self_match_rows_seed_without_corrupting(self):
        # The pre-F-1 bug rows (a tag playing itself) must not move ratings.
        records = recompute_glickos(
            [{"checkpoint": "a", "opponent": "a", "games": 2, "wins": 0, "draws": 2, "elo": 800.0}]
        )
        assert records["a"] == GlickoRating()

    def test_elo_recompute_still_agrees_with_tonights_ledger(self):
        # The independent Elo path is untouched: 808 at promotion, then four
        # 16-draw holds vs 800-seeded candidates each pull ~0.37 back ->
        # ~806.5, matching the live feed's 806.6 sparkline point.
        elos = recompute_elos(_TONIGHT_ROWS)
        assert elos["ckpt-000128-0064"] == pytest.approx(806.5, abs=0.3)


class TestResumeAcrossU9:
    @staticmethod
    def _restore(state: dict[str, object]):
        """restore_run_state with the tiny model/buffer plumbing it needs."""
        import numpy as np
        import torch

        from training.model import MODEL_TINY, GambitNet
        from training.trainer import TrainingBuffer

        torch.manual_seed(0)
        model = GambitNet(MODEL_TINY)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "buffer_npz": None,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "play_rng_state": np.random.default_rng(0).bit_generator.state,
            **state,
        }
        buffer: TrainingBuffer = TrainingBuffer(max_size=10)
        play_rng = np.random.default_rng(0)
        return restore_run_state(
            state, model=model, optimizer=optimizer, buffer=buffer, play_rng=play_rng
        )

    def test_legacy_state_without_glicko_restores(self):
        """A state file written before U9 has no champion_glicko key; restore
        must tolerate that (a crash-restart tonight loads THIS code)."""
        state = {
            "iterations_completed": 3,
            "steps": 12,
            "games_played": 8,
            "champion_tag": "ckpt-000128-0064",
            "champion_elo": 792.0,
            "profile_signature": {},
        }
        run_state = self._restore(state)
        assert run_state.champion_glicko is None
        assert run_state.champion_elo == pytest.approx(792.0)

    def test_new_state_round_trips_the_glicko_record(self):
        record = GlickoRating(r=1537.0, rd=88.4, sigma=0.059).to_dict()
        state = {
            "iterations_completed": 4,
            "steps": 16,
            "games_played": 12,
            "champion_tag": "ckpt-000128-0064",
            "champion_elo": 793.37,
            "profile_signature": {},
            "champion_glicko": record,
        }
        run_state = self._restore(state)
        assert run_state.champion_glicko == record
        assert GlickoRating.from_dict(run_state.champion_glicko).rd == pytest.approx(88.4)

    def test_corrupt_glicko_payload_degrades_to_none(self):
        state = {
            "iterations_completed": 1,
            "steps": 1,
            "games_played": 1,
            "champion_tag": None,
            "champion_elo": 800.0,
            "profile_signature": {},
            "champion_glicko": "not-a-dict",
        }
        assert self._restore(state).champion_glicko is None


class TestConfigKnobs:
    def test_ladder_rejects_bad_sprt_config(self):
        with pytest.raises(ValueError):
            LadderConfig(sprt_elo0=10.0, sprt_elo1=0.0)
        with pytest.raises(ValueError):
            LadderConfig(sprt_alpha=0.0)
        with pytest.raises(ValueError):
            LadderConfig(promote_rd_cap=0.0)

    def test_expected_score_logistic(self):
        assert expected_score(GlickoRating(), GlickoRating()) == pytest.approx(0.5)
        # +120 Elo vs a PRECISE opponent (RD 30): ~0.665 (g() ~= 1 there).
        assert expected_score(
            GlickoRating(r=1620.0), GlickoRating(r=1500.0, rd=30.0)
        ) == pytest.approx(0.6654, abs=0.002)
        # vs the default RD-350 prior the g() down-weight compresses toward 0.5.
        assert expected_score(GlickoRating(r=1620.0), GlickoRating()) == pytest.approx(
            0.6135, abs=0.002
        )
