"""Checkpoint ladder + arena gating (plan §5.1 — binding).

Ladder: snapshot the training model on a wall-clock schedule — 5/10/15/20
minutes, then every 20 minutes. Time, not game count, is the experiment axis.
Configurable via `LadderConfig`.

Arena gate: every snapshot plays the reigning champion a match, both colors,
reduced sims. Promotion per the U9 decision rule (`decide_promotion`):
SPRT [elo0, elo1] alpha=beta accept H1, OR the §5.1 >=55% threshold with the
candidate's Glicko-2 RD under `promote_rd_cap`. The ladder IS the
multi-model experiment; the ledger records who beat whom (`training.ledger`),
and per-checkpoint Elo AND Glicko-2 records flow to the metrics feed.

Match play is sequential tonight (slice 1); `ArenaConfig` carries the knobs so
parallelizing later is config work, not a rewrite.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .glicko import GlickoRating, rate_player
from .openings import sample_gate_openings
from .selfplay import ArenaConfig, Evaluator, play_match_game
from .sprt import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_ELO0,
    DEFAULT_ELO1,
    sprt_bounds,
    sprt_state,
)

log = logging.getLogger(__name__)

_DEFAULT_ARENA_CONFIG = ArenaConfig()


@dataclass(frozen=True)
class LadderConfig:
    """Wall-clock snapshot schedule + gate rule (§5.1, configurable)."""

    markers_minutes: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)
    repeat_every_minutes: float = 20.0
    promote_win_rate: float = 0.55
    base_elo: float = 800.0
    k_factor: float = 32.0
    # U9: SPRT [elo0, elo1] at alpha=beta (RESEARCH_NOTES §3) and the RD cap
    # for the threshold fallback. promote_rd_cap=150 admits exactly the
    # 16-game gates the ladder plays (a fresh candidate lands at RD ~ 122
    # after 16 games; an 8-game gate fails at ~ 162), so the §5.1 fallback
    # stays operative at the current match size while sub-gate smoke matches
    # can no longer promote on noise.
    sprt_elo0: float = DEFAULT_ELO0
    sprt_elo1: float = DEFAULT_ELO1
    sprt_alpha: float = DEFAULT_ALPHA
    sprt_beta: float = DEFAULT_BETA
    promote_rd_cap: float = 150.0
    arena: ArenaConfig = field(default_factory=ArenaConfig)

    def __post_init__(self) -> None:
        if not self.markers_minutes:
            raise ValueError("markers_minutes must not be empty")
        if self.repeat_every_minutes <= 0:
            raise ValueError("repeat_every_minutes must be positive")
        if not 0.5 < self.promote_win_rate <= 1.0:
            raise ValueError("promote_win_rate must be in (0.5, 1.0]")
        if not 0.0 <= self.sprt_elo0 < self.sprt_elo1:
            raise ValueError("sprt_elo0 must be < sprt_elo1 (and non-negative)")
        if not 0.0 < self.sprt_alpha < 1.0 or not 0.0 < self.sprt_beta < 1.0:
            raise ValueError("sprt_alpha/sprt_beta must be in (0, 1)")
        if self.promote_rd_cap <= 0:
            raise ValueError("promote_rd_cap must be positive")

    def scheduled_minutes(self) -> tuple[float, ...]:
        """5/10/15/20 then every `repeat_every_minutes` (40, 60, ...)."""
        tail_start = self.markers_minutes[-1] + self.repeat_every_minutes
        return (*self.markers_minutes, *np.arange(tail_start, 1441.0, self.repeat_every_minutes))


def due_snapshot_minutes(
    config: LadderConfig, elapsed_minutes: float, last_snapshot_minute: float
) -> list[float]:
    """Ladder times that have come due since the last snapshot.

    Pure function of (schedule, elapsed, last snapshot) so it is unit-testable
    with a fake clock.
    """
    return [
        m
        for m in config.scheduled_minutes()
        if last_snapshot_minute < m <= elapsed_minutes
    ]


def score_rate(wins: int, draws: int, games: int) -> float:
    """Match score rate: (wins + 0.5 * draws) / games; 0.0 for an empty match."""
    if games <= 0:
        return 0.0
    return (wins + 0.5 * draws) / games


@dataclass(frozen=True)
class MatchResult:
    """A completed candidate-vs-champion match (candidate perspective)."""

    candidate: str
    champion: str
    games: int
    wins: int
    draws: int
    losses: int
    # FB-1: the opening FENs of the games actually played, in game order.
    # Empty for matches run without opening randomization (opening_seed=None),
    # so every pre-FB-1 consumer keeps its exact shape.
    openings: tuple[str, ...] = ()

    @property
    def win_rate(self) -> float:
        return score_rate(self.wins, self.draws, self.games)


def should_promote(match: MatchResult, threshold: float) -> bool:
    """§5.1 promotion rule: candidate replaces champion at >= threshold."""
    return match.games > 0 and match.win_rate >= threshold


def update_match_pair(
    candidate: GlickoRating, champion: GlickoRating, match: MatchResult
) -> tuple[GlickoRating, GlickoRating]:
    """One rating period for BOTH parties of a match (U9).

    The match's games are the period's results: the candidate played them
    with its own scores, the champion with the mirrored scores, each
    against the other's pre-period rating (paper §5 model). Returns
    (candidate', champion'); both records move even though only one of the
    two can end up reigning.
    """
    if match.games <= 0:
        return candidate, champion
    candidate_scores = [1.0] * match.wins + [0.5] * match.draws + [0.0] * match.losses
    champion_scores = [1.0 - s for s in candidate_scores]
    candidate_prime = rate_player(candidate, [(champion, s) for s in candidate_scores])
    champion_prime = rate_player(champion, [(candidate, s) for s in champion_scores])
    return candidate_prime, champion_prime


@dataclass(frozen=True)
class PromotionDecision:
    """The gate's full verdict, carried to the ledger/journal/metrics."""

    promote: bool
    reason: Literal[
        "no_games",
        "sprt_accept_h1",
        "threshold_with_rd_ok",
        "hold_below_threshold",
        "hold_rd_above_cap",
        "hold_sprt_accept_h0",
    ]
    score_rate: float
    threshold: float
    rd: float
    rd_cap: float
    llr: float
    lower: float
    upper: float
    sprt_verdict: str

    def to_dict(self) -> dict[str, object]:
        return {
            "promote": self.promote,
            "reason": self.reason,
            "score_rate": round(self.score_rate, 4),
            "threshold": self.threshold,
            "rd": round(self.rd, 2),
            "rd_cap": self.rd_cap,
            "llr": round(self.llr, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "sprt_verdict": self.sprt_verdict,
        }


def decide_promotion(
    match: MatchResult, candidate_rating: GlickoRating, config: LadderConfig
) -> PromotionDecision:
    """U9 promotion decision — the rule, as a table:

    | # | Path                | Condition (in order)                                | Verdict                |
    |---|---------------------|-----------------------------------------------------|------------------------|
    | 0 | no games            | match.games == 0 (baseline row)                     | HOLD (`no_games`)      |
    | 1 | SPRT accepts H1     | LLR >= ln((1-alpha)/beta) over the candidate's      | PROMOTE                |
    |   |                     | accumulated games vs the champion                   | (`sprt_accept_h1`)     |
    | 2 | threshold + RD cap  | score >= promote_win_rate AND candidate RD <=       | PROMOTE                |
    |   | (§5.1 fallback)     | promote_rd_cap                                      | (`threshold_with_rd_ok`)|
    | 3 | SPRT accepts H0     | LLR <= ln(beta/(1-alpha))                           | HOLD (`hold_sprt_accept_h0`) |
    | 4 | threshold, RD high  | score >= threshold but RD > cap (match too small    | HOLD (`hold_rd_above_cap`)   |
    |   |                     | to trust the score — surface, do not promote)       |                        |
    | 5 | otherwise           | —                                                   | HOLD (`hold_below_threshold`)|

    Rationale (RESEARCH_NOTES §3, U9): 16-game gates carry ~ ±120 Elo of
    noise at 95%, so the bare 0.55 threshold is noise-dominated; SPRT
    [0, 10] alpha=beta=0.05 is the statistically meaningful test but needs
    on the order of 10^3 games to fire at a true +10 (see
    `glicko.expected_games_to_accept`) — impossible at overnight gate
    sizes. Hence the OR: SPRT accepts when it can, the §5.1 threshold
    remains the operative fallback but only above the RD cap (i.e. only
    from matches big enough for its RD to be meaningful). The score/LLR
    relationship makes cases 1 vs 3 mutually exclusive with case 2 in
    practice: at score >= 0.55 the per-game LLR is positive, so a
    threshold pass can never coincide with an H0 accept.
    """
    lower, upper = sprt_bounds(config.sprt_alpha, config.sprt_beta)
    if match.games <= 0:
        return PromotionDecision(
            promote=False,
            reason="no_games",
            score_rate=match.win_rate,
            threshold=config.promote_win_rate,
            rd=candidate_rating.rd,
            rd_cap=config.promote_rd_cap,
            llr=0.0,
            lower=lower,
            upper=upper,
            sprt_verdict="undecided",
        )
    state = sprt_state(
        match.wins,
        match.draws,
        match.losses,
        elo0=config.sprt_elo0,
        elo1=config.sprt_elo1,
        alpha=config.sprt_alpha,
        beta=config.sprt_beta,
    )
    common: dict[str, object] = {
        "score_rate": match.win_rate,
        "threshold": config.promote_win_rate,
        "rd": candidate_rating.rd,
        "rd_cap": config.promote_rd_cap,
        "llr": state.llr,
        "lower": state.lower,
        "upper": state.upper,
        "sprt_verdict": state.verdict,
    }
    if state.verdict == "accept_h1":
        return PromotionDecision(promote=True, reason="sprt_accept_h1", **common)
    if match.win_rate >= config.promote_win_rate and candidate_rating.rd <= config.promote_rd_cap:
        return PromotionDecision(promote=True, reason="threshold_with_rd_ok", **common)
    if state.verdict == "accept_h0":
        return PromotionDecision(promote=False, reason="hold_sprt_accept_h0", **common)
    if match.win_rate >= config.promote_win_rate:
        return PromotionDecision(promote=False, reason="hold_rd_above_cap", **common)
    return PromotionDecision(promote=False, reason="hold_below_threshold", **common)


class Arena:
    """Runs a candidate-vs-champion match, both colors, sequential games.

    FB-1: with `ArenaConfig.opening_seed` set, every game starts from a
    seeded opening position — one FEN per game, mirrored across color pairs
    (`training.openings.sample_gate_openings`) — instead of all games sharing
    the initial position (which made deterministic arena play an effective
    sample size of ~2). The seed is a pure function of (run_seed, gate_index)
    supplied by the pipeline, so each gate event reproduces its own set.
    """

    def __init__(
        self,
        candidate_evaluator: Evaluator,
        champion_evaluator: Evaluator,
        config: ArenaConfig = _DEFAULT_ARENA_CONFIG,
    ) -> None:
        self.candidate_evaluator = candidate_evaluator
        self.champion_evaluator = champion_evaluator
        self.config = config

    def _opening_fens(self, games: int) -> tuple[str, ...]:
        """One FEN per planned game, or () when randomization is off."""
        if self.config.opening_seed is None or games < 2:
            return ()
        return sample_gate_openings(games, self.config.opening_seed)

    def run(self, candidate_tag: str, champion_tag: str) -> Iterator[tuple[str, str, str]]:
        """Play the match game by game; yields (result, candidate_color, termination).

        With `config.parallel > 1` the games run concurrently and are yielded
        in completion order. SPRT early stopping (disabled by default) stops
        consuming the iterator, so with parallelism a few already-in-flight
        games finish unobserved — the played set is a prefix-in-spirit, never
        a different decision rule.
        """
        games = self.config.games + (self.config.games % 2)  # balance colors
        openings = self._opening_fens(games)
        parallel = max(1, min(self.config.parallel, games))

        def play_one(game_index: int) -> tuple[str, str, str]:
            candidate_is_white = game_index % 2 == 0
            if candidate_is_white:
                white_eval, black_eval = self.candidate_evaluator, self.champion_evaluator
            else:
                white_eval, black_eval = self.champion_evaluator, self.candidate_evaluator
            opening_fen = openings[game_index] if openings else None
            result, termination = play_match_game(
                white_eval,
                black_eval,
                self.config.num_simulations,
                self.config.max_plies,
                opening_fen=opening_fen,
            )
            return result, ("white" if candidate_is_white else "black"), termination

        if parallel == 1:
            for game_index in range(games):
                yield play_one(game_index)
            return

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(play_one, i) for i in range(games)]
            for future in as_completed(futures):
                yield future.result()

    def match(
        self, candidate_tag: str, champion_tag: str, *, stop_on_sprt: bool | None = None
    ) -> MatchResult:
        """Play the match and aggregate from the candidate's perspective.

        With `stop_on_sprt` (or the config's `sprt_early_stop`) the match is
        SEQUENTIAL (U9 gate economics): after every even game count — even,
        so the two colors stay balanced at the stop point — the SPRT LLR is
        recomputed and the match stops the moment a bound is crossed. A
        decisive mismatch therefore costs fewer games than the configured
        maximum; at the default gate size the bounds are unreachable, making
        the stop a provable no-op (see RESEARCH_NOTES §3 pre-registration).
        """
        early_stop = self.config.sprt_early_stop if stop_on_sprt is None else stop_on_sprt
        planned = self.config.games + (self.config.games % 2)
        openings = self._opening_fens(planned)
        wins = draws = losses = 0
        games = 0
        played_openings: list[str] = []
        for result, _candidate_color, _termination in self.run(candidate_tag, champion_tag):
            games += 1
            if openings:
                played_openings.append(openings[games - 1])
            if result == "1/2-1/2":
                draws += 1
            elif result == "1-0":
                wins += 1
            else:
                losses += 1
            if (
                early_stop
                and games % 2 == 0
                and sprt_state(
                    wins, draws, losses,
                    elo0=self.config.sprt_elo0,
                    elo1=self.config.sprt_elo1,
                    alpha=self.config.sprt_alpha,
                    beta=self.config.sprt_beta,
                ).verdict
                != "undecided"
            ):
                log.info(
                    "[ARENA] sequential stop at %d games (SPRT %s)",
                    games,
                    sprt_state(wins, draws, losses,
                               elo0=self.config.sprt_elo0,
                               elo1=self.config.sprt_elo1,
                               alpha=self.config.sprt_alpha,
                               beta=self.config.sprt_beta).verdict,
                )
                break
        match_result = MatchResult(
            candidate=candidate_tag,
            champion=champion_tag,
            games=games,
            wins=wins,
            draws=draws,
            losses=losses,
            openings=tuple(played_openings),
        )
        log.info(
            "[ARENA] %s vs %s: %d games (of %d planned), +%.1f-%.1f-%d (score %.3f)%s",
            candidate_tag,
            champion_tag,
            games,
            planned,
            wins,
            draws,
            losses,
            match_result.win_rate,
            f", {len(match_result.openings)} openings" if match_result.openings else "",
        )
        return match_result


def update_elo(
    candidate_elo: float,
    champion_elo: float,
    match: MatchResult,
    k_factor: float,
) -> tuple[float, float]:
    """One K-factor Elo update from a match; returns (candidate, champion)."""
    if match.games <= 0:
        return candidate_elo, champion_elo
    expected = 1.0 / (1.0 + 10.0 ** ((champion_elo - candidate_elo) / 400.0))
    delta = k_factor * (match.win_rate - expected)
    return candidate_elo + delta, champion_elo - delta
