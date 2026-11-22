"""SPRT [elo0, elo1] over arena games — U9 (RESEARCH_NOTES §3).

Sequential probability ratio test (Wald 1945) as used by the engine-testing
community (CPW SPRT methodology; cutechess-cli): accumulate the log-likelihood
ratio of the observed W/D/L counts under H1 ("candidate is `elo1` stronger")
vs H0 ("candidate is `elo0` stronger"), with bounds
lower = ln(beta/(1-alpha)) (accept H0) and upper = ln((1-alpha)/beta)
(accept H1).

Model choice — the TRINOMIAL (win/draw/loss) LLR with the draw rate fixed at
the observed rate under both hypotheses, NOT the pentanomial:
  * the pentanomial treats color-reversed PAIRS as one observation and needs
    an explicit pair-outcome model; our arena games share the standard
    initial position with colors alternated, but the pair correlation comes
    from the engines, not from randomized shared openings, so the pairing
    assumption buys nothing here;
  * the fixed-draw trinomial is the CPW standard for unpaired games,
    degrades exactly to the binomial when there are no draws, and stays
    well-defined at the extreme draw rates this arena actually produces
    (tonight's gates: 50-100% draws);
  * consequence, stated honestly: an all-draw sample carries zero
    information under this model (LLR = 0), so SPRT stays undecided on
    draw-only gates — that is the truthful verdict, not a bug.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

DEFAULT_ELO0 = 0.0
DEFAULT_ELO1 = 10.0
DEFAULT_ALPHA = 0.05
DEFAULT_BETA = 0.05
# Clamp for logistic-model cells that hit 0 at extreme observed draw rates
# (keeps ln() finite; only reachable on samples the fixed-draw model cannot
# score, e.g. mixed results at a 100% observed draw rate).
_CELL_EPSILON = 1e-12


@dataclass(frozen=True)
class SPRTState:
    """One SPRT snapshot after a gate: LLR, bounds, verdict."""

    llr: float
    lower: float
    upper: float
    games: int

    @property
    def verdict(self) -> Literal["accept_h1", "accept_h0", "undecided"]:
        if self.llr >= self.upper:
            return "accept_h1"
        if self.llr <= self.lower:
            return "accept_h0"
        return "undecided"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "llr": round(self.llr, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "games": self.games,
            "verdict": self.verdict,
        }


def expected_score_for(elo_delta: float) -> float:
    """Logistic expected score for an Elo advantage (CPW convention)."""
    return 1.0 / (1.0 + 10.0 ** (-elo_delta / 400.0))


def sprt_bounds(alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> tuple[float, float]:
    """(lower, upper) LLR bounds: H0 accepted at/below lower, H1 at/above upper."""
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError("alpha and beta must be in (0, 1)")
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - alpha) / beta)


def trinomial_llr(
    wins: int,
    draws: int,
    losses: int,
    elo0: float = DEFAULT_ELO0,
    elo1: float = DEFAULT_ELO1,
    draw_rate: float | None = None,
) -> float:
    """LLR of W/D/L counts under H1 vs H0 (fixed-draw-rate trinomial).

    `draw_rate` overrides the observed draw share (kept as a parameter for
    the honesty projections); None uses draws / (wins+draws+losses).
    """
    games = wins + draws + losses
    if games <= 0:
        return 0.0
    observed_draw_rate = draws / games if draw_rate is None else float(draw_rate)
    if observed_draw_rate >= 1.0:
        # An all-draw sample pins the model's win/loss cells to zero under
        # BOTH hypotheses (s0 = 0.5 makes them exactly zero): zero
        # information about the score difference.
        return 0.0
    half_draws = observed_draw_rate / 2.0
    s0, s1 = expected_score_for(elo0), expected_score_for(elo1)
    w0 = max(s0 - half_draws, _CELL_EPSILON)
    w1 = max(s1 - half_draws, _CELL_EPSILON)
    l0 = max(1.0 - s0 - half_draws, _CELL_EPSILON)
    l1 = max(1.0 - s1 - half_draws, _CELL_EPSILON)
    llr = 0.0
    if wins:
        llr += wins * math.log(w1 / w0)
    if losses:
        llr += losses * math.log(l1 / l0)
    # draws term: ln(d1/d0) = ln(1) = 0 by construction (fixed draw rate).
    return llr


def sprt_state(
    wins: int,
    draws: int,
    losses: int,
    *,
    elo0: float = DEFAULT_ELO0,
    elo1: float = DEFAULT_ELO1,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> SPRTState:
    """Full SPRT snapshot over a candidate's accumulated games vs one opponent."""
    lower, upper = sprt_bounds(alpha, beta)
    llr = trinomial_llr(wins, draws, losses, elo0, elo1)
    return SPRTState(llr=llr, lower=lower, upper=upper, games=wins + draws + losses)
