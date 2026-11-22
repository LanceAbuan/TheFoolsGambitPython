"""Glicko-2 ratings (r, RD, sigma) for the arena ladder — U9 (RESEARCH_NOTES §3).

Implements the full Glicko-2 update from Mark Glickman's paper
("Parameter estimation in large dynamic paired comparison experiments",
1999, http://www.glicko.net/glicko/glicko2.pdf): the 2-step scale
conversion (Glicko -> Glicko-2 -> back), the v/Delta variance-improvement
terms, and the Illinois-algorithm volatility iteration. The numbers are
verified against the paper's worked example to two decimals
(tests/test_training_glicko.py).

Scale choice — the standard 1500/350 Glicko scale, NOT the run's 800 base:
  * every constant in the paper (initial r=1500, RD=350, sigma=0.06, the
    173.7178 scale factor) and the worked example live on the 1500 scale;
  * RD magnitudes and the published anchor literature (CCRL, CPW) are
    quoted on that scale, so keeping it makes our numbers comparable and
    the worked-example test exact;
  * the run-internal `base_elo=800` Elo series answers a different
    question (a continuous run-facing rating feed) and keeps its meaning
    untouched — the two series coexist as separate additive fields.

Both parties of an arena match are updated from the same games: the
candidate (a fresh unrated checkpoint every gate — its own r/RD record is
kept even when it does not promote) and the champion lineage. A match is
one rating period; its games are the period's results (Glicko's model
treats each game as one result of the period; same-opponent correlation
across the match's games is ignored, which is standard practice and only
mildly overstates information — the SPRT/honesty layer is the check on
over-claiming, see training.sprt).

A player who sits out a rating period decays via `rate_inactive`
(paper: r unchanged, phi' = sqrt(phi^2 + sigma^2) — RD widens slowly, in
Glicko-2 through the player's own volatility rather than a fixed c).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .sprt import expected_score_for, sprt_bounds

# --- Glicko-2 system constants (paper §5 / worked example) -----------------
GLICKO2_SCALE = 173.7178  # RD/1000 of the original Glicko scale (paper §5)
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_SIGMA = 0.06  # volatility prior (paper's worked example; small by design)
DEFAULT_TAU = 0.5  # system constant constraining volatility change
_VOLATILITY_EPSILON = 0.000001  # paper's convergence tolerance for the Illinois loop
_MAX_ILLINOIS_ITERATIONS = 100
# Glicko caps RD at 350 on the 1500 scale (paper §3: "the Glicko system
# bounds RD"); the decay helper respects the same ceiling.
_MAX_RD = 350.0


@dataclass(frozen=True)
class GlickoRating:
    """One player's Glicko-2 state: rating r, deviation RD, volatility sigma."""

    r: float = DEFAULT_RATING
    rd: float = DEFAULT_RD
    sigma: float = DEFAULT_SIGMA

    def to_dict(self) -> dict[str, float]:
        """Plain-dict form for JSONL/state payloads."""
        return {"r": self.r, "rd": self.rd, "sigma": self.sigma}

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> GlickoRating:
        """Rebuild from a payload dict; tolerates missing keys and None."""
        if not isinstance(data, dict):
            return cls()
        return cls(
            r=float(data.get("r", DEFAULT_RATING)),  # type: ignore[arg-type]
            rd=float(data.get("rd", DEFAULT_RD)),  # type: ignore[arg-type]
            sigma=float(data.get("sigma", DEFAULT_SIGMA)),  # type: ignore[arg-type]
        )


def _g(phi: float) -> float:
    """Glicko-2 g(phi) down-weight of an opponent's imprecision (paper §5)."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _expected(mu: float, mu_opponent: float, phi_opponent: float) -> float:
    """E(mu, mu_j, phi_j): expected score against one opponent (paper §5)."""
    return 1.0 / (1.0 + math.exp(-_g(phi_opponent) * (mu - mu_opponent)))


def expected_score(rating: GlickoRating, opponent: GlickoRating) -> float:
    """Expected score of `rating` vs `opponent` on the Glicko scale."""
    mu = (rating.r - DEFAULT_RATING) / GLICKO2_SCALE
    mu_j = (opponent.r - DEFAULT_RATING) / GLICKO2_SCALE
    phi_j = min(opponent.rd, _MAX_RD) / GLICKO2_SCALE
    return _expected(mu, mu_j, phi_j)


def _volatility(
    delta: float, phi: float, v: float, sigma: float, tau: float
) -> float:
    """Step 4 of the paper: sigma' via the Illinois algorithm (regula falsi)."""
    a = math.log(sigma * sigma)
    phi_sq = phi * phi

    def f(x: float) -> float:
        exp_x = math.exp(x)
        numerator = exp_x * (delta * delta - phi_sq - v - exp_x)
        denominator = 2.0 * (phi_sq + v + exp_x) ** 2
        return numerator / denominator - (x - a) / (tau * tau)

    big_a = a
    if delta * delta > phi_sq + v:
        big_b = math.log(delta * delta - phi_sq - v)
    else:
        k = 1
        while f(a - k * tau) < 0.0 and k < _MAX_ILLINOIS_ITERATIONS:
            k += 1
        big_b = a - k * tau

    f_a, f_b = f(big_a), f(big_b)
    iterations = 0
    while abs(big_b - big_a) > _VOLATILITY_EPSILON and iterations < _MAX_ILLINOIS_ITERATIONS:
        big_c = big_a + (big_a - big_b) * f_a / (f_b - f_a)
        f_c = f(big_c)
        if f_c * f_b <= 0.0:
            big_a, f_a = big_b, f_b
        else:
            f_a = f_a / 2.0
        big_b, f_b = big_c, f_c
        iterations += 1
    return math.exp(big_a / 2.0)


def rate_player(
    rating: GlickoRating,
    results: Sequence[tuple[GlickoRating, float]],
    *,
    tau: float = DEFAULT_TAU,
) -> GlickoRating:
    """One rating period of play: the paper's steps 2-6.

    `results` carries one (opponent rating, score) pair per game played in
    the period, score in {0.0, 0.5, 1.0}. An empty period returns the
    rating unchanged (use `rate_inactive` for the RD decay of a period
    missed entirely).
    """
    if not results:
        return rating
    mu = (rating.r - DEFAULT_RATING) / GLICKO2_SCALE
    phi = min(rating.rd, _MAX_RD) / GLICKO2_SCALE

    v_inverse = 0.0
    delta_sum = 0.0
    for opponent, score in results:
        mu_j = (opponent.r - DEFAULT_RATING) / GLICKO2_SCALE
        phi_j = min(opponent.rd, _MAX_RD) / GLICKO2_SCALE
        g_j = _g(phi_j)
        expected = _expected(mu, mu_j, phi_j)
        v_inverse += g_j * g_j * expected * (1.0 - expected)
        delta_sum += g_j * (float(score) - expected)
    v = 1.0 / v_inverse
    delta = v * delta_sum

    sigma_prime = _volatility(delta, phi, v, rating.sigma, tau)
    phi_star = math.sqrt(phi * phi + sigma_prime * sigma_prime)
    phi_prime = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_prime = mu + phi_prime * phi_prime * delta_sum
    return GlickoRating(
        r=DEFAULT_RATING + GLICKO2_SCALE * mu_prime,
        rd=GLICKO2_SCALE * phi_prime,
        sigma=sigma_prime,
    )


def rate_inactive(rating: GlickoRating) -> GlickoRating:
    """One rating period without play (paper: r unchanged, RD widens).

    Glicko-2 decays through the player's own volatility: phi' =
    sqrt(phi^2 + sigma^2) — much gentler than Glicko-1's linear c*t, and
    self-limiting at the 350 RD ceiling.
    """
    phi = min(rating.rd, _MAX_RD) / GLICKO2_SCALE
    phi_prime = math.sqrt(phi * phi + rating.sigma * rating.sigma)
    return GlickoRating(
        r=rating.r,
        rd=min(GLICKO2_SCALE * phi_prime, _MAX_RD),
        sigma=rating.sigma,
    )


# ---------------------------------------------------------------------------
# Match-size honesty (U9: "gate match size vs overnight cost — pre-register")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchHonesty:
    """What a gate of `games` games can actually claim, given the RDs.

    `min_significant_score` is the observed score above which the 95%
    interval on the TRUE expected score excludes a 0.5 coin-flip — the
    honest bar a threshold rule is implicitly claiming to clear. The
    decomposition: sampling noise of the observed score (per_game_sd /
    sqrt(games)) plus the RD-driven uncertainty of the strength difference
    itself (rd_sd). Normal approximation, documented as such.
    """

    games: int
    expected_score: float
    per_game_sd: float
    rd_sd: float
    ci95_half_width: float
    min_significant_score: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "games": self.games,
            "expected_score": round(self.expected_score, 4),
            "per_game_sd": round(self.per_game_sd, 4),
            "rd_sd": round(self.rd_sd, 4),
            "ci95_half_width": round(self.ci95_half_width, 4),
            "min_significant_score": round(self.min_significant_score, 4),
        }


_Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def match_honesty(
    candidate: GlickoRating, champion: GlickoRating, games: int
) -> MatchHonesty:
    """Score bar a `games`-game match must clear to be RD-aware significant."""
    expected = expected_score(candidate, champion)
    per_game_sd = math.sqrt(max(expected * (1.0 - expected), 0.0) / max(games, 1))
    # Uncertainty of the TRUE expected score from both players' RDs: dE/dmu
    # is g(phi_j)*E*(1-E), and the rating difference carries sd
    # sqrt(phi_c^2 + phi_h^2) (independent deviations).
    phi_j = min(champion.rd, _MAX_RD) / GLICKO2_SCALE
    rd_sd = _g(phi_j) * expected * (1.0 - expected) * math.sqrt(
        (candidate.rd / GLICKO2_SCALE) ** 2 + (champion.rd / GLICKO2_SCALE) ** 2
    )
    ci95 = _Z_95 * (per_game_sd + rd_sd)
    return MatchHonesty(
        games=games,
        expected_score=expected,
        per_game_sd=per_game_sd,
        rd_sd=rd_sd,
        ci95_half_width=ci95,
        min_significant_score=expected + ci95,
    )


def expected_games_to_accept(
    true_score: float,
    draw_rate: float,
    *,
    elo0: float = 0.0,
    elo1: float = 10.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> float | None:
    """Expected games for the SPRT to accept H1 at a given true score.

    Under the fixed-draw-rate trinomial (training.sprt), the per-game LLR
    at true score `true_score` with draw share `draw_rate` is
    w*ln(w1/w0) + l*ln(l1/l0); the accept boundary needs ~upper/mean
    games (Wald's approximation). None when H1 is unreachable at this
    score/draw combination (mean LLR <= 0 or an infeasible cell).
    """
    if not 0.0 <= draw_rate < 1.0 or not 0.0 < true_score <= 1.0:
        return None
    half_draws = draw_rate / 2.0
    wins = true_score - half_draws
    losses = 1.0 - true_score - half_draws
    if wins < 0.0 or losses < 0.0:
        return None
    s0 = expected_score_for(elo0)
    s1 = expected_score_for(elo1)
    w0, w1 = s0 - half_draws, s1 - half_draws
    l0, l1 = 1.0 - s0 - half_draws, 1.0 - s1 - half_draws
    if w0 <= 0.0 or w1 <= 0.0 or l0 <= 0.0 or l1 <= 0.0:
        return None
    mean_llr = wins * math.log(w1 / w0) + losses * math.log(l1 / l0)
    if mean_llr <= 0.0:
        return None
    _, upper = sprt_bounds(alpha, beta)
    return upper / mean_llr
