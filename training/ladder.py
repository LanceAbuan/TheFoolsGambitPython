"""Checkpoint ladder + arena gating (plan §5.1 — binding).

Ladder: snapshot the training model on a wall-clock schedule — 5/10/15/20
minutes, then every 20 minutes. Time, not game count, is the experiment axis.
Configurable via `LadderConfig`.

Arena gate: every snapshot plays the reigning champion a match, both colors,
reduced sims. Promotion at >=55% match win rate (draws count half — standard
score rate). The ladder IS the multi-model experiment; the ledger records who
beat whom (`training.ledger`), and per-checkpoint Elo flows to the metrics
feed.

Match play is sequential tonight (slice 1); `ArenaConfig` carries the knobs so
parallelizing later is config work, not a rewrite.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .selfplay import ArenaConfig, Evaluator, play_match_game

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
    arena: ArenaConfig = field(default_factory=ArenaConfig)

    def __post_init__(self) -> None:
        if not self.markers_minutes:
            raise ValueError("markers_minutes must not be empty")
        if self.repeat_every_minutes <= 0:
            raise ValueError("repeat_every_minutes must be positive")
        if not 0.5 < self.promote_win_rate <= 1.0:
            raise ValueError("promote_win_rate must be in (0.5, 1.0]")

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

    @property
    def win_rate(self) -> float:
        return score_rate(self.wins, self.draws, self.games)


def should_promote(match: MatchResult, threshold: float) -> bool:
    """§5.1 promotion rule: candidate replaces champion at >= threshold."""
    return match.games > 0 and match.win_rate >= threshold


class Arena:
    """Runs a candidate-vs-champion match, both colors, sequential games."""

    def __init__(
        self,
        candidate_evaluator: Evaluator,
        champion_evaluator: Evaluator,
        config: ArenaConfig = _DEFAULT_ARENA_CONFIG,
    ) -> None:
        self.candidate_evaluator = candidate_evaluator
        self.champion_evaluator = champion_evaluator
        self.config = config

    def run(self, candidate_tag: str, champion_tag: str) -> Iterator[tuple[str, str, str]]:
        """Play the match game by game; yields (result, candidate_color, termination)."""
        games = self.config.games + (self.config.games % 2)  # balance colors
        for game_index in range(games):
            candidate_is_white = game_index % 2 == 0
            if candidate_is_white:
                white_eval, black_eval = self.candidate_evaluator, self.champion_evaluator
            else:
                white_eval, black_eval = self.champion_evaluator, self.candidate_evaluator
            result, termination = play_match_game(
                white_eval, black_eval, self.config.num_simulations, self.config.max_plies
            )
            yield result, ("white" if candidate_is_white else "black"), termination

    def match(self, candidate_tag: str, champion_tag: str) -> MatchResult:
        """Play the full match and aggregate from the candidate's perspective."""
        wins = draws = losses = 0
        games = 0
        for result, _candidate_color, _termination in self.run(candidate_tag, champion_tag):
            games += 1
            if result == "1/2-1/2":
                draws += 1
            elif result == "1-0":
                wins += 1
            else:
                losses += 1
        match_result = MatchResult(
            candidate=candidate_tag, champion=champion_tag, games=games, wins=wins, draws=draws, losses=losses
        )
        log.info(
            "[ARENA] %s vs %s: %d games, +%.1f-%.1f-%d (score %.3f)",
            candidate_tag,
            champion_tag,
            games,
            wins,
            draws,
            losses,
            match_result.win_rate,
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
