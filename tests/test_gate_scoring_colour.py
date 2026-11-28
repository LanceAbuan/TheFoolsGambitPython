"""Gate scoring must be from the CANDIDATE's perspective.

Regression for 2026-11-28: `Arena.match` counted `result == "1-0"` (White won)
as a candidate win regardless of which colour the candidate played. With mirrored
openings that cancels any consistent strength gap to exactly 0.500 - a stronger
candidate wins as White ("1-0" -> win) and as Black ("0-1" -> loss) - which froze
promotion and produced the false "no measurable progress" conclusion.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from training.ladder import Arena, ArenaConfig


class _ScriptedEvaluator:
    """Evaluator stub; the scripted game outcomes decide the match."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def evaluate(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("scripted run does not evaluate")


class _ScriptedArena(Arena):
    """Arena whose games are scripted, so colour handling is testable alone."""

    def __init__(self, script):  # script: list of (result, candidate_color)
        super().__init__(
            _ScriptedEvaluator("cand"),
            _ScriptedEvaluator("champ"),
            ArenaConfig(games=len(script), num_simulations=2, max_plies=4),
        )
        self._script = script

    def run(self, candidate_tag, champion_tag):  # type: ignore[override]
        for result, candidate_color in self._script:
            yield result, candidate_color, "game-over"


def test_strength_gap_is_not_cancelled_by_mirrored_colours():
    """Candidate wins BOTH mirrored games -> win_rate must be 1.0, not 0.5."""
    arena = _ScriptedArena([("1-0", "white"), ("0-1", "black")])
    match = arena.match("cand", "champ")
    assert (match.wins, match.draws, match.losses) == (2, 0, 0)
    assert match.win_rate == pytest.approx(1.0)


def test_weaker_candidate_scores_zero():
    arena = _ScriptedArena([("0-1", "white"), ("1-0", "black")])
    match = arena.match("cand", "champ")
    assert (match.wins, match.draws, match.losses) == (0, 0, 2)
    assert match.win_rate == pytest.approx(0.0)


def test_even_match_stays_fifty_percent():
    arena = _ScriptedArena([("1-0", "white"), ("0-1", "white")])
    match = arena.match("cand", "champ")
    assert (match.wins, match.draws, match.losses) == (1, 0, 1)
    assert match.win_rate == pytest.approx(0.5)
