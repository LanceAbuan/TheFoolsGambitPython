"""U2 playout cap randomization tests (KataGo, Wu 2020 sec 3.1).

The critical regression is flag-off bit-identity: with
playout_randomization=False (and search_kind="puct") self-play must be
BYTE-identical to the pre-U1/U2 pipeline. The pre-change golden lives in
tests/fixtures/search_golden.json and was captured BEFORE any search edit.
"""
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest

from training.encoder import legal_move_map
from training.fastpath import disable_fastpath
from training.selfplay import SearchConfig, SelfPlayConfig, self_play_game

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "search_golden.json"

# name -> (sims, plies, temperature_plies, seed) - matches the capture script
GOLDEN_CASES = {
    "tau1_sims8_plies6_seed3": (8, 6, 30, 3),
    "argmax_sims4_plies5_seed11": (4, 5, 0, 11),
    "mixed_sims16_plies8_seed5": (16, 8, 2, 5),
}


class StubEvaluator:
    """Uniform legal priors, constant value - matches the golden capture."""

    def evaluate(self, boards):
        n = len(boards)
        return np.zeros((n, 4672), dtype=np.float32), np.full(n, 0.0, dtype=np.float32)


class CountingStubEvaluator(StubEvaluator):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, boards):
        self.calls += 1
        return super().evaluate(boards)


@pytest.fixture(autouse=True)
def _fastpath_off():
    disable_fastpath()
    yield


def golden_config(sims: int, plies: int, temperature_plies: int) -> SelfPlayConfig:
    return SelfPlayConfig(
        search=SearchConfig(num_simulations=sims, root_noise=False),
        max_plies=plies,
        temperature_plies=temperature_plies,
    )


def digest(result) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(result.moves).encode())
    h.update(result.result.encode())
    h.update(result.pgn.encode())
    for ex in result.examples:
        h.update(ex.planes.tobytes())
        h.update(np.ascontiguousarray(ex.policy).tobytes())
        h.update(np.asarray(ex.value, dtype=np.float64).tobytes())
    return h.hexdigest()


class TestFlagOffBitIdentity:
    def test_matches_pre_change_golden(self):
        """P7 regression: flag-off self-play reproduces the pre-U1/U2 bytes."""
        golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
        for name, (sims, plies, temperature_plies, seed) in GOLDEN_CASES.items():
            result = self_play_game(
                StubEvaluator(),
                golden_config(sims, plies, temperature_plies),
                rng=np.random.default_rng(seed),
            )
            assert digest(result) == golden[name]["digest"], name
            assert result.ply_count == golden[name]["plies"], name
            assert len(result.examples) == golden[name]["examples"], name


class TestFlagOnSemantics:
    def _cap_config(self, **overrides) -> SelfPlayConfig:
        base = SelfPlayConfig(
            search=SearchConfig(num_simulations=8, root_noise=False),
            max_plies=1,
            temperature_plies=0,
            playout_randomization=True,
            playout_cap_start_ply=0,
        )
        return replace(base, **overrides)

    def test_full_turn_call_count(self):
        """A full turn costs 1 root eval + num_simulations leaf evals."""
        ev = CountingStubEvaluator()
        self_play_game(ev, self._cap_config(playout_cap_full_prob=1.0), rng=np.random.default_rng(0))
        assert ev.calls == 9

    def test_fast_turn_call_count_and_no_record(self):
        """Fast turns cost 1 + n evals (n = max(fast_min, fraction*sims) = 2)
        and are NOT recorded for training (paper default)."""
        config = self._cap_config(playout_cap_full_prob=0.0)
        ev = CountingStubEvaluator()
        result = self_play_game(ev, config, rng=np.random.default_rng(0))
        assert ev.calls == 3  # 1 root + fast_n=2, no terminal leaves in 1 ply
        assert result.ply_count == 1  # the move was still played
        assert len(result.examples) == 0  # ...but not recorded

    def test_full_turn_records_example(self):
        config = self._cap_config(playout_cap_full_prob=1.0)
        result = self_play_game(StubEvaluator(), config, rng=np.random.default_rng(0))
        assert len(result.examples) == 1 == result.ply_count

    def test_start_ply_boundary(self):
        """Plies before start_ply are always full AND recorded."""
        base = SelfPlayConfig(
            search=SearchConfig(num_simulations=4, root_noise=False),
            max_plies=6,
            temperature_plies=0,
            playout_randomization=True,
            playout_cap_full_prob=0.0,
            playout_cap_start_ply=4,
        )
        result = self_play_game(StubEvaluator(), base, rng=np.random.default_rng(5))
        assert result.ply_count == 6  # moves were still played on fast turns
        assert len(result.examples) == 4  # plies 0-3 recorded, 4-5 not

    def test_record_fast_records_correct_targets(self):
        """playout_cap_record_fast=True: fast plies land in the buffer with
        policy targets that are proper distributions over legal moves."""
        config = SelfPlayConfig(
            search=SearchConfig(num_simulations=4, root_noise=False),
            max_plies=6,
            temperature_plies=0,
            playout_randomization=True,
            playout_cap_full_prob=0.0,
            playout_cap_start_ply=0,
            playout_cap_record_fast=True,
        )
        result = self_play_game(StubEvaluator(), config, rng=np.random.default_rng(5))
        assert len(result.examples) == result.ply_count
        board = chess.Board()
        for example, san in zip(result.examples, result.moves):
            legal = set(legal_move_map(board).keys())
            nonzero = set(np.nonzero(example.policy)[0])
            assert nonzero <= legal
            assert abs(float(example.policy.sum()) - 1.0) < 1e-5
            board.push_san(san)

    def test_flag_on_deterministic_and_diverges_from_flag_off(self):
        config_off = SelfPlayConfig(
            search=SearchConfig(num_simulations=8, root_noise=False),
            max_plies=8,
            temperature_plies=0,
        )
        config_on = replace(
            config_off,
            playout_randomization=True,
            playout_cap_full_prob=0.5,
            playout_cap_start_ply=0,
        )
        off = self_play_game(StubEvaluator(), config_off, rng=np.random.default_rng(9))
        on_a = self_play_game(StubEvaluator(), config_on, rng=np.random.default_rng(9))
        on_b = self_play_game(StubEvaluator(), config_on, rng=np.random.default_rng(9))
        assert digest(on_a) == digest(on_b)  # seeded -> reproducible
        assert digest(on_a) != digest(off)  # cap draws consume the rng stream

    def test_gumbel_with_playout_cap_is_a_config_error(self):
        """Pinned U1xU2 interaction (documented behavior, not an accident).

        KataGo cap semantics (which plies are capped/recorded) presume PUCT
        visit-count targets; a fast gumbel ply would record the Eq. 11
        improved policy of a reduced-budget search - an unvalidated target
        construction. Both flags on -> ValueError, never silent reshaping.
        """
        with pytest.raises(ValueError, match="PUCT search path only"):
            SelfPlayConfig(
                search=SearchConfig(num_simulations=8, search_kind="gumbel"),
                playout_randomization=True,
            )
