"""Audit F-2 regression tests: compact state files + newest-2 pruning.

The old save_run_state pickled list(buffer.buffer): ~26 KB per example,
~2.6 GB per state file at night-profile fill, never pruned. Now the buffer
persists as an fp16 npz blob inside the single state file, and old state
files are pruned to the newest two. Planes (0/1) and game-outcome values are
exact in fp16, so resume-reproduces-fresh semantics are unchanged; policy
targets take ~1e-3 relative rounding, asserted (and thereby documented)
below.
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from training.model import MODEL_TINY, GambitNet
from training.regimes import create_run, get_regime
from training.resume import load_latest_run_state, restore_run_state, save_run_state
from training.trainer import TrainingBuffer


def tiny_profile():
    from training.pipeline import TrainingProfile

    return TrainingProfile(
        name="smoke",
        model=MODEL_TINY,
        regime="scratch",
        iterations=1,
        games_per_iteration=1,
        max_plies=4,
        sims=2,
        train_steps=1,
        batch_size=4,
        min_examples=1,
        buffer_size=100,
        seed=0,
    )


def _one_hot_policy(index: int, value: float) -> np.ndarray:
    policy = np.zeros(4672, dtype=np.float32)
    policy[index] = value
    return policy


def _save(paths, step: int, buffer: TrainingBuffer) -> None:
    model = GambitNet(MODEL_TINY)
    save_run_state(
        paths,
        iterations_completed=step,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        buffer=buffer,
        play_rng=np.random.default_rng(step),
        steps=step,
        games_played=step,
        champion_tag=None,
        champion_elo=800.0,
        profile=tiny_profile(),
    )


def _restore_into(state, buffer: TrainingBuffer) -> None:
    fresh_model = GambitNet(MODEL_TINY)
    restore_run_state(
        state,
        model=fresh_model,
        optimizer=torch.optim.Adam(fresh_model.parameters(), lr=1e-3),
        buffer=buffer,
        play_rng=np.random.default_rng(99),
    )


class TestNpzRoundtrip:
    def test_exact_for_fp16_exact_values(self, tmp_path):
        paths = create_run(get_regime("scratch"), "npz-rt", base_dir=tmp_path)
        buffer = TrainingBuffer(max_size=100)
        planes = np.zeros((29, 8, 8), dtype=np.float32)
        planes[0, 0, 0] = 1.0
        examples = [
            {"board_tensor": planes, "policy": _one_hot_policy(7, 1.0), "value": 1.0},
            {"board_tensor": -planes, "policy": _one_hot_policy(11, 0.5), "value": -1.0},
            {"board_tensor": planes * 0.5, "policy": _one_hot_policy(3, 0.25), "value": 0.0},
        ]
        buffer.add(examples)
        _save(paths, 5, buffer)
        state = load_latest_run_state(paths)
        assert state is not None
        restored = TrainingBuffer(max_size=100)
        _restore_into(state, restored)
        assert len(restored.buffer) == 3
        for got, want in zip(restored.buffer, examples):
            np.testing.assert_array_equal(got["board_tensor"], want["board_tensor"])
            np.testing.assert_array_equal(got["policy"], want["policy"])
            assert float(got["value"]) == want["value"]

    def test_fp16_policy_rounding_is_bounded(self, tmp_path):
        """The documented fp16 caveat: policy targets round at ~1e-3."""
        paths = create_run(get_regime("scratch"), "npz-round", base_dir=tmp_path)
        buffer = TrainingBuffer(max_size=10)
        buffer.add(
            [
                {
                    "board_tensor": np.zeros((29, 8, 8), dtype=np.float32),
                    "policy": _one_hot_policy(0, 1.0 / 3.0),
                    "value": 0.25,
                }
            ]
        )
        _save(paths, 1, buffer)
        state = load_latest_run_state(paths)
        assert state is not None
        restored = TrainingBuffer(max_size=10)
        _restore_into(state, restored)
        got = float(restored.buffer[0]["policy"][0])
        delta = abs(got - 1.0 / 3.0)
        assert 0.0 < delta < 1e-3  # rounded (fp16), and only by ~1e-3


class TestPruning:
    def test_only_newest_two_state_files_remain(self, tmp_path):
        paths = create_run(get_regime("scratch"), "prune", base_dir=tmp_path)
        buffer = TrainingBuffer(max_size=10)
        buffer.add(
            [
                {
                    "board_tensor": np.zeros((29, 8, 8), dtype=np.float32),
                    "policy": _one_hot_policy(1, 1.0),
                    "value": 1.0,
                }
            ]
        )
        for step in (1, 2, 3, 4):
            _save(paths, step, buffer)
        remaining = sorted(p.name for p in paths.state.glob("state-*.pt"))
        assert remaining == ["state-000003.pt", "state-000004.pt"]
        state = load_latest_run_state(paths)
        assert state is not None
        assert int(state["steps"]) == 4

    def test_empty_buffer_saves_and_restores(self, tmp_path):
        paths = create_run(get_regime("scratch"), "prune-empty", base_dir=tmp_path)
        _save(paths, 1, TrainingBuffer(max_size=10))
        assert len(list(paths.state.glob("state-*.pt"))) == 1
        state = load_latest_run_state(paths)
        assert state is not None
        restored = TrainingBuffer(max_size=10)
        _restore_into(state, restored)
        assert len(restored.buffer) == 0


class TestLegacyCompat:
    def test_pre_npz_state_still_restores(self):
        model = GambitNet(MODEL_TINY)
        example = {
            "board_tensor": np.zeros((29, 8, 8), dtype=np.float32),
            "policy": _one_hot_policy(2, 1.0),
            "value": 0.5,
        }
        state = {
            "iterations_completed": 1,
            "steps": 3,
            "games_played": 2,
            "model_state": model.state_dict(),
            "optimizer_state": torch.optim.Adam(model.parameters(), lr=1e-3).state_dict(),
            "buffer_examples": [example],
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "play_rng_state": np.random.default_rng(0).bit_generator.state,
            "champion_tag": None,
            "champion_elo": 800.0,
            "profile_signature": {},
        }
        buffer = TrainingBuffer(max_size=10)
        _restore_into(state, buffer)
        assert len(buffer.buffer) == 1
        assert float(buffer.buffer[0]["value"]) == 0.5


class TestSize:
    def test_state_file_is_materially_smaller_than_pickled_buffer(self, tmp_path):
        rng = np.random.default_rng(0)
        examples = [
            {
                "board_tensor": rng.random((29, 8, 8), dtype=np.float32),
                "policy": rng.random(4672, dtype=np.float32) / 100.0,
                "value": 1.0,
            }
            for _ in range(64)
        ]
        buffer = TrainingBuffer(max_size=1000)
        buffer.add(examples)
        pickled = io.BytesIO()
        torch.save(examples, pickled)
        paths = create_run(get_regime("scratch"), "size", base_dir=tmp_path)
        _save(paths, 1, buffer)
        state_size = (paths.state / "state-000001.pt").stat().st_size
        assert state_size < 0.8 * len(pickled.getvalue())
