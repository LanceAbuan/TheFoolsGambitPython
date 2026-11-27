"""Frozen eval-set tests (A88: proper scoring rules, comparable series)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from training.eval_set import (
    EVAL_SET_FILENAME,
    build_eval_set,
    build_from_run,
    load_eval_set,
    score,
)
from training.model import GambitNet, ModelConfig


def _example(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    policy = rng.random(4672).astype(np.float32)
    policy /= policy.sum()
    return {
        "board_tensor": rng.random((104, 8, 8)).astype(np.float32),
        "policy": policy,
        "value": float(rng.choice([-1.0, 0.0, 1.0])),
    }


def _tiny_model() -> GambitNet:
    return GambitNet(
        ModelConfig(
            in_channels=104, num_blocks=1, filters=8, value_channels=4,
            value_fc=16, value_head="wdl",
        )
    )


def test_build_and_load_roundtrip(tmp_path):
    path = build_eval_set([_example(i) for i in range(10)], tmp_path / "e.npz", count=8)
    data = load_eval_set(path)
    assert data["planes"].shape == (8, 104, 8, 8)
    assert data["policies"].shape == (8, 4672)
    assert data["values"].shape == (8,)


def test_build_is_deterministic_and_strided(tmp_path):
    examples = [_example(i) for i in range(20)]
    a = build_eval_set(examples, tmp_path / "a.npz", count=5, seed=7)
    b = build_eval_set(examples, tmp_path / "b.npz", count=5, seed=7)
    assert np.array_equal(load_eval_set(a)["planes"], load_eval_set(b)["planes"])


def test_build_rejects_empty():
    with pytest.raises(ValueError, match="zero examples"):
        build_eval_set([], "unused.npz")


def test_score_returns_finite_proper_metrics(tmp_path):
    data = load_eval_set(build_eval_set([_example(i) for i in range(16)], tmp_path / "e.npz", count=16))
    scores = score(_tiny_model(), data, torch.device("cpu"))
    for key in (
        "eval_policy_ce",
        "eval_policy_top1",
        "eval_value_ce",
        "eval_value_brier",
        "eval_mean_p_true",
        "eval_score_mae",
    ):
        assert key in scores and np.isfinite(scores[key]), key
    assert scores["eval_n"] == 16
    assert 0.0 <= scores["eval_policy_top1"] <= 1.0
    assert 0.0 <= scores["eval_mean_p_true"] <= 1.0


def test_score_prefers_the_matching_model(tmp_path):
    """A model trained to imitate the eval set must score better than noise.

    This is the property the whole metric rests on: if it cannot order an
    imitator above a random net, it cannot detect progress.
    """
    data = load_eval_set(
        build_eval_set([_example(i) for i in range(32)], tmp_path / "e.npz", count=32)
    )
    device = torch.device("cpu")
    planes = torch.from_numpy(data["planes"])
    policies = torch.from_numpy(data["policies"])

    random_model = _tiny_model()
    random_scores = score(random_model, data, device)

    imitator = _tiny_model()
    opt = torch.optim.Adam(imitator.parameters(), lr=3e-3)
    imitator.train()
    for _ in range(300):
        out_p, _ = imitator(planes)
        loss = -(policies * torch.log_softmax(out_p, dim=-1)).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    imitator_scores = score(imitator, data, device)

    assert imitator_scores["eval_policy_ce"] < random_scores["eval_policy_ce"]
    assert imitator_scores["eval_policy_top1"] >= random_scores["eval_policy_top1"]


def test_build_from_run_uses_newest_state(tmp_path, monkeypatch):
    """build_from_run reads the newest state's replay blob."""
    from training import resume

    run_root = tmp_path / "runs" / "scratch" / "evalrun"
    (run_root / "state").mkdir(parents=True)
    examples = [_example(i) for i in range(6)]
    blob = resume._buffer_npz_bytes(examples)
    torch.save({"buffer_npz": blob}, run_root / "state" / "state-000096.pt")
    path = build_from_run(run_root, count=6)
    assert path.name == EVAL_SET_FILENAME
    assert load_eval_set(path)["planes"].shape[0] == 6
def test_build_from_spool_filters_worker(tmp_path):
    """A NEUTRAL set can be built from engine-labelled rows only."""
    import base64

    import numpy as np

    from training.eval_set import build_from_spool
    from training.rust_workers import write_game_record

    rng = np.random.default_rng(0)
    for worker in ("sf", "w0"):
        policy = np.zeros(4672, dtype=np.float32)
        policy[0] = 1.0
        record = {
            "schema": "gambit.selfplay.v2", "encoding": "f32.le.b64", "weight_version": 1,
            "worker_id": worker, "seed": 1, "opening_seed": None, "sims": 0,
            "max_plies": 4, "temperature": 0.0, "temperature_plies": 0,
            "opening_fen": "start", "result": "1/2-1/2", "termination": "game-over",
            "ucis": ["e2e4"], "moves": ["e4"], "fens": ["x"], "ply_count": 1,
            "examples": [{
                "board_tensor_b64": base64.b64encode(
                    np.zeros((104, 8, 8), dtype="<f4").tobytes()).decode(),
                "policy_b64": base64.b64encode(policy.astype("<f4").tobytes()).decode(),
                "value": 0.0,
            }],
        }
        write_game_record(tmp_path / "spool", record)

    path = build_from_spool(tmp_path / "spool", worker_prefix="sf", count=8)
    data = np.load(path)
    assert data["planes"].shape[0] == 1, "only the sf game contributes"
