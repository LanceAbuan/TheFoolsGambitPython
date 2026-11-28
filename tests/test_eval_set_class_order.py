

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch


class _StubWinner(torch.nn.Module):
    """Always predicts P(win)=1 for the model's index 0."""

    def forward(self, x):
        logits = torch.full((x.shape[0], 3), -20.0)
        logits[:, 0] = 20.0
        return torch.zeros((x.shape[0], 4672)), logits


def test_value_class_order_is_win_draw_loss():
    """Index 0 must be WIN (trainer.wdl_targets), not loss.

    Regression: the first scorer assumed -1..+1 -> 0..2, so p(true) was read off
    the wrong class. A model that always predicts index 0 must score a WIN
    target as near-certain and a LOSS target as near-impossible.
    """
    from training.eval_set import score

    data = {
        "planes": np.zeros((2, 104, 8, 8), dtype=np.float32),
        "policies": np.zeros((2, 4672), dtype=np.float32),
        "values": np.array([1.0, -1.0], dtype=np.float32),
    }
    data["policies"][:, 0] = 1.0
    scores = score(_StubWinner(), data, torch.device("cpu"))
    # one win target (p_true ~ 1.0) + one loss target (p_true ~ 0.0) -> mean ~0.5
    assert scores["eval_mean_p_true"] == pytest.approx(0.5, abs=0.05)
