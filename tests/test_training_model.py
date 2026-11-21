"""Model v2 tests: shapes, param budget, masked softmax, CPU (+CUDA if present)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import pytest
import torch

from training.encoder import NUM_PLANES, POLICY_MOVES, encode_state, legal_mask
from training.model import (
    MODEL_BASE,
    MODEL_SMALL,
    MODEL_TINY,
    GambitNet,
    ModelConfig,
    masked_policy_softmax,
)


class TestModelConfig:
    def test_policy_moves(self):
        assert MODEL_TINY.policy_moves == 4672
        assert POLICY_MOVES == 4672

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            ModelConfig(num_blocks=0)
        with pytest.raises(ValueError):
            ModelConfig(filters=0)


class TestGambitNet:
    def test_forward_shapes_cpu(self):
        model = GambitNet(MODEL_TINY)
        model.eval()
        x = torch.randn(2, NUM_PLANES, 8, 8)
        with torch.no_grad():
            policy, value = model(x)
        assert policy.shape == (2, POLICY_MOVES)
        assert value.shape == (2,)
        assert float(value.abs().max()) <= 1.0

    def test_tiny_param_count_under_1m(self):
        model = GambitNet(MODEL_TINY)
        count = model.count_parameters()
        assert count < 1_000_000, f"tiny config must stay under 1M params, got {count}"
        print(f"tiny config params: {count:,}")

    def test_small_and_base_param_targets(self):
        small = GambitNet(MODEL_SMALL).count_parameters()
        base = GambitNet(MODEL_BASE).count_parameters()
        assert 400_000 < small < 2_000_000
        assert 3_000_000 < base < 8_000_000

    def test_backward_pass_flows(self):
        model = GambitNet(MODEL_TINY)
        x = torch.randn(4, NUM_PLANES, 8, 8)
        policy, value = model(x)
        loss = policy.sum() + value.sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert any(float(g.abs().sum()) > 0 for g in grads)

    def test_forward_on_encoded_position(self):
        """The encoder and model agree on layout: a real position runs through."""
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        model = GambitNet(MODEL_TINY)
        model.eval()
        tensor = torch.from_numpy(encode_state(board)).unsqueeze(0)
        with torch.no_grad():
            policy, value = model(tensor)
        assert policy.shape == (1, POLICY_MOVES)
        assert -1.0 <= float(value.item()) <= 1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_cuda(self):
        model = GambitNet(MODEL_TINY).to(torch.device("cuda"))
        model.eval()
        x = torch.randn(8, NUM_PLANES, 8, 8, device="cuda")
        with torch.no_grad():
            policy, value = model(x)
        assert policy.shape == (8, POLICY_MOVES)
        assert policy.device.type == "cuda"
        assert -1.0 <= float(value.abs().max()) <= 1.0

    def test_config_roundtrip(self):
        config = ModelConfig.from_dict(MODEL_SMALL.to_dict())
        assert config == MODEL_SMALL


class TestMaskedSoftmax:
    def test_illegal_moves_get_zero(self):
        board = chess.Board()
        model = GambitNet(MODEL_TINY)
        model.eval()
        tensor = torch.randn(1, NUM_PLANES, 8, 8)
        with torch.no_grad():
            logits, _ = model(tensor)
        mask = torch.from_numpy(legal_mask(board)).unsqueeze(0)
        probs = masked_policy_softmax(logits, mask)
        assert probs.shape == (1, POLICY_MOVES)
        illegal = probs[0][mask[0] == 0]
        assert float(illegal.abs().sum()) == 0.0
        assert abs(float(probs[0][mask[0] == 1].sum()) - 1.0) < 1e-5
        assert board.legal_moves.count() == int((probs[0] > 0).sum())
