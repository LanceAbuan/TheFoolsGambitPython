"""Tests for training/model.py — CPU-only, no GPU required."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pytest

# Force CPU for tests
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from training.model import (
    ChessNet,
    ResidualBlock,
    NUM_CHANNELS,
    NUM_RESIDUAL_BLOCKS,
    RESIDUAL_FILTERS,
    POLICY_HEAD_CHANNELS,
    VALUE_HEAD_CHANNELS,
    NUM_POSSIBLE_MOVES,
)


class TestResidualBlock:
    def test_forward_shape(self):
        block = ResidualBlock(32)
        x = torch.randn(1, 32, 8, 8)
        out = block(x)
        assert out.shape == (1, 32, 8, 8)

    def test_residual_connection(self):
        """Output should include identity + transform."""
        block = ResidualBlock(16)
        x = torch.randn(2, 16, 8, 8)
        out = block(x)
        assert out.shape == x.shape


class TestChessNet:
    @pytest.fixture
    def model(self):
        return ChessNet()

    def test_forward_shapes(self, model):
        x = torch.randn(1, 8, 8, NUM_CHANNELS)
        policy, value = model(x)
        assert policy.shape == (1, NUM_POSSIBLE_MOVES)
        assert value.shape == (1, 1)

    def test_batch_forward(self, model):
        x = torch.randn(4, 8, 8, NUM_CHANNELS)
        policy, value = model(x)
        assert policy.shape == (4, NUM_POSSIBLE_MOVES)
        assert value.shape == (4, 1)

    def test_value_range(self, model):
        model.eval()
        x = torch.randn(1, 8, 8, NUM_CHANNELS)
        with torch.no_grad():
            _, value = model(x)
            assert -1.0 <= value.item() <= 1.0

    def test_policy_sum(self, model):
        model.eval()
        x = torch.randn(1, 8, 8, NUM_CHANNELS)
        with torch.no_grad():
            policy_logits, _ = model(x)
            probs = torch.softmax(policy_logits, dim=-1)
            assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_deterministic_forward(self, model):
        model.eval()
        x = torch.randn(1, 8, 8, NUM_CHANNELS)
        torch.manual_seed(42)
        with torch.no_grad():
            p1, v1 = model(x)
        torch.manual_seed(42)
        with torch.no_grad():
            p2, v2 = model(x)
        assert torch.allclose(p1, p2)
        assert torch.allclose(v1, v2)

    def test_custom_config(self):
        model = ChessNet(num_residual_blocks=4, residual_filters=64)
        x = torch.randn(1, 8, 8, NUM_CHANNELS)
        policy, value = model(x)
        assert policy.shape == (1, NUM_POSSIBLE_MOVES)
        assert value.shape == (1, 1)

    def test_evaluate(self, model):
        # evaluate() expects (H, W, C) — matches board_to_tensor output shape
        board_tensor = np.random.randn(8, 8, NUM_CHANNELS).astype(np.float32)
        policy_probs, value = model.evaluate(board_tensor)
        assert policy_probs.shape == (NUM_POSSIBLE_MOVES,)
        assert -1.0 <= value <= 1.0
        # Policy should sum to ~1
        assert abs(policy_probs.sum() - 1.0) < 1e-4

    def test_evaluate_with_legal_mask(self, model):
        board_tensor = np.random.randn(8, 8, NUM_CHANNELS).astype(np.float32)
        legal_mask = torch.zeros(NUM_POSSIBLE_MOVES)
        legal_mask[0] = 1.0
        legal_mask[100] = 1.0
        policy_probs, value = model.evaluate(board_tensor, legal_mask)
        # Only legal moves should have non-zero probability
        illegal_probs = policy_probs[legal_mask == 0]
        assert torch.allclose(illegal_probs, torch.zeros_like(illegal_probs), atol=1e-8)


class TestConstants:
    def test_channels(self):
        assert NUM_CHANNELS == 16

    def test_possible_moves(self):
        assert NUM_POSSIBLE_MOVES == 4096

    def test_default_residual_blocks(self):
        assert NUM_RESIDUAL_BLOCKS == 2
