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
    value_head_scalar,
)
from training.trainer import wdl_targets, wdl_value_loss


class TestModelConfig:
    def test_policy_moves(self):
        assert MODEL_TINY.policy_moves == 4672
        assert POLICY_MOVES == 4672

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            ModelConfig(num_blocks=0)
        with pytest.raises(ValueError):
            ModelConfig(filters=0)

    def test_value_head_defaults_to_scalar(self):
        """WP-C: every pre-existing config keeps the scalar tanh head."""
        assert MODEL_TINY.value_head == "scalar"
        assert MODEL_SMALL.value_head == "scalar"
        assert MODEL_BASE.value_head == "scalar"
        assert ModelConfig().value_head == "scalar"
        assert ModelConfig().value_outputs == 1

    def test_value_head_wdl_accepted_and_validated(self):
        wdl = ModelConfig(value_head="wdl")
        assert wdl.value_outputs == 3
        with pytest.raises(ValueError):
            ModelConfig(value_head="categorical")  # type: ignore[arg-type]


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


class TestWDLValueHead:
    """WP-C: the 3-logit (W, D, L) value head and its scalar default."""

    WDL_TINY = ModelConfig(num_blocks=1, filters=16, value_channels=8, value_fc=32, value_head="wdl")

    def test_wdl_forward_shapes(self):
        """Acceptance: wdl forward -> (N, 4672) policy + (N, 3) value logits."""
        model = GambitNet(self.WDL_TINY)
        model.eval()
        x = torch.randn(4, NUM_PLANES, 8, 8)
        with torch.no_grad():
            policy, value = model(x)
        assert policy.shape == (4, POLICY_MOVES)
        assert value.shape == (4, 3)
        # Raw logits: NOT tanh-bounded.
        assert model.value_fc2.out_features == 3

    def test_scalar_path_identical_to_today(self):
        """The default head is unchanged: (N,) tanh-bounded, 1-output linear."""
        model = GambitNet(MODEL_TINY)
        model.eval()
        x = torch.randn(2, NUM_PLANES, 8, 8)
        with torch.no_grad():
            policy, value = model(x)
        assert policy.shape == (2, POLICY_MOVES)
        assert value.shape == (2,)
        assert -1.0 <= float(value.abs().max()) <= 1.0
        assert model.value_fc2.out_features == 1
        assert model.value_fc2.weight.shape == (1, MODEL_TINY.value_fc)

    def test_pre_wpc_config_dict_loads_as_scalar(self):
        """Old checkpoints' config dicts lack `value_head` and must load as
        scalar — the default preserves every existing checkpoint path."""
        legacy = {k: v for k, v in MODEL_TINY.to_dict().items() if k != "value_head"}
        config = ModelConfig.from_dict(legacy)
        assert config == MODEL_TINY
        assert config.value_head == "scalar"

    def test_wdl_config_roundtrip(self):
        config = ModelConfig.from_dict(self.WDL_TINY.to_dict())
        assert config == self.WDL_TINY

    def test_wdl_backward_pass_flows(self):
        model = GambitNet(self.WDL_TINY)
        x = torch.randn(4, NUM_PLANES, 8, 8)
        policy, value = model(x)
        assert value.shape == (4, 3)
        (policy.sum() + value.sum()).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert any(float(g.abs().sum()) > 0 for g in grads)


class TestValueHeadScalar:
    """The single wdl->scalar derivation every evaluator shares."""

    def test_wdl_logits_map_to_p_win_minus_p_loss(self):
        logits = torch.tensor([[10.0, 0.0, -10.0], [-10.0, 0.0, 10.0], [0.0, 10.0, 0.0]])
        scalar = value_head_scalar(logits)
        # softmax([10,0,-10]) ~ one-hot W -> +1; mirrored -> -1; draw-heavy -> ~0
        assert float(scalar[0]) == pytest.approx(1.0, abs=1e-4)
        assert float(scalar[1]) == pytest.approx(-1.0, abs=1e-4)
        assert float(scalar[2]) == pytest.approx(0.0, abs=1e-3)
        assert scalar.shape == (3,)

    def test_exact_softmax_math(self):
        logits = torch.tensor([[0.0, 0.0, 0.0]])
        # uniform -> 1/3 - 1/3 = 0
        assert float(value_head_scalar(logits)[0]) == pytest.approx(0.0, abs=1e-6)
        logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
        assert float(value_head_scalar(logits)[0]) == pytest.approx(0.5, abs=1e-6)

    def test_scalar_heads_pass_through(self):
        flat = torch.tensor([0.25, -0.75])
        assert torch.equal(value_head_scalar(flat), flat)
        column = torch.tensor([[0.25], [-0.75]])
        assert torch.equal(value_head_scalar(column), flat)


class TestWDLLoss:
    """WP-C: cross-entropy on the one-hot game result for the wdl head."""

    def test_wdl_targets_mapping(self):
        values = torch.tensor([1.0, 0.0, -1.0])
        assert wdl_targets(values).tolist() == [0, 1, 2]

    def test_wdl_targets_tolerate_buffer_rounding(self):
        """fp16 replay-buffer round-trips must never flip a label class."""
        values = torch.tensor([0.999, -0.999, 0.001, -0.001, 0.5, -0.5])
        assert wdl_targets(values).tolist() == [0, 2, 1, 1, 1, 1]

    def test_perfect_predictions_zero_loss(self):
        logits = torch.tensor([[100.0, 0.0, -100.0], [0.0, 100.0, 0.0], [-100.0, 0.0, 100.0]])
        values = torch.tensor([1.0, 0.0, -1.0])
        assert float(wdl_value_loss(logits, values)) == pytest.approx(0.0, abs=1e-4)

    def test_loss_decreases_over_20_steps_fixed_batch(self):
        """Acceptance: the wdl loss decreases over 20 steps on a fixed batch."""
        torch.manual_seed(0)
        config = ModelConfig(num_blocks=1, filters=16, value_channels=8, value_fc=32, value_head="wdl")
        model = GambitNet(config)
        tensors = torch.randn(16, NUM_PLANES, 8, 8)
        policies = torch.full((16, POLICY_MOVES), 1.0 / POLICY_MOVES)
        values = torch.tensor([1.0, 0.0, -1.0] * 5 + [1.0])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        def step_loss() -> float:
            optimizer.zero_grad()
            pred_policy, pred_value = model(tensors)
            log_probs = torch.log_softmax(pred_policy, dim=-1)
            policy_loss = -(policies * log_probs).sum(dim=-1).mean()
            value_loss = wdl_value_loss(pred_value, values)
            (policy_loss + value_loss).backward()
            optimizer.step()
            return float((policy_loss + value_loss).item())

        first = step_loss()
        last = first
        for _ in range(19):
            last = step_loss()
        assert last < first, f"wdl loss did not decrease: {first:.4f} -> {last:.4f}"
        # The value head genuinely learned the fixed outcomes: near-zero CE.
        model.eval()
        with torch.no_grad():
            _, pred_value = model(tensors)
        assert float(wdl_value_loss(pred_value, values)) < 0.25

    def test_scalar_loss_decreases_over_20_steps_fixed_batch(self):
        """The flag-off branch keeps the exact pre-WP-C MSE composition."""
        torch.manual_seed(0)
        config = ModelConfig(num_blocks=1, filters=16, value_channels=8, value_fc=32)
        model = GambitNet(config)
        tensors = torch.randn(16, NUM_PLANES, 8, 8)
        policies = torch.full((16, POLICY_MOVES), 1.0 / POLICY_MOVES)
        values = torch.zeros(16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        def step_loss() -> float:
            optimizer.zero_grad()
            pred_policy, pred_value = model(tensors)
            log_probs = torch.log_softmax(pred_policy, dim=-1)
            policy_loss = -(policies * log_probs).sum(dim=-1).mean()
            value_loss = torch.nn.functional.mse_loss(pred_value, values)
            (policy_loss + value_loss).backward()
            optimizer.step()
            return float((policy_loss + value_loss).item())

        first = step_loss()
        last = first
        for _ in range(19):
            last = step_loss()
        assert last < first


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
