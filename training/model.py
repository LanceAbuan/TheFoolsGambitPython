"""Neural network models for chess evaluation.

Legacy v1 (`ChessNet`, below): 16-plane input, flat 4096 from-to policy head.
Kept importable for the legacy trainer/server; tainted by the §10 promotion
bug (all promotions share one logit) — do not train new models with it.

Model v2 (`GambitNet`, bottom of this file): AlphaZero-canonical 29-plane
input, 73x64 = 4672 policy logits (real underpromotion planes), dataclass
config, CPU + CUDA. Value head is configurable (WP-C): "scalar" tanh by
default, or "wdl" — 3 class logits whose softmax gives (p_win, p_draw,
p_loss) from the side-to-move's perspective.

Architecture (both): residual tower with batch norm, conv policy head, tanh
value head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

NUM_CHANNELS = 16
NUM_RESIDUAL_BLOCKS = 2
RESIDUAL_FILTERS = 32
POLICY_HEAD_CHANNELS = 4
VALUE_HEAD_CHANNELS = 32
NUM_POSSIBLE_MOVES = 4096


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and ReLU."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ChessNet(nn.Module):
    """AlphaZero-style chess neural network.
    
    Args:
        num_residual_blocks: Number of residual blocks in the tower.
        residual_filters: Number of filters in each residual block.
    """
    
    def __init__(self, num_residual_blocks=NUM_RESIDUAL_BLOCKS, residual_filters=RESIDUAL_FILTERS):
        super().__init__()
        self.residual_filters = residual_filters
        
        self.input_conv = nn.Conv2d(NUM_CHANNELS, residual_filters, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(residual_filters)
        
        self.residual_tower = nn.Sequential(*[
            ResidualBlock(residual_filters) for _ in range(num_residual_blocks)
        ])
        
        self.policy_conv = nn.Conv2d(residual_filters, POLICY_HEAD_CHANNELS, 1)
        self.policy_bn = nn.BatchNorm2d(POLICY_HEAD_CHANNELS)
        self.policy_fc = nn.Linear(POLICY_HEAD_CHANNELS * 64, NUM_POSSIBLE_MOVES)
        
        self.value_conv = nn.Conv2d(residual_filters, VALUE_HEAD_CHANNELS, 1)
        self.value_bn = nn.BatchNorm2d(VALUE_HEAD_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_HEAD_CHANNELS * 64, VALUE_HEAD_CHANNELS)
        self.value_fc2 = nn.Linear(VALUE_HEAD_CHANNELS, 1)
    
    def forward(self, x):
        x = x.to(self.input_conv.weight.device)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.residual_tower(out)
        
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.reshape(-1, POLICY_HEAD_CHANNELS * 64)
        policy = self.policy_fc(policy)
        
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.reshape(-1, VALUE_HEAD_CHANNELS * 64)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def evaluate(self, board_tensor, legal_mask=None):
        """Evaluate a single board position.
        
        Args:
            board_tensor: numpy array of shape (16, 8, 8)
            legal_mask: optional tensor of shape (NUM_POSSIBLE_MOVES,) with 1 for legal moves
        
        Returns:
            tuple: (policy_probs, value) where policy_probs is over legal moves,
                   value is in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(board_tensor).unsqueeze(0)
            device = next(self.parameters()).device
            x = x.to(device)
            policy_logits, value = self(x)
            
            if legal_mask is not None:
                mask = legal_mask.unsqueeze(0).float().to(device)
                policy_logits = policy_logits + (1 - mask) * -1e10
                policy_logits = policy_logits.masked_fill(~mask.bool(), float('-inf'))
            
            policy_probs = F.softmax(policy_logits, dim=-1)
            
            return policy_probs.squeeze(0), value.squeeze(0).item()


# ---------------------------------------------------------------------------
# Model v2 (WP-C): AlphaZero-canonical policy head over 73x64 = 4672 logits.
# ---------------------------------------------------------------------------

NUM_POLICY_MOVES_V2 = 4672

# WP-C: value-head layout. "scalar" is the pre-WP-C tanh head (every existing
# checkpoint and test path); "wdl" is 3 logits (W, D, L) from the side-to-move's
# perspective, trained with cross-entropy on the one-hot game result.
ValueHeadKind = Literal["scalar", "wdl"]
VALUE_HEAD_KINDS: tuple[str, ...] = ("scalar", "wdl")


@dataclass(frozen=True)
class ModelConfig:
    """Shape knobs for `GambitNet`. Presets below match plan §4/M2 targets."""

    in_channels: int = 29
    num_blocks: int = 2
    filters: int = 32
    conv_width: int = 3
    policy_planes: int = 73
    value_channels: int = 8
    value_fc: int = 64
    # WP-C: "scalar" (default) keeps the tanh value head bit-identical to the
    # pre-WP-C model; "wdl" swaps the final linear for 3 class logits.
    value_head: ValueHeadKind = "scalar"

    def __post_init__(self) -> None:
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.filters < 1 or self.in_channels < 1:
            raise ValueError("filters and in_channels must be >= 1")
        if self.value_head not in VALUE_HEAD_KINDS:
            raise ValueError(
                f"value_head must be one of {VALUE_HEAD_KINDS}, got {self.value_head!r}"
            )

    @property
    def policy_moves(self) -> int:
        return self.policy_planes * 64

    @property
    def value_outputs(self) -> int:
        """Width of the value head's final linear: 3 for wdl, 1 for scalar."""
        return 3 if self.value_head == "wdl" else 1

    def to_dict(self) -> dict[str, object]:
        return {
            "in_channels": self.in_channels,
            "num_blocks": self.num_blocks,
            "filters": self.filters,
            "conv_width": self.conv_width,
            "policy_planes": self.policy_planes,
            "value_channels": self.value_channels,
            "value_fc": self.value_fc,
            # WP-C: stamped into checkpoints so the loader rebuilds the right
            # head. Pre-WP-C checkpoints lack the key; `from_dict` defaults
            # them to "scalar" exactly as before.
            "value_head": self.value_head,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ModelConfig:
        return cls(**data)  # type: ignore[arg-type]


# tiny: smoke/tests/CI — well under 1M params (~81k)
MODEL_TINY = ModelConfig(num_blocks=2, filters=32)
# small: ~530k params — the "6-block/64-f ~1M" M2 preset
MODEL_SMALL = ModelConfig(num_blocks=6, filters=64, value_fc=128)
# base: ~5.1M params — the "10-block/128-f ~6M" M2 preset
MODEL_BASE = ModelConfig(num_blocks=10, filters=128, value_channels=32, value_fc=512)


class ResidualBlockV2(nn.Module):
    """Pre-activation residual block (conv-bn-relu-conv-bn + identity)."""

    def __init__(self, channels: int, conv_width: int) -> None:
        super().__init__()
        padding = conv_width // 2
        self.conv1 = nn.Conv2d(channels, channels, conv_width, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, conv_width, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class GambitNet(nn.Module):
    """Model v2: 29-plane input, 73x64 policy head, configurable value head.

    Policy logits are indexed in side-to-move perspective coordinates, exactly
    matching `training.encoder` (see encoder module docstring) — including real
    underpromotion planes, so the §10 promotion bug cannot reappear.

    Value head (WP-C): `value_head="scalar"` (default) returns one tanh-bounded
    value per position; `value_head="wdl"` returns 3 raw class logits
    (W, D, L from the side-to-move's perspective). Search consumes the derived
    scalar ``p_win − p_loss`` via `value_head_scalar` — the MCTS interface is
    unchanged.
    """

    def __init__(self, config: ModelConfig = MODEL_TINY) -> None:
        super().__init__()
        self.config = config
        padding = config.conv_width // 2
        self.input_conv = nn.Conv2d(
            config.in_channels, config.filters, config.conv_width, padding=padding, bias=False
        )
        self.input_bn = nn.BatchNorm2d(config.filters)
        self.residual_tower = nn.Sequential(
            *[ResidualBlockV2(config.filters, config.conv_width) for _ in range(config.num_blocks)]
        )
        # Policy: 1x1 conv straight to 73 planes -> flatten to 4672 logits.
        self.policy_conv = nn.Conv2d(config.filters, config.policy_planes, 1, bias=True)
        # Value: 1x1 conv -> MLP -> tanh scalar (or 3 W/D/L logits, WP-C).
        self.value_conv = nn.Conv2d(config.filters, config.value_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(config.value_channels)
        self.value_fc1 = nn.Linear(config.value_channels * 64, config.value_fc)
        self.value_fc2 = nn.Linear(config.value_fc, config.value_outputs)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, 8, 8) -> (policy_logits (B, 4672), value).

        value is (B,) tanh scalars for `value_head="scalar"`, or (B, 3) raw
        W/D/L logits for `value_head="wdl"` (side-to-move perspective).
        """
        x = x.to(self.device)
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.residual_tower(out)

        policy = self.policy_conv(out)
        policy = policy.reshape(-1, self.config.policy_moves)

        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.reshape(value.shape[0], -1)
        value = F.relu(self.value_fc1(value))
        if self.config.value_head == "wdl":
            return policy, self.value_fc2(value)
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)
        return policy, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def value_head_scalar(values: torch.Tensor) -> torch.Tensor:
    """Collapse a value-head output to the scalar MCTS consumes (WP-C).

    scalar head: (B,) or (B, 1) tanh values -> (B,) unchanged. wdl head:
    (B, 3) logits -> (B,) ``p_win − p_loss`` from the side-to-move's
    perspective (softmax over the W/D/L logits). This is the ONLY place the
    wdl->scalar derivation happens, so every evaluator that feeds
    `MCTSSearch` keeps the exact pre-WP-C interface.
    """
    if values.dim() == 2 and values.shape[-1] == 3:
        probs = torch.softmax(values, dim=-1)
        return probs[..., 0] - probs[..., 2]
    return values.reshape(values.shape[0], -1)[..., 0]


def masked_policy_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax restricted to legal moves.

    logits: (B, 4672) raw policy logits; mask: (B, 4672) with 1.0/True at
    legal indices. Illegal entries get exactly 0 probability.
    """
    mask_bool = mask.bool()
    masked = logits.masked_fill(~mask_bool, float("-inf"))
    return F.softmax(masked, dim=-1)
