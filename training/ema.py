"""WP4: exponential moving average of model weights.

The EMA weights are exported to the workers (ONNX) and used as the gate
candidate. Rationale (research cycle 13): parameter EMA beat the live iterate
by 30-65 Elo in large-scale self-play RL; it costs one extra float copy of the
model state and one fused update per optimizer step.
"""
from __future__ import annotations

import torch


class EmaWeights:
    """Shadow copy of the floating-point model state, updated per optimizer step."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.enabled = self.decay > 0.0
        self._shadow: dict[str, torch.Tensor] = {
            name: tensor.detach().clone().float()
            for name, tensor in model.state_dict().items()
            if torch.is_floating_point(tensor)
        }

    def update(self, model: torch.nn.Module) -> None:
        """theta_ema <- decay * theta_ema + (1 - decay) * theta (per step)."""
        if not self.enabled:
            return
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                shadow = self._shadow.get(name)
                if shadow is not None:
                    shadow.mul_(self.decay).add_(tensor.detach().float(), alpha=1.0 - self.decay)

    def apply_to(self, target: torch.nn.Module) -> None:
        """Copy the EMA state into `target` (an eval copy of the same model)."""
        if not self.enabled:
            return
        with torch.no_grad():
            for name, tensor in target.state_dict().items():
                shadow = self._shadow.get(name)
                if shadow is not None and tensor.is_floating_point():
                    tensor.copy_(shadow.to(device=tensor.device, dtype=tensor.dtype))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.cpu().clone() for name, tensor in self._shadow.items()}

    def load_state_dict(self, data: dict[str, torch.Tensor]) -> None:
        for name, tensor in self._shadow.items():
            saved = data.get(name)
            if saved is not None:
                tensor.copy_(saved.to(device=tensor.device, dtype=tensor.dtype))
