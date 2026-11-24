"""WP4: EMA weights — math, application, persistence, profile defaults."""
import torch
from torch import nn

from training.ema import EmaWeights


def _set(model, value):
    with torch.no_grad():
        model.weight.fill_(value)
        model.bias.fill_(value)


def test_ema_update_and_apply():
    model = nn.Linear(2, 1)
    _set(model, 0.0)
    ema = EmaWeights(model, decay=0.5)
    _set(model, 2.0)
    ema.update(model)  # shadow = 0.5 * 0 + 0.5 * 2 = 1
    target = nn.Linear(2, 1)
    ema.apply_to(target)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 1.0))
    assert torch.allclose(target.bias, torch.full_like(target.bias, 1.0))


def test_ema_state_roundtrip():
    model = nn.Linear(2, 1)
    _set(model, 0.0)
    ema = EmaWeights(model, decay=0.5)
    _set(model, 2.0)
    ema.update(model)
    saved = ema.state_dict()
    other = nn.Linear(2, 1)
    _set(other, 0.0)
    ema2 = EmaWeights(other, decay=0.5)
    ema2.load_state_dict(saved)
    target = nn.Linear(2, 1)
    ema2.apply_to(target)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 1.0))


def test_ema_disabled_when_decay_zero():
    model = nn.Linear(2, 1)
    _set(model, 3.0)
    ema = EmaWeights(model, decay=0.0)
    assert ema.enabled is False
    _set(model, 9.0)
    ema.update(model)  # no-op
    target = nn.Linear(2, 1)
    _set(target, -1.0)
    ema.apply_to(target)  # no-op
    assert torch.allclose(target.weight, torch.full_like(target.weight, -1.0))


def test_profiles_keep_ema_off_by_default():
    from training.pipeline import NIGHT_V2_PROFILE, TrainingProfile

    profile = TrainingProfile(name="t", model=NIGHT_V2_PROFILE.model)
    assert profile.ema_decay == 0.0
    assert profile.ema_min_iteration == 2000
    assert NIGHT_V2_PROFILE.ema_decay == 0.0
