"""WP10: optimizer package — warmup/cosine schedule math + profile guards."""
import math

import pytest

from training.pipeline import SMOKE_PROFILE, NIGHT_V2_PROFILE, TrainingProfile, _iteration_lr


def _profile(**kwargs):
    return TrainingProfile(name="t", model=NIGHT_V2_PROFILE.model, **kwargs)


def test_legacy_constant_lr_when_no_schedule():
    profile = _profile()
    assert _iteration_lr(profile, 0, 0, 1e-3, 100) == 1e-3
    assert _iteration_lr(profile, 500, 50, 1e-3, 100) == 1e-3


def test_warmup_ramps_from_zero():
    profile = _profile(lr_warmup_steps=10)
    assert _iteration_lr(profile, 0, 0, 1e-3, 100) == pytest.approx(1e-4)
    assert _iteration_lr(profile, 4, 0, 1e-3, 100) == pytest.approx(5e-4)
    # after warmup the base rate applies (no other schedule configured)
    assert _iteration_lr(profile, 10, 1, 1e-3, 100) == pytest.approx(1e-3)


def test_cosine_endpoints_and_midpoint():
    profile = _profile(lr_cosine=True, lr_floor_ratio=0.01)
    base = 1e-3
    assert _iteration_lr(profile, 0, 0, base, 100) == pytest.approx(base)
    end = _iteration_lr(profile, 0, 99, base, 100)
    assert end == pytest.approx(base * 0.01, rel=1e-6)
    mid = _iteration_lr(profile, 0, 50, base, 100)
    assert base * 0.01 < mid < base
    assert mid == pytest.approx(
        base * 0.01 + 0.5 * (base - base * 0.01) * (1 + math.cos(math.pi * 50 / 99)), rel=1e-6
    )


def test_base_lr_override():
    profile = _profile(base_lr=3e-4)
    assert _iteration_lr(profile, 0, 0, 1e-3, 100) == pytest.approx(3e-4)


def test_defaults_unchanged():
    assert SMOKE_PROFILE.optimizer_kind == "adam"
    assert NIGHT_V2_PROFILE.optimizer_kind == "adam"
    assert NIGHT_V2_PROFILE.grad_clip == 0.0
    assert NIGHT_V2_PROFILE.lr_warmup_steps == 0
    with pytest.raises(ValueError, match="optimizer_kind"):
        _profile(optimizer_kind="sgd")
