"""WP6: two-stage gate — Wilson bound math + config validation."""
import pytest

from training.ladder import LadderConfig, wilson_lower_bound


def test_wilson_lower_bound_math():
    # 64 games at exactly 0.5 -> LB slightly below 0.5
    lb = wilson_lower_bound(0.5, 64)
    assert 0.37 < lb < 0.5
    # Stronger score and more games -> higher bound
    assert wilson_lower_bound(0.6, 128) > wilson_lower_bound(0.6, 32)
    assert wilson_lower_bound(0.0, 10) == 0.0
    assert wilson_lower_bound(0.7, 0) == 0.0


def test_two_stage_config_defaults_are_off():
    cfg = LadderConfig()
    assert cfg.screen_games == 0
    assert cfg.confirm_games == 0
    assert cfg.screen_threshold == pytest.approx(0.52)
    assert cfg.confirm_lower_bound == pytest.approx(0.5)


def test_negative_gate_sizes_rejected():
    with pytest.raises(ValueError, match="screen_games"):
        LadderConfig(screen_games=-1)
