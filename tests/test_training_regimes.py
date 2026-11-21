"""Regime config tests (§5.2): five named regimes, lineage paths, data hooks."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from training.regimes import (
    REGIME_NAMES,
    REGIMES,
    FinetuneProvider,
    HybridProvider,
    SelfPlayProvider,
    SFDistillProvider,
    SFGamesProvider,
    StubDataProvider,
    all_regime_names,
    build_data_provider,
    config_hash,
    create_run,
    get_regime,
)
from training.selfplay import SearchConfig, SelfPlayConfig


class TestRegimeConfigs:
    def test_five_named_regimes_exist(self):
        assert set(REGIME_NAMES) == {"scratch", "sf-games", "sf-distill", "hybrid", "finetune"}
        assert set(REGIMES) == set(REGIME_NAMES)

    def test_factory_returns_configs(self):
        for name in REGIME_NAMES:
            config = get_regime(name)
            assert config.name == name
            assert config.description
            assert config.num_simulations > 0
            assert config.games_per_iteration > 0
            assert config.batch_size > 0
            assert 0 < config.learning_rate < 1

    def test_unknown_regime_lists_choices(self):
        with pytest.raises(ValueError, match="scratch"):
            get_regime("bogus")

    def test_regimes_differ_in_data_source(self):
        sources = {name: REGIMES[name].data_source for name in REGIME_NAMES}
        assert sources["scratch"] == "selfplay"
        assert sources["sf-games"] == "sf-games"
        assert sources["sf-distill"] == "sf-distill"
        assert sources["hybrid"] == "hybrid"
        assert sources["finetune"] == "finetune"

    def test_hybrid_warmstarts_from_distill(self):
        assert REGIMES["hybrid"].warmstart_from == "sf-distill"

    def test_config_hash_stable_and_sensitive(self):
        base = get_regime("scratch")
        assert config_hash(base) == config_hash(get_regime("scratch"))
        tweaked = base.__class__(**{**base.to_dict(), "num_simulations": 999})
        assert config_hash(base) != config_hash(tweaked)


class TestRunLineage:
    def test_lineage_paths_created(self, tmp_path):
        config = get_regime("scratch")
        paths = create_run(config, "run-abc", base_dir=tmp_path)
        expected_root = tmp_path / "scratch" / "run-abc"
        assert paths.root == expected_root
        assert expected_root.is_dir()
        assert (expected_root / "checkpoints").is_dir()
        assert (expected_root / "games").is_dir()
        assert paths.checkpoints == expected_root / "checkpoints"

    def test_run_metadata_written(self, tmp_path):
        config = get_regime("sf-distill")
        paths = create_run(config, "run-xyz", base_dir=tmp_path)
        meta = json.loads(paths.meta_path.read_text(encoding="utf-8"))
        assert meta["regime"] == "sf-distill"
        assert meta["run_id"] == "run-xyz"
        assert meta["config_hash"] == config_hash(config)
        assert meta["config"]["data_source"] == "sf-distill"

    def test_every_regime_can_create_lineage(self, tmp_path):
        for name in REGIME_NAMES:
            create_run(get_regime(name), f"run-{name}", base_dir=tmp_path)
            assert (tmp_path / name / f"run-{name}").is_dir()


class TestDataProviders:
    def test_scratch_provider_is_selfplay(self):
        assert isinstance(build_data_provider(get_regime("scratch")), SelfPlayProvider)

    @pytest.mark.parametrize(
        ("name", "provider_type"),
        [
            ("sf-games", SFGamesProvider),
            ("sf-distill", SFDistillProvider),
            ("hybrid", HybridProvider),
            ("finetune", FinetuneProvider),
        ],
    )
    def test_slice2_providers_replaced_stubs(self, name, provider_type):
        """FS-3: the slice-1 NotImplementedError stubs are now live providers.

        (Updated from `test_stub_providers_fail_with_clear_message`, which
        pinned the pre-slice-2 state that these regimes raised
        NotImplementedError — that contract was replaced by FS-3.)
        """
        provider = build_data_provider(get_regime(name))
        assert isinstance(provider, provider_type)

    def test_unknown_data_source_still_stubs(self):
        """An unrecognised data_source keeps the clear stub error."""
        provider = build_data_provider(get_regime("scratch").__class__(
            name="x", description="x", data_source="mystery", num_simulations=1,
            games_per_iteration=1, train_steps_per_iteration=1, batch_size=1, learning_rate=0.1,
        ))
        assert isinstance(provider, StubDataProvider)
        with pytest.raises(NotImplementedError, match="no data provider wired"):
            provider.collect(evaluator=None, num_games=1, config=SelfPlayConfig())

    def test_selfplay_provider_collects_games(self):
        """The scratch provider actually produces self-play results (tiny)."""
        from training.model import MODEL_TINY, GambitNet
        from training.selfplay import BatchNNEvaluator

        model = GambitNet(MODEL_TINY)
        model.eval()
        provider = SelfPlayProvider()
        config = SelfPlayConfig(
            search=SearchConfig(num_simulations=2, root_noise=False), max_plies=2
        )
        results = provider.collect(BatchNNEvaluator(model), 2, config)
        assert len(results) == 2
        assert all(r.ply_count <= 2 for r in results)

    def test_all_regime_names_matches_dict(self):
        assert list(all_regime_names()) == list(REGIMES)
