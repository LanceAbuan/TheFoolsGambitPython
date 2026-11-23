"""End-to-end smoke: tiny profile produces checkpoint + ledger + metrics (<2 min CPU)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from pathlib import Path

import pytest
import torch

from training.journal import TrainingJournal
from training.ledger import ArenaLedger
from training.model import MODEL_TINY, GambitNet, ModelConfig
from training.pipeline import (
    SMOKE_PROFILE,
    ArenaConfig,
    LadderConfig,
    LRStepSchedule,
    TrainingProfile,
    encoder_version_for,
    run_training,
)
from training.regimes import get_regime


def test_smoke_run_produces_lineage_ledger_metrics(tmp_path):
    """WP-C slice 1 acceptance: the full loop end-to-end on CPU, tiny profile."""
    profile = SMOKE_PROFILE
    assert profile.name == "smoke"

    summary = run_training(
        profile,
        run_id="smoke-test",
        base_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
    )

    # 1. checkpoint files under runs/<regime>/<run-id>/checkpoints/
    run_root = tmp_path / "runs" / "scratch" / "smoke-test"
    assert run_root.is_dir()
    checkpoints = sorted((run_root / "checkpoints").glob("*.pt"))
    assert len(checkpoints) >= 2  # at least one per iteration
    assert len(summary.checkpoints) == len(checkpoints)

    # checkpoint content loads back into the same architecture
    data = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    assert data["regime"] == "scratch"
    assert data["run_id"] == "smoke-test"
    assert "model_state" in data and "config" in data

    # 2. run metadata with regime provenance (§5.2)
    meta = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    assert meta["regime"] == "scratch"
    assert meta["run_id"] == "smoke-test"

    # 3. ledger has a baseline line + an arena line
    ledger = ArenaLedger(tmp_path / "data" / "arena.jsonl")
    rows = ledger.read()
    assert len(rows) >= 2
    assert rows[0]["checkpoint"] == summary.checkpoints[0]
    assert rows[0]["opponent"] == "none"  # first checkpoint = baseline champion
    assert rows[0]["kind"] == "baseline"  # §5.1: seed row, no games
    arena_row = rows[-1]
    for key in ("run_id", "regime", "checkpoint", "opponent", "games", "wins", "draws", "elo"):
        assert key in arena_row
    assert arena_row["games"] > 0
    assert arena_row["checkpoint"] == summary.checkpoints[-1]
    assert arena_row["kind"] == "arena"  # §5.1: kind='arena' on match rows

    # 4. metrics feed carries the contract keys
    metrics = json.loads((tmp_path / "data" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["games_played"] == summary.games_played + summary.arena_games
    assert metrics["regime"] == "scratch"
    assert metrics["champion"] in summary.checkpoints
    assert isinstance(metrics["elo"], (int, float))
    assert 0.0 <= metrics["win_rate"] <= 1.0
    assert set(metrics["checkpoints"]) >= set(summary.checkpoints)

    # 5. champion pointer written for the next run's gate
    champion_pointer = tmp_path / "runs" / "scratch" / "champion.json"
    pointer = json.loads(champion_pointer.read_text(encoding="utf-8"))
    assert pointer["tag"] in summary.checkpoints
    assert pointer["checkpoint_path"].endswith(".pt")

    # smoke must be genuinely fast
    assert summary.elapsed_seconds < 120.0


def test_second_run_gates_against_first_champion(tmp_path):
    """The champion pointer from run 1 gates run 2 (arena actually runs)."""
    base = tmp_path / "runs"
    data = tmp_path / "data"
    first = run_training(SMOKE_PROFILE, run_id="run-1", base_dir=base, data_dir=data)
    assert first.champion is not None

    second = run_training(SMOKE_PROFILE, run_id="run-2", base_dir=base, data_dir=data)
    assert second.champion is not None
    rows = ArenaLedger(data / "arena.jsonl").read()
    # run-1: baseline + arena; run-2: arena vs the surviving champion
    assert len(rows) >= 4
    run2_rows = [r for r in rows if r["run_id"] == "run-2"]
    assert len(run2_rows) >= 1
    assert run2_rows[0]["opponent"] in first.checkpoints or run2_rows[0]["opponent"] == "none"


def test_regime_config_flows_into_run(tmp_path):
    """Regime stays a swappable config: a different regime name gives its own lineage."""
    profile = SMOKE_PROFILE
    object.__setattr__(profile, "regime", "hybrid")  # frozen dataclass override for the test
    summary = run_training(profile, run_id="hybrid-run", base_dir=tmp_path / "runs",
                           data_dir=tmp_path / "data")
    assert (tmp_path / "runs" / "hybrid" / "hybrid-run").is_dir()
    assert summary.regime == "hybrid"
    meta = json.loads(
        (tmp_path / "runs" / "hybrid" / "hybrid-run" / "run.json").read_text(encoding="utf-8")
    )
    assert meta["config_hash"] is not None
    assert get_regime("hybrid").data_source == "hybrid"


class TestWPAThroughputKnobs:
    """WP-A: night-v2 profile + resume-stable step-decay LR schedule."""

    def test_night_v2_profile_values(self):
        """The exact acceptance assertion from the WP-A spec."""
        from training.pipeline import NIGHT_V2_PROFILE as p

        assert (p.parallel_games, p.sims, p.train_steps, p.batch_size) == (32, 600, 96, 512)
        assert p.min_examples == 128
        assert p.name == "night-v2"

    def test_night_profile_untouched(self):
        """The live run's profile stays byte-identical (resume safety): the
        v2 knobs live ONLY on NIGHT_V2_PROFILE."""
        from training.pipeline import NIGHT_PROFILE

        assert NIGHT_PROFILE.name == "night"
        assert (NIGHT_PROFILE.parallel_games, NIGHT_PROFILE.sims) == (16, 400)
        assert (NIGHT_PROFILE.train_steps, NIGHT_PROFILE.batch_size) == (64, 256)
        assert NIGHT_PROFILE.min_examples == 64
        assert NIGHT_PROFILE.lr_schedule is None
        # arena unchanged: 128 sims / 16 games (section 5.1)
        assert NIGHT_PROFILE.ladder.arena.num_simulations == 128
        assert NIGHT_PROFILE.ladder.arena.games == 16

    def test_night_v2_parallel_games_at_regime_cap(self):
        """32 sits exactly at MAX_PARALLEL_GAMES — no regime change needed."""
        from training.pipeline import NIGHT_V2_PROFILE
        from training.regimes import MAX_PARALLEL_GAMES

        assert NIGHT_V2_PROFILE.parallel_games == MAX_PARALLEL_GAMES == 32

    def test_lr_schedule_rungs(self):
        """199 -> 1e-3, 200 -> 5e-4, 400 -> 2.5e-4 (x0.5 per milestone)."""
        from training.pipeline import LRStepSchedule

        schedule = LRStepSchedule(milestones=(200, 400, 600, 800), gamma=0.5)
        assert schedule.lr_at(199, 1e-3) == pytest.approx(1e-3)
        assert schedule.lr_at(200, 1e-3) == pytest.approx(5e-4)
        assert schedule.lr_at(201, 1e-3) == pytest.approx(5e-4)
        assert schedule.lr_at(399, 1e-3) == pytest.approx(5e-4)
        assert schedule.lr_at(400, 1e-3) == pytest.approx(2.5e-4)
        assert schedule.lr_at(799, 1e-3) == pytest.approx(1.25e-4)
        assert schedule.lr_at(800, 1e-3) == pytest.approx(6.25e-5)

    def test_lr_schedule_resume_from_350_lands_on_right_rung(self):
        """Resume stability: the rung depends only on the ABSOLUTE iteration,
        so a run resumed at 350 gets the same LR it never stopped for."""
        from training.pipeline import LRStepSchedule

        schedule = LRStepSchedule(milestones=(200, 400, 600, 800), gamma=0.5)
        assert schedule.lr_at(350, 1e-3) == pytest.approx(5e-4)

    def test_lr_schedule_validation(self):
        from training.pipeline import LRStepSchedule

        with pytest.raises(ValueError):
            LRStepSchedule(milestones=(400, 200))  # not increasing
        with pytest.raises(ValueError):
            LRStepSchedule(milestones=(200, 200))  # not unique
        with pytest.raises(ValueError):
            LRStepSchedule(milestones=(200,), gamma=1.5)

    def test_trainer_loop_applies_schedule_per_iteration(self, tmp_path):
        """The loop's LR follows the schedule rungs; the journal records it."""
        profile = TrainingProfile(
            name="lr-sched",
            model=MODEL_TINY,
            regime="scratch",
            iterations=3,
            games_per_iteration=2,
            max_plies=4,
            sims=2,
            train_steps=2,
            batch_size=8,
            min_examples=4,
            buffer_size=1000,
            ladder=LadderConfig(arena=ArenaConfig(num_simulations=2, max_plies=4, games=2)),
            seed=0,
            snapshot_every_iteration=True,
            parallel_games=1,
            lr_schedule=LRStepSchedule(milestones=(1, 2), gamma=0.5),
        )
        summary = run_training(profile, run_id="lr-sched", base_dir=tmp_path / "runs",
                               data_dir=tmp_path / "data")
        assert summary.iterations_completed == 3
        events = TrainingJournal(tmp_path / "data" / "train_log.jsonl").read()
        iteration_events = [e for e in events if e["kind"] == "iteration"]
        lrs = [float(e["lr"]) for e in iteration_events]
        assert lrs == [pytest.approx(1e-3), pytest.approx(5e-4), pytest.approx(2.5e-4)]

    def test_trainer_loop_constant_lr_without_schedule(self, tmp_path):
        """Default profiles keep the regime's constant LR (flag-off parity)."""
        profile = TrainingProfile(
            name="lr-const",
            model=MODEL_TINY,
            regime="scratch",
            iterations=2,
            games_per_iteration=2,
            max_plies=4,
            sims=2,
            train_steps=2,
            batch_size=8,
            min_examples=4,
            buffer_size=1000,
            ladder=LadderConfig(arena=ArenaConfig(num_simulations=2, max_plies=4, games=2)),
            seed=0,
            snapshot_every_iteration=True,
            parallel_games=1,
        )
        run_training(profile, run_id="lr-const", base_dir=tmp_path / "runs",
                     data_dir=tmp_path / "data")
        events = TrainingJournal(tmp_path / "data" / "train_log.jsonl").read()
        lrs = [float(e["lr"]) for e in events if e["kind"] == "iteration"]
        assert lrs == [pytest.approx(1e-3), pytest.approx(1e-3)]

    def test_profile_signature_carries_lr_schedule(self):
        from training.pipeline import NIGHT_PROFILE, NIGHT_V2_PROFILE, LRStepSchedule
        from training.resume import profile_signature

        assert profile_signature(NIGHT_PROFILE)["lr_schedule"] is None
        sig = profile_signature(NIGHT_V2_PROFILE)
        assert sig["lr_schedule"] == {"milestones": [200, 400, 600, 800], "gamma": 0.5}
        # round-trips through from_dict
        schedule = LRStepSchedule.from_dict(sig["lr_schedule"])  # type: ignore[arg-type]
        assert schedule == NIGHT_V2_PROFILE.lr_schedule

    def test_cli_night_v2_profile_builds(self):
        """`--profile night-v2` is launchable: the CLI builds the v2 profile."""
        import importlib.util

        script = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
        spec = importlib.util.spec_from_file_location("train_cli_under_test", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.build_parser().parse_args(["--profile", "night-v2"])
        profile = module.build_profile(args)
        assert profile.name == "night-v2"
        assert (profile.parallel_games, profile.sims) == (32, 600)
        assert (profile.train_steps, profile.batch_size, profile.min_examples) == (96, 512, 128)
        assert profile.lr_schedule is not None
        assert profile.lr_schedule.lr_at(200, 1e-3) == pytest.approx(5e-4)
        # smoke profile untouched by the refactor
        smoke_args = module.build_parser().parse_args(["--profile", "smoke"])
        smoke = module.build_profile(smoke_args)
        assert smoke.name == "smoke" and smoke.model == MODEL_TINY


class TestWPBEncoderStamping:
    """WP-B: encoder-version stamps on checkpoint metadata + journal records."""

    def testencoder_version_for_known_plane_counts(self):
        from training.encoder import ENCODER_VERSION, NUM_PLANES, NUM_PLANES_V2

        assert encoder_version_for(NUM_PLANES) == 1
        assert encoder_version_for(NUM_PLANES_V2) == ENCODER_VERSION == 2
        assert NUM_PLANES_V2 == 104
        with pytest.raises(ValueError):
            encoder_version_for(32)

    def test_checkpoint_metadata_stamps_encoder_layout(self, tmp_path):
        from training.pipeline import _save_checkpoint

        v1 = GambitNet(MODEL_TINY)
        ckpt = tmp_path / "v1.pt"
        _save_checkpoint(v1, ckpt, "tag-1", "scratch", "run", 1, 1)
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert data["in_channels"] == 29
        assert data["encoder_version"] == 1

        # A v2-shaped model (104 input planes) stamps version 2 automatically.
        v2 = GambitNet(ModelConfig(in_channels=104, num_blocks=1, filters=8, value_channels=8, value_fc=16))
        ckpt2 = tmp_path / "v2.pt"
        _save_checkpoint(v2, ckpt2, "tag-2", "scratch", "run", 2, 1)
        data2 = torch.load(ckpt2, map_location="cpu", weights_only=False)
        assert data2["in_channels"] == 104
        assert data2["encoder_version"] == 2

    def test_journal_records_stamp_encoder_version(self, tmp_path):
        summary = run_training(
            SMOKE_PROFILE,
            run_id="stamp-test",
            base_dir=tmp_path / "runs",
            data_dir=tmp_path / "data",
        )
        assert summary.iterations_completed >= 1
        events = [
            json.loads(line)
            for line in (tmp_path / "data" / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert events, "journal produced no events"
        stamped = [e for e in events if e.get("kind") in ("iteration", "arena")]
        assert stamped, "no iteration/arena journal events"
        # The smoke model is 29-plane: the honest stamp is v1.
        assert all(e["encoder_version"] == 1 for e in stamped)
