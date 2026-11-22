"""End-to-end smoke: tiny profile produces checkpoint + ledger + metrics (<2 min CPU)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import torch

from training.ledger import ArenaLedger
from training.pipeline import SMOKE_PROFILE, run_training
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
