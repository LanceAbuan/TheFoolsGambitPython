"""Audit F-1/F-5 regression tests: frozen champion snapshots + honest Elo.

F-1: the arena champion used to be the LIVE training model object (the gate
aliased champion_model = model), so every gate after the first played the
model against itself: deterministic 0.5, promotion impossible after the
baseline, and the champion kept learning in place. The fix re-loads the
champion from its just-saved checkpoint (frozen snapshot). These tests fail
on the aliased code: the champion object handed to the arena must be a
distinct, exact, frozen re-load of the checkpoint file.

F-5: candidate Elo used to be recorded as ladder.base_elo on every gate row,
fabricating a rating history for one-match candidates. Non-promoted
candidates now carry elo=None (no claim); a promoted candidate inherits the
champion lineage's post-match rating; the run's rating series is the
champion lineage (journal + metrics), seeded from the reigning pointer.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from training import pipeline
from training.journal import TrainingJournal
from training.ladder import ArenaConfig, LadderConfig, MatchResult
from training.ledger import ArenaLedger
from training.model import MODEL_TINY, GambitNet
from training.pipeline import TrainingProfile, _champion_snapshot, run_training
from training.regimes import create_run, get_regime


def _digest_state(state: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for tensor in state.values():
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _digest(model: torch.nn.Module) -> str:
    return _digest_state(model.state_dict())


def _weights_of(checkpoint: Path) -> str:
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return _digest_state(data["model_state"])


def gate_profile() -> TrainingProfile:
    return TrainingProfile(
        name="gate",
        model=MODEL_TINY,
        regime="scratch",
        iterations=3,
        games_per_iteration=2,
        max_plies=6,
        sims=2,
        train_steps=2,
        batch_size=8,
        min_examples=4,
        buffer_size=1000,
        ladder=LadderConfig(arena=ArenaConfig(num_simulations=2, max_plies=6, games=2)),
        seed=0,
        snapshot_every_iteration=True,
        parallel_games=1,
    )


class TestChampionSnapshot:
    def _saved_checkpoint(self, tmp_path: Path) -> tuple[Path, GambitNet]:
        torch.manual_seed(0)
        model = GambitNet(MODEL_TINY)
        paths = create_run(get_regime("scratch"), "snap-run", base_dir=tmp_path)
        ckpt = paths.checkpoints / "ckpt-000001.pt"
        pipeline._save_checkpoint(model, ckpt, "ckpt-000001", "scratch", "snap-run", 1, 1)
        return ckpt, model

    def test_snapshot_is_distinct_exact_and_frozen(self, tmp_path):
        ckpt, model = self._saved_checkpoint(tmp_path)
        snapshot = _champion_snapshot(ckpt, torch.device("cpu"), model)
        assert snapshot is not model
        assert _digest(snapshot) == _weights_of(ckpt)
        assert not snapshot.training
        assert all(not parameter.requires_grad for parameter in snapshot.parameters())

    def test_snapshot_does_not_move_when_the_live_model_trains(self, tmp_path):
        ckpt, model = self._saved_checkpoint(tmp_path)
        snapshot = _champion_snapshot(ckpt, torch.device("cpu"), model)
        before = _digest(snapshot)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        tensors = torch.randn(4, MODEL_TINY.in_channels, 8, 8)
        policy, value = model(tensors)
        (policy.sum() + value.sum()).backward()
        optimizer.step()
        assert _digest(snapshot) == before  # frozen across a training step

    def test_unreadable_checkpoint_falls_back_to_deep_copy(self, tmp_path):
        torch.manual_seed(0)
        model = GambitNet(MODEL_TINY)
        snapshot = _champion_snapshot(tmp_path / "missing.pt", torch.device("cpu"), model)
        assert snapshot is not model
        assert _digest(snapshot) == _digest(model)
        assert all(not parameter.requires_grad for parameter in snapshot.parameters())


class TestWithinRunPromotion:
    def test_promotion_possible_and_champion_frozen(self, tmp_path, monkeypatch):
        """A candidate that WINS its arena promotes mid-run, and every later
        gate arenas a frozen champion snapshot: a distinct object whose weights
        equal the promoted checkpoint exactly. Match outcomes are scripted
        (16-game win then 16-game loss) so the gate mechanics are
        deterministic; measuring real strength is the ablation bench's job,
        not a unit test's. Matches are gate-sized (16 games) because U9's RD
        cap deliberately refuses promotion to sub-gate matches (a 4-game
        match lands at RD ~208 > cap 150).
        """
        runs = tmp_path / "runs"
        data = tmp_path / "data"
        calls: list[dict[str, object]] = []

        def scripted(candidate, champion, _device, _arena_config, ctag, htag):
            calls.append(
                {
                    "candidate_tag": ctag,
                    "champion_tag": htag,
                    "champion_is_candidate": champion is candidate,
                    "champion_digest": _digest(champion),
                }
            )
            if len(calls) == 1:  # first match: candidate wins 12-4 -> promote
                return MatchResult(candidate=ctag, champion=htag, games=16, wins=12, draws=4, losses=0)
            return MatchResult(candidate=ctag, champion=htag, games=16, wins=0, draws=0, losses=16)

        monkeypatch.setattr(pipeline, "_arena_match", scripted)

        summary = run_training(gate_profile(), run_id="gate-e2e", base_dir=runs, data_dir=data)

        # iterations=3 with gates every iteration: gate 0 is the baseline
        # (no match), gates 1 and 2 run the two scripted matches.
        assert len(calls) == 2
        promoted_tag = str(calls[0]["candidate_tag"])
        assert summary.champion == promoted_tag

        # F-1: the second gate did NOT play the model against itself - the
        # champion is a distinct frozen snapshot of the promoted checkpoint.
        # (On the pre-fix code champion_is_candidate is True and the champion
        # weights have drifted with training.)
        assert calls[1]["champion_tag"] == promoted_tag
        assert calls[1]["champion_is_candidate"] is False
        promoted_ckpt = runs / "scratch" / "gate-e2e" / "checkpoints" / f"{promoted_tag}.pt"
        assert calls[1]["champion_digest"] == _weights_of(promoted_ckpt)

        # F-5: the promoted row carries the inherited lineage rating; the
        # non-promoted row carries NO rating claim; the baseline row is the
        # honest starting rating.
        rows = ArenaLedger(data / "arena.jsonl").read()
        by_checkpoint = {str(row["checkpoint"]): row for row in rows}
        held_tag = str(calls[1]["candidate_tag"])
        assert by_checkpoint[promoted_tag]["promoted"] is True
        assert isinstance(by_checkpoint[promoted_tag]["elo"], float)
        assert by_checkpoint[held_tag]["promoted"] is False
        assert by_checkpoint[held_tag]["elo"] is None

        # The rating series is the champion lineage and it actually MOVES
        # (no more pinned-at-base chart): journal arena events agree with the
        # promoted row, the final summary, and the metrics feed.
        events = TrainingJournal(data / "train_log.jsonl").read()
        arena_events = [event for event in events if event["kind"] == "arena" and event["games"] > 0]
        assert len(arena_events) == 2
        promoted_row_elo = float(by_checkpoint[promoted_tag]["elo"])  # type: ignore[arg-type]
        assert float(arena_events[0]["elo"]) == pytest.approx(promoted_row_elo)
        assert float(arena_events[1]["elo"]) == pytest.approx(summary.elo)
        assert summary.elo != pytest.approx(800.0)
        metrics = json.loads((data / "metrics.json").read_text(encoding="utf-8"))
        assert float(metrics["elo"]) == pytest.approx(summary.elo)
        pointer = json.loads((runs / "scratch" / "champion.json").read_text(encoding="utf-8"))
        assert float(pointer["elo"]) == pytest.approx(summary.elo)
