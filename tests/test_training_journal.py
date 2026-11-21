"""FS-6 training journal tests: structured events on data/train_log.jsonl."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from training.journal import TrainingJournal
from training.ladder import ArenaConfig, LadderConfig
from training.model import MODEL_TINY
from training.pipeline import TrainingProfile, run_training

REQUIRED_ITERATION_KEYS = {
    "iteration",
    "loss",
    "policy_loss",
    "value_loss",
    "elo",
    "arena_result",
    "ts",
}
REQUIRED_ARENA_KEYS = {
    "iteration",
    "checkpoint",
    "opponent",
    "games",
    "arena_result",
    "elo",
    "loss",
    "policy_loss",
    "value_loss",
    "ts",
}


def tiny_profile(**overrides) -> TrainingProfile:
    base: dict[str, object] = {
        "name": "smoke",
        "model": MODEL_TINY,
        "regime": "scratch",
        "iterations": 2,
        "games_per_iteration": 2,
        "max_plies": 8,
        "sims": 2,
        "train_steps": 2,
        "batch_size": 8,
        "min_examples": 4,
        "buffer_size": 100_000,
        "ladder": LadderConfig(arena=ArenaConfig(num_simulations=2, max_plies=8, games=2)),
        "seed": 0,
        "snapshot_every_iteration": True,
        "parallel_games": 1,
    }
    base.update(overrides)
    return TrainingProfile(**base)  # type: ignore[arg-type]


class TestTrainingJournalUnit:
    def test_append_stamps_ts_and_writes_one_json_per_line(self, tmp_path):
        journal = TrainingJournal(tmp_path / "train_log.jsonl")
        journal.append({"kind": "iteration", "iteration": 1})
        journal.append({"kind": "iteration", "iteration": 2})
        lines = (tmp_path / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["kind"] == "iteration"
        assert isinstance(first["ts"], float)
        assert first["ts"] <= time.time()

    def test_reader_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "train_log.jsonl"
        path.write_text(
            '{"kind": "iteration"}\nnot json at all\n\n{"kind": "arena"}\n', encoding="utf-8"
        )
        events = TrainingJournal(path).read()
        assert [e["kind"] for e in events] == ["iteration", "arena"]

    def test_reader_empty_when_missing(self, tmp_path):
        assert TrainingJournal(tmp_path / "missing.jsonl").read() == []


class TestPipelineJournalIntegration:
    @pytest.fixture()
    def journal_events(self, tmp_path):
        summary = run_training(
            tiny_profile(),
            run_id="journaled",
            base_dir=tmp_path / "runs",
            data_dir=tmp_path / "data",
        )
        path = tmp_path / "data" / "train_log.jsonl"
        events = TrainingJournal(path).read()
        return summary, events, path

    def test_file_lands_in_data_dir(self, journal_events):
        _summary, events, path = journal_events
        assert path.name == "train_log.jsonl"
        assert events, "the pipeline must journal events"

    def test_every_line_is_one_json_object(self, journal_events):
        _summary, _events, path = journal_events
        for line in path.read_text(encoding="utf-8").splitlines():
            data = json.loads(line)  # raises on any torn/multi-object line
            assert isinstance(data, dict)

    def test_iteration_events_carry_required_keys(self, journal_events):
        summary, events, _path = journal_events
        iteration_events = [e for e in events if e["kind"] == "iteration"]
        assert len(iteration_events) == 2  # one per smoke iteration
        for event in iteration_events:
            assert REQUIRED_ITERATION_KEYS <= set(event)
            assert event["loss"] is not None
            assert event["policy_loss"] is not None
            assert event["value_loss"] is not None
            assert event["ts"] > 0
            assert event["run_id"] == "journaled"
            assert event["regime"] == "scratch"
        assert [e["iteration"] for e in iteration_events] == [0, 1]
        assert summary.games_played == sum(e["games"] for e in iteration_events)

    def test_arena_events_carry_required_keys(self, journal_events):
        summary, events, _path = journal_events
        arena_events = [e for e in events if e["kind"] == "arena"]
        assert len(arena_events) == len(summary.checkpoints)
        for event in arena_events:
            assert REQUIRED_ARENA_KEYS <= set(event)
            assert event["checkpoint"] in summary.checkpoints
            assert isinstance(event["arena_result"], dict) or event["arena_result"] is None
            if event["arena_result"] is not None:
                assert 0.0 <= event["arena_result"]["win_rate"] <= 1.0
                assert isinstance(event["arena_result"]["promoted"], bool)
            assert event["elo"] is not None
            assert event["ts"] > 0
        # The first checkpoint is the baseline champion (no match played).
        assert arena_events[0]["opponent"] == "none"
        assert arena_events[0]["arena_result"] is None
