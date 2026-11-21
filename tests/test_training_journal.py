"""FS-6 training journal tests: structured events on data/train_log.jsonl."""
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import pytest

from training.journal import TrainingJournal
from training.ladder import ArenaConfig, LadderConfig
from training.model import MODEL_TINY, GambitNet
from training.pipeline import TrainingProfile, run_training
from training.selfplay import (
    BatchNNEvaluator,
    SearchConfig,
    SelfPlayConfig,
    StreamContext,
    parallel_self_play,
)

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

    def test_live_game_events_stream_by_default(self, journal_events):
        """FS-6b: a pipeline run journals every ply + one game_end per game."""
        summary, events, _path = journal_events
        live = [e for e in events if e["kind"] == "live"]
        ends = [e for e in events if e["kind"] == "game_end"]
        assert live, "smoke run must stream live moves by default"
        assert len(ends) == summary.games_played  # one game_end per finished game
        assert len(live) == sum(e["plies"] for e in ends)
        for event in live:
            assert event["run_id"] == "journaled"
            assert event["regime"] == "scratch"
            chess.Board(event["fen"])  # every streamed position parses

    def test_game_slots_are_unique_across_iterations(self, journal_events):
        """FS-6b: iteration i owns slots [i*N, (i+1)*N) — no cross-iteration
        collision, so the client never sees two games in one board slot."""
        _summary, events, _path = journal_events
        slots = {e["game"] for e in events if e["kind"] in ("live", "game_end")}
        # 2 iterations x 2 games -> exactly 4 distinct run-unique slots.
        assert slots == {0, 1, 2, 3}
        # Each slot's moves replay as one coherent game from startpos.
        live = [e for e in events if e["kind"] == "live"]
        for slot in slots:
            board = chess.Board()
            for event in sorted(
                (e for e in live if e["game"] == slot), key=lambda e: e["ply"]
            ):
                board.push_uci(event["move"])
                assert board.fen() == event["fen"]

    def test_no_stream_games_flag_stops_streaming(self, tmp_path):
        """--no-stream-games: zero live/game_end events, aggregates unchanged."""
        summary = run_training(
            tiny_profile(stream_games=False),
            run_id="quiet-run",
            base_dir=tmp_path / "runs",
            data_dir=tmp_path / "data",
        )
        events = TrainingJournal(tmp_path / "data" / "train_log.jsonl").read()
        assert not [e for e in events if e["kind"] in ("live", "game_end")]
        assert [e for e in events if e["kind"] == "iteration"]  # aggregates still flow
        assert summary.games_played == 4  # collection itself is unchanged


# ---------------------------------------------------------------------------
# FS-6b: live per-move streaming
# ---------------------------------------------------------------------------

STREAM_CONFIG = SelfPlayConfig(search=SearchConfig(num_simulations=4, root_noise=False), max_plies=6)

LIVE_KEYS = {"kind", "run_id", "regime", "game", "ply", "move", "fen", "ts"}
GAME_END_KEYS = {"kind", "run_id", "regime", "game", "result", "plies", "ts"}


def tiny_model_evaluator() -> BatchNNEvaluator:
    model = GambitNet(MODEL_TINY)
    model.eval()
    return BatchNNEvaluator(model)


class TestLiveStreaming:
    """FS-6b: every ply journaled, game_end per game, replayable FENs."""

    @pytest.fixture()
    def streamed(self, tmp_path):
        journal = TrainingJournal(tmp_path / "stream.jsonl")
        stream = StreamContext(journal=journal, run_id="live-run", regime="scratch")
        evaluator = tiny_model_evaluator()
        results = parallel_self_play(evaluator, 2, STREAM_CONFIG, parallel_games=2, stream=stream)
        events = journal.read()
        return results, events

    def test_live_events_have_required_keys_and_valid_fen(self, streamed):
        _results, events = streamed
        live = [e for e in events if e["kind"] == "live"]
        assert live, "streaming must emit live events"
        for event in live:
            assert LIVE_KEYS <= set(event)
            assert isinstance(event["game"], int) and event["game"] >= 0
            assert isinstance(event["ply"], int) and event["ply"] >= 1
            chess.Move.from_uci(event["move"])  # valid UCI
            chess.Board(event["fen"])  # parses as a position

    def test_fen_replays_from_the_move_list(self, streamed):
        """Replaying each game's uci moves from startpos reproduces every FEN."""
        _results, events = streamed
        live = [e for e in events if e["kind"] == "live"]
        by_game: dict[int, list[dict]] = {}
        for event in live:
            by_game.setdefault(event["game"], []).append(event)
        assert len(by_game) == 2
        for game_events in by_game.values():
            game_events.sort(key=lambda e: e["ply"])
            board = chess.Board()
            for expected_ply, event in enumerate(game_events, start=1):
                assert event["ply"] == expected_ply
                board.push_uci(event["move"])
                assert board.fen() == event["fen"]

    def test_game_end_fires_once_per_game_with_consistent_plies(self, streamed):
        results, events = streamed
        ends = [e for e in events if e["kind"] == "game_end"]
        assert len(ends) == len(results) == 2
        assert {e["game"] for e in ends} == {0, 1}
        live_by_game: dict[int, int] = {}
        for event in events:
            if event["kind"] == "live":
                live_by_game[event["game"]] = live_by_game.get(event["game"], 0) + 1
        for end in ends:
            assert GAME_END_KEYS <= set(end)
            assert end["result"] in {"1-0", "0-1", "1/2-1/2"}
            # (c) live-event count per game == that game's final ply count
            assert live_by_game[end["game"]] == end["plies"] == results[end["game"]].ply_count

    def test_total_live_events_equal_sum_of_plies(self, streamed):
        _results, events = streamed
        live = [e for e in events if e["kind"] == "live"]
        ends = [e for e in events if e["kind"] == "game_end"]
        assert len(live) == sum(e["plies"] for e in ends)

    def test_streaming_disabled_emits_nothing(self, tmp_path):
        journal = TrainingJournal(tmp_path / "quiet.jsonl")
        stream = StreamContext(journal=journal, run_id="r", regime="scratch", enabled=False)
        evaluator = tiny_model_evaluator()
        parallel_self_play(evaluator, 2, STREAM_CONFIG, parallel_games=2, stream=stream)
        assert journal.read() == []
        assert (tmp_path / "quiet.jsonl").exists() is False  # nothing was ever written

    def test_none_journal_emits_nothing(self):
        stream = StreamContext(journal=None, run_id="r", regime="scratch")
        assert stream.for_game(0) is None
        evaluator = tiny_model_evaluator()
        results = parallel_self_play(evaluator, 1, STREAM_CONFIG, stream=stream)
        assert len(results) == 1

    def test_stream_context_shift_keeps_slots_unique(self, tmp_path):
        """A provider running two batches offsets indices so slots never collide."""
        journal = TrainingJournal(tmp_path / "shift.jsonl")
        base = StreamContext(journal=journal, run_id="r", regime="sf-games")
        shifted = base.shifted(3)
        assert shifted.game_offset == 3
        assert shifted.for_game(0).game == 3  # type: ignore[union-attr]
        assert base.for_game(0).game == 0  # type: ignore[union-attr]


class TestConcurrentAppendIntegrity:
    def test_sixteen_thread_append_stress_produces_parseable_lines(self, tmp_path):
        """16 threads x 25 events: every line parses, none interleaved (FS-6b)."""
        journal = TrainingJournal(tmp_path / "stress.jsonl")
        threads = 16
        per_thread = 25
        errors: list[Exception] = []

        def hammer(thread_id: int) -> None:
            try:
                for i in range(per_thread):
                    journal.append(
                        {
                            "kind": "live",
                            "thread": thread_id,
                            "i": i,
                            "payload": "x" * 40,  # force multi-write-sized lines
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - collected below
                errors.append(exc)

        workers = [threading.Thread(target=hammer, args=(t,)) for t in range(threads)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30.0)

        assert not errors
        events = journal.read()
        assert len(events) == threads * per_thread  # every line parsed as one JSON object
        per_thread_counts: dict[int, int] = {}
        for event in events:
            per_thread_counts[event["thread"]] = per_thread_counts.get(event["thread"], 0) + 1
        assert all(count == per_thread for count in per_thread_counts.values())
        assert len(per_thread_counts) == threads

    def test_unknown_event_kind_is_tolerated(self, tmp_path):
        """Reader (and consumers) never crash on kinds they don't know (FS-6b)."""
        journal = TrainingJournal(tmp_path / "future.jsonl")
        journal.append({"kind": "live", "ply": 1})
        journal.append({"kind": "hologram_projection", "data": {"whatever": True}})
        journal.append({"kind": "game_end", "plies": 9})
        events = journal.read()
        assert [e["kind"] for e in events] == ["live", "hologram_projection", "game_end"]
        # Consumers filter by kind; unknown kinds simply never match.
        assert [e for e in events if e["kind"] == "live"]  # type: ignore[comparison-overlap]
        assert not [e for e in events if e["kind"] == "iteration"]

