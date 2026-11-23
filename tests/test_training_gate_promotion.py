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

import chess
import numpy as np
import pytest
import torch

from training import pipeline
from training.journal import TrainingJournal
from training.ladder import Arena, ArenaConfig, LadderConfig, MatchResult
from training.ledger import ArenaLedger
from training.model import MODEL_TINY, GambitNet
from training.openings import gate_opening_seed, random_openings, sample_gate_openings
from training.pipeline import TrainingProfile, _champion_snapshot, _save_champion, run_training
from training.regimes import create_run, get_regime
from training.selfplay import play_match_game


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


class _UniformStub:
    """Deterministic arena evaluator: uniform legal priors, fixed value."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.calls: list[chess.Board] = []

    def evaluate(self, boards):
        # Store COPIES: search hands the caller's live game board to the root
        # evaluation, and that object mutates as the game progresses.
        self.calls.extend(board.copy() for board in boards)
        n = len(boards)
        return (
            np.zeros((n, 4672), dtype="float32"),
            np.full(n, self.value, dtype="float32"),
        )


class TestGateOpeningRandomization:
    """FB-1: gate openings were effectively deterministic (every game from
    the initial position under argmax play -> effective sample size ~2).
    Each gate event now seeds a fresh opening set from (run_seed, gate_index)
    and logs the played FENs in the gate record."""

    def test_gate_opening_seed_is_deterministic_per_event(self):
        assert gate_opening_seed(0, 3) == gate_opening_seed(0, 3)
        assert gate_opening_seed(None, 3) == gate_opening_seed(None, 3)
        assert gate_opening_seed(0, 3) != gate_opening_seed(0, 4)
        assert gate_opening_seed(0, 3) != gate_opening_seed(1, 3)
        assert 0 <= gate_opening_seed(12345, 77) < 2**31

    def test_two_gate_events_different_sets_same_index_reproduces(self):
        """THE acceptance criterion: different gate_index -> different opening
        sets; the same index reproduces the same set exactly."""
        first = sample_gate_openings(16, gate_opening_seed(7, 1))
        second = sample_gate_openings(16, gate_opening_seed(7, 2))
        first_again = sample_gate_openings(16, gate_opening_seed(7, 1))
        assert first != second
        assert first == first_again
        assert len(first) == len(second) == 16

    def test_openings_mirror_across_color_pairs(self):
        """Game 2k (candidate white) and 2k+1 (candidate black) share an
        opening, so opening strength cancels across the pair."""
        fens = sample_gate_openings(6, 99)
        assert fens[0] == fens[1]
        assert fens[2] == fens[3]
        assert fens[4] == fens[5]
        assert len(set(fens)) == 3

    def test_random_openings_are_legal_unique_positions(self):
        fens = random_openings(5, 123)
        assert len(set(fens)) == 5
        for fen in fens:
            board = chess.Board(fen)  # raises on a malformed/illegal FEN
            assert board.legal_moves.count() > 0
            assert not board.is_game_over()

    def test_arena_with_opening_seed_plays_and_records_openings(self):
        config = ArenaConfig(games=4, num_simulations=2, max_plies=4, opening_seed=42)
        white_eval, black_eval = _UniformStub(), _UniformStub()
        arena = Arena(white_eval, black_eval, config)
        match = arena.match("cand", "champ")
        assert match.games == 4
        assert match.openings == sample_gate_openings(4, 42)
        # The games actually START from the logged opening positions.
        assert white_eval.calls[0].fen() == match.openings[0]
        # Two arenas with the same gate seed play the same opening set.
        again = Arena(_UniformStub(), _UniformStub(), config).match("cand", "champ")
        assert again.openings == match.openings

    def test_arena_without_opening_seed_is_unchanged(self):
        """Flag-off parity: no seed -> startpos games, empty openings record."""
        config = ArenaConfig(games=2, num_simulations=2, max_plies=4)
        white_eval = _UniformStub()
        match = Arena(white_eval, _UniformStub(), config).match("cand", "champ")
        assert match.openings == ()
        assert white_eval.calls[0].fen() == chess.STARTING_FEN

    def test_play_match_game_honors_opening_fen(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        white_eval = _UniformStub()
        result, termination = play_match_game(
            white_eval, _UniformStub(), num_simulations=2, max_plies=4, opening_fen=fen
        )
        assert result in ("1-0", "0-1", "1/2-1/2")
        assert termination == "max-plies"
        assert white_eval.calls[0].fen() == fen

    def test_pipeline_gate_record_logs_openings_and_gate_index(self, tmp_path):
        """The journal's arena events carry the gate event identity and the
        opening FENs actually played; different gate events differ."""
        summary = run_training(gate_profile(), run_id="fb1-record",
                               base_dir=tmp_path / "runs", data_dir=tmp_path / "data")
        events = TrainingJournal(tmp_path / "data" / "train_log.jsonl").read()
        arena_events = [e for e in events if e["kind"] == "arena"]
        assert len(arena_events) >= 3  # baseline + 2 gates (gate every iteration)
        baseline = arena_events[0]
        assert baseline["games"] == 0
        assert baseline["openings"] == []
        assert baseline["gate_index"] == 0
        for event in arena_events[1:]:
            assert event["gate_index"] == event["iteration"]
            assert len(event["openings"]) == event["games"]  # one FEN per game
            for fen in event["openings"]:
                chess.Board(fen)  # legal, parseable position
        # Gate 1 and gate 2 are distinct events -> distinct opening sets.
        assert arena_events[1]["openings"] != arena_events[2]["openings"]
        # Reproducing the sets from the recorded identity matches the record.
        assert tuple(arena_events[1]["openings"]) == sample_gate_openings(
            arena_events[1]["games"], gate_opening_seed(0, arena_events[1]["gate_index"])
        )
        assert summary.champion is not None


class TestChampionPointerIntegrity:
    """FB-2: champion.json's tag and checkpoint_path must be ONE resolved
    checkpoint object; the writer asserts tag == Path(path).stem."""

    def _make_checkpoint(self, tmp_path: Path, filename: str) -> Path:
        ckpt = tmp_path / "checkpoints" / filename
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tag": ckpt.stem, "model_state": {}}, ckpt)
        return ckpt

    def test_writer_roundtrips_consistent_pair(self, tmp_path):
        ckpt = self._make_checkpoint(tmp_path, "ckpt-000042-0007.pt")
        champion_file = tmp_path / "runs" / "scratch" / "champion.json"
        _save_champion(champion_file, GambitNet(MODEL_TINY), "ckpt-000042-0007",
                       812.5, "run-1", ckpt, None)
        data = json.loads(champion_file.read_text(encoding="utf-8"))
        assert data["tag"] == "ckpt-000042-0007"
        assert Path(data["checkpoint_path"]).stem == data["tag"]
        loaded = pipeline._load_champion_pointer(champion_file)
        assert loaded is not None
        assert loaded[0] == ckpt

    def test_writer_raises_on_deliberate_mismatch(self, tmp_path):
        good = self._make_checkpoint(tmp_path, "ckpt-000001-0001.pt")
        other = self._make_checkpoint(tmp_path, "ckpt-000999-0009.pt")
        with pytest.raises(ValueError, match="FB-2"):
            _save_champion(tmp_path / "champion.json", GambitNet(MODEL_TINY),
                           good.stem, 800.0, "run-1", other, None)

    def test_hold_gate_writes_paired_pointer(self, tmp_path, monkeypatch):
        """THE FB-2 regression: a champion that HOLDS its gate used to be
        written with its own tag but the LOSING CANDIDATE's checkpoint path
        (observed: tag ckpt-001536-0768 pointing at a different file)."""
        runs = tmp_path / "runs"
        data = tmp_path / "data"

        def scripted(candidate, champion, _device, _arena_config, ctag, htag):
            return MatchResult(candidate=ctag, champion=htag, games=16, wins=0, draws=0, losses=16)

        monkeypatch.setattr(pipeline, "_arena_match", scripted)
        summary = run_training(gate_profile(), run_id="hold-integrity",
                               base_dir=runs, data_dir=data)
        pointer = json.loads((runs / "scratch" / "champion.json").read_text(encoding="utf-8"))
        assert Path(pointer["checkpoint_path"]).stem == pointer["tag"]
        # The pointer describes the champion that held (the baseline), and the
        # file it points at really is that champion's checkpoint.
        assert pointer["tag"] == summary.checkpoints[0]
        assert Path(pointer["checkpoint_path"]).exists()

    def test_promote_gate_writes_promoted_checkpoint_pointer(self, tmp_path, monkeypatch):
        runs = tmp_path / "runs"
        data = tmp_path / "data"

        def scripted(candidate, champion, _device, _arena_config, ctag, htag):
            if not scripted.called:  # type: ignore[attr-defined]
                scripted.called = True  # type: ignore[attr-defined]
                return MatchResult(candidate=ctag, champion=htag, games=16, wins=12, draws=4, losses=0)
            return MatchResult(candidate=ctag, champion=htag, games=16, wins=0, draws=0, losses=16)

        scripted.called = False  # type: ignore[attr-defined]
        monkeypatch.setattr(pipeline, "_arena_match", scripted)
        summary = run_training(gate_profile(), run_id="promote-integrity",
                               base_dir=runs, data_dir=data)
        pointer = json.loads((runs / "scratch" / "champion.json").read_text(encoding="utf-8"))
        assert summary.champion is not None
        assert pointer["tag"] == summary.champion
        assert Path(pointer["checkpoint_path"]).stem == pointer["tag"]
        assert Path(pointer["checkpoint_path"]).exists()
