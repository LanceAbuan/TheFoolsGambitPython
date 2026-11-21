"""FS-5 resumable runs + HF publishing tests.

Key proof: a run killed after N iterations and resumed via a second
`run_training(resume=...)` call rebuilds exactly — the iteration counter
continues, the replay buffer contents carry over, and the resumed run's
final loss matches a fresh uninterrupted run of the same seed.
"""
import os
import sys
import types
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from training.hf import resolve_hf_token, upload_checkpoint
from training.journal import TrainingJournal
from training.model import MODEL_TINY, GambitNet
from training.pipeline import TrainingProfile, run_training
from training.regimes import create_run, get_regime
from training.resume import (
    load_latest_run_state,
    resolve_warmstart,
    restore_run_state,
    save_run_state,
)
from training.trainer import TrainingBuffer


def tiny_profile(**overrides) -> TrainingProfile:
    from training.ladder import ArenaConfig, LadderConfig

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
        # Sequential games: MCTS Dirichlet draws from the GLOBAL numpy stream,
        # so only the sequential path is bit-reproducible across processes.
        "parallel_games": 1,
    }
    base.update(overrides)
    return TrainingProfile(**base)  # type: ignore[arg-type]


def buffer_fingerprint(buffer: TrainingBuffer) -> tuple[int, str]:
    """(size, content summary) for exact-rebuild assertions."""
    first = buffer.buffer[0]
    summary = (
        float(first["value"]),
        int(np.argmax(first["policy"])),
        first["board_tensor"].tobytes()[:64],
    )
    return len(buffer), repr(summary)


class TestRunStateRoundtrip:
    def test_save_then_restore_rebuilds_everything(self, tmp_path):
        torch.manual_seed(0)
        np.random.seed(0)
        paths = create_run(get_regime("scratch"), "rt-1", base_dir=tmp_path)
        model = GambitNet(MODEL_TINY)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        buffer = TrainingBuffer(max_size=1000)
        planes = np.zeros((29, 8, 8), dtype=np.float32)
        policy = np.zeros(4672, dtype=np.float32)
        policy[5] = 1.0
        buffer.add(
            [
                {"board_tensor": planes, "policy": policy, "value": 0.5},
                {"board_tensor": planes, "policy": policy, "value": -0.5},
            ]
        )
        play_rng = np.random.default_rng(7)
        torch.randn(1)  # advance torch rng so the restore point is observable
        saved_torch_state = torch.get_rng_state()
        saved_np_state = np.random.get_state()

        save_run_state(
            paths,
            iterations_completed=3,
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            play_rng=play_rng,
            steps=12,
            games_played=9,
            champion_tag="ckpt-000012",
            champion_elo=812.5,
            profile=tiny_profile(),
        )
        assert len(list(paths.state.glob("state-*.pt"))) == 1

        # Mutate everything, then rebuild from the persisted state.
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        buffer.clear()
        torch.randn(1)
        np.random.rand(3)
        fresh_rng = np.random.default_rng(999)

        loaded = load_latest_run_state(paths)
        assert loaded is not None
        restored = restore_run_state(
            loaded, model=model, optimizer=optimizer, buffer=buffer, play_rng=fresh_rng
        )
        assert restored.iterations_completed == 3
        assert restored.steps == 12
        assert restored.games_played == 9
        assert restored.champion_tag == "ckpt-000012"
        assert restored.champion_elo == pytest.approx(812.5)
        # rng streams back (must run BEFORE anything consumes torch RNG again)
        assert torch.equal(torch.get_rng_state(), saved_torch_state)
        assert np.array_equal(np.random.get_state()[1], saved_np_state[1])
        assert fresh_rng.bit_generator.state == play_rng.bit_generator.state
        # weights back
        clone = GambitNet(MODEL_TINY)
        clone.load_state_dict(loaded["model_state"])
        for got, want in zip(model.parameters(), clone.parameters()):
            assert torch.allclose(got, want)
        # buffer back
        assert buffer_fingerprint(buffer)[0] == 2

    def test_no_state_returns_none(self, tmp_path):
        paths = create_run(get_regime("scratch"), "empty-run", base_dir=tmp_path)
        assert load_latest_run_state(paths) is None


class TestKillAndResume:
    def test_two_calls_continue_iteration_counter_and_buffer(self, tmp_path):
        runs = tmp_path / "runs"
        data = tmp_path / "data"
        profile = tiny_profile()

        first = run_training(tiny_profile(iterations=1), run_id="killed-run",
                             base_dir=runs, data_dir=data)
        assert first.iterations_completed == 1
        assert first.steps >= 1
        state_dir = runs / "scratch" / "killed-run" / "state"
        first_states = sorted(state_dir.glob("state-*.pt"))
        assert len(first_states) == 1
        first_state = torch.load(first_states[0], map_location="cpu", weights_only=False)
        buffer_size_after_iter0 = len(first_state["buffer_examples"])

        second = run_training(profile, run_id="killed-run", base_dir=runs, data_dir=data,
                              resume=True)
        assert second.resumed is True
        assert second.iterations_completed == 1  # only the remaining iteration ran
        assert second.steps > first.steps
        assert len(list(state_dir.glob("state-*.pt"))) == 2
        final_state = torch.load(
            max(state_dir.glob("state-*.pt")), map_location="cpu", weights_only=False
        )
        assert final_state["iterations_completed"] == 2
        # The replay buffer carried over and kept growing (never restarted).
        assert len(final_state["buffer_examples"]) > buffer_size_after_iter0

    def test_streamed_game_slots_unique_across_resume(self, tmp_path):
        """FS-6b: a resumed session continues the slot blocks, never reusing
        ids from the original session's journal events."""
        runs = tmp_path / "runs"
        data = tmp_path / "data"

        run_training(tiny_profile(iterations=1), run_id="streamed",
                     base_dir=runs, data_dir=data)
        run_training(tiny_profile(), run_id="streamed",
                     base_dir=runs, data_dir=data, resume=True)

        events = TrainingJournal(data / "train_log.jsonl").read()
        per_kind: dict[str, list[int]] = {}
        for event in events:
            if event["kind"] in ("live", "game_end"):
                per_kind.setdefault(event["kind"], []).append(event["game"])
        # Session 1 = iteration 0 -> slots {0, 1}; resumed session = iteration
        # 1 -> slots {2, 3}. Every slot distinct across the whole journal.
        assert set(per_kind["game_end"]) == {0, 1, 2, 3}
        assert set(per_kind["live"]) == {0, 1, 2, 3}

    def test_resumed_loss_matches_fresh_run(self, tmp_path, monkeypatch):
        """Exact rebuild: kill/resume reaches the same loss as one clean run.

        Pinned to CPU: CUDA conv-backward is bit-nondeterministic across runs
        (atomicAdd), which would defeat bit-exactness for reasons unrelated to
        resume. The rebuild property itself is device-agnostic.
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        # Isolated lineage roots: the regime champion pointer is shared per
        # base_dir, so the two comparison runs must not gate on each other.
        fresh = run_training(tiny_profile(), run_id="fresh",
                             base_dir=tmp_path / "fresh" / "runs",
                             data_dir=tmp_path / "shared" / "data")
        assert fresh.final_loss is not None

        run_training(tiny_profile(iterations=1), run_id="split",
                     base_dir=tmp_path / "split" / "runs",
                     data_dir=tmp_path / "shared" / "data")
        resumed = run_training(tiny_profile(), run_id="split",
                               base_dir=tmp_path / "split" / "runs",
                               data_dir=tmp_path / "shared" / "data",
                               resume=True)

        assert resumed.final_loss is not None
        assert resumed.final_loss == pytest.approx(fresh.final_loss, abs=1e-5)
        assert resumed.steps == fresh.steps

    def test_profile_mismatch_refuses_resume(self, tmp_path):
        runs = tmp_path / "runs"
        data = tmp_path / "data"
        run_training(tiny_profile(iterations=1), run_id="strict", base_dir=runs, data_dir=data)
        with pytest.raises(ValueError, match="cannot resume run"):
            run_training(tiny_profile(buffer_size=555), run_id="strict", base_dir=runs,
                         data_dir=data, resume=True)

    def test_resume_without_state_starts_fresh(self, tmp_path):
        summary = run_training(tiny_profile(iterations=1), run_id="ghost-run",
                               base_dir=tmp_path / "runs", data_dir=tmp_path / "data",
                               resume=True)
        assert summary.resumed is False
        assert summary.iterations_completed == 1


class TestWarmstartResolution:
    def _write_champion(self, base: Path, regime: str, updated_at: float) -> Path:
        model = GambitNet(MODEL_TINY)
        ckpt = base / regime / "ck.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": model.state_dict(), "config": MODEL_TINY.to_dict(), "tag": "t"},
            ckpt,
        )
        pointer = base / regime / "champion.json"
        pointer.write_text(
            '{"tag": "t", "checkpoint_path": '
            f'"{ckpt.as_posix()}", "elo": 900.0, "updated_at": {updated_at}}}',
            encoding="utf-8",
        )
        return ckpt

    def test_regime_name_spec(self, tmp_path):
        ckpt = self._write_champion(tmp_path, "sf-distill", updated_at=1.0)
        weights = resolve_warmstart("sf-distill", tmp_path)
        assert weights is not None
        reference = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state"]
        key = next(iter(reference))
        assert torch.equal(weights[key], reference[key])

    def test_champion_spec_picks_newest(self, tmp_path):
        self._write_champion(tmp_path, "scratch", updated_at=1.0)
        newest = self._write_champion(tmp_path, "hybrid", updated_at=2.0)
        weights = resolve_warmstart("champion", tmp_path)
        assert weights is not None
        reference = torch.load(newest, map_location="cpu", weights_only=False)["model_state"]
        key = next(iter(reference))
        assert torch.equal(weights[key], reference[key])

    def test_direct_path_spec(self, tmp_path):
        ckpt = self._write_champion(tmp_path, "scratch", updated_at=1.0)
        weights = resolve_warmstart(str(ckpt), tmp_path)
        assert weights is not None

    def test_missing_spec_degrades_to_none(self, tmp_path):
        assert resolve_warmstart("sf-distill", tmp_path) is None
        assert resolve_warmstart("no/such/repo", tmp_path) is None

    def test_broken_checkpoint_degrades(self, tmp_path):
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a checkpoint")
        assert resolve_warmstart(str(bad), tmp_path) is None


class FakeHfApi:
    calls: ClassVar[list[tuple]] = []
    fail: ClassVar[bool] = False

    def __init__(self, token=None) -> None:
        self.token = token

    def create_repo(self, repo_id, repo_type=None, exist_ok=False):
        FakeHfApi.calls.append(("create_repo", repo_id))
        if FakeHfApi.fail:
            raise RuntimeError("hub down")


def fake_upload_file(path_or_fileobj, path_in_repo, repo_id, token):
    FakeHfApi.calls.append(("upload_file", repo_id, str(path_in_repo)))
    if FakeHfApi.fail:
        raise RuntimeError("upload failed")
    return path_in_repo


@pytest.fixture()
def fake_hub(monkeypatch):
    module = types.ModuleType("huggingface_hub")
    module.HfApi = FakeHfApi  # type: ignore[attr-defined]
    module.upload_file = fake_upload_file  # type: ignore[attr-defined]
    FakeHfApi.calls = []
    FakeHfApi.fail = False
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module


class TestHuggingFacePublish:
    def test_no_token_is_graceful_noop(self, tmp_path, monkeypatch, fake_hub):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        checkpoint = tmp_path / "c.pt"
        torch.save({}, checkpoint)
        assert resolve_hf_token() == ""
        assert upload_checkpoint("user/gambit-x", checkpoint) is False
        assert FakeHfApi.calls == []  # never touched the Hub

    def test_upload_with_token(self, tmp_path, monkeypatch, fake_hub):
        monkeypatch.setenv("HF_TOKEN", "tok")
        checkpoint = tmp_path / "c.pt"
        torch.save({"model_state": {}}, checkpoint)
        ok = upload_checkpoint("LanceAbuan/gambit-scratch", checkpoint, metadata={"tag": "t"})
        assert ok is True
        repos = [call[1] for call in FakeHfApi.calls if call[0] == "create_repo"]
        uploads = [call[1] for call in FakeHfApi.calls if call[0] == "upload_file"]
        assert repos == ["LanceAbuan/gambit-scratch"]
        assert uploads == ["LanceAbuan/gambit-scratch", "LanceAbuan/gambit-scratch"]

    def test_upload_failure_never_raises(self, tmp_path, monkeypatch, fake_hub):
        monkeypatch.setenv("HF_TOKEN", "tok")
        FakeHfApi.fail = True
        checkpoint = tmp_path / "c.pt"
        torch.save({}, checkpoint)
        assert upload_checkpoint("user/gambit-x", checkpoint) is False

    def test_missing_checkpoint_skips(self, tmp_path, monkeypatch, fake_hub):
        monkeypatch.setenv("HF_TOKEN", "tok")
        assert upload_checkpoint("user/gambit-x", tmp_path / "nope.pt") is False
        assert FakeHfApi.calls == []

    def test_pipeline_uploads_per_regime_repo(self, tmp_path, monkeypatch, fake_hub):
        monkeypatch.setenv("HF_TOKEN", "tok")
        summary = run_training(
            tiny_profile(iterations=1),
            run_id="hf-run",
            base_dir=tmp_path / "runs",
            data_dir=tmp_path / "data",
        )
        assert summary.checkpoints
        repos = [call[1] for call in FakeHfApi.calls if call[0] == "create_repo"]
        assert repos and all(r == "LanceAbuan/gambit-scratch" for r in repos)
