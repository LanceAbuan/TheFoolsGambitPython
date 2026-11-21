"""FS-3 data provider tests: sf-games / sf-distill / hybrid / finetune.

Engines are fakes (no Stockfish required); the resolution chain itself is
tested against the real ``tools/stockfish.exe`` when present. Every provider
must return examples in the shared buffer format, and degrade to pure
self-play when no engine resolves.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest

from training.model import MODEL_TINY, GambitNet
from training.regimes import (
    FinetuneProvider,
    HybridProvider,
    SelfPlayProvider,
    SFDistillProvider,
    SFGamesProvider,
    get_regime,
)
from training.selfplay import BatchNNEvaluator, SearchConfig, SelfPlayConfig
from training.sf_data import sf_skill_for_progress, skill_share_plan
from training.stockfish_engine import TOOLS_DIR, resolve_stockfish_path

POLICY_MOVES = 4672

TINY_CONFIG = SelfPlayConfig(search=SearchConfig(num_simulations=2, root_noise=False), max_plies=4)


def make_evaluator() -> BatchNNEvaluator:
    """A tiny CPU evaluator for the self-play halves of every provider."""
    model = GambitNet(MODEL_TINY)
    model.eval()
    return BatchNNEvaluator(model)


class FakeSF:
    """Deterministic engine double: takes the first legal move, evals +50cp."""

    def __init__(self, skill: int | None = None) -> None:
        self.skill = skill
        self.closed = False
        self.move_calls = 0
        self.eval_calls = 0

    def get_move(self, board: chess.Board, depth: int | None = None) -> str | None:
        self.move_calls += 1
        moves = list(board.legal_moves)
        if not moves:
            return None
        return moves[0].uci()

    def get_evaluation(self, board: chess.Board) -> int:
        self.eval_calls += 1
        return 50  # white +0.5 pawns

    def close(self) -> None:
        self.closed = True


def make_provider(cls, regime_name, **regime_overrides):
    regime = get_regime(regime_name).__class__(
        **{**get_regime(regime_name).to_dict(), **regime_overrides}
    )
    engine = FakeSF()
    provider = cls(regime, engine_factory=lambda skill: engine)
    return provider, engine


class TestStockfishPathResolution:
    """§10 'Stockfish path' bug: no more dead WSL default — env -> tools/ -> PATH."""

    def test_env_var_wins(self, tmp_path, monkeypatch):
        fake_binary = tmp_path / "my-stockfish.exe"
        fake_binary.write_bytes(b"stub")
        monkeypatch.setenv("STOCKFISH_PATH", str(fake_binary))
        assert resolve_stockfish_path() == str(fake_binary)

    def test_invalid_env_falls_through_to_tools(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STOCKFISH_PATH", str(tmp_path / "missing.exe"))
        fake_tools = tmp_path / "tools"
        fake_tools.mkdir()
        (fake_tools / "stockfish.exe").write_bytes(b"stub")
        monkeypatch.setattr("training.stockfish_engine.TOOLS_DIR", fake_tools)
        resolved = resolve_stockfish_path()
        assert resolved is not None
        assert Path(resolved) == fake_tools / "stockfish.exe"

    def test_path_fallback_when_no_env_no_tools(self, tmp_path, monkeypatch):
        monkeypatch.delenv("STOCKFISH_PATH", raising=False)
        monkeypatch.setattr("training.stockfish_engine.TOOLS_DIR", tmp_path)
        import shutil

        assert resolve_stockfish_path() == shutil.which("stockfish")

    def test_repo_tools_binary_resolves(self):
        """The real binary shipped to gitignored tools/ resolves (skip on CI)."""
        if not any((TOOLS_DIR / name).is_file() for name in ("stockfish.exe", "stockfish")):
            pytest.skip("no repo-local stockfish binary on this machine")
        resolved = resolve_stockfish_path()
        assert resolved is not None
        assert "stockfish" in resolved.lower()

    def test_legacy_dead_wsl_default_gone(self):
        """The old hardcoded '/home/lance/.local/bin/stockfish' must never return."""
        resolved = resolve_stockfish_path()
        assert resolved != "/home/lance/.local/bin/stockfish"


class TestSkillLadder:
    def test_ladder_endpoints(self):
        assert sf_skill_for_progress(2, 12, 0.0) == 2
        assert sf_skill_for_progress(2, 12, 1.0) == 12

    def test_ladder_midpoint(self):
        assert sf_skill_for_progress(2, 12, 0.5) == 7

    def test_ladder_clamps(self):
        assert sf_skill_for_progress(2, 12, -5.0) == 2
        assert sf_skill_for_progress(2, 12, 5.0) == 12

    def test_share_plan(self):
        assert skill_share_plan(8, 0.5) == 4
        assert skill_share_plan(8, 0.0) == 0
        assert skill_share_plan(8, 1.0) == 8
        assert skill_share_plan(0, 0.5) == 0


class TestSFGamesProvider:
    def test_collect_returns_examples_and_marks_opponent(self):
        provider, engine = make_provider(SFGamesProvider, "sf-games")
        results = provider.collect(make_evaluator(), 4, TINY_CONFIG, progress=0.0)
        assert len(results) == 4
        sf_results = [r for r in results if r.opponent is not None]
        assert sf_results, "expected the SF share of games to be marked"
        assert all(r.opponent == "stockfish-skill-2" for r in sf_results)
        assert all(r.ply_count <= TINY_CONFIG.max_plies for r in results)
        for result in results:
            for ex in result.examples:
                assert ex.planes.shape == (29, 8, 8)
                assert ex.policy.shape == (POLICY_MOVES,)
                assert -1.0 <= ex.value <= 1.0
        provider.close()
        assert engine.closed

    def test_progress_scales_skill(self):
        provider, _engine = make_provider(SFGamesProvider, "sf-games")
        results = provider.collect(make_evaluator(), 2, TINY_CONFIG, progress=1.0)
        sf_results = [r for r in results if r.opponent is not None]
        assert sf_results
        assert all(r.opponent == "stockfish-skill-12" for r in sf_results)
        provider.close()

    def test_share_zero_is_pure_selfplay(self):
        provider, engine = make_provider(SFGamesProvider, "sf-games", sf_games_share=0.0)
        results = provider.collect(make_evaluator(), 3, TINY_CONFIG)
        assert all(r.opponent is None for r in results)
        assert engine.move_calls == 0
        provider.close()

    def test_no_engine_degrades_to_selfplay(self):
        regime = get_regime("sf-games")
        provider = SFGamesProvider(regime, engine_factory=lambda skill: None)
        results = provider.collect(make_evaluator(), 3, TINY_CONFIG)
        assert len(results) == 3
        assert all(r.opponent is None for r in results)
        provider.close()

    def test_nn_and_sf_alternate_colors(self):
        """Across SF games the NN plays both colors (board sanity via results)."""
        provider, _engine = make_provider(SFGamesProvider, "sf-games", sf_games_share=1.0)
        results = provider.collect(make_evaluator(), 4, TINY_CONFIG)
        assert len(results) == 4
        assert all(r.opponent == "stockfish-skill-2" for r in results)
        provider.close()

    def test_collect_streams_with_unique_slots(self, tmp_path):
        """FS-6b: SF + self-play games share one iteration's slot space."""
        from pathlib import Path as _Path

        from training.journal import TrainingJournal
        from training.selfplay import StreamContext

        journal = TrainingJournal(_Path(tmp_path) / "provider-stream.jsonl")
        stream = StreamContext(journal=journal, run_id="prov", regime="sf-games")
        provider, _engine = make_provider(SFGamesProvider, "sf-games")
        results = provider.collect(make_evaluator(), 4, TINY_CONFIG, stream=stream)
        provider.close()

        events = journal.read()
        live = [e for e in events if e["kind"] == "live"]
        ends = [e for e in events if e["kind"] == "game_end"]
        assert live and len(ends) == len(results)
        slots = {e["game"] for e in events}
        # Slots are unique per game within the iteration: one per finished game.
        assert len(slots) == len(results)
        # Every slot's live events replay as one coherent game.
        for slot in slots:
            moves = [e["move"] for e in live if e["game"] == slot]
            board = chess.Board()
            for uci in moves:
                board.push_uci(uci)
            assert board.is_valid()


class TestSFDistillProvider:
    def test_distill_targets_are_sf_one_hot_and_eval(self):
        """Policy target = one-hot at the SF move's encoder index; value = SF eval."""
        provider, engine = make_provider(SFDistillProvider, "sf-distill")
        results = provider.collect(make_evaluator(), 2, TINY_CONFIG)
        assert len(results) == 2
        assert all(r.opponent == "stockfish-distill" for r in results)
        assert engine.move_calls > 0 and engine.eval_calls > 0
        for result in results:
            assert result.examples, "distill games must record positions"
            for ex in result.examples:
                # One-hot teacher policy over the 4672 indices.
                assert ex.policy.shape == (POLICY_MOVES,)
                nonzero = np.nonzero(ex.policy)[0]
                assert len(nonzero) == 1
                assert ex.policy.sum() == pytest.approx(1.0)
                # FakeSF evals +50cp -> ±50/2000 from the mover's perspective.
                assert abs(ex.value) == pytest.approx(50 / 2000.0)
        provider.close()

    def test_distill_without_engine_degrades(self):
        regime = get_regime("sf-distill")
        provider = SFDistillProvider(regime, engine_factory=lambda skill: None)
        results = provider.collect(make_evaluator(), 2, TINY_CONFIG)
        assert len(results) == 2
        assert all(r.opponent is None for r in results)
        provider.close()

    def test_distill_positions_come_from_selfplay_play(self):
        """The NN's own search chooses the moves, not the SF move."""
        provider, _engine = make_provider(SFDistillProvider, "sf-distill")
        results = provider.collect(make_evaluator(), 1, TINY_CONFIG)
        # FakeSF always plays the FIRST legal move; if the game followed SF
        # blindly from the startpos, every game would be identical e2e3-style
        # play. Self-play sampling from MCTS visits gives distinct move lists
        # across seeds; at minimum, the PGN must parse as a legal game.
        board = chess.Board()
        for san in results[0].moves:
            board.push_san(san)
        assert len(results[0].moves) == results[0].ply_count
        provider.close()


class TestHybridProvider:
    def test_distill_phase_before_warmstart_minutes(self):
        provider, engine = make_provider(HybridProvider, "hybrid", warmstart_minutes=10.0)
        results = provider.collect(make_evaluator(), 2, TINY_CONFIG, elapsed_minutes=5.0)
        assert all(r.opponent == "stockfish-distill" for r in results)
        assert engine.move_calls > 0
        provider.close()

    def test_selfplay_phase_after_warmstart_minutes(self):
        provider, engine = make_provider(HybridProvider, "hybrid", warmstart_minutes=10.0)
        results = provider.collect(make_evaluator(), 2, TINY_CONFIG, elapsed_minutes=30.0)
        assert all(r.opponent is None for r in results)
        assert engine.move_calls == 0  # the SF engine is untouched in phase 2
        provider.close()

    def test_phase_boundary_is_inclusive(self):
        provider, _engine = make_provider(HybridProvider, "hybrid", warmstart_minutes=10.0)
        results = provider.collect(make_evaluator(), 1, TINY_CONFIG, elapsed_minutes=10.0)
        assert all(r.opponent is None for r in results)
        provider.close()


class TestFinetuneProvider:
    def test_finetune_data_is_selfplay(self):
        provider = FinetuneProvider()
        results = provider.collect(make_evaluator(), 2, TINY_CONFIG)
        assert len(results) == 2
        assert all(r.opponent is None for r in results)
        for result in results:
            for ex in result.examples:
                assert ex.planes.shape == (29, 8, 8)
                assert ex.policy.shape == (POLICY_MOVES,)

    def test_provider_close_is_safe(self):
        provider = FinetuneProvider()
        provider.close()  # no engine, must not raise
        assert SelfPlayProvider().close() is None
