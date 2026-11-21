"""FS-4 queued evaluator + parallel self-play tests.

The binding contract: N concurrent games submit evaluation requests, the
service batches them (<=256 / 4 ms window), runs ONE forward, and fans results
back out — with ZERO cross-game leakage. Proven here with interleaved fake
requests whose answers are unique per board.
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess
import numpy as np
import pytest

from training.evaluator import QueuedNNEvaluator
from training.model import MODEL_TINY, GambitNet
from training.regimes import get_regime
from training.selfplay import SearchConfig, SelfPlayConfig, parallel_self_play

POLICY_MOVES = 4672


def _board_signature(board: chess.Board) -> float:
    """Deterministic unique-per-position value (leakage canary)."""
    return ((hash(board.fen()) % 10000) / 10000.0) - 0.5


def _signature_index(board: chess.Board) -> int:
    return hash(board.fen()) % POLICY_MOVES


def fake_forward(calls: list[int] | None = None, gate: threading.Event | None = None):
    """Batch forward stub: each board's answer encodes its own identity."""

    def forward(batch):
        if calls is not None:
            calls.append(len(batch))
        if gate is not None:
            gate.wait(timeout=10.0)
        logits = np.zeros((len(batch), POLICY_MOVES), dtype=np.float32)
        values = np.zeros(len(batch), dtype=np.float32)
        for i, board in enumerate(batch):
            logits[i, _signature_index(board)] = 1.0
            values[i] = _board_signature(board)
        return logits, values

    return forward


class TestQueuedNNEvaluator:
    def test_single_request_roundtrip(self):
        evaluator = QueuedNNEvaluator(forward_fn=fake_forward())
        try:
            board = chess.Board()
            logits, values = evaluator.evaluate([board])
            assert logits.shape == (1, POLICY_MOVES)
            assert values.shape == (1,)
            assert values[0] == pytest.approx(_board_signature(board))
            assert logits[0, _signature_index(board)] == 1.0
        finally:
            evaluator.close()

    def test_empty_batch_returns_empty(self):
        evaluator = QueuedNNEvaluator(forward_fn=fake_forward())
        try:
            logits, values = evaluator.evaluate([])
            assert logits.shape[0] == 0 and values.shape[0] == 0
        finally:
            evaluator.close()

    def test_zero_cross_game_leakage_interleaved(self):
        """4 games x 8 interleaved requests: every future gets ONLY its own rows."""
        games = 4
        plies = 8
        calls: list[int] = []
        evaluator = QueuedNNEvaluator(forward_fn=fake_forward(calls))
        try:
            # Each "game" walks its own distinct position sequence.
            game_boards: list[list[chess.Board]] = []
            for game in range(games):
                boards = [chess.Board()]
                for _ in range(plies - 1):
                    nxt = boards[-1].copy()
                    nxt.push(list(nxt.legal_moves)[game % len(list(nxt.legal_moves))])
                    boards.append(nxt)
                game_boards.append(boards)

            errors: list[Exception] = []

            def submit(game: int, ply: int) -> None:
                try:
                    logits, values = evaluator.evaluate([game_boards[game][ply]])
                    board = game_boards[game][ply]
                    assert values.shape == (1,)
                    if values[0] != pytest.approx(_board_signature(board)):
                        raise AssertionError(
                            f"LEAK: game {game} ply {ply} got value {values[0]} "
                            f"expected {_board_signature(board)}"
                        )
                    if logits[0, _signature_index(board)] != 1.0:
                        raise AssertionError(f"LEAK: game {game} ply {ply} got foreign logits")
                except Exception as exc:  # noqa: BLE001 - collected below
                    errors.append(exc)

            with ThreadPoolExecutor(max_workers=games) as pool:
                futures = [
                    pool.submit(submit, game, ply)
                    for ply in range(plies)
                    for game in range(games)  # round-robin interleave
                ]
                for future in futures:
                    future.result()
            assert not errors, errors
            # The shared forward actually merged requests into batches.
            assert calls and max(calls) > 1, f"expected batching, got batch sizes {calls}"
        finally:
            evaluator.close()

    def test_batch_cap_respected_threaded(self):
        """max_batch_size=4 caps every forwarded batch (10 concurrent requests)."""
        calls: list[int] = []
        gate = threading.Event()
        evaluator = QueuedNNEvaluator(
            forward_fn=fake_forward(calls, gate=gate), max_batch_size=4, collect_window_seconds=0.2
        )
        results: list[tuple] = []
        errors: list[Exception] = []

        def submit(i: int):
            try:
                board = chess.Board()
                board.push(list(board.legal_moves)[i % 20])
                _logits, values = evaluator.evaluate([board])
                results.append((board.fen(), float(values[0])))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        try:
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = []
                for i in range(10):
                    futures.append(pool.submit(submit, i))
                time.sleep(0.3)  # let all 10 queue up behind the gated forward
                gate.set()
                for future in futures:
                    future.result()
            assert not errors, errors
            assert len(results) == 10
            assert all(size <= 4 for size in calls), calls
            assert max(calls) == 4, calls  # the cap actually bit
            for fen, value in results:
                assert value == pytest.approx(_board_signature(chess.Board(fen)))
        finally:
            gate.set()
            evaluator.close()

    def test_window_collects_late_requests(self):
        """A request arriving inside the collect window joins the same batch."""
        calls: list[int] = []
        evaluator = QueuedNNEvaluator(forward_fn=fake_forward(calls), collect_window_seconds=0.15)
        try:
            def late_submit():
                time.sleep(0.05)
                board = chess.Board()
                board.push_san("e4")
                evaluator.evaluate([board])

            thread = threading.Thread(target=late_submit)
            thread.start()
            evaluator.evaluate([chess.Board()])
            thread.join(timeout=5.0)
            assert calls == [2], f"expected one merged batch of 2, got {calls}"
        finally:
            evaluator.close()

    def test_evaluate_after_close_raises(self):
        evaluator = QueuedNNEvaluator(forward_fn=fake_forward())
        evaluator.close()
        with pytest.raises(RuntimeError, match="closed"):
            evaluator.evaluate([chess.Board()])

    def test_invalid_batch_size_rejected(self):
        with pytest.raises(ValueError):
            QueuedNNEvaluator(forward_fn=fake_forward(), max_batch_size=0)


class TestParallelSelfPlay:
    def test_parallel_games_complete_in_order(self):
        model = GambitNet(MODEL_TINY)
        model.eval()
        evaluator = QueuedNNEvaluator(model, max_batch_size=64, collect_window_seconds=0.002)
        config = SelfPlayConfig(search=SearchConfig(num_simulations=2, root_noise=False), max_plies=4)
        try:
            results = parallel_self_play(evaluator, 6, config, parallel_games=4)
        finally:
            evaluator.close()
        assert len(results) == 6
        assert all(r.ply_count <= 4 for r in results)
        assert all(len(ex.policy) == POLICY_MOVES for r in results for ex in r.examples)
        assert all(ex.planes.shape == (29, 8, 8) for r in results for ex in r.examples)

    def test_sequential_path_matches_batch_size(self):
        model = GambitNet(MODEL_TINY)
        model.eval()
        evaluator = QueuedNNEvaluator(model, max_batch_size=64, collect_window_seconds=0.002)
        config = SelfPlayConfig(search=SearchConfig(num_simulations=2, root_noise=False), max_plies=4)
        try:
            results = parallel_self_play(evaluator, 3, config, parallel_games=1)
        finally:
            evaluator.close()
        assert len(results) == 3

    def test_invalid_parallelism_rejected(self):
        model = GambitNet(MODEL_TINY)
        evaluator = QueuedNNEvaluator(model, collect_window_seconds=0.001)
        try:
            with pytest.raises(ValueError):
                parallel_self_play(evaluator, 2, SelfPlayConfig(), parallel_games=0)
        finally:
            evaluator.close()


class TestRegimeParallelConfig:
    def test_parallel_games_bounds(self):
        for bad in (0, 33, -1):
            with pytest.raises(ValueError, match="parallel_games"):
                get_regime("scratch").__class__(
                    name="x", description="x", data_source="selfplay", num_simulations=1,
                    games_per_iteration=1, train_steps_per_iteration=1, batch_size=1,
                    learning_rate=0.1, parallel_games=bad,
                )

    def test_regimes_carry_parallel_games(self):
        for name in ("scratch", "sf-games", "sf-distill", "hybrid", "finetune"):
            config = get_regime(name)
            assert 1 <= config.parallel_games <= 32

    def test_sf_share_and_warmstart_bounds(self):
        base = {
            "name": "x", "description": "x", "data_source": "sf-games", "num_simulations": 1,
            "games_per_iteration": 1, "train_steps_per_iteration": 1, "batch_size": 1,
            "learning_rate": 0.1,
        }
        regime_cls = get_regime("scratch").__class__
        with pytest.raises(ValueError, match="sf_games_share"):
            regime_cls(**base, sf_games_share=1.5)
        with pytest.raises(ValueError, match="warmstart_minutes"):
            regime_cls(**base, warmstart_minutes=-1.0)
