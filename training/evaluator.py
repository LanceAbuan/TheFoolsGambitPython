"""Queued GPU evaluator (plan §5 lever 2 — "NNEvaluator micro-batching").

N concurrent self-play games submit ``(board ...)`` evaluation requests; a
single worker thread collects up to ``max_batch_size`` (default 256) requests
or until ``collect_window_seconds`` (default 4 ms) elapses since the first
pending request, runs ONE model forward, and fans the result rows back out to
each waiting caller via its own ``Future``.

Correctness contract (FS-4):
- **Zero cross-game leakage** — every request keeps its own boards and its own
  future; the batch is a pure concatenation and the fan-out is a pure slice.
  `tests/test_training_evaluator.py` proves it with interleaved fake requests.
- The :class:`~training.selfplay.Evaluator` protocol is preserved
  (``evaluate(boards) -> (logits (N, 4672), values (N,))``), so
  :class:`~training.selfplay.MCTSSearch` is unaware of the queue.

Latency contract: with a single caller the request still waits out the collect
window, so sequential self-play keeps using the synchronous
``BatchNNEvaluator`` — the queue is only worth its window once more than one
game feeds it (``RegimeConfig.parallel_games > 1``).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Self

import chess
import numpy as np
import torch

from .encoder import encode_batch
from .model import value_head_scalar

log = logging.getLogger(__name__)

MAX_BATCH_SIZE = 256        # §5: "up to 256"
COLLECT_WINDOW_SECONDS = 0.004  # §5: "4 ms window"
_DRAIN_POLL_SECONDS = 0.0005
_WORKER_POLL_SECONDS = 0.05

# Type of the pluggable batch forward: boards -> (logits (N, POLICY), values (N,)).
ForwardFn = Callable[[Sequence[chess.Board]], tuple[np.ndarray, np.ndarray]]


@dataclass
class _EvalRequest:
    """One caller's submission: its boards + the future that gets its rows."""

    boards: list[chess.Board]
    future: Future = field(default_factory=Future)


class QueuedNNEvaluator:
    """Batching evaluation service shared by N concurrent games."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        device: torch.device | None = None,
        max_batch_size: int = MAX_BATCH_SIZE,
        collect_window_seconds: float = COLLECT_WINDOW_SECONDS,
        forward_fn: ForwardFn | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if collect_window_seconds < 0:
            raise ValueError("collect_window_seconds must be >= 0")
        if forward_fn is not None:
            self._forward: ForwardFn = forward_fn
        else:
            if model is None:
                raise ValueError("either model or forward_fn is required")
            self._forward = _model_forward(model, device)
        self._max_batch_size = int(max_batch_size)
        self._collect_window_seconds = float(collect_window_seconds)
        self._queue: queue.Queue[_EvalRequest] = queue.Queue()
        self._running = True
        self._worker = threading.Thread(
            target=self._run, name="queued-nn-evaluator", daemon=True
        )
        self._worker.start()

    # -- Evaluator protocol -------------------------------------------------

    def evaluate(self, boards: Sequence[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        """Submit `boards` and block until the shared forward resolves them."""
        if not boards:
            return (
                np.zeros((0, 0), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        if not self._running:
            raise RuntimeError("queued evaluator is closed")
        request = _EvalRequest(boards=list(boards))
        self._queue.put(request)
        return request.future.result()

    # -- service internals ---------------------------------------------------

    def _run(self) -> None:
        while self._running or not self._queue.empty():
            try:
                first = self._queue.get(timeout=_WORKER_POLL_SECONDS)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.perf_counter() + self._collect_window_seconds
            while len(batch) < self._max_batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                    continue
                except queue.Empty:
                    pass
                if time.perf_counter() >= deadline:
                    break
                time.sleep(_DRAIN_POLL_SECONDS)
            self._process(batch)

    def _process(self, batch: list[_EvalRequest]) -> None:
        boards = [board for request in batch for board in request.boards]
        try:
            logits, values = self._forward(boards)
        except Exception as exc:  # noqa: BLE001 - fan the failure out to every waiter
            log.error("[EVALQ] forward failed (batch of %d requests): %s", len(batch), exc)
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)
            return
        offset = 0
        for request in batch:
            size = len(request.boards)
            request.future.set_result(
                (
                    np.asarray(logits[offset : offset + size], dtype=np.float32),
                    np.asarray(values[offset : offset + size], dtype=np.float32),
                )
            )
            offset += size

    # -- lifecycle -----------------------------------------------------------

    def close(self, timeout: float = 5.0) -> None:
        """Stop the worker; requests still queued are failed, not dropped silently."""
        self._running = False
        self._worker.join(timeout=timeout)
        while True:
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                break
            if not request.future.done():
                request.future.set_exception(RuntimeError("evaluator closed before serving request"))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


def _model_forward(model: torch.nn.Module, device: torch.device | None) -> ForwardFn:
    """Build the real batched forward from a GambitNet (+ device)."""
    resolved = device if device is not None else next(model.parameters()).device

    def forward(boards: Sequence[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        batch = torch.from_numpy(encode_batch(boards)).to(resolved)
        with torch.inference_mode():
            logits, values = model(batch)
        # WP-C: a wdl value head returns (B, 3) logits; the queued evaluator
        # fans out the derived scalar (p_win - p_loss) so the Evaluator
        # protocol — and every MCTSSearch behind it — is unchanged.
        scalar_values = value_head_scalar(values)
        return (
            logits.detach().cpu().numpy().astype(np.float32),
            scalar_values.detach().cpu().numpy().astype(np.float32),
        )

    return forward
