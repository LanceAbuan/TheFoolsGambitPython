"""WP-F C6 gate (b): python vs rust search throughput (nodes/sec).

One node = one simulation (selection + expansion + backup). The evaluator is
a deterministic in-process stub, so the comparison isolates TREE work from NN
latency - the NN call is identical for both backends in production (the same
batched service sits behind the bridge callback).

Usage:
    python scripts/bench_search_backends.py [--sims 400] [--searches 5]

Rust rows run through the real PyO3 bridge (GIL round-trips included).
Prints the throughput table the PR must include.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Sequence

import chess
import numpy as np

import gambit_core
from training.selfplay import MCTSSearch, SearchConfig

log = logging.getLogger(__name__)

BENCH_FENS: tuple[str, ...] = (
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
)

POLICY_MOVES = 4672


def _canon_key(board: chess.Board) -> tuple:
    """Position identity without move-stack fields (parity with the bridge)."""
    return (board.board_fen(), board.turn, board.castling_rights, board.ep_square)


def _uniform(boards: Sequence[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
    """Zero-cost evaluator: uniform priors, zero value (tree-only cost)."""
    n = len(boards)
    return np.zeros((n, POLICY_MOVES), dtype=np.float32), np.zeros(n, dtype=np.float32)


class UniformPythonEvaluator:
    """Evaluator protocol double feeding the python MCTSSearch (tree-only)."""

    def evaluate(self, boards: Sequence[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        return _uniform(boards)


def rust_callback(fens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Bridge callback feeding the pyfunction search (stub, no NN)."""
    return _uniform([chess.Board(fen) for fen in fens])




def _bench(call: Callable[[], object], searches: int) -> float:
    """Seconds per run of `searches` full searches (median of 3 rounds)."""
    rounds: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(searches):
            call()
        rounds.append(time.perf_counter() - start)
    return min(rounds) / searches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=400, help="simulations per search")
    parser.add_argument("--searches", type=int, default=5, help="searches per timing round")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[8, 16],
                        help="rust wave sizes to bench beyond the sequential mode")
    args = parser.parse_args()

    rows: list[tuple[str, float]] = []

    def py_search() -> None:
        for fen in BENCH_FENS:
            MCTSSearch(
                UniformPythonEvaluator(), SearchConfig(num_simulations=args.sims, root_noise=False)
            ).search(chess.Board(fen))

    # Warmup (first-call allocs / imports), then bench.
    py_search()
    seconds = _bench(py_search, args.searches)
    rows.append(("python MCTSSearch (batch=1)", args.sims * len(BENCH_FENS) / seconds))

    for batch in [1, *args.batch_sizes]:
        def rust_search(batch: int = batch) -> None:
            cfg = gambit_core.SearchConfig(
                num_simulations=args.sims, root_noise=False, batch_size=batch
            )
            for fen in BENCH_FENS:
                gambit_core.search(fen, rust_callback, cfg)

        rust_search()
        seconds = _bench(rust_search, args.searches)
        rows.append((f"rust gambit_core.search (batch={batch})", args.sims * len(BENCH_FENS) / seconds))

    base = rows[0][1]
    print()
    print(f"nodes/sec over {len(BENCH_FENS)} fixed positions, {args.sims} sims each (stub evaluator, noise off)")
    print("-" * 72)
    for name, nps in rows:
        print(f"{name:<38} {nps:>12,.0f} nodes/s  ({nps / base:>5.1f}x python)")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
