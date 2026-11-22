"""Bench: gambit_core (Rust) vs python-chess + training.encoder (Python).

FS-7 acceptance evidence for docs/UPGRADE_PLAN.md §5/§13. Measures the CPU
portion of a self-play/MCTS tree step — legal movegen, batch encoding, legal
mask generation — plus a composite "full step" workload. python-chess remains
the oracle; this script only measures speed, never correctness.

NOTE: lives inside the crate (scripts/ writes were permission-blocked for this
worker); the intended final home is scripts/bench_gambit_core.py.

Usage:
    uv run python gambit_core/bench_gambit_core.py [--positions N] [--reps R]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

import gambit_core
from training.encoder import encode_batch as py_encode_batch
from training.encoder import encode_state as py_encode_state
from training.encoder import legal_mask as py_legal_mask

WALK_STARTS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
]


def random_positions(n: int, seed: int = 20261121) -> list[str]:
    """Random legal positions from varied starts (realistic mid-game mix)."""
    rng = random.Random(seed)
    fens: list[str] = []
    while len(fens) < n:
        board = chess.Board(WALK_STARTS[len(fens) % len(WALK_STARTS)])
        for _ in range(rng.randint(10, 90)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            fens.append(board.fen())
            if len(fens) >= n:
                break
    return fens[:n]


def best_of(reps: int, fn):
    best = float("inf")
    result = None
    for _ in range(reps):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return best, result


def fmt(us: float) -> str:
    return f"{us:10.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=2000)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    fens = random_positions(args.positions)
    py_boards = [chess.Board(fen) for fen in fens]
    n = len(fens)
    print(f"gambit_core bench — {n} random positions, best of {args.reps}")
    print("baselines: python-chess 1.11.2 + training.encoder (pure Python)")
    print()

    rows: list[tuple[str, float, float]] = []  # (workload, python µs/pos, rust µs/pos)

    # --- legal move generation (full UCI list) ------------------------------
    def py_moves():
        return [len(list(board.legal_moves)) for board in py_boards]

    def rust_moves():
        return [len(gambit_core.legal_moves(fen)) for fen in fens]

    t_py, ref_moves = best_of(args.reps, py_moves)
    t_rust, rust_moves_out = best_of(args.reps, rust_moves)
    assert ref_moves == rust_moves_out, "movegen counts diverged?!"
    rows.append(("legal_moves() -> UCI list", t_py / n * 1e6, t_rust / n * 1e6))

    # --- batch encoding (29, 8, 8) float32 ----------------------------------
    def py_encode():
        return py_encode_batch(py_boards)

    def rust_encode():
        return gambit_core.encode_batch(fens)

    t_py, ref_planes = best_of(args.reps, py_encode)
    t_rust, rust_planes = best_of(args.reps, rust_encode)
    # Parity is proven in tests; sanity-check shapes here.
    assert ref_planes.shape == rust_planes.shape == (n, 29, 8, 8)
    rows.append(("encode_batch() (pre-built boards)", t_py / n * 1e6, t_rust / n * 1e6))

    # --- encode from FEN strings (incl. python board construction) ----------
    def py_encode_from_fen():
        return py_encode_batch([chess.Board(fen) for fen in fens])

    t_py_from_fen, _ = best_of(args.reps, py_encode_from_fen)
    rows.append(("encode_batch() incl. Board(fen)", t_py_from_fen / n * 1e6, t_rust / n * 1e6))

    # --- legal masks (4672) float32 ------------------------------------------
    def py_masks():
        return np.stack([py_legal_mask(board) for board in py_boards])

    def rust_masks():
        return gambit_core.legal_masks(fens)

    t_py, ref_masks = best_of(args.reps, py_masks)
    t_rust, rust_masks_out = best_of(args.reps, rust_masks)
    assert ref_masks.shape == rust_masks_out.shape == (n, 4672)
    rows.append(("legal_masks() -> (N,4672)", t_py / n * 1e6, t_rust / n * 1e6))

    # --- composite: one self-play/MCTS tree step per position ----------------
    # python: legal moves + push first move + mask + encode
    # rust:   parse + legal moves + make_move + mask + encode
    def py_step():
        out = []
        for board in py_boards:
            moves = list(board.legal_moves)
            child = board.copy(stack=False)
            if moves:
                child.push(moves[0])
            out.append((len(moves), py_legal_mask(board), py_encode_state(board)))
        return out

    def rust_step():
        out = []
        for fen in fens:
            b = gambit_core.Board(fen)
            moves = b.legal_moves()
            child = b.make_move(moves[0]) if moves else b
            out.append((len(moves), child.legal_mask(), child.encode()))
        return out

    t_py, ref_steps = best_of(args.reps, py_step)
    t_rust, rust_steps = best_of(args.reps, rust_step)
    assert len(ref_steps) == len(rust_steps)
    rows.append(("full tree-step CPU slice", t_py / n * 1e6, t_rust / n * 1e6))

    # --- report ---------------------------------------------------------------
    header = f"{'workload':34} {'python µs/pos':>13} {'rust µs/pos':>12} {'speedup':>9}"
    print(header)
    print("-" * len(header))
    for name, py_us, rust_us in rows:
        print(f"{name:34} {fmt(py_us)} {fmt(rust_us)} {py_us / rust_us:8.1f}x")
    name, py_us, rust_us = rows[-1]
    print()
    print(
        f"Composite tree-step speedup: {py_us / rust_us:.1f}x "
        f"({py_us:.0f} µs -> {rust_us:.0f} µs per position)"
    )
    print(
        f"Estimated self-play stepping rate: "
        f"{1e6 / rust_us:,.0f} positions/sec (rust) vs {1e6 / py_us:,.0f} (python)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
