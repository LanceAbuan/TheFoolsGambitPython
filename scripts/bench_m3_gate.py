"""M3 gate bench: Python-path vs Rust-path end-to-end self-play throughput.

Gate (docs/RESEARCH_STANDARD.md principles 1-7 + slice M3 brief):
    >= 3x end-to-end sims/sec at parallel_games=16, same seed, best-of-3
    with variance, or an honest report of why the ceiling is lower.

Methodology (contended-machine safe):
- The machine is under external load (live training run + desktop apps), so
  ABSOLUTE numbers are inflated and only INTERLEAVED PAIRED ratios are
  trusted. Arms alternate order per pair to decorrelate load drift.
- Each arm: parallel_self_play with the production queued evaluator
  (QueuedNNEvaluator), GAMBIT_CORE_FASTPATH off/on, identical seeding.
- sims metric = sum(ply_count) * num_simulations / wall_seconds (the same
  metric as the recorded Python baseline 745.9 / 777.4 sims/sec).

Usage:
    uv run python scripts/bench_m3_gate.py [--pairs 3] [--plies 64]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU-pinned, before torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from training import fastpath as fp
from training.evaluator import QueuedNNEvaluator
from training.model import MODEL_TINY, GambitNet
from training.selfplay import (
    SearchConfig,
    SelfPlayConfig,
    parallel_self_play,
)

SEED = 123
NUM_GAMES = 16
PARALLEL_GAMES = 16
NUM_SIMULATIONS = 32


def run_arm(config: SelfPlayConfig, model: torch.nn.Module, seed: int) -> tuple[int, float]:
    """One parallel_self_play arm; returns (total plies, wall seconds)."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    evaluator = QueuedNNEvaluator(model)
    t0 = time.perf_counter()
    results = parallel_self_play(
        evaluator,
        NUM_GAMES,
        config,
        parallel_games=PARALLEL_GAMES,
        rng=rng,
    )
    wall = time.perf_counter() - t0
    evaluator.close()
    plies = sum(r.ply_count for r in results)
    return plies, wall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=3, help="interleaved pairs (best-of-N)")
    parser.add_argument("--plies", type=int, default=64, help="max_plies per game (primary)")
    parser.add_argument("--plies-short", type=int, default=16, help="secondary short-game arm")
    args = parser.parse_args()

    torch.set_num_threads(4)
    model = GambitNet(MODEL_TINY)
    model.eval()

    print(
        f"M3 GATE BENCH: games={NUM_GAMES} parallel_games={PARALLEL_GAMES} "
        f"sims={NUM_SIMULATIONS} pairs={args.pairs} plies={args.plies}+{args.plies_short} "
        f"(CPU-pinned, interleaved arms)"
    )

    verdicts: list[tuple[int, float]] = []
    for plies_cap in (args.plies, args.plies_short):
        config = SelfPlayConfig(
            search=SearchConfig(num_simulations=NUM_SIMULATIONS),
            max_plies=plies_cap,
            temperature_plies=plies_cap,
        )
        rows: list[tuple[str, float, float, float]] = []
        for pair in range(args.pairs):
            seed = SEED + pair
            # Alternate arm order per pair to decorrelate machine-load drift.
            order = ("python", "rust") if pair % 2 == 0 else ("rust", "python")
            outcomes: dict[str, tuple[int, float]] = {}
            for arm in order:
                if arm == "python":
                    fp.disable_fastpath()
                else:
                    fp.enable_fastpath()
                plies, wall = run_arm(config, model, seed)
                outcomes[arm] = (plies, wall)
                label = f"plies_cap={plies_cap} pair={pair + 1} {arm:>6}"
                print(
                    f"  {label}: plies={plies} wall={wall:.2f}s "
                    f"sims_per_sec={plies * NUM_SIMULATIONS / wall:.1f} "
                    f"games_per_hour={NUM_GAMES / wall * 3600:.1f}"
                )
            fp.disable_fastpath()

            py_plies, py_wall = outcomes["python"]
            rs_plies, rs_wall = outcomes["rust"]
            py_rate = py_plies * NUM_SIMULATIONS / py_wall
            rs_rate = rs_plies * NUM_SIMULATIONS / rs_wall
            ratio = rs_rate / py_rate
            rows.append(("python", py_rate, py_wall, 1.0))
            rows.append(("rust", rs_rate, rs_wall, ratio))
            print(f"  --> plies_cap={plies_cap} pair={pair + 1} RUST/PYTHON = {ratio:.3f}x")
            verdicts.append((plies_cap, ratio))

        py_rates = [r[1] for r in rows if r[0] == "python"]
        rs_ratios = [r[3] for r in rows if r[0] == "rust"]
        mean_ratio = sum(rs_ratios) / len(rs_ratios)
        print(
            f"== plies_cap={plies_cap}: mean ratio={mean_ratio:.3f}x "
            f"spread=[{min(rs_ratios):.3f}..{max(rs_ratios):.3f}] "
            f"python rates={[f'{r:.1f}' for r in py_rates]}"
        )

    print(
        "CAVEAT: contended machine (live training run + desktop load); "
        "absolute rates are inflated, paired ratios are the evidence."
    )
    gates = [r for r in verdicts if r[0] == args.plies]
    gate_mean = sum(r for _, r in gates) / len(gates)
    verdict = "PASS" if gate_mean >= 3.0 else "FAIL"
    print(f"GATE VERDICT (plies={args.plies} primary): {gate_mean:.3f}x vs 3.0x -> {verdict}")
    return 0 if gate_mean >= 3.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
