"""Leaf-batching A/B at FIXED simulations (research A100 / cycle 33).

Wave search with virtual loss trades search fidelity for fewer ORT round-trips:
at the same sims, a batch>1 search descends into sibling subtrees that a
sequential search would have pruned. This harness measures both sides of that
trade on the SAME seeds and openings:

  * throughput  - games/s and sims/s
  * target sharpness - mean policy entropy and mean max-probability of the
    recorded visit distribution (flatter targets = weaker search)
  * value sanity - sign(root_value) agreement with the game result

Usage (fleet stopped, so the GPU is not shared):
    python scripts/bench_batch_search.py --batches 1 4 8 --games 24 --sims 128
"""

from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_GLOB = "runs/scratch/second-light-v2/onnx/model-*.onnx"


def _register_cuda_dlls() -> str:
    """Make the CUDA 13 / cuDNN 9 runtime visible to ORT's CUDA EP.

    Without this the first `selfplay_game` call logs "CUDA EP FAILED ... cublasLt64_13.dll
    is missing" and silently falls back to the CPU EP - which makes every timing in
    this benchmark meaningless. torch ships all the DLLs ORT links against.
    """
    added: list[str] = []
    try:
        import torch

        lib = pathlib.Path(torch.__file__).parent / "lib"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
            os.environ["PATH"] = f"{lib}{os.pathsep}" + os.environ.get("PATH", "")
            added.append(str(lib))
    except Exception:  # noqa: BLE001 - torch is optional for this script
        pass
    return ", ".join(added) if added else "none"


def latest_model() -> pathlib.Path:
    models = sorted(
        ROOT.glob(MODEL_GLOB), key=lambda p: int(p.stem.split("-")[1])
    )
    if not models:
        raise SystemExit(f"no exported model found under {MODEL_GLOB}")
    return models[-1]


def run_config(model_path: pathlib.Path, batch: int, games: int, sims: int) -> dict:
    import gambit_core

    entropies: list[float] = []
    max_probs: list[float] = []
    sign_hits = 0
    sign_total = 0
    plies_total = 0
    started = time.perf_counter()

    for index in range(games):
        game = gambit_core.selfplay_game(
            str(model_path),
            seed=90_000 + index,          # identical seed set across configs
            sims=sims,
            max_plies=200,
            temperature=1.0,
            temperature_plies=30,
            opening_seed=50_000 + index,  # identical openings across configs
            weight_version=0,
            forced_playouts=True,
            fp_k=2.0,
            pcr_p_full=0.0,               # isolate batching: no PCR
            pcr_fast_sims=100,
            soft_resign=False,
            search_batch=batch,
        )
        plies_total += int(game["ply_count"])
        result = game["result"]
        final = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0
        for ex in game["examples"]:
            policy = np.asarray(ex["policy"], dtype=np.float64)
            total = float(policy.sum())
            if total <= 0.0:
                continue
            prob = policy / total
            nz = prob[prob > 0.0]
            entropies.append(float(-(nz * np.log(nz)).sum()))
            max_probs.append(float(prob.max()))
            root_value = float(ex.get("root_value", 0.0))
            if abs(root_value) > 0.2 and abs(final) > 0.5:
                sign_total += 1
                sign_hits += int(np.sign(root_value) == np.sign(final))

    elapsed = time.perf_counter() - started
    return {
        "batch": batch,
        "games": games,
        "elapsed_s": elapsed,
        "games_per_s": games / elapsed,
        "sims_per_s": (plies_total * sims) / elapsed,
        "plies": plies_total,
        "entropy": statistics.fmean(entropies) if entropies else float("nan"),
        "max_prob": statistics.fmean(max_probs) if max_probs else float("nan"),
        "sign_agreement": (sign_hits / sign_total) if sign_total else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    model = pathlib.Path(args.model) if args.model else latest_model()
    print(f"[batch-ab] cuda dlls: {_register_cuda_dlls()}", flush=True)
    print(f"[batch-ab] model {model.name} | {args.games} games/config, {args.sims} sims", flush=True)
    print(f"{'batch':>5} {'games/s':>8} {'sims/s':>9} {'entropy':>8} {'max_p':>7} {'sign':>6}", flush=True)
    for batch in args.batches:
        row = run_config(model, batch, args.games, args.sims)
        print(
            f"{row['batch']:>5} {row['games_per_s']:>8.3f} {row['sims_per_s']:>9.0f} "
            f"{row['entropy']:>8.3f} {row['max_prob']:>7.3f} {row['sign_agreement']:>6.2f}",
            flush=True,
        )
    print(
        "\nRead: batch 1 is the reference. Higher entropy / lower max_p at the "
        "SAME sims = flatter (weaker) targets; games/s is the throughput it buys.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
