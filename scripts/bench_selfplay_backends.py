"""WP-H H6(b): games/hour - python self-play backend vs rust worker backend.

CPU-vs-CPU on the SAME ONNX model (models/gambitnet-v2.onnx, CPU EP both
sides), sequential single-game semantics, standard start position:

    python   training.selfplay.self_play_game (python-chess rules + python
             MCTS + onnxruntime-python eval)
    rust     gambit_core.selfplay_game (rust rules + rust MCTS + ort-rust eval)
    rustxN   N worker PROCESSES playing concurrently (the H4 topology)

Numbers are indicative, not the flip gate: short games, few simulations, and
whatever else this machine is running (e.g. live training) share the cores.
The flip gate stays the plan's H6(b): >= 2x the WP-F games/hr on the real
night-v2 config with GPU util >= 85% during training steps.

Usage:
    .venv/Scripts/python.exe scripts/bench_selfplay_backends.py --quick
    .venv/Scripts/python.exe scripts/bench_selfplay_backends.py \
        --games 4 --sims 16 --max-plies 40 --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "gambitnet-v2.onnx"


def _find_dylib() -> str | None:
    import os

    env = os.environ.get("ORT_DYLIB_PATH")
    if env and pathlib.Path(env).is_file():
        return env
    for profile in ("release", "debug"):
        hit = ROOT / "target" / profile / "onnxruntime.dll"
        if hit.is_file():
            return str(hit)
    return None


# --- python backend ----------------------------------------------------------


class _OrtCpuEvaluator:
    """BatchNNEvaluator-shaped evaluator backed by onnxruntime CPU (v2 net)."""

    def __init__(self, model_path: pathlib.Path) -> None:
        import onnxruntime as ort

        self._sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        from training.encoder import encode_batch_v2

        self._encode = encode_batch_v2

    def evaluate(self, boards):
        import numpy as np

        from training.model import value_head_scalar  # re-exported helper
        import torch

        batch = self._encode(boards)
        logits, wdl = self._sess.run(None, {"input": batch})
        wdl_t = torch.from_numpy(wdl)
        scalar = value_head_scalar(wdl_t).numpy().astype(np.float32)
        return logits.astype(np.float32), scalar


def bench_python(games: int, sims: int, max_plies: int) -> tuple[float, int]:
    from training.selfplay import SelfPlayConfig, SearchConfig, self_play_game

    evaluator = _OrtCpuEvaluator(MODEL_PATH)
    config = SelfPlayConfig(
        search=SearchConfig(num_simulations=sims, search_backend="python"),
        max_plies=max_plies,
        temperature_plies=max_plies,
        encoder_version=2,
    )
    import numpy as np

    rng = np.random.default_rng(2026)
    plies = 0
    started = time.perf_counter()
    for i in range(games):
        result = self_play_game(evaluator, config=config, rng=rng)
        plies += result.ply_count
    return time.perf_counter() - started, plies


# --- rust backend ------------------------------------------------------------

_RUST_STATE: dict = {}


def _rust_worker_init(model_path: str, sims: int, max_plies: int) -> None:
    import os

    dylib = _find_dylib()
    if dylib:
        os.environ["ORT_DYLIB_PATH"] = dylib
    import gambit_core

    assert hasattr(gambit_core, "selfplay_game"), "rebuild the wheel with --features onnx"
    _RUST_STATE.update({"core": gambit_core, "model": model_path, "sims": sims, "max_plies": max_plies})


def _rust_play_one(game_index: int) -> int:
    core = _RUST_STATE["core"]
    game = core.selfplay_game(
        _RUST_STATE["model"],
        seed=9000 + game_index,
        sims=_RUST_STATE["sims"],
        max_plies=_RUST_STATE["max_plies"],
        temperature=1.0,
        temperature_plies=_RUST_STATE["max_plies"],
        opening_seed=None,  # standard start: apples-to-apples with the python loop
        weight_version=0,
    )
    return int(game["ply_count"])


def bench_rust(games: int, sims: int, max_plies: int) -> tuple[float, int]:
    _rust_worker_init(str(MODEL_PATH), sims, max_plies)
    plies = 0
    started = time.perf_counter()
    for i in range(games):
        plies += _rust_play_one(i)
    return time.perf_counter() - started, plies


def bench_rust_parallel(games: int, sims: int, max_plies: int, workers: int) -> tuple[float, int]:
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_rust_worker_init, initargs=(str(MODEL_PATH), sims, max_plies)) as pool:
        started = time.perf_counter()
        plies = sum(pool.map(_rust_play_one, range(games)))
        elapsed = time.perf_counter() - started
    return elapsed, plies


# --- driver -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="WP-H H6(b): self-play backend games/hr.")
    parser.add_argument("--games", type=int, default=4, help="games per backend")
    parser.add_argument("--sims", type=int, default=16)
    parser.add_argument("--max-plies", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--backends",
        default="python,rust,rustxN",
        help="comma list from: python, rust, rustxN",
    )
    parser.add_argument("--quick", action="store_true", help="games=2 sims=8 max-plies=24")
    args = parser.parse_args()
    if args.quick:
        args.games, args.sims, args.max_plies = 2, 8, 24

    if not MODEL_PATH.is_file():
        print("models/gambitnet-v2.onnx missing (run scripts/export_onnx.py)", file=sys.stderr)
        return 1
    import gambit_core

    wanted = [b.strip() for b in args.backends.split(",") if b.strip()]
    if ("rust" in wanted or "rustxN" in wanted) and not hasattr(gambit_core, "selfplay_game"):
        print("wheel lacks selfplay_game; build: maturin develop --release --features onnx", file=sys.stderr)
        return 1

    print(
        f"[bench] games={args.games} sims={args.sims} max_plies={args.max_plies} "
        f"workers={args.workers} model={MODEL_PATH.name} (CPU EP both sides)",
        flush=True,
    )
    rows: list[tuple[str, float, int]] = []
    if "python" in wanted:
        elapsed, plies = bench_python(args.games, args.sims, args.max_plies)
        rows.append(("python", elapsed, plies))
    if "rust" in wanted:
        elapsed, plies = bench_rust(args.games, args.sims, args.max_plies)
        rows.append(("rust (1 proc)", elapsed, plies))
    if "rustxN" in wanted:
        elapsed, plies = bench_rust_parallel(args.games, args.sims, args.max_plies, args.workers)
        rows.append((f"rust x{args.workers} procs", elapsed, plies))

    print()
    print(f"{'backend':>18} | {'games':>5} | {'elapsed_s':>9} | {'plies':>6} | {'plies/hr':>10} | {'games/hr':>9}")
    print("-" * 74)
    has_python_row = any(name == "python" for name, _, _ in rows)
    base = None
    for name, elapsed, plies in rows:
        gph = args.games / elapsed * 3600.0 if elapsed > 0 else float("nan")
        plies_hr = plies / elapsed * 3600.0 if elapsed > 0 else float("nan")
        if base is None:
            base = plies_hr
        delta = (
            f" ({plies_hr / base:.2f}x python plies/hr)"
            if has_python_row and name != "python"
            else ""
        )
        print(f"{name:>18} | {args.games:>5} | {elapsed:>9.1f} | {plies:>6} | {plies_hr:>10.0f} | {gph:>9.0f}{delta}")
    print()
    print("[bench] indicative CPU numbers only - see the H6(b) flip gate in the docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())