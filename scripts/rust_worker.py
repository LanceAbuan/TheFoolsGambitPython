"""WP-H H4: ONE Rust self-play worker process (the fleet unit of run_rust_workers.py).

Loop (all per-move work inside ``gambit_core.selfplay_game``, zero Python in
the game path):

    1. resolve the current model via the trainer-published pointer
       (<models-dir>/latest.json) - checked BETWEEN games only, never mid-game
    2. play ONE game against that model file (seeded, opening sampled)
    3. atomically append the finished game to the spool
       (<spool-dir>/inbox/game-<worker>-<version>-<uuid>.jsonl)
    4. repeat until --duration-seconds elapses (0 = until Ctrl+C)

Every game record is tagged with the weight_version it played against, so the
trainer can always tell which weights produced which examples.

Exit codes: 0 clean (deadline / Ctrl+C / idle shutdown), 1 fatal config, 3
consecutive game failures (the supervisor restarts with backoff).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_CONSECUTIVE_FAILURES = 3
LOG = logging.getLogger("rust_worker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WP-H H4: one Rust self-play worker.")
    parser.add_argument("--models-dir", required=True, help="dir holding latest.json + model-*.onnx")
    parser.add_argument("--spool-dir", required=True, help="trainer spool dir (inbox/ is created)")
    parser.add_argument("--worker-id", default=None, help="unique worker tag (default w<pid>)")
    parser.add_argument("--sims", type=int, default=64, help="MCTS simulations per ply")
    parser.add_argument("--max-plies", type=int, default=200, help="ply cap per game")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-plies", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=0, help="game seed = seed-base + game index")
    parser.add_argument(
        "--opening-seed-base",
        type=int,
        default=0,
        help="opening sample seed = opening-seed-base + game index",
    )
    parser.add_argument("--no-openings", action="store_true", help="always start from the standard position")
    parser.add_argument("--forced-playouts", action="store_true",
                        help="WP1: KataGo forced playouts + policy target pruning")
    parser.add_argument("--fp-k", type=float, default=2.0, help="WP1: forced-playout scale k")
    parser.add_argument("--pcr-p-full", type=float, default=0.0,
                        help="WP2: playout cap randomization p_full (0=off, KataGo 0.25)")
    parser.add_argument("--pcr-fast-sims", type=int, default=100,
                        help="WP2: simulations for value-only PCR searches")
    parser.add_argument("--duration-seconds", type=float, default=0.0, help="0 = run until Ctrl+C")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="idle wait between pointer checks")
    return parser.parse_args()


def find_dylib() -> str | None:
    """ORT_DYLIB_PATH wins; else the ort-sys build cache under target/."""
    env = os.environ.get("ORT_DYLIB_PATH")
    if env and pathlib.Path(env).is_file():
        return env
    for profile in ("release", "debug"):
        hit = ROOT / "target" / profile / "onnxruntime.dll"
        if hit.is_file():
            return str(hit)
    for hit in sorted(ROOT.glob("target/**/onnxruntime.dll")):
        return str(hit)
    return None


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format=f"[{args.worker_id or 'w' + str(os.getpid())}][%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    worker_id = args.worker_id or f"w{os.getpid()}"
    models_dir = pathlib.Path(args.models_dir)
    spool_dir = pathlib.Path(args.spool_dir)
    if not models_dir.is_dir():
        LOG.error("models dir %s does not exist", models_dir)
        return 1

    dylib = find_dylib()
    if dylib is None:
        LOG.error("onnxruntime dylib not found (set ORT_DYLIB_PATH); cannot run ort inference")
        return 1
    os.environ["ORT_DYLIB_PATH"] = dylib
    LOG.info("dylib: %s", dylib)

    import gambit_core

    if not hasattr(gambit_core, "selfplay_game"):
        LOG.error("gambit_core wheel lacks selfplay_game (rebuild with --features onnx)")
        return 1

    from training.rust_workers import (
        ModelWatcher,
        game_record_from_rust,
        write_game_record,
    )

    watcher = ModelWatcher(models_dir)
    deadline = time.monotonic() + args.duration_seconds if args.duration_seconds > 0 else None
    games_done = 0
    failures = 0
    current_version = -1
    LOG.info(
        "worker up: sims=%d max_plies=%d temp=%.2f/%d spool=%s models=%s",
        args.sims,
        args.max_plies,
        args.temperature,
        args.temperature_plies,
        spool_dir,
        models_dir,
    )

    while deadline is None or time.monotonic() < deadline:
        # --- hot-reload check: BETWEEN games only, never mid-game ---------
        try:
            pointer = watcher.current()
        except Exception as exc:  # noqa: BLE001 - a bad pointer must not kill the worker
            LOG.warning("pointer unusable (%s); keeping the previous model", exc)
            pointer = None if games_done == 0 else watcher._cached
        if pointer is None:
            if games_done == 0 and deadline is not None and time.monotonic() >= deadline:
                break
            LOG.info("waiting for the trainer to publish a model (v%s)...", current_version)
            time.sleep(args.poll_seconds)
            continue
        if pointer.weight_version != current_version:
            LOG.info("hot-reload -> weight_version %d (%s)", pointer.weight_version, pointer.path.name)
            current_version = pointer.weight_version

        seed = args.seed_base + games_done
        opening_seed = None if args.no_openings else args.opening_seed_base + games_done
        try:
            game = gambit_core.selfplay_game(
                str(pointer.path),
                seed=seed,
                sims=args.sims,
                max_plies=args.max_plies,
                temperature=args.temperature,
                temperature_plies=args.temperature_plies,
                opening_seed=opening_seed,
                weight_version=pointer.weight_version,
                forced_playouts=args.forced_playouts,
                fp_k=args.fp_k,
                pcr_p_full=args.pcr_p_full,
                pcr_fast_sims=args.pcr_fast_sims,
            )
        except BaseException as exc:
            # pyo3's PanicException (rust panic inside selfplay_game)
            # derives from BaseException, not Exception - catch it too,
            # but let Ctrl+C / SystemExit through untouched.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failures += 1
            LOG.exception("game failed (%d consecutive)", failures)
            if failures >= MAX_CONSECUTIVE_FAILURES:
                LOG.error("too many consecutive failures; exiting for supervisor restart")
                return 1
            time.sleep(args.poll_seconds)
            continue
        failures = 0

        record = game_record_from_rust(
            game,
            weight_version=pointer.weight_version,
            worker_id=worker_id,
            seed=seed,
            opening_seed=opening_seed,
            sims=args.sims,
            max_plies=args.max_plies,
            temperature=args.temperature,
            temperature_plies=args.temperature_plies,
        )
        path = write_game_record(spool_dir, record)
        games_done += 1
        LOG.info(
            "game %d done: %s %s in %d plies -> %s (%.1f KB)",
            games_done,
            record["result"],
            record["termination"],
            record["ply_count"],
            path.name,
            path.stat().st_size / 1024,
        )

    LOG.info("worker done: %d games (last weight_version %d)", games_done, current_version)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"event": "interrupted"}), flush=True)
        raise SystemExit(0) from None