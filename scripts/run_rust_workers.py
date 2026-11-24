"""WP-H H4: launch + supervise the N-worker Rust self-play fleet.

    python scripts/run_rust_workers.py --workers 4 \
        --models-dir <run-root>/onnx --spool-dir <run-root>/spool

Supervision (risk register: "worker crash isolation - supervisor restarts a
dead worker; trainer keeps buffered examples"): each worker runs
``scripts/rust_worker.py`` as an independent process; a worker that exits is
restarted with a fixed backoff up to --max-restarts per slot. Workers pick a
disjoint seed range (``seed_base + i * 1e9 + game_index``) so no two workers
ever replay the same stream.

SOAK (WP-H H6c - orchestrator-run, NOT part of CI):
    python scripts/run_rust_workers.py --workers 8 --duration-seconds 86400 \
        --models-dir runs/<regime>/<run-id>/onnx \
        --spool-dir runs/<regime>/<run-id>/spool \
        --sims 600 --max-plies 200
Pass criteria: zero restart-exhausted slots, trainer buffer intake rate
stable (watch the trainer's [RUST-SPOOL] / games_per_hour journal lines).
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER_SCRIPT = ROOT / "scripts" / "rust_worker.py"
POLL_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WP-H H4: run the rust self-play worker fleet.")
    parser.add_argument("--workers", type=int, default=4, help="number of worker processes")
    parser.add_argument("--models-dir", required=True, help="trainer-published model dir (latest.json)")
    parser.add_argument("--spool-dir", required=True, help="trainer spool dir")
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-plies", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=0, help="worker i uses seed_base + i*1e9")
    parser.add_argument("--opening-seed-base", type=int, default=0)
    parser.add_argument("--no-openings", action="store_true")
    parser.add_argument("--forced-playouts", action="store_true",
                        help="WP1: KataGo forced playouts + policy target pruning")
    parser.add_argument("--fp-k", type=float, default=2.0, help="WP1: forced-playout scale k")
    parser.add_argument("--duration-seconds", type=float, default=0.0, help="0 = until Ctrl+C")
    parser.add_argument("--max-restarts", type=int, default=10, help="per-slot restart budget")
    parser.add_argument("--restart-backoff", type=float, default=5.0)
    parser.add_argument("--python", default=sys.executable, help="interpreter for the workers")
    return parser.parse_args()


def worker_argv(args: argparse.Namespace, slot: int) -> list[str]:
    return [
        args.python,
        str(WORKER_SCRIPT),
        "--models-dir", args.models_dir,
        "--spool-dir", args.spool_dir,
        "--worker-id", f"w{slot}",
        "--sims", str(args.sims),
        "--max-plies", str(args.max_plies),
        "--temperature", str(args.temperature),
        "--temperature-plies", str(args.temperature_plies),
        "--seed-base", str(args.seed_base + slot * 1_000_000_000),
        "--opening-seed-base", str(args.opening_seed_base + slot * 1_000_000_000),
        "--duration-seconds", str(args.duration_seconds),
    ] + (["--no-openings"] if args.no_openings else []) + (
        ["--forced-playouts", "--fp-k", str(args.fp_k)] if args.forced_playouts else []
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        print("--workers must be >= 1", file=sys.stderr)
        return 1
    if not pathlib.Path(args.models_dir).is_dir():
        print(f"models dir {args.models_dir} does not exist (start the trainer first or export once)", file=sys.stderr)
        return 1

    deadline = time.monotonic() + args.duration_seconds if args.duration_seconds > 0 else None
    slots: list[dict] = []
    for slot in range(args.workers):
        slots.append({"slot": slot, "restarts": 0, "proc": None, "next_start": 0.0})
    print(f"[fleet] {args.workers} workers; models={args.models_dir} spool={args.spool_dir}", flush=True)

    try:
        while True:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            alive = 0
            for entry in slots:
                proc = entry["proc"]
                if proc is not None and proc.poll() is None:
                    alive += 1
                    continue
                if proc is not None:
                    code = proc.returncode
                    print(f"[fleet] w{entry['slot']} exited code={code}", flush=True)
                    entry["proc"] = None
                    if code == 0:
                        continue  # clean shutdown (deadline) - do not restart
                    entry["restarts"] += 1
                    if entry["restarts"] > args.max_restarts:
                        print(
                            f"[fleet] w{entry['slot']} exhausted {args.max_restarts} restarts; slot down",
                            flush=True,
                        )
                        continue
                    entry["next_start"] = now + args.restart_backoff
                if entry["restarts"] <= args.max_restarts and now >= entry["next_start"]:
                    argv = worker_argv(args, entry["slot"])
                    entry["proc"] = subprocess.Popen(argv)
                    print(
                        f"[fleet] w{entry['slot']} started (restarts={entry['restarts']})",
                        flush=True,
                    )
            if all(e["proc"] is None or e["proc"].poll() is not None for e in slots) and all(
                e["restarts"] > args.max_restarts for e in slots
            ):
                print("[fleet] all slots exhausted; supervisor exiting", flush=True)
                return 1
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("[fleet] interrupted", flush=True)
    finally:
        for entry in slots:
            proc = entry["proc"]
            if proc is not None and proc.poll() is None:
                proc.terminate()
        for entry in slots:
            proc = entry["proc"]
            if proc is not None:
                with_suppress = getattr(proc, "wait", None)
                if with_suppress:
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
    exhausted = sum(1 for e in slots if e["restarts"] > args.max_restarts)
    print(f"[fleet] done: {exhausted}/{args.workers} slots exhausted their restart budget", flush=True)
    return 0 if exhausted == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())