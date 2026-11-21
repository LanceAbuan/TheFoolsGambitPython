"""WP-C bench — measure-first numbers (plan §5). Prints, does not train.

Reports for the tiny config on the current device:
  - NN batch latency (ms) for inference and one fwd+bwd step
  - MCTS sims/sec (search-bound, tiny net)
  - games/hour (extrapolated from a capped smoke-style game)
  - FS-4 parallel sweep: sims/sec + games/hour for parallel_games in {1,8,16,32}
    (p=1 is the slice-1 sequential path; p>1 shares the queued evaluator)
  - FS-6b streaming guard: games/hour with per-move journal streaming on vs off
  - GPU availability + name (one-shot CUDA smoke, no long runs)
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np
import torch

from training.evaluator import QueuedNNEvaluator
from training.model import MODEL_TINY, GambitNet
from training.selfplay import (
    BatchNNEvaluator,
    MCTSSearch,
    SearchConfig,
    SelfPlayConfig,
    parallel_self_play,
    self_play_game,
)

# Slice-1 recorded sequential baseline (3080, tiny config) for old-vs-new (§5).
SLICE1_BASELINE_SIMS_PER_SEC = 390.0
SLICE1_BASELINE_GAMES_PER_HOUR = 10_600.0


def bench_nn_batch_latency(model: GambitNet, batch: int, iters: int, warmup: int = 5) -> float:
    """Average forward latency (ms) for one inference batch."""
    device = model.device
    x = torch.randn(batch, model.config.in_channels, 8, 8, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def bench_train_step_latency(model: GambitNet, batch: int, iters: int, warmup: int = 3) -> float:
    """Average fwd+bwd+step latency (ms) — the optimize-step number."""
    device = model.device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch, model.config.in_channels, 8, 8, device=device)
    target_p = torch.softmax(torch.randn(batch, model.config.policy_moves, device=device), dim=-1)
    target_v = torch.zeros(batch, device=device)
    for _ in range(warmup):
        optimizer.zero_grad()
        policy, value = model(x)
        loss = torch.nn.functional.cross_entropy(policy, target_p) + torch.nn.functional.mse_loss(value, target_v)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad()
        policy, value = model(x)
        loss = torch.nn.functional.cross_entropy(policy, target_p) + torch.nn.functional.mse_loss(value, target_v)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def bench_sims_per_second(model: GambitNet, sims: int) -> float:
    """MCTS throughput on the start position (search + NN eval, tiny net)."""
    evaluator = BatchNNEvaluator(model)
    search = MCTSSearch(evaluator, SearchConfig(num_simulations=sims))
    import chess

    board = chess.Board()
    start = time.perf_counter()
    search.search(board)
    elapsed = time.perf_counter() - start
    return sims / elapsed if elapsed > 0 else float("inf")


def bench_games_per_hour(model: GambitNet, max_plies: int, sims: int) -> tuple[float, float]:
    """Play one capped smoke-style game; return (estimated games/hour, seconds)."""
    evaluator = BatchNNEvaluator(model)
    config = SelfPlayConfig(
        search=SearchConfig(num_simulations=sims),
        max_plies=max_plies,
        temperature_plies=max_plies,
    )
    start = time.perf_counter()
    result = self_play_game(evaluator, config)
    elapsed = time.perf_counter() - start
    per_game = elapsed / max(result.ply_count, 1) * min(max_plies, 120)
    return 3600.0 / per_game if per_game > 0 else float("inf"), elapsed


def bench_parallel_sims_per_second(model: GambitNet, parallel_games: int, sims: int) -> float:
    """Aggregate MCTS throughput while `parallel_games` root searches run at once.

    parallel_games == 1 keeps the slice-1 synchronous evaluator (the old path);
    > 1 shares the queued batch service (FS-4) so the forwards merge.
    """
    evaluator: object
    if parallel_games <= 1:
        evaluator = BatchNNEvaluator(model)
    else:
        evaluator = QueuedNNEvaluator(model)
    searches = [
        MCTSSearch(evaluator, SearchConfig(num_simulations=sims, root_noise=False))  # type: ignore[arg-type]
        for _ in range(parallel_games)
    ]
    board = chess.Board()
    try:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=parallel_games) as pool:
            list(pool.map(lambda search: search.search(board), searches))
        elapsed = time.perf_counter() - start
    finally:
        if isinstance(evaluator, QueuedNNEvaluator):
            evaluator.close()
    total_sims = parallel_games * sims
    return total_sims / elapsed if elapsed > 0 else float("inf")


def bench_parallel_games_per_hour(
    model: GambitNet, parallel_games: int, max_plies: int, sims: int
) -> tuple[float, float]:
    """Play `parallel_games` capped games at once; (aggregate games/hour, seconds)."""
    if parallel_games <= 1:
        evaluator: object = BatchNNEvaluator(model)
    else:
        evaluator = QueuedNNEvaluator(model)
    config = SelfPlayConfig(
        search=SearchConfig(num_simulations=sims),
        max_plies=max_plies,
        temperature_plies=max_plies,
    )
    try:
        start = time.perf_counter()
        results = parallel_self_play(evaluator, parallel_games, config, parallel_games=parallel_games)  # type: ignore[arg-type]
        elapsed = time.perf_counter() - start
    finally:
        if isinstance(evaluator, QueuedNNEvaluator):
            evaluator.close()
    games_per_hour = len(results) / elapsed * 3600.0 if elapsed > 0 else float("inf")
    return games_per_hour, elapsed


def bench_streaming_overhead(
    model: GambitNet, parallel_games: int, max_plies: int, sims: int
) -> dict[str, float]:
    """FS-6b perf guard: pN games/hour with live streaming ON vs OFF.

    Both runs go through the same `parallel_self_play` path with the SAME
    seeded rng and root noise off — so both batches play byte-identical games
    and the games/hour delta isolates the per-move journal cost (game-length
    variance would otherwise swamp it). ON additionally journals one event per
    ply + one game_end per game into a temp journal.
    """
    import tempfile

    from training.journal import TrainingJournal
    from training.selfplay import StreamContext

    config = SelfPlayConfig(
        search=SearchConfig(num_simulations=sims, root_noise=False),
        max_plies=max_plies,
        temperature_plies=max_plies,
    )
    stats: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmp:
        journal_path = Path(tmp) / "stream-bench.jsonl"
        for mode, enabled in (("off", False), ("on", True)):
            journal = TrainingJournal(journal_path)
            stream = StreamContext(journal=journal, run_id="bench", regime="bench", enabled=enabled)
            evaluator: object = (
                QueuedNNEvaluator(model) if parallel_games > 1 else BatchNNEvaluator(model)
            )
            play_rng = np.random.default_rng(0)  # identical games in both modes
            try:
                start = time.perf_counter()
                games = parallel_self_play(  # type: ignore[arg-type]
                    evaluator,
                    parallel_games,
                    config,
                    parallel_games=parallel_games,
                    rng=play_rng,
                    stream=stream,
                )
                elapsed = time.perf_counter() - start
            finally:
                if isinstance(evaluator, QueuedNNEvaluator):
                    evaluator.close()
                journal.close()  # release the handle so the tmp dir can be removed
            stats[f"games_per_hour_{mode}"] = (
                len(games) / elapsed * 3600.0 if elapsed > 0 else float("inf")
            )
            stats[f"plies_{mode}"] = float(sum(game.ply_count for game in games))
            events = len(journal.read())
            stats[f"events_{mode}"] = float(events)
            stats[f"events_per_sec_{mode}"] = events / elapsed if elapsed > 0 else 0.0
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="training bench (tiny config, current device)")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--plies", type=int, default=40, help="ply cap for the games/hour probe")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--parallel", type=str, default="1,8,16,32",
                        help="comma list of parallel_games levels for the FS-4 sweep")
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else "CPU only"
    print(f"device: {'CUDA' if cuda else 'CPU'} — {device_name}")
    print(f"torch {torch.__version__}, cudnn {torch.backends.cudnn.version()}")

    device = torch.device("cuda" if cuda else "cpu")
    model = GambitNet(MODEL_TINY).to(device)
    model.eval()
    print(f"tiny config: {model.config.to_dict()} -> {model.count_parameters():,} params")

    inference_ms = bench_nn_batch_latency(model, args.batch, args.iters)
    train_ms = bench_train_step_latency(model, min(args.batch, 64), max(args.iters // 3, 5))
    sims_per_sec = bench_sims_per_second(model, args.sims)
    games_per_hour, probe_seconds = bench_games_per_hour(model, args.plies, sims=2)

    print("--- bench results (tiny config) ---")
    print(f"NN batch latency:  {inference_ms:8.2f} ms / batch of {args.batch} ({device.type})")
    print(f"train step:        {train_ms:8.2f} ms / fwd+bwd batch 64 ({device.type})")
    print(f"MCTS throughput:   {sims_per_sec:8.1f} sims/sec ({args.sims} sims, startpos)")
    print(f"games/hour (est):  {games_per_hour:8.1f}  (one {args.plies}-ply game in {probe_seconds:.2f}s, extrapolated)")

    # --- FS-4: parallel self-play sweep (queued evaluator) vs the old path ---
    levels = [int(level) for level in args.parallel.split(",") if level.strip()]
    rows: list[tuple[int, float, float]] = []
    for level in levels:
        sweep_sims = bench_parallel_sims_per_second(model, level, min(args.sims, 64))
        sweep_gph, sweep_elapsed = bench_parallel_games_per_hour(model, level, args.plies, sims=2)
        rows.append((level, sweep_sims, sweep_gph))
        print(
            f"parallel={level:>2}: {sweep_sims:9.1f} sims/sec | {sweep_gph:9.1f} games/hour "
            f"| ({level} games in {sweep_elapsed:.2f}s)"
        )
    baseline_gph = rows[0][2] if rows else 1.0
    print("--- parallel sweep vs slice-1 sequential baseline ---")
    print(f"slice-1 recorded (GPU, sequential): ~{SLICE1_BASELINE_SIMS_PER_SEC:.0f} sims/sec, "
          f"~{SLICE1_BASELINE_GAMES_PER_HOUR:.0f} games/hour est")
    for level, sweep_sims, sweep_gph in rows:
        speedup_sims = sweep_sims / rows[0][1] if rows and rows[0][1] > 0 else 0.0
        speedup_gph = sweep_gph / baseline_gph if baseline_gph > 0 else 0.0
        print(
            f"parallel={level:>2}: {speedup_sims:5.2f}x sims/sec vs p=1 | "
            f"{speedup_gph:5.2f}x games/hour vs p=1"
        )

    # --- FS-6b perf guard: live move streaming ON vs OFF (p8 + p16) ---
    print("--- FS-6b streaming on/off (games/hour through the per-move journal) ---")
    for level in (8, min(16, max(levels)) if levels else 16):
        stats = bench_streaming_overhead(model, level, min(args.plies, 24), sims=2)
        gph_off, gph_on = stats["games_per_hour_off"], stats["games_per_hour_on"]
        overhead = (gph_off - gph_on) / gph_off * 100.0 if gph_off > 0 else 0.0
        print(
            f"p={level:>2}: off {gph_off:8.1f} games/h | on {gph_on:8.1f} games/h | "
            f"overhead {overhead:5.1f}% | identical games? plies off={stats['plies_off']:.0f} "
            f"on={stats['plies_on']:.0f} | {stats['events_on']:.0f} events "
            f"({stats['events_per_sec_on']:.1f} events/sec)"
        )
    if cuda:
        # One-shot GPU smoke: device verdict + a single synced pass, no long runs.
        torch.cuda.synchronize()
        print(f"GPU smoke: cuda available, {torch.cuda.get_device_name(0)}, "
              f"mem allocated {torch.cuda.memory_allocated(0) / 1024 ** 2:.1f} MiB")
    else:
        print("GPU smoke: cuda NOT available — running CPU-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
