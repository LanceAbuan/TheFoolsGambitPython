"""WP-C bench — measure-first numbers (plan §5). Prints, does not train.

Reports for the tiny config on the current device:
  - NN batch latency (ms) for inference and one fwd+bwd step
  - MCTS sims/sec (search-bound, tiny net)
  - games/hour (extrapolated from a capped smoke-style game)
  - GPU availability + name (one-shot CUDA smoke, no long runs)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from training.model import MODEL_TINY, GambitNet
from training.selfplay import (
    BatchNNEvaluator,
    MCTSSearch,
    SearchConfig,
    SelfPlayConfig,
    self_play_game,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="training bench (tiny config, current device)")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--plies", type=int, default=40, help="ply cap for the games/hour probe")
    parser.add_argument("--iters", type=int, default=30)
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
