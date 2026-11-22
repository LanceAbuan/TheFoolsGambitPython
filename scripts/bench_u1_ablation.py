"""U1/U2 ablation bench (RESEARCH_STANDARD: measure first, one gate at a time).

Pre-registered predictions live in tmp_u1u2/PREDICTIONS.md (P1-P7). This
script measures, on CPU (CUDA_VISIBLE_DEVICES=-1 is set BEFORE torch import):

  1. Self-play throughput, 8 games each, tiny net, sequential batch evaluator:
       PUCT@400 (baseline) | gumbel@64 | gumbel@100 | PUCT@400 + U2 cap.
       U2 is PUCT-only by config validation (the gumbel x cap combination is a
       config error - see SelfPlayConfig docstring), so there is no gumbel-cap
       run; the U2 overhead/gain (P6) is measured on the PUCT baseline.
  2. Head-to-head arena, 12 games per pairing (6 per color), root_noise off:
       gumbel@64  vs PUCT@400   (U1 acceptance: >=4x games/h AND win-rate >=0.35)
       gumbel@100 vs gumbel@64
  3. U2 overhead/gain from the capped vs uncapped throughput runs (P6), plus
     the recorded-example share per game.

Every game prints one progress line; a JSON summary is written next to this
script's --out path for the report. Never touches runs/ or data/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU-pinned BEFORE torch import

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

torch.set_num_threads(1)

from training.fastpath import disable_fastpath
from training.model import MODEL_TINY, GambitNet
from training.selfplay import (
    BatchNNEvaluator,
    Evaluator,
    SearchConfig,
    SelfPlayConfig,
    play_match_game,
    self_play_game,
)

SEED_BASE = 20261121
MAX_PLIES = 60
GAMES_PER_CONFIG = 8
H2H_GAMES = 12  # 6 per color


class CountingEvaluator(Evaluator):
    """Thin proxy that counts NN evaluate() calls (sims/sec evidence)."""

    def __init__(self, inner: Evaluator) -> None:
        self.inner = inner
        self.calls = 0

    def evaluate(self, boards):
        self.calls += 1
        return self.inner.evaluate(boards)


def selfplay_config(search: SearchConfig, capped: bool) -> SelfPlayConfig:
    return SelfPlayConfig(
        search=search,
        max_plies=MAX_PLIES,
        temperature_plies=MAX_PLIES,  # sample throughout (self-play regime)
        playout_randomization=capped,
    )


def puct(sims: int) -> SearchConfig:
    return SearchConfig(num_simulations=sims, root_noise=True)


def gumbel(sims: int, candidates: int = 16) -> SearchConfig:
    return SearchConfig(
        num_simulations=sims, search_kind="gumbel", gumbel_candidates=candidates, root_noise=True
    )


def arena_config(search: SearchConfig) -> SearchConfig:
    return replace_root_noise(search, False)


def replace_root_noise(config: SearchConfig, root_noise: bool) -> SearchConfig:
    import dataclasses

    return dataclasses.replace(config, root_noise=root_noise)


def bench_throughput(model: GambitNet, configs: dict[str, SearchConfig], capped: dict[str, bool]) -> dict:
    results: dict[str, dict[str, object]] = {}
    for name, search in configs.items():
        evaluator = CountingEvaluator(BatchNNEvaluator(model))
        config = selfplay_config(search, capped[name])
        total_seconds = 0.0
        total_plies = 0
        total_examples = 0
        total_calls = 0
        for game_index in range(GAMES_PER_CONFIG):
            np.random.seed(SEED_BASE + game_index)  # dirichlet draws use the global stream
            started = time.perf_counter()
            result = self_play_game(evaluator, config, rng=np.random.default_rng(SEED_BASE + game_index))
            total_seconds += time.perf_counter() - started
            total_plies += result.ply_count
            total_examples += len(result.examples)
            total_calls += evaluator.calls
            per_game = time.perf_counter() - started
            print(
                f"[throughput] {name}: game {game_index + 1}/{GAMES_PER_CONFIG} "
                f"plies={result.ply_count} examples={len(result.examples)} "
                f"calls={evaluator.calls} {per_game:.1f}s",
                flush=True,
            )
            evaluator.calls = 0
        results[name] = {
            "games": GAMES_PER_CONFIG,
            "seconds": round(total_seconds, 2),
            "games_per_hour": round(GAMES_PER_CONFIG / total_seconds * 3600.0, 1),
            "avg_plies": round(total_plies / GAMES_PER_CONFIG, 1),
            "avg_examples_per_game": round(total_examples / GAMES_PER_CONFIG, 1),
            "recorded_share": round(total_examples / max(total_plies, 1), 3),
            "avg_nn_calls_per_game": round(total_calls / GAMES_PER_CONFIG, 1),
            "nn_calls_per_second": round(total_calls / total_seconds, 1),
        }
        print(f"[throughput] {name}: {results[name]}", flush=True)
    return results


def bench_head_to_head(
    model: GambitNet,
    candidate_name: str,
    candidate_search: SearchConfig,
    champion_name: str,
    champion_search: SearchConfig,
) -> dict:
    candidate_eval = BatchNNEvaluator(model)
    champion_eval = BatchNNEvaluator(model)
    cand_wins = draws = cand_losses = 0
    started = time.perf_counter()
    for game_index in range(H2H_GAMES):
        candidate_is_white = game_index % 2 == 0
        white_search = arena_config(candidate_search if candidate_is_white else champion_search)
        black_search = arena_config(champion_search if candidate_is_white else candidate_search)
        result, termination = play_match_game(
            candidate_eval,
            champion_eval,
            num_simulations=400,
            max_plies=MAX_PLIES,
            white_config=white_search,
            black_config=black_search,
            rng=np.random.default_rng(SEED_BASE + 1000 + game_index),
        )
        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == candidate_is_white:
            cand_wins += 1
        else:
            cand_losses += 1
        print(
            f"[h2h] {candidate_name} vs {champion_name}: game {game_index + 1}/{H2H_GAMES} "
            f"cand={'W' if candidate_is_white else 'B'} {result} ({termination})",
            flush=True,
        )
    games = cand_wins + draws + cand_losses
    summary = {
        "candidate": candidate_name,
        "champion": champion_name,
        "games": games,
        "candidate_wins": cand_wins,
        "draws": draws,
        "candidate_losses": cand_losses,
        "win_rate": round((cand_wins + 0.5 * draws) / games, 3),
        "seconds": round(time.perf_counter() - started, 1),
    }
    print(f"[h2h] {candidate_name} vs {champion_name}: {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="U1/U2 ablation bench (CPU-pinned)")
    parser.add_argument("--out", default=str(Path(__file__).parent.parent / "tmp_u1u2" / "ablation_results.json"))
    parser.add_argument("--quick", action="store_true", help="1 game per config, 12 plies, 3 h2h games")
    args = parser.parse_args()
    if args.quick:
        global MAX_PLIES, GAMES_PER_CONFIG, H2H_GAMES
        MAX_PLIES = 12
        GAMES_PER_CONFIG = 1
        H2H_GAMES = 3

    disable_fastpath()
    device = torch.device("cpu")
    model = GambitNet(MODEL_TINY).to(device).eval()
    print(
        f"[bench] cpu threads={torch.get_num_threads()} model=tiny "
        f"plies<={MAX_PLIES} games/config={GAMES_PER_CONFIG} h2h games/pairing={H2H_GAMES}",
        flush=True,
    )

    throughput = bench_throughput(
        model,
        {
            "puct400": puct(400),
            "gumbel64": gumbel(64),
            "gumbel100": gumbel(100),
            "puct400_cap": puct(400),
        },
        capped={"puct400": False, "gumbel64": False, "gumbel100": False, "puct400_cap": True},
    )

    head_to_head = [
        bench_head_to_head(model, "gumbel64", gumbel(64), "puct400", puct(400)),
        bench_head_to_head(model, "gumbel100", gumbel(100), "gumbel64", gumbel(64)),
    ]

    summary = {
        "meta": {
            "cpu_threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "max_plies": MAX_PLIES,
            "games_per_config": GAMES_PER_CONFIG,
            "h2h_games_per_pairing": H2H_GAMES,
            "seed_base": SEED_BASE,
        },
        "throughput": throughput,
        "head_to_head": head_to_head,
        "u1_gate": {
            "games_per_hour_ratio": round(
                throughput["gumbel64"]["games_per_hour"] / throughput["puct400"]["games_per_hour"], 2
            ),
            "gumbel64_win_rate_vs_puct400": head_to_head[0]["win_rate"],
            "acceptance": ">= 4.0x games/h AND win-rate >= 0.35",
        },
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["u1_gate"], indent=2), flush=True)
    print(f"[bench] results written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
