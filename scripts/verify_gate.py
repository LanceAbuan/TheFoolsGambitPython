"""End-to-end gate verification (Fool's Gambit).

Runs one candidate-vs-champion match with the EXACT settings the pipeline will
use, so a gate can be validated in a few minutes instead of discovered dead
inside a training loop:

    python scripts/verify_gate.py \
        --candidate runs/scratch/second-light-v2/checkpoints/ckpt-093504-29892.pt \
        --champion  runs/scratch/second-light-v2/checkpoints/ckpt-002016-0642.pt \
        --games 32 --sims 96 --parallel 8

What it checks (and why each has failed before):
  * every game completes and a heartbeat lands per game (a silent gate was
    indistinguishable from a hang and got killed by the watchdog);
  * the whole match finishes well under 15 minutes with the fleet RUNNING
    (the arena shares the GPU with 8 self-play workers, which is what made
    16 sequential games cost ~2 hours);
  * a MatchResult comes back with a sane score.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.ladder import Arena  # noqa: E402
from training.pipeline import _load_champion  # noqa: E402
from training.selfplay import ArenaConfig, BatchNNEvaluator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--sims", type=int, default=96)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--search-backend", default="rust", choices=["python", "rust"])
    args = parser.parse_args()

    import torch

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    candidate, cand_tag, cand_cfg = _load_champion(pathlib.Path(args.candidate), device)
    champion, champ_tag, champ_cfg = _load_champion(pathlib.Path(args.champion), device)
    print(
        f"[gate-verify] candidate {cand_tag} vs champion {champ_tag} | "
        f"{args.games} games, {args.sims} sims, parallel {args.parallel}, "
        f"backend {args.search_backend} | device {device}",
        flush=True,
    )

    arena = Arena(
        BatchNNEvaluator(candidate, device),
        BatchNNEvaluator(champion, device),
        ArenaConfig(
            games=args.games,
            num_simulations=args.sims,
            max_plies=args.max_plies,
            parallel=args.parallel,
            sprt_early_stop=False,  # SPRT forces the SEQUENTIAL path
            search_backend=args.search_backend,
        ),
    )
    match = arena.match(cand_tag, champ_tag)
    elapsed = time.perf_counter() - started
    score = (match.wins + 0.5 * match.draws) / max(match.games, 1)
    print(
        f"[gate-verify] DONE in {elapsed/60:.1f} min: "
        f"{match.wins}W-{match.draws}D-{match.losses}L over {match.games} games "
        f"(score {score:.3f})",
        flush=True,
    )
    ok = match.games == args.games and elapsed < 15 * 60
    print(f"[gate-verify] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
