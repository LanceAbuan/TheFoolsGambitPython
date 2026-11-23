"""WP-C training CLI — thin wrapper over training.pipeline.run_training.

Usage:
    uv run python scripts/train.py --profile smoke                 # <2 min, CPU-safe
    uv run python scripts/train.py --profile night --regime scratch # real run (orchestrator only)

Never start a `night` run unattended without the orchestrator asking.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

# Make `training` importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.encoder import NUM_PLANES
from training.model import MODEL_TINY
from training.pipeline import (
    NIGHT_PROFILE,
    NIGHT_V2_PROFILE,
    SMOKE_PROFILE,
    TrainingProfile,
    encoder_version_for,
    run_training,
)
from training.regimes import REGIME_NAMES

_PROFILE_BY_NAME: dict[str, TrainingProfile] = {
    "smoke": SMOKE_PROFILE,
    "night": NIGHT_PROFILE,
    "night-v2": NIGHT_V2_PROFILE,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fool's Gambit v2 training (WP-C)")
    parser.add_argument("--profile", choices=list(_PROFILE_BY_NAME), default="smoke",
                        help="smoke = tiny/fast sanity run; night = full run; "
                             "night-v2 = night + WP-A throughput knobs + LR schedule")
    parser.add_argument("--regime", choices=list(REGIME_NAMES), default="scratch",
                        help="training regime (§5.2)")
    parser.add_argument("--run-id", default=None, help="run id (default: UTC timestamp)")
    parser.add_argument("--runs-dir", default="runs", help="lineage root (default: runs/)")
    parser.add_argument("--data-dir", default="data", help="ledger/metrics dir (default: data/)")
    parser.add_argument("--iterations", type=int, default=None, help="override profile iterations")
    parser.add_argument("--games", type=int, default=None, help="override games per iteration")
    parser.add_argument("--plies", type=int, default=None, help="override max plies per game")
    parser.add_argument("--sims", type=int, default=None, help="override MCTS sims per move")
    parser.add_argument("--seed", type=int, default=None, help="override RNG seed")
    parser.add_argument("--parallel-games", type=int, default=None,
                        help="override concurrent self-play games (1-32, FS-4)")
    parser.add_argument("--no-stream-games", action="store_true",
                        help="disable live per-move journal streaming (FS-6b, headless/bulk)")
    parser.add_argument("--resume", nargs="?", const=True, default=None, metavar="RUN_ID",
                        help="resume a run: --resume RUN_ID, or --resume with --run-id")
    parser.add_argument("--init-checkpoint", default=None, metavar="PATH",
                        help="WP-D: seed a FRESH run's weights from a checkpoint file "
                             "(e.g. runs/v2-warmstart/ckpt-v2-0000.pt); ignored with --resume")
    parser.add_argument("--search", choices=["puct", "gumbel"], default=None,
                        help="root search policy (U1): puct (regime default) or gumbel")
    parser.add_argument("--playout-cap", action="store_true",
                        help="enable KataGo playout cap randomization (U2, Wu 2020)")
    return parser


def build_profile(args: argparse.Namespace) -> TrainingProfile:
    base = _PROFILE_BY_NAME[args.profile]
    # WP-D: the model comes from the PROFILE (smoke's MODEL_TINY preset,
    # night's 10x128 preset, night-v2's 104-plane W/D/L preset). Overriding
    # it here used to clobber night-v2's encoder/head config with MODEL_BASE.
    model = MODEL_TINY if args.profile == "smoke" else base.model
    # The encoder layout must be a known one (29-plane v1 or 104-plane v2);
    # the pipeline's evaluators dispatch on the model's plane count (WP-D).
    encoder_version_for(model.in_channels)
    overrides: dict[str, object] = {}
    if args.iterations is not None:
        overrides["iterations"] = args.iterations
    if args.games is not None:
        overrides["games_per_iteration"] = args.games
    if args.plies is not None:
        overrides["max_plies"] = args.plies
    if args.sims is not None:
        overrides["sims"] = args.sims
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.parallel_games is not None:
        overrides["parallel_games"] = args.parallel_games
    if args.no_stream_games:
        overrides["stream_games"] = False
    # WP-A: dataclasses.replace keeps every un-overridden base field —
    # including `lr_schedule`, `train_steps`, `batch_size`, `min_examples`
    # and the ladder — so a new profile flag can never silently drop one.
    return replace(
        base,
        name=base.name,
        model=model,
        regime=args.regime,
        **overrides,  # type: ignore[arg-type]
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args()
    if args.profile == "night":
        logging.getLogger(__name__).warning(
            "profile=night is a long run — start it only deliberately, not from CI"
        )
    if args.resume is True and args.run_id is None:
        logging.getLogger(__name__).error("--resume without a run id: pass --resume RUN_ID or --run-id")
        return 2
    resume: str | bool = args.run_id if args.resume is True else (args.resume or False)
    if args.init_checkpoint and resume:
        logging.getLogger(__name__).error(
            "--init-checkpoint cannot be combined with --resume (a resumed run rebuilds its own weights)"
        )
        return 2
    profile = build_profile(args)
    summary = run_training(profile, run_id=args.run_id,
                           base_dir=Path(args.runs_dir), data_dir=Path(args.data_dir),
                           resume=resume,
                           search_kind=args.search,
                           playout_randomization=True if args.playout_cap else None,
                           init_checkpoint=args.init_checkpoint)
    logging.getLogger(__name__).info(
        "run complete: %s/%s checkpoints=%s champion=%s elo=%s games=%d",
        summary.regime, summary.run_id, summary.checkpoints,
        summary.champion, summary.elo, summary.games_played,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
