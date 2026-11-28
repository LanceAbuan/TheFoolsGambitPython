"""Restore champion.json from a checkpoint + the trainer's known state.

The pointer was missing after the 2026-11-27 crash cascade (the run's state knew
champion=ckpt-002016-0642 elo=808.8, but runs/<regime>/<run>/champion.json had
been lost), so the play server had nothing to serve. This writes it back in the
exact format `pipeline._save_champion` produces.

    python scripts/restore_champion.py \
        --checkpoint runs/scratch/second-light-v2/checkpoints/ckpt-002016-0642.pt
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--run-root",
        default=None,
        help="directory holding champion.json; default = the REGIME dir derived "
             "from the checkpoint path (runs/<regime>), which is where the trainer "
             "writes it and where server/model_server.py scans",
    )
    parser.add_argument("--elo", type=float, default=808.8)
    parser.add_argument("--glicko-r", type=float, default=1524.0)
    parser.add_argument("--glicko-rd", type=float, default=36.5)
    parser.add_argument("--run-id", default="second-light-v2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import torch

    from training.model import ModelConfig

    ckpt = pathlib.Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        print(f"checkpoint not found: {ckpt}", file=sys.stderr)
        return 1
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(data["config"])
    tag = ckpt.stem
    payload = {
        "tag": tag,
        "checkpoint_path": str(ckpt),
        "elo": float(args.elo),
        "glicko": {"r": float(args.glicko_r), "rd": float(args.glicko_rd)},
        "run_id": args.run_id,
        "model_config": config.to_dict(),
        "updated_at": time.time(),
    }
    target_dir = (
        pathlib.Path(args.run_root)
        if args.run_root
        # runs/<regime>/<run-id>/checkpoints/x.pt -> runs/<regime>
        else ckpt.parents[2]
    )
    target = target_dir / "champion.json"
    print(json.dumps(payload, indent=2))
    if args.dry_run:
        print(f"[champion] dry run; would write {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: a torn champion.json disables the served model (it vanished once
    # already and had to be hand-restored).
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    print(f"[champion] wrote {target} -> {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
