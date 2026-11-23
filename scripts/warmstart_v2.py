"""WP-D: warm-start the v2 net (104ch encoder, W/D/L head) from the final v1 champion.

What transfers (shapes identical at 10x128):
  - residual tower + policy head: copied verbatim
  - batch-norm running stats: copied verbatim
What is adapted:
  - input conv [128,29,3,3] -> [128,104,3,3]: v1 planes are a strict PREFIX of
    v2 (hard WP-B requirement), so old weights copy into channels 0:29 and the
    new channels (29:104) zero-init - the net starts seeing history/rep planes
    as no-ops and learns them from data.
What starts fresh (no v1 equivalent):
  - W/D/L value head (v1 had a single tanh scalar), Adam optimizer moments
    (v1 checkpoints carry no optimizer state).

Proof of transfer: on 200 seeded random-play positions the warm-started v2
policy must sit far closer to the v1 teacher's policy than a freshly
initialized v2 does (mean KL divergence, nats). The v1 model is evaluated on
planes[:29] of the SAME v2 tensor - valid because of the prefix rule.

Usage:
  .venv/Scripts/python.exe scripts/warmstart_v2.py \
      [--src runs/scratch/second-light/checkpoints/ckpt-001536-0768.pt] \
      [--dst runs/v2-warmstart/ckpt-v2-0000.pt]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.encoder import ENCODER_VERSION, encode_v2  # noqa: E402
from training.model import GambitNet, ModelConfig  # noqa: E402
from training.pipeline import NIGHT_V2_PROFILE  # noqa: E402


def load_v1_model(src: Path) -> tuple[GambitNet, dict]:
    data = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(data["config"])  # pre-WP-C dict -> value_head defaults to "scalar"
    model = GambitNet(cfg)
    model.load_state_dict(data["model_state"], strict=True)
    model.eval()
    return model, data


def transfer_weights(v1: GambitNet, v2: GambitNet) -> dict[str, object]:
    sd1, sd2 = v1.state_dict(), v2.state_dict()
    copied = partial = fresh = 0
    fresh_names: list[str] = []
    with torch.no_grad():
        for name, t2 in sd2.items():
            if name not in sd1:
                fresh += 1
                fresh_names.append(name)
                continue
            t1 = sd1[name]
            if t1.shape == t2.shape:
                t2.copy_(t1)
                copied += 1
            elif (
                t1.dim() == 4
                and t2.dim() == 4
                and t1.shape[0] == t2.shape[0]
                and t1.shape[2:] == t2.shape[2:]
                and t1.shape[1] != t2.shape[1]
            ):
                # input-conv channel widening via the prefix rule
                n = min(t1.shape[1], t2.shape[1])
                t2[:, :n].copy_(t1)
                t2[:, n:].zero_()
                partial += 1
            else:
                fresh += 1
                fresh_names.append(name)
    return {"copied": copied, "partial": partial, "fresh": fresh, "fresh_names": fresh_names}


def probe_planes(n_plies: int, seed: int) -> np.ndarray:
    """Seeded random-play position stack, encoded as v2 tensors."""
    import chess

    board = chess.Board()
    rng = np.random.default_rng(seed)
    history: list[chess.Board] = []
    out: list[np.ndarray] = []
    for _ply in range(n_plies):
        if board.is_game_over(claim_draw=True):
            break
        out.append(encode_v2(board, history[-7:] if history else None))
        history.append(board.copy())
        moves = list(board.legal_moves)
        board.push(moves[int(rng.integers(len(moves)))])
    return np.stack(out)


def mean_kl(model: GambitNet, planes: torch.Tensor, teacher: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits, _value = model(planes)
    logq = F.log_softmax(logits, dim=-1)
    return F.kl_div(logq, teacher, reduction="batchmean").item()


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm-start the v2 net from the final v1 champion")
    parser.add_argument("--src", default="runs/scratch/second-light/checkpoints/ckpt-001728-0864.pt")
    parser.add_argument("--dst", default="runs/v2-warmstart/ckpt-v2-0000.pt")
    parser.add_argument("--probe-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    src, dst = ROOT / args.src, ROOT / args.dst
    if not src.is_file():
        print(f"[warm] source checkpoint missing: {src}", file=sys.stderr)
        return 1

    torch.manual_seed(0)
    v1, data = load_v1_model(src)
    print(f"[warm] v1 teacher loaded: {data.get('tag')} (step {data.get('step')}, "
          f"value_head={data['config'].get('value_head', 'scalar')})")

    v2 = GambitNet(NIGHT_V2_PROFILE.model)
    torch.manual_seed(1234)  # fresh reference with a DIFFERENT init than v2 above
    fresh = GambitNet(NIGHT_V2_PROFILE.model)

    stats = transfer_weights(v1, v2)
    print(f"[warm] weights: {stats['copied']} copied, {stats['partial']} channel-widened, "
          f"{stats['fresh']} fresh ({', '.join(stats['fresh_names']) or '-'})")

    planes = probe_planes(args.probe_plies, args.seed)
    x2 = torch.from_numpy(planes)
    with torch.no_grad():
        v1_logits, _ = v1(x2[:, :29])  # prefix rule: v1 planes are v2[:29]
        teacher = F.softmax(v1_logits, dim=-1)
    kl_warm = mean_kl(v2, x2, teacher)
    kl_fresh = mean_kl(fresh, x2, teacher)
    print(f"[warm] transfer proof on {len(planes)} positions: "
          f"mean KL(v1 || warm v2) = {kl_warm:.4f} nats vs "
          f"mean KL(v1 || fresh v2) = {kl_fresh:.4f} nats")

    ok = kl_warm < 1.0 and kl_warm * 3 < kl_fresh
    if not ok:
        print("[warm] TRANSFER NOT PROVEN - refusing to write the checkpoint", file=sys.stderr)
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": v2.state_dict(),
            "config": NIGHT_V2_PROFILE.model.to_dict(),
            "tag": dst.stem,
            "step": 0,
            "games": 0,
            "ts": time.time(),
            "regime": "scratch",
            "run_id": None,
            "warm_started_from": data.get("tag"),
            "source_step": data.get("step"),
            "encoder_version": ENCODER_VERSION,
            "value_head": "wdl",
        },
        dst,
    )
    print(f"[warm] wrote {dst} (TRANSFER PROVEN: warm KL {kl_warm:.4f} << fresh KL {kl_fresh:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
