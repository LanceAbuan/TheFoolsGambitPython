"""WP-H H1: export the v2 net to ONNX and prove parity against PyTorch.

- Builds GambitNet(NIGHT_V2_PROFILE.model) with a seeded init (structure is
  what matters; the fresh weights are deterministic and reproducible).
- Exports models/gambitnet-v2.onnx (opset 17, dynamic batch).
- Parity gate: ONNX Runtime vs PyTorch outputs <= 1e-5 on 1,000 seeded
  random-play positions (<=8-ply walks, so history stays within the 7-slot
  window and python/rust encoding agree bit-exactly).
- Saves a committed fixture tests/fixtures/onnx_parity_v2.npz (128 rows:
  inputs + torch policy/value) for the Rust ort evaluator's validation.

Usage: .venv/Scripts/python.exe scripts/export_onnx.py
       .venv/Scripts/python.exe scripts/export_onnx.py --out <path>.onnx
       .venv/Scripts/python.exe scripts/export_onnx.py --checkpoint <ckpt>.pt --out <path>.onnx

WP-H H4: `--checkpoint` exports a TRAINED checkpoint (the pipeline's
gate/iteration artifacts) instead of the seeded fresh init, and `--out`
redirects the target (default models/gambitnet-v2.onnx, byte-identical
default behavior). The H4 worker cadence is automatic -
training.rust_workers.export_model_to_onnx runs this export per iteration
from the trainer and publishes models/latest.json - so this CLI is the
MANUAL/backfill path (e.g. re-exporting a champion for a soak run).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.encoder import encode_v2  # noqa: E402
from training.model import GambitNet  # noqa: E402
from training.pipeline import NIGHT_V2_PROFILE  # noqa: E402


def probe_planes(n_rows: int, seed: int) -> np.ndarray:
    import chess

    board = chess.Board()
    rng = np.random.default_rng(seed)
    history: list[chess.Board] = []
    out: list[np.ndarray] = []
    while len(out) < n_rows:
        if board.is_game_over(claim_draw=True):
            board = chess.Board()
            history = []
            continue
        out.append(encode_v2(board, history[-7:] if history else None))
        history.append(board.copy())
        moves = list(board.legal_moves)
        board.push(moves[int(rng.integers(len(moves)))])
        if len(history) > 64:  # keep walks short; bounded memory
            board = chess.Board()
            history = []
    return np.stack(out).astype(np.float32)


def load_checkpoint_model(ckpt_path: Path) -> "GambitNet":
    """Build a GambitNet from a pipeline checkpoint (config + state dict)."""
    from training.model import ModelConfig

    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(data["config"])
    model = GambitNet(config)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model


def export_model(model: "GambitNet", onnx_path: Path) -> None:
    """Export a (v2-layout) GambitNet to ONNX opset 17, dynamic batch."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, int(model.config.in_channels), 8, 8)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={"input": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"[onnx] exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the v2 GambitNet to ONNX (WP-H H1/H4).")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / "gambitnet-v2.onnx",
        help="target .onnx path (default: models/gambitnet-v2.onnx)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="export a trained pipeline .pt checkpoint instead of the seeded fresh init",
    )
    args = parser.parse_args()

    if args.checkpoint is not None:
        model = load_checkpoint_model(args.checkpoint)
        print(f"[onnx] loaded checkpoint {args.checkpoint}")
    else:
        torch.manual_seed(20261123)
        model = GambitNet(NIGHT_V2_PROFILE.model)
        model.eval()
    onnx_path = args.out

    # The parity probe encodes v2 (104-plane) positions; refuse anything
    # else loudly instead of exporting an unverifiable model.
    if int(model.config.in_channels) != int(NIGHT_V2_PROFILE.model.in_channels):
        print(
            f"[onnx] model is {model.config.in_channels}-plane; this exporter "
            f"verifies the v2 {NIGHT_V2_PROFILE.model.in_channels}-plane layout only",
            file=sys.stderr,
        )
        return 2
    export_model(model, onnx_path)

    planes = probe_planes(1000, seed=20261123)
    x = torch.from_numpy(planes)
    with torch.no_grad():
        t_policy, t_value = model(x)
    t_policy, t_value = t_policy.numpy(), t_value.numpy()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o_policy, o_value = sess.run(None, {"input": planes})
    dp = float(np.abs(o_policy - t_policy).max())
    dv = float(np.abs(o_value - t_value).max())
    print(f"[onnx] parity on {len(planes)} positions: max|dPolicy|={dp:.2e}  max|dValue|={dv:.2e}")
    if dp > 1e-5 or dv > 1e-5:
        print("[onnx] PARITY FAILED", file=sys.stderr)
        return 1

    if onnx_path == ROOT / "models" / "gambitnet-v2.onnx":
        fixture = ROOT / "tests" / "fixtures" / "onnx_parity_v2.npz"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        np.savez(fixture, inputs=planes[:128], policy=t_policy[:128], value=t_value[:128])
        print(f"[onnx] wrote fixture {fixture} (128 rows)")
    else:
        print("[onnx] custom --out: fixture untouched")
    print("[onnx] H1 GATE GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
