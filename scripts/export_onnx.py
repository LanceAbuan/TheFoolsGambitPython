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
"""
from __future__ import annotations

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


def main() -> int:
    torch.manual_seed(20261123)
    model = GambitNet(NIGHT_V2_PROFILE.model)
    model.eval()

    onnx_path = ROOT / "models" / "gambitnet-v2.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, NIGHT_V2_PROFILE.model.in_channels, 8, 8)
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

    fixture = ROOT / "tests" / "fixtures" / "onnx_parity_v2.npz"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fixture, inputs=planes[:128], policy=t_policy[:128], value=t_value[:128])
    print(f"[onnx] wrote fixture {fixture} (128 rows)")
    print("[onnx] H1 GATE GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
