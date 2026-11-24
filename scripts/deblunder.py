"""WP3 deblunder: re-label rows where a temperature-induced low-visit move
diverged sharply from the search's own estimate.

Lc0's deblunder rescorer fixes training rows whose value target was polluted
by an actively induced blunder. Our cheap version operates on spool records
(schema v2, per-example `root_value`):

    if played_move_share < min_share  AND  |z - q| > delta:
        value := (1 - blend) * z + blend * q

`z` is the game outcome target, `q` the search's root value for that ply. The
policy target is untouched. Never modifies the source files: writes
`<name>.deblundered.jsonl` into --out-dir (or alongside with --in-place).

Usage:
  .venv/Scripts/python.exe scripts/deblunder.py --dir runs/<run>/spool/processed --out-dir data/deblundered
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np


def _b64_encode(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def deblunder_record(
    record: dict,
    *,
    min_share: float = 0.05,
    delta: float = 0.5,
    blend: float = 0.5,
    min_ply: int = 8,
) -> int:
    """Re-label one decoded record in place; returns the number of rows fixed.

    The record must carry decoded tensors (``board_tensor``/``policy`` numpy
    arrays, ``root_value`` floats) - pass records through
    ``training.rust_workers.read_game_record`` first.
    """
    import chess

    from training.encoder import encode_move

    board = chess.Board(record["opening_fen"])
    changed = 0
    for i, (uci, ex) in enumerate(zip(record["ucis"], record["examples"])):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal recorded move {uci} at ply {i}")
        idx = encode_move(board, move)
        policy = np.asarray(ex["policy"])
        played_share = float(policy[idx]) if idx < policy.shape[0] else 0.0
        z = float(ex["value"])
        q = float(ex.get("root_value", z))
        if i >= min_ply and played_share < min_share and abs(z - q) > delta:
            ex["value"] = float((1.0 - blend) * z + blend * q)
            changed += 1
        board.push(move)
    return changed


def encode_record(record: dict) -> dict:
    """Inverse of read_game_record: numpy tensors -> base64 JSON payload."""
    out = dict(record)
    out["examples"] = [
        {
            "board_tensor_b64": _b64_encode(np.asarray(ex["board_tensor"])),
            "policy_b64": _b64_encode(np.asarray(ex["policy"])),
            "value": float(ex["value"]),
            "policy_masked": bool(ex.get("policy_masked", False)),
            "root_value": float(ex.get("root_value", 0.0)),
        }
        for ex in record["examples"]
    ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="WP3 deblunder rescorer")
    parser.add_argument("--dir", required=True, help="directory of spool game JSONL files")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-share", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 = all files")
    args = parser.parse_args()

    from training.rust_workers import read_game_record

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.dir).glob("*.jsonl"))
    if args.limit:
        files = files[: args.limit]
    total_rows = 0
    changed_files = 0
    for path in files:
        record = read_game_record(path)
        changed = deblunder_record(
            record,
            min_share=args.min_share,
            delta=args.delta,
            blend=args.blend,
            min_ply=args.min_ply,
        )
        if changed:
            changed_files += 1
            total_rows += changed
        (out_dir / f"{path.stem}.deblundered.jsonl").write_text(
            json.dumps(encode_record(record)), encoding="utf-8"
        )
    print(f"[deblunder] {len(files)} games, {changed_files} touched, {total_rows} rows re-labelled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
