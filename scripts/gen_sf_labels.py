"""Generate Stockfish-labelled spool games (policy distillation from a stronger engine).

Why this exists: pure self-play from a weak net is a slow, self-referential
signal. Research A35 measured that mixing ~25% engine-labelled rows into the
training data is a net win (+0.05 in the cited run) while 50% hurts - it adds
*volume with a better teacher* without displacing self-play.

What a record contains: SF plays both sides from a random short opening, and at
every position we store
  * policy = SF's chosen move as a one-hot over the 4672-action index space, and
  * value  = SF's normalized evaluation (a smoother, stronger value target than
             a single game result).
Records land in the normal spool inbox (worker_id "sf"), so the trainer consumes
them exactly like self-play games - no pipeline changes, and the staleness
horizon treats them as current.

Usage:
    python scripts/gen_sf_labels.py --games 40 --depth 8 --skill 8 \
        --spool-dir runs/scratch/second-light-v2/spool

Mix guidance: aim for SF rows at ~25% of new positions per iteration. At 600
games/hr self-play and ~60 usable rows/game, 40 SF games (~2400 rows) is a
reasonable daily top-up; check the [RUST-SPOOL] weight_version line.
"""

from __future__ import annotations

import argparse
import base64
import pathlib
import random
import sys

import chess
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.encoder import encode_move, encode_v2  # noqa: E402
from training.rust_workers import (  # noqa: E402
    SPOOL_ENCODING,
    SPOOL_SCHEMA_ID,
    write_game_record,
)


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode()


def _random_opening(rng: random.Random, plies: int = 6) -> chess.Board:
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


def _coerce_move(board: chess.Board, move) -> chess.Move | None:
    """Accept a chess.Move, a UCI string or a SAN string; None if unplayable."""
    if isinstance(move, chess.Move):
        return move if move in board.legal_moves else None
    if isinstance(move, str):
        for parse in (chess.Move.from_uci, board.parse_san):
            try:
                candidate = parse(move)
            except Exception:  # noqa: BLE001 - try the next notation
                continue
            if candidate in board.legal_moves:
                return candidate
    return None


def build_record(
    engine,
    *,
    seed: int,
    weight_version: int,
    max_plies: int = 120,
    opening_plies: int = 6,
    worker_id: str = "sf",
    black_engine=None,
) -> dict:
    """Play one SF-vs-SF game and return a schema-valid spool record.

    ``engine`` only needs ``get_move(board)`` and
    ``get_evaluation_normalized(board)`` - that is what makes this testable
    without a Stockfish binary.
    """
    rng = random.Random(seed)
    board = _random_opening(rng, opening_plies)
    opening_fen = board.fen()
    history = [board.copy(stack=False)]
    records: list[dict] = []
    ucis: list[str] = []
    sans: list[str] = []
    fens: list[str] = []

    while not board.is_game_over() and len(ucis) < max_plies:
        active = black_engine if (black_engine is not None and not board.turn) else engine
        move = _coerce_move(board, active.get_move(board))
        if move is None:
            break
        eval_norm = float(active.get_evaluation_normalized(board))
        policy = np.zeros(4672, dtype=np.float32)
        index = encode_move(board, move)
        if index is not None:
            policy[index] = 1.0
        planes = encode_v2(board, history)
        records.append(
            {
                "board_tensor_b64": _b64(planes),
                "policy_b64": _b64(policy),
                "value": max(-1.0, min(1.0, eval_norm)),
                "policy_masked": False,
            }
        )
        ucis.append(move.uci())
        sans.append(board.san(move))
        board.push(move)
        history.append(board.copy(stack=False))
        fens.append(board.fen())

    if not ucis:
        raise ValueError("engine produced no legal move; nothing to record")

    result = board.result(claim_draw=True) if board.is_game_over() else "1/2-1/2"
    if result not in ("1-0", "0-1", "1/2-1/2"):
        result = "1/2-1/2"
    return {
        "schema": SPOOL_SCHEMA_ID,
        "encoding": SPOOL_ENCODING,
        "weight_version": int(weight_version),
        "worker_id": worker_id,
        "seed": int(seed),
        "opening_seed": None,
        "sims": 0,
        "max_plies": int(max_plies),
        "temperature": 0.0,
        "temperature_plies": 0,
        "opening_fen": opening_fen,
        "result": result,
        "termination": "game-over" if board.is_game_over() else "max-plies",
        "ucis": ucis,
        "moves": sans,
        "fens": fens,
        "ply_count": len(ucis),
        "examples": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--skill", type=int, default=None, help="UCI skill 0-20 (None = full)")
    parser.add_argument("--skill-black", type=int, default=None,
                        help="optional second skill for Black; a disparity (e.g. white 20 vs "
                             "black 0) produces DECISIVE games, which is what makes the "
                             "resulting positions useful as a value benchmark")
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--spool-dir", default="runs/scratch/second-light-v2/spool")
    parser.add_argument("--weight-version", type=int, default=None,
                        help="default: the published latest.json version")
    args = parser.parse_args()

    from training.rust_workers import read_model_pointer
    from training.stockfish_engine import StockfishPlayer, resolve_stockfish_path

    if resolve_stockfish_path() is None:
        print(
            "no Stockfish binary found: set STOCKFISH_PATH or drop stockfish.exe in tools/",
            file=sys.stderr,
        )
        return 1
    version = args.weight_version
    if version is None:
        pointer = read_model_pointer(pathlib.Path(args.spool_dir).parent / "onnx")
        version = pointer.weight_version if pointer is not None else 0

    written = 0
    black_engine = None
    with StockfishPlayer(depth=args.depth, skill=args.skill) as engine:
        if args.skill_black is not None and args.skill_black != args.skill:
            black_engine = StockfishPlayer(depth=args.depth, skill=args.skill_black)
            black_engine.__enter__()
        try:
            for i in range(args.games):
                record = build_record(
                    engine,
                    seed=args.seed + i,
                    weight_version=version,
                    max_plies=args.max_plies,
                    opening_plies=args.opening_plies,
                    black_engine=black_engine,
                )
                path = write_game_record(args.spool_dir, record)
                written += 1
                print(
                    f"[sf-labels] {i+1}/{args.games} {record['result']} "
                    f"{record['ply_count']} plies -> {path.name}",
                    flush=True,
                )
        finally:
            if black_engine is not None:
                black_engine.__exit__(None, None, None)
    print(f"[sf-labels] wrote {written} games (weight_version {version})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
