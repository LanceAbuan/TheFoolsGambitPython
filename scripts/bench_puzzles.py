"""WP9: Lichess puzzle benchmark — tactical accuracy that moves before Elo.

Input file (data/puzzles.jsonl), one JSON object per line:
    {"puzzle_id": "...", "fen": "<position BEFORE the opponent's first move>",
     "moves": ["<uci>", "<uci>", ...], "rating": 1500, "themes": [...]}

Scoring (ChessBench-style, move by move):
  * the bot must play exactly the Lichess solution move (or give mate);
  * wrong / illegal move ends the puzzle as unsolved;
  * the opponent's replies come from the puzzle line (never guessed);
  * only the bot's plies are scored; a puzzle counts as solved when every
    bot move matched.

The DB is public: https://database.lichess.org/#puzzles (zstd CSV). Sample a
few hundred puzzles across rating bands and convert with the loader below.

Usage:
  .venv/Scripts/python.exe scripts/bench_puzzles.py --puzzles data/puzzles.jsonl --sims 128 --limit 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_puzzles(path: Path) -> list[dict]:
    puzzles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            puzzles.append(json.loads(line))
    return puzzles


def play_puzzle(board, moves: list[str], choose_move, max_bot_plies: int = 8) -> bool:
    """Replay one puzzle; True when every bot move matches the solution.

    `choose_move(board)` returns a `chess.Move` (the bot's answer).
    `moves` alternates opponent/bot UCI moves, Lichess format.
    """
    import chess

    bot_to_move = False  # Lichess FEN is BEFORE the opponent's move: the
    # solver answers moves[1], moves[3], ... (opponent replies come from Film line).
    for i, uci in enumerate(moves):
        solution = chess.Move.from_uci(uci)
        if bot_to_move:
            if i >= max_bot_plies * 2:
                return False
            answer = choose_move(board)
            if answer is None:
                return False
            if answer != solution and not gives_mate(board, answer):
                return False
        board.push(solution)
        bot_to_move = not bot_to_move
        if i + 1 >= len(moves):
            break
    return True


def gives_mate(board, move) -> bool:
    board.push(move)
    mated = board.is_checkmate()
    board.pop()
    return mated


def main() -> int:
    parser = argparse.ArgumentParser(description="Lichess puzzle benchmark (WP9)")
    parser.add_argument("--puzzles", default="data/puzzles.jsonl")
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    path = Path(args.puzzles)
    if not path.is_file():
        print(
            f"[puzzles] {path} missing - sample the Lichess puzzle DB first\n"
            "  https://database.lichess.org/#puzzles (zstd CSV)\n"
            "  convert rows to {puzzle_id, fen, moves, rating, themes} JSONL",
            file=sys.stderr,
        )
        return 1
    puzzles = load_puzzles(path)
    if args.limit:
        puzzles = puzzles[: args.limit]
    if not puzzles:
        print("[puzzles] no puzzles loaded", file=sys.stderr)
        return 1

    sys.path.insert(0, str(ROOT))
    import chess
    import torch

    from training.model import GambitNet, ModelConfig
    from training.pipeline import NIGHT_V2_PROFILE
    from training.selfplay import BatchNNEvaluator, MCTSSearch, SearchConfig

    ckpt_path = args.checkpoint or str(
        ROOT / "runs" / "scratch" / "second-light-v2" / "checkpoints"
    )
    if Path(ckpt_path).is_dir():
        files = sorted(Path(ckpt_path).glob("ckpt-*.pt"))
        if not files:
            print("[puzzles] no checkpoints found", file=sys.stderr)
            return 1
        ckpt_path = str(files[-1])
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(data["config"])
    model = GambitNet(config)
    model.load_state_dict(data["model_state"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    evaluator = BatchNNEvaluator(model, device)
    search = MCTSSearch(
        evaluator, SearchConfig(num_simulations=args.sims, root_noise=False)
    )

    solved = 0
    failures = 0
    for i, puzzle in enumerate(puzzles):
        board = chess.Board(puzzle["fen"])

        def choose(b):
            move = search.search(b).chosen_move
            return move

        try:
            ok = play_puzzle(board, puzzle["moves"], choose)
        except Exception:  # noqa: BLE001 - a bad puzzle row must not stop the bench
            failures += 1
            continue
        solved += 1 if ok else 0
        if (i + 1) % 25 == 0:
            print(f"[puzzles] {i + 1}/{len(puzzles)} solved={solved}", flush=True)
    total = len(puzzles) - failures
    pct = 100.0 * solved / total if total else 0.0
    print(
        f"[puzzles] solved {solved}/{total} ({pct:.1f}%) at {args.sims} sims; "
        f"{failures} unreadable rows; ckpt={Path(ckpt_path).name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
