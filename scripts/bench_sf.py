"""SF benchmark: champion NN vs Stockfish presets — the external Elo axis.

The arena gate is self-referential by construction (NN vs NN, and argmax
determinism means a 16-game match is really 2 distinct games — see TODO).
This script gives the champion a reference that exists outside the training
lineage: Stockfish at the *same* strength presets humans face on the Play
page (server/stockfish.py — wandering/curious/enlightened).

Design:
- Champion checkpoint loads exactly like the play server loads it
  (server/model_server.py: embedded ModelConfig + state dict, CUDA).
- NN side plays at gate strength (root_noise=False, argmax, --sims default
  128 = LadderConfig arena sims) so numbers are comparable to gate play.
- Randomized seeded openings (6 half-moves, unique per pairing, balanced
  colors) make every game independent — without this, deterministic NN vs
  deterministic SF would repeat the same game.
- One SF subprocess, Threads=1 + Hash=16 — mirrors the server's per-game
  caps so the bench never saturates the box alongside training.
- Read-only for training: touches nothing under runs/ except reading the
  champion checkpoint; appends results to data/sf_bench.jsonl.

Usage (repo root):
    .venv/Scripts/python.exe scripts/bench_sf.py                 # all presets
    .venv/Scripts/python.exe scripts/bench_sf.py --presets curious
    .venv/Scripts/python.exe scripts/bench_sf.py --games-per-preset 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Mirrors server/stockfish.py exactly (preset table + CONFIG_OPTIONS).
SF_BINARY = Path("tools/stockfish.exe")
PRESETS: dict[str, dict[str, int]] = {
    "wandering": {"skill": 2, "depth": 1},
    "curious": {"skill": 6, "depth": 4},
    "enlightened": {"skill": 12, "depth": 12},
}
SF_OPTIONS: dict[str, int] = {"Threads": 1, "Hash": 16}
OPENING_PLIES = 6  # half-moves of seeded random play before the engines take over


def load_champion(checkpoint: Path):
    """Load the champion .pt the same way server/model_server does."""
    import torch

    from training.model import GambitNet, ModelConfig
    from training.selfplay import BatchNNEvaluator

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = ModelConfig.from_dict(payload["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GambitNet(model_config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    evaluator = BatchNNEvaluator(model, device)
    tag = str(payload.get("tag", "unknown"))
    return evaluator, device, tag, model_config


def champion_checkpoint() -> Path:
    """Resolve the champion checkpoint from runs/<regime>/champion.json (newest mtime)."""
    pointers = sorted(ROOT.glob("runs/*/champion.json"), key=lambda p: p.stat().st_mtime)
    for pointer in reversed(pointers):
        data = json.loads(pointer.read_text(encoding="utf-8"))
        raw = data.get("checkpoint_path", "")
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return path
    raise FileNotFoundError("no readable champion checkpoint under runs/*/champion.json")


def random_openings(count: int, seed: int) -> list[str]:
    """`count` unique opening position FENs via seeded random play (6 half-moves)."""
    import numpy as np
    import chess

    rng_seed = seed
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < count:
        rng_seed += 1
        board = chess.Board()
        rng = np.random.default_rng(rng_seed)
        for _ in range(OPENING_PLIES):
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
        fen = board.fen()
        if fen not in seen:
            seen.add(fen)
            out.append(fen)
    return out


def play_game(evaluator, opening_fen: str, nn_is_white: bool, engine, depth: int, sims: int, max_plies: int):
    """One game: NN vs SF from `opening_fen`; mirrors ladder.play_match_game NN path."""
    import chess
    from training.selfplay import MCTSSearch, SearchConfig

    board = chess.Board(fen=opening_fen)
    search = MCTSSearch(evaluator, SearchConfig(root_noise=False, num_simulations=sims), rng=None)
    termination = "max-plies"
    for _ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            termination = "game-over"
            break
        nn_turn = (board.turn == chess.WHITE) == nn_is_white
        if nn_turn:
            move = search.search(board).chosen_move
        else:
            move = engine.play(board, chess.engine.Limit(depth=depth)).move
        assert move is not None, "engine returned no move on a live position"
        board.push(move)
    result = board.result(claim_draw=True)
    if result == "*":
        result = "1/2-1/2"  # capped games score as draws (arena aggregation)
    return result, termination


def main() -> int:
    parser = argparse.ArgumentParser(description="Champion NN vs Stockfish presets benchmark")
    parser.add_argument("--presets", default="wandering,curious,enlightened")
    parser.add_argument("--games-per-preset", type=int, default=12)
    parser.add_argument("--sims", type=int, default=128, help="NN simulations (gate uses 128)")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--checkpoint", default=None, help="override champion resolution")
    parser.add_argument("--out", default="data/sf_bench.jsonl")
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import chess.engine

    ckpt = Path(args.checkpoint) if args.checkpoint else champion_checkpoint()
    evaluator, device, tag, model_config = load_champion(ckpt)
    print(f"[bench] champion {tag} loaded on {device} ({ckpt.name})", flush=True)

    if not SF_BINARY.is_file():
        print(f"[bench] SF binary missing: {SF_BINARY}", file=sys.stderr)
        return 1
    engine = chess.engine.SimpleEngine.popen_uci(str(SF_BINARY))
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    overall = time.time()
    try:
        for preset_name in [p.strip() for p in args.presets.split(",") if p.strip()]:
            if preset_name not in PRESETS:
                print(f"[bench] unknown preset '{preset_name}' — skipping", file=sys.stderr)
                continue
            preset = PRESETS[preset_name]
            n = args.games_per_preset + (args.games_per_preset % 2)  # balance colors
            openings = random_openings(n // 2, args.seed + abs(hash(preset_name)) % 1000)
            engine.configure({**SF_OPTIONS, "Skill Level": preset["skill"]})

            wins = draws = losses = 0
            games_detail: list[dict] = []
            t0 = time.time()
            for opening_fen in openings:
                for nn_is_white in (True, False):
                    result, termination = play_game(
                        evaluator, opening_fen, nn_is_white, engine,
                        preset["depth"], args.sims, args.max_plies,
                    )
                    if result == "1/2-1/2":
                        draws += 1
                    elif (result == "1-0") == nn_is_white:
                        wins += 1
                    else:
                        losses += 1
                    games_detail.append({
                        "opening": opening_fen,
                        "nn_color": "white" if nn_is_white else "black",
                        "result": result,
                        "termination": termination,
                    })
                    played = wins + draws + losses
                    print(
                        f"[bench] {preset_name} {played:>2}/{n}  nn={'W' if nn_is_white else 'B'}  "
                        f"{result}  ({termination})",
                        flush=True,
                    )

            played = wins + draws + losses
            record = {
                "kind": "sf_bench",
                "ts": time.time(),
                "iso": datetime.now(timezone.utc).isoformat(),
                "checkpoint": ckpt.name,
                "tag": tag,
                "regime": "scratch",
                "sims": args.sims,
                "max_plies": args.max_plies,
                "preset": preset_name,
                "skill": preset["skill"],
                "depth": preset["depth"],
                "games": played,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "score_rate": round((wins + 0.5 * draws) / played, 4) if played else None,
                "model_config": {"blocks": model_config.num_blocks, "filters": model_config.filters},
                "games_detail": games_detail,
                "wall_seconds": round(time.time() - t0, 1),
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            print(
                f"[bench] == {preset_name}: {wins}W-{draws}D-{losses}L  "
                f"score {record['score_rate']}  ({round(time.time() - t0)}s)",
                flush=True,
            )
    finally:
        engine.quit()
        print(f"[bench] done in {round(time.time() - overall)}s -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
