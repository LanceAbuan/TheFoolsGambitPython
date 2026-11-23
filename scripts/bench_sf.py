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
import itertools
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Mirrors server/stockfish.py presets plus extended rungs for the ladder.
# Ordered weak -> strong. "max" = strongest SF we have (skill 20, deep but
# one thread) — the anchor rung. The bot's Elo is only measurable where it
# scores ~50% vs some rung, so the ladder must bracket the bot from below.
SF_BINARY = Path("tools/stockfish.exe")
# Node-limited rungs (fishtest-style): bounded per-move time, reproducible.
# The server presets keep their depth semantics for humans; these rungs are
# the bench's own reproducible ladder. Ordered weak -> strong; "max" anchors.
RUNGS: dict[str, dict[str, int]] = {
    "floor":       {"skill": 0,  "nodes": 100},
    "wandering":   {"skill": 2,  "nodes": 300},
    "s4":          {"skill": 4,  "nodes": 1_000},
    "curious":     {"skill": 6,  "nodes": 5_000},
    "s9":          {"skill": 9,  "nodes": 20_000},
    "enlightened": {"skill": 12, "nodes": 80_000},
    "s16":         {"skill": 16, "nodes": 300_000},
    "max":         {"skill": 20, "nodes": 1_200_000},
}
PRESETS = RUNGS  # back-compat alias
LADDER_ORDER = ["floor", "wandering", "s4", "curious", "s9", "enlightened", "s16", "max"]
# Elo anchor assumption (documented, not measured): 1-thread 1.2M-node SF.
ANCHOR_ELO = 3400.0
CALIBRATION_PATH = Path("data/sf_calibration.json")
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
    import chess
    import numpy as np

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


# Curated real openings (FB-3): the bot only ever trains from the startpos, so
# random 6-ply openings measure it in alien territory. These are mainstream
# positions after 2-4 natural plies.
BOOK_OPENINGS: list[str] = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq -",   # Italian
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -",      # Sicilian
    "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq -",        # French
    "rnbqkbnr/pp1ppppp/2p5/8/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq -",      # Caro-Kann
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq -",    # Closed Sicilian
    "rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR b KQkq -",        # Scandinavian
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",        # Open game
    "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -",   # Italian, ...Bf6
]


def book_openings(count: int, seed: int) -> list[str]:
    import numpy as np

    idx = np.random.default_rng(seed).permutation(len(BOOK_OPENINGS))[:count]
    return [BOOK_OPENINGS[i] for i in idx]


def play_game(evaluator, opening_fen: str, nn_is_white: bool, engine, nodes: int, sims: int, max_plies: int):
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
            move = engine.play(board, chess.engine.Limit(nodes=nodes)).move
        assert move is not None, "engine returned no move on a live position"
        board.push(move)
    result = board.result(claim_draw=True)
    if result == "*":
        result = "1/2-1/2"  # capped games score as draws (arena aggregation)
    return result, termination


def main() -> int:
    parser = argparse.ArgumentParser(description="Champion NN vs Stockfish ladder benchmark")
    parser.add_argument("--mode", default="bench", choices=["bench", "calibrate", "fit"],
                        help="bench: NN vs rungs; calibrate: SF vs SF rung gaps; fit: Elo from results")
    parser.add_argument("--presets", default="wandering,curious,enlightened")
    parser.add_argument("--games-per-preset", type=int, default=12)
    parser.add_argument("--games-per-pair", type=int, default=20, help="calibrate mode")
    parser.add_argument("--sims", type=int, default=128, help="NN simulations (gate uses 128)")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--checkpoint", default=None, help="override champion resolution")
    parser.add_argument("--out", default="data/sf_bench.jsonl")
    parser.add_argument("--openings", default="random", choices=["random", "book"],
                        help="random: seeded random 6-ply positions; book: curated real openings (FB-3)")
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    if args.mode == "calibrate":
        return run_calibrate(args)
    if args.mode == "fit":
        return run_fit(args)
    return run_bench(args)


def run_bench(args) -> int:
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
            openings = (
                book_openings(n // 2, args.seed + abs(hash(preset_name)) % 1000)
                if args.openings == "book"
                else random_openings(n // 2, args.seed + abs(hash(preset_name)) % 1000)
            )
            engine.configure({**SF_OPTIONS, "Skill Level": preset["skill"]})

            wins = draws = losses = 0
            games_detail: list[dict] = []
            t0 = time.time()
            for opening_fen in openings:
                for nn_is_white in (True, False):
                    result, termination = play_game(
                        evaluator, opening_fen, nn_is_white, engine,
                        preset["nodes"], args.sims, args.max_plies,
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
                "iso": datetime.now(UTC).isoformat(),
                "checkpoint": ckpt.name,
                "tag": tag,
                "regime": "scratch",
                "sims": args.sims,
                "max_plies": args.max_plies,
                "preset": preset_name,
                "skill": preset["skill"],
                "nodes": preset["nodes"],
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


def _sf_move(engine, board, nodes):
    import chess.engine

    return engine.play(board, chess.engine.Limit(nodes=nodes)).move


def sf_vs_sf(engines, white_rung, black_rung, opening_fen, max_plies):
    """One SF-vs-SF game; result from white's perspective ('1-0'/'0-1'/'1/2-1/2')."""
    import chess

    board = chess.Board(fen=opening_fen)
    for _ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        turn = white_rung if board.turn == chess.WHITE else black_rung
        engines[turn].configure({"Skill Level": RUNGS[turn]["skill"]})
        board.push(_sf_move(engines[turn], board, RUNGS[turn]["nodes"]))
    result = board.result(claim_draw=True)
    return "1/2-1/2" if result == "*" else result


def elo_gap(score: float) -> float:
    """Elo difference implied by a score in [0,1] (clamped away from 0/1)."""
    import math

    s = min(max(score, 0.02), 0.98)
    return -400.0 * math.log10(1.0 / s - 1.0)


def run_calibrate(args) -> int:
    """Adjacent-rung SF-vs-SF matches -> per-rung Elo ratings anchored at the top."""
    import chess.engine

    if not SF_BINARY.is_file():
        print(f"[cal] SF binary missing: {SF_BINARY}", file=sys.stderr)
        return 1
    engine = chess.engine.SimpleEngine.popen_uci(str(SF_BINARY))
    engine.configure({**SF_OPTIONS})
    pairs: list[dict] = []
    try:
        for lo, hi in itertools.pairwise(LADDER_ORDER):
            n = args.games_per_pair + (args.games_per_pair % 2)
            openings = random_openings(n // 2, args.seed + abs(hash(lo + ">" + hi)) % 1000)
            wins_hi = draws = 0
            for fen in openings:
                for white, black in ((hi, lo), (lo, hi)):
                    result = sf_vs_sf({white: engine, black: engine}, white, black, fen, args.max_plies)
                    hi_won = (result == "1-0") == (white == hi)
                    if result == "1/2-1/2":
                        draws += 1
                    elif hi_won:
                        wins_hi += 1
                    print(f"[cal] {hi} vs {lo}: {result}", flush=True)
            played = wins_hi + draws + (n - wins_hi - draws)
            score_hi = (wins_hi + 0.5 * draws) / played
            gap = elo_gap(score_hi)  # hi above lo by `gap`
            pairs.append({"lo": lo, "hi": hi, "games": played, "score_hi": round(score_hi, 4),
                          "elo_gap": round(gap, 1)})
            print(f"[cal] == {hi} - {lo}: score {score_hi:.3f} -> gap {gap:.0f}", flush=True)
    finally:
        engine.quit()

    ratings: dict[str, float] = {LADDER_ORDER[-1]: ANCHOR_ELO}
    for pair in reversed(pairs):
        ratings[pair["lo"]] = ratings[pair["hi"]] - pair["elo_gap"]
    payload = {
        "kind": "sf_calibration", "ts": time.time(),
        "iso": datetime.now(UTC).isoformat(),
        "anchor": {"rung": LADDER_ORDER[-1], "elo": ANCHOR_ELO},
        "pairs": pairs, "ratings": {k: round(v, 1) for k, v in ratings.items()},
    }
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[cal] ratings: {payload['ratings']}", flush=True)
    print(f"[cal] wrote {CALIBRATION_PATH}", flush=True)
    return 0


def run_fit(args) -> int:
    """Fit the bot's Elo by MLE from sf_bench.jsonl results + ladder calibration."""
    import math

    cal = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    ratings = cal["ratings"]

    games: list[tuple[float, float]] = []  # (score, opponent_rating)
    for line in (ROOT / args.out).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") != "sf_bench" or rec.get("preset") not in ratings:
            continue
        opp = ratings[rec["preset"]]
        played = rec["games"]
        if not played:
            continue
        games.append((rec["wins"] + 0.5 * rec["draws"], played, opp))

    def neg_ll(r_bot: float) -> float:
        ll = 0.0
        for score_sum, n, opp in games:
            expected = 1.0 / (1.0 + 10 ** ((opp - r_bot) / 400.0))
            ll += score_sum * math.log(expected) + (n - score_sum) * math.log(1 - expected)
        return -ll

    r = 800.0
    step = 50.0
    for _ in range(60):
        lo, hi = neg_ll(r - step), neg_ll(r + step)
        if neg_ll(r) <= lo and neg_ll(r) <= hi:
            break
        r = r + step if hi < lo else r - step
        step *= 0.98

    total_score = sum(g[0] for g in games)
    total_n = sum(g[1] for g in games)
    if total_score == 0.0:
        bound = min(g[2] for g in games) - elo_gap(0.02)
        print(f"[fit] 0/{total_n} scored — point estimate undefined (NLL saturated).")
        print(f"[fit] honest bound: bot Elo <= {bound:.0f} (clamp limit vs strongest rung played)")
        return 0
    if total_score == total_n:
        bound = max(g[2] for g in games) + elo_gap(0.98)
        print(f"[fit] {total_n}/{total_n} scored — point estimate undefined (NLL saturated).")
        print(f"[fit] honest bound: bot Elo >= {bound:.0f}")
        return 0

    print(f"[fit] games used: {len(games)} across rungs {sorted({g[2] for g in games})}")
    print(f"[fit] fitted bot Elo: {r:.0f}  (anchor: {cal['anchor']['rung']} = {cal['anchor']['elo']})")
    for score_sum, n, opp in games:
        expected = 1.0 / (1.0 + 10 ** ((opp - r) / 400.0))
        print(f"[fit]   vs opp {opp:.0f}: actual {score_sum}/{n} = {score_sum / n:.3f}, expected {expected:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
