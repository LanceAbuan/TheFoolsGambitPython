"""Live-board feeder for the Rust self-play path (WP-H flip companion).

Rust workers publish FINISHED games only, so the /watch page's per-move
journal stream goes quiet on the rust_workers backend. This script replays
the newest finished spool game onto the journal (data/train_log.jsonl,
kind="live") at a human pace, then follows new games as they land - so the
Watch page always shows a real, per-move self-play game.

Record shape mirrors the trainer's own live stream exactly:
  {"fen": <position AFTER the move>, "game": N, "kind": "live",
   "move": <uci>, "ply": i+1, "regime": ..., "run_id": ..., "ts": now}

Usage:
  .venv/Scripts/python.exe scripts/replay_latest_game.py \
      --processed runs/scratch/second-light-v2/spool/inbox/processed \
      --journal data/train_log.jsonl --run-id second-light-v2 --pacing 2
  --once  replay the newest finished game once and exit (smoke test)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def list_games(processed: Path) -> list[Path]:
    return sorted(processed.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)


def replay(path: Path, journal: Path, run_id: str, pacing: float, game_no: int) -> int:
    try:
        with open(path, encoding="utf-8") as fh:
            rec = json.loads(fh.readline())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[replay] skip {path.name}: {exc}", flush=True)
        return game_no

    fens = rec.get("fens") or []
    ucis = rec.get("ucis") or []
    if not fens or len(fens) != len(ucis):
        print(f"[replay] skip {path.name}: bad fens/ucis", flush=True)
        return game_no

    print(
        f"[replay] game {game_no}: {path.name} — {len(fens)} plies, "
        f"result {rec.get('result')} ({rec.get('termination')}), "
        f"weights v{rec.get('weight_version')}",
        flush=True,
    )
    with open(journal, "a", encoding="utf-8") as out:
        for i, (fen, uci) in enumerate(zip(fens, ucis)):
            out.write(json.dumps({
                "fen": fen,
                "game": game_no,
                "kind": "live",
                "move": uci,
                "ply": i + 1,
                "regime": "scratch",
                "run_id": run_id,
                "ts": time.time(),
            }) + "\n")
            out.flush()
            if pacing > 0:
                time.sleep(pacing)
    return game_no + 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay newest rust self-play game onto the journal")
    parser.add_argument("--processed", default="runs/scratch/second-light-v2/spool/inbox/processed")
    parser.add_argument("--journal", default="data/train_log.jsonl")
    parser.add_argument("--run-id", default="second-light-v2")
    parser.add_argument("--pacing", type=float, default=2.0, help="seconds per ply")
    parser.add_argument("--poll", type=float, default=5.0, help="seconds between spool checks")
    parser.add_argument("--once", action="store_true", help="replay newest finished game once, then exit")
    args = parser.parse_args()

    processed = Path(args.processed)
    journal = Path(args.journal)
    if not processed.is_dir():
        print(f"[replay] processed dir missing: {processed}", file=sys.stderr)
        return 1

    seen: set[str] = set()
    game_no = 1000
    replayed_any = False
    # Startup: skip the backlog, start from the newest finished game.
    backlog = list_games(processed)
    if backlog:
        seen.update(g.name for g in backlog[:-1])
    while True:
        games = list_games(processed)
        fresh = [g for g in games if g.name not in seen]
        if fresh:
            for game in fresh:
                seen.add(game.name)
                game_no = replay(game, journal, args.run_id, args.pacing, game_no)
                replayed_any = True
                if args.once:
                    return 0
        elif args.once and not replayed_any:
            print("[replay] no finished games in the spool yet", file=sys.stderr)
            return 1
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
