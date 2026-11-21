"""Validate FS-6b live-stream events in a train_log.jsonl journal.

Diagnostic CLI (prints, like bench_train.py): parses every line, checks the
``live``/``game_end`` key contracts, replays each streamed game move-by-move
from startpos (FENs must match exactly), asserts game-slot uniqueness, and
reports the observed live-event rate.

Usage:
    python scripts/verify_journal_stream.py [path/to/train_log.jsonl]

Defaults to ``data/train_log.jsonl``. Exit code 1 on any contract violation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import chess

LIVE_KEYS = {"kind", "run_id", "regime", "game", "ply", "move", "fen", "ts"}
END_KEYS = {"kind", "run_id", "regime", "game", "result", "plies", "ts"}
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}


def verify(path: Path) -> int:
    """Print a contract report for `path`; return a process exit code."""
    lines = path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]

    kinds: dict[str, int] = {}
    for event in events:
        kinds[event["kind"]] = kinds.get(event["kind"], 0) + 1
    print(f"{path}: {len(events)} events, kinds={kinds}")

    live = [e for e in events if e["kind"] == "live"]
    ends = [e for e in events if e["kind"] == "game_end"]

    failures: list[str] = []
    for event in live:
        missing = LIVE_KEYS - set(event)
        if missing:
            failures.append(f"live event missing keys {sorted(missing)}")
    for event in ends:
        missing = END_KEYS - set(event)
        if missing:
            failures.append(f"game_end event missing keys {sorted(missing)}")
        if event.get("result") not in VALID_RESULTS:
            failures.append(f"game_end bad result {event.get('result')!r}")

    end_slots = [e["game"] for e in ends]
    if len(end_slots) != len(set(end_slots)):
        failures.append(f"duplicate game_end slots: {sorted(end_slots)}")

    for slot in sorted({e["game"] for e in live + ends}):
        board = chess.Board()
        game_events = sorted((e for e in live if e["game"] == slot), key=lambda e: e["ply"])
        for expected_ply, event in enumerate(game_events, start=1):
            if event["ply"] != expected_ply:
                failures.append(f"slot {slot}: ply {event['ply']} out of sequence")
                break
            try:
                board.push_uci(event["move"])
            except ValueError:
                failures.append(f"slot {slot} ply {event['ply']}: illegal uci {event['move']!r}")
                break
            if board.fen() != event["fen"]:
                failures.append(f"slot {slot} ply {event['ply']}: FEN mismatch")
                break
        else:
            end = next((e for e in ends if e["game"] == slot), None)
            if end is not None and len(game_events) != end["plies"]:
                failures.append(
                    f"slot {slot}: {len(game_events)} live events != plies {end['plies']}"
                )

    span = max((e["ts"] for e in live), default=0.0) - min((e["ts"] for e in live), default=0.0)
    rate = len(live) / span if span > 0 else 0.0
    print(
        f"live={len(live)} game_end={len(ends)} | "
        f"rate {rate:.1f} live events/sec over a {span:.2f}s collection window"
    )

    if failures:
        for failure in failures[:20]:
            print(f"FAIL: {failure}")
        return 1
    print("OK: every streamed game replays exactly; contracts hold")
    return 0


def main() -> None:
    journal_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train_log.jsonl")
    if not journal_path.exists():
        print(f"journal not found: {journal_path}")
        raise SystemExit(2)
    raise SystemExit(verify(journal_path))


if __name__ == "__main__":
    main()
