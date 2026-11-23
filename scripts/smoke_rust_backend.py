"""WP-F C6 gate (c): end-to-end smoke-profile training run with backend=rust.

Runs the canonical SMOKE_PROFILE with ``search_backend="rust"`` (self-play AND
arena searches go through ``gambit_core.search``), writing EVERYTHING inside
the repo worktree (``tmp_e2e/runs``, ``tmp_e2e/data``) so the live training
run and the shared ``runs/`` / ``data/`` trees are never touched.

Assertions:
- the run completes and plays > 0 games;
- every game replays as legal chess from its PGN;
- training examples exist and the run summary reports steps > 0.

Usage:
    python scripts/smoke_rust_backend.py
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import chess

sys_path = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(sys_path))

from training.pipeline import SMOKE_PROFILE, run_training
from training.rust_search import RUST_SEARCH_AVAILABLE

log = logging.getLogger(__name__)


def _assert_legal_journal(journal_path: Path, run_id: str) -> int:
    """Replay every journaled live move of `run_id`; assert legality + FEN.

    The v2 pipeline journals one ``live`` event per ply (move + resulting
    FEN) instead of writing PGNs, so legality of the rust-backend games is
    proven against the journal: each move must be legal in the replayed
    position and the recorded FEN must match exactly.
    """
    boards: dict[int, chess.Board] = {}
    total_moves = 0
    with journal_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("kind") != "live" or event.get("run_id") != run_id:
                continue
            slot = int(event["game"])
            board = boards.setdefault(slot, chess.Board())
            move = chess.Move.from_uci(event["move"])
            if move not in board.legal_moves:
                raise AssertionError(
                    f"illegal journaled move {event["move"]} in game slot {slot}"
                )
            board.push(move)
            if event.get("fen") and board.fen() != event["fen"]:
                raise AssertionError(f"FEN drift in game slot {slot} after {event["move"]}")
            total_moves += 1
    return total_moves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep tmp_e2e artifacts")
    args = parser.parse_args()

    if not RUST_SEARCH_AVAILABLE:
        raise RuntimeError(
            "gambit_core.search unavailable: build the wheel (maturin build --release) "
            "and install it into this environment first"
        )

    profile = replace(SMOKE_PROFILE, search_backend="rust", name="smoke-rust")
    run_id = "smoke-rust-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base_dir = sys_path / "tmp_e2e" / "runs"
    data_dir = sys_path / "tmp_e2e" / "data"

    log.info("[E2E] run_id=%s base_dir=%s", run_id, base_dir)
    summary = run_training(profile, run_id=run_id, base_dir=base_dir, data_dir=data_dir)

    assert summary.games_played > 0, "no games were played"
    assert summary.steps > 0, "no training steps ran"
    total_moves = _assert_legal_journal(data_dir / "train_log.jsonl", run_id)
    assert total_moves > 0, "no legal moves across all games"

    print()
    print("[E2E] rust-backend smoke run GREEN")
    print(f"  journal moves: {total_moves} legal moves replayed")
    print(f"  games played : {summary.games_played}")
    print(f"  arena games  : {summary.arena_games}")
    print(f"  steps        : {summary.steps}  final loss: {summary.final_loss}")
    if not args.keep:
        import shutil

        shutil.rmtree(sys_path / "tmp_e2e", ignore_errors=True)
        print("  artifacts    : cleaned (pass --keep to inspect)")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
