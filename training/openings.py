"""Seeded opening-position sampling for fair arena gates (FB-1).

Before FB-1 every gate game started from the initial position with
deterministic-strength play (no root noise, argmax moves), so a 16-game
match was the SAME game played out twice — one per color — and the gate
decided promotion on an effective sample size of ~2 (observed: gate 22's
8W/8D/0L). The fix: every gate event seeds a FRESH opening set from
``seed = f(run_seed, gate_index)``; games are paired so each opening is
played twice with colors swapped (candidate-as-white and candidate-as-black),
and the played FENs are logged in the gate record for reproducibility.

Determinism contract: for a fixed (run_seed, gate_index) the opening set is
byte-identical across processes and resumes — the seed derivation uses
SHA-256 (Python's salted ``hash()`` is process-random and would not
reproduce), and the playout RNG is a fresh ``np.random.default_rng`` per
position, mirroring ``scripts/bench_sf.random_openings``.
"""
from __future__ import annotations

import hashlib
import logging

import chess
import numpy as np

log = logging.getLogger(__name__)

OPENING_PLIES = 6  # half-moves of random play (3 per side), as in the SF bench
_MAX_OPENING_ATTEMPTS_FACTOR = 200  # safety bound on unique-FEN retries


def gate_opening_seed(run_seed: int | None, gate_index: int) -> int:
    """Stable per-event seed: f(run_seed, gate_index) (FB-1).

    Pure function of the run's seed and the gate event's absolute index, so a
    resumed run reproduces the exact opening set of the gate it stopped at.
    ``run_seed=None`` (unseeded profiles) hashes its repr — still deterministic
    per event, shared across seedless runs.
    """
    payload = f"gambit-gate:{run_seed!s}:{gate_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def random_openings(count: int, seed: int, plies: int = OPENING_PLIES) -> list[str]:
    """`count` unique opening-position FENs via seeded random play.

    Each candidate position replays `plies` uniformly random half-moves from
    the initial position under its own fresh ``default_rng`` (seed, seed+1, ...
    — the bench_sf pattern). Duplicates are discarded; the retry bound makes
    termination unconditional even at pathological collision rates.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return []
    seen: set[str] = set()
    out: list[str] = []
    attempts = 0
    max_attempts = count * _MAX_OPENING_ATTEMPTS_FACTOR
    while len(out) < count:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"random_openings could not find {count} unique positions "
                f"after {attempts} attempts ({plies} plies each)"
            )
        attempts += 1
        rng_seed = seed + attempts
        board = chess.Board()
        rng = np.random.default_rng(rng_seed)
        for _ in range(plies):
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
        fen = board.fen()
        if fen not in seen:
            seen.add(fen)
            out.append(fen)
    return out


def sample_gate_openings(games: int, seed: int) -> tuple[str, ...]:
    """One opening FEN per gate game, mirrored across color pairs (FB-1).

    ``games`` must be a positive even count (the arena already forces even).
    Game 2k (candidate as white) and game 2k+1 (candidate as black) play the
    SAME opening, so opening strength cancels across the pair and colors stay
    balanced — the fair-match pairing used by the SF bench.
    """
    if games < 2 or games % 2 != 0:
        raise ValueError(f"games must be a positive even count, got {games}")
    half = random_openings(games // 2, seed)
    return tuple(fen for fen in half for _ in range(2))
