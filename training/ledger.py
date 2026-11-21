"""Arena ledger (JSONL) + metrics feed writer (plan §5.1 / §14.0 contract).

`data/arena.jsonl` — one line per arena match:
    {run_id, regime, checkpoint, opponent, games, wins, draws, elo, ...}

`data/metrics.json` — flat dict consumed by the backend's `GET /api/metrics`
(server/api/train.py merges it over {elo, loss, win_rate, games_played}).
The writer is read-modify-write with an atomic replace so a concurrent reader
never sees a torn file.

Per-checkpoint Elo is recomputed from the ledger itself (§5.1: "per-checkpoint
Elo computed from the ledger") with a simple symmetric K-factor update per
match. SPRT/uncertainty bounds are optional later per §5.1.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

LEDGER_FILENAME = "arena.jsonl"
METRICS_FILENAME = "metrics.json"


@dataclass(frozen=True)
class LedgerEntry:
    """One arena result line. Required keys per §5.1 + provenance extras.

    `kind` mirrors the SQLite `games.kind` column (§5.1): "arena" for a real
    candidate-vs-champion match, "baseline" for the first-checkpoint seed row
    (no games played). Slice 2's SQLite migration inserts these rows 1:1 with
    `kind='arena'`.
    """

    run_id: str
    regime: str
    checkpoint: str
    opponent: str
    games: int
    wins: int
    draws: int
    elo: float
    kind: str = "arena"
    losses: int = 0
    promoted: bool = False
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ArenaLedger:
    """Append-only JSONL arena ledger with a reader for Elo recomputation."""

    def __init__(self, path: Path | str = Path("data") / LEDGER_FILENAME) -> None:
        self.path = Path(path)

    def append(self, entry: LedgerEntry) -> None:
        """Append one match line (parent dirs created as needed)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry.to_dict(), sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        log.info("[LEDGER] %s: %s vs %s -> elo %.1f", self.path, entry.checkpoint, entry.opponent, entry.elo)

    def read(self) -> list[dict[str, object]]:
        """All recorded entries in order; unreadable/corrupt lines are skipped."""
        if not self.path.exists():
            return []
        entries: list[dict[str, object]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                log.warning("[LEDGER] skipping corrupt line in %s", self.path)
                continue
            if isinstance(data, dict):
                entries.append(data)
        return entries


def recompute_elos(
    entries: list[dict[str, object]],
    base_elo: float = 800.0,
    k_factor: float = 32.0,
) -> dict[str, float]:
    """Replay the ledger chronologically and return per-checkpoint Elo.

    A checkpoint's first appearance starts at `base_elo`; each match applies a
    symmetric K-factor update from the recorded score rate.
    """
    elos: dict[str, float] = {}
    for entry in entries:
        candidate = str(entry.get("checkpoint", ""))
        opponent = str(entry.get("opponent", ""))
        games = int(entry.get("games", 0) or 0)
        if not candidate or games <= 0 or opponent in ("none", ""):
            if candidate and candidate not in elos:
                elos[candidate] = base_elo
            continue
        wins = float(entry.get("wins", 0) or 0)
        draws = float(entry.get("draws", 0) or 0)
        rate = (wins + 0.5 * draws) / games
        cand_elo = elos.get(candidate, base_elo)
        opp_elo = elos.get(opponent, base_elo)
        expected = 1.0 / (1.0 + 10.0 ** ((opp_elo - cand_elo) / 400.0))
        delta = k_factor * (rate - expected)
        elos[candidate] = cand_elo + delta
        elos[opponent] = opp_elo - delta
    return elos


class MetricsWriter:
    """Read-modify-write the flat metrics dict the backend feeds on.

    Required keys per the pinned contract: `elo`, `loss`, `win_rate`,
    `games_played`. Extra keys (per-checkpoint Elo map, champion tag, regime)
    pass through untouched by the server merge.
    """

    def __init__(self, path: Path | str = Path("data") / METRICS_FILENAME) -> None:
        self.path = Path(path)

    def _read_existing(self) -> dict[str, object]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def update(self, **metrics: object) -> dict[str, object]:
        """Merge `metrics` into the file atomically; returns the new content."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged = self._read_existing()
        merged.update({k: v for k, v in metrics.items() if v is not None})
        merged["updated_at"] = time.time()
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.path)
        log.info("[METRICS] %s updated: elo=%s win_rate=%s games=%s",
                 self.path, merged.get("elo"), merged.get("win_rate"), merged.get("games_played"))
        return merged

    def read(self) -> dict[str, object]:
        return self._read_existing()
