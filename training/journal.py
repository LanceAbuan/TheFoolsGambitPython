"""Training journal (FS-6): structured events on ``data/train_log.jsonl``.

One JSON object per line, appended at every training iteration and every
arena gate; the backend tails this file on ``train``-topic WS subscribes
(FS-6's backend half reads the same contract). Events are append-only and
self-describing — every line carries ``kind`` and ``ts`` so the tailer can
route without knowing the writer.

Line contracts (superset of the FS-6 required keys — iteration, loss,
policy_loss, value_loss, elo, arena_result, ts):

    {"kind": "iteration", "iteration": int, "loss": float|null,
     "policy_loss": float|null, "value_loss": float|null, "elo": float|null,
     "arena_result": null, "games": int, "buffer_size": int,
     "games_per_hour": float|null, "regime": str, "run_id": str, "ts": float}

    {"kind": "arena", "iteration": int, "checkpoint": str, "opponent": str,
     "games": int, "wins": int, "draws": int, "losses": int,
     "arena_result": {"win_rate": float, "promoted": bool},
     "elo": float|null, "regime": str, "run_id": str, "ts": float}
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

JOURNAL_FILENAME = "train_log.jsonl"


class TrainingJournal:
    """Append-only JSONL event log with one object per line."""

    def __init__(self, path: Path | str = Path("data") / JOURNAL_FILENAME) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: Mapping[str, Any]) -> None:
        """Append one event line; `ts` is stamped here when absent."""
        record = dict(event)
        record.setdefault("ts", time.time())
        line = json.dumps(record, sort_keys=True, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        log.debug("[JOURNAL] %s: %s", self.path, record.get("kind"))

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Parse the journal; corrupt lines are skipped (reader-side helper)."""
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                log.warning("[JOURNAL] skipping corrupt line in %s", self.path)
                continue
            if isinstance(data, dict):
                events.append(data)
        return events[-limit:] if limit is not None else events
