"""WP8: Syzygy tablebase integration (perfect endgame labels).

With the 3-4-5 WDL tables present, any position with <= 5 pieces gets a
PERFECT value target from the tablebase instead of the game outcome - the
strongest available fix for endgame label noise. Without tables the corrector
is silently disabled (training continues unchanged).

Fetching the tables (one-time, ~1 GB, hosted by Lichess):
    https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/
    (point --tb-path at the directory containing the .rtbw files)

WDL probe mapping (mover perspective):
    2 (win)          -> +1.0
    1 (cursed win)   -> +0.5
    0 (draw)         ->  0.0
   -1 (blessed loss) -> -0.5
   -2 (loss)         -> -1.0
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

MAX_TB_PIECES = 5  # 3-4-5 WDL tables


class TablebaseCorrector:
    """Probe up to 5-man positions; `None` when unavailable/out of range."""

    def __init__(self, path: str | Path | None) -> None:
        self.tablebase = None
        self.enabled = False
        if not path:
            return
        try:
            import chess.syzygy

            directory = Path(path)
            if not directory.is_dir():
                log.warning("[TB] %s is not a directory - tablebase disabled", directory)
                return
            self.tablebase = chess.syzygy.open_tablebase(str(directory))
            self.enabled = True
            log.info("[TB] Syzygy WDL tables loaded from %s", directory)
        except Exception as exc:  # noqa: BLE001 - never kill training
            log.warning("[TB] tablebase unavailable (%s) - continuing without", exc)

    def value_for(self, board) -> float | None:
        """Perfect WDL value from the side-to-move's perspective, or None."""
        if not self.enabled or board is None:
            return None
        if board.is_game_over(claim_draw=True):
            return None
        if chess_piece_count(board) > MAX_TB_PIECES:
            return None
        try:
            wdl = self.tablebase.probe_wdl(board)
        except Exception:  # noqa: BLE001 - missing table for this material
            return None
        return {2: 1.0, 1: 0.5, 0: 0.0, -1: -0.5, -2: -1.0}.get(int(wdl))

    def close(self) -> None:
        if self.tablebase is not None:
            self.tablebase.close()
            self.tablebase = None
            self.enabled = False


def chess_piece_count(board) -> int:
    return int(board.occupied.bit_count())
