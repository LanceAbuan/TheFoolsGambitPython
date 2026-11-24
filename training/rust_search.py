"""WP-F C3: Python side of the Rust search backend (``gambit_core.search``).

``RustMCTSSearch`` is a drop-in replacement for
``training.selfplay.MCTSSearch``: same ``search(board) -> MCTSResult``
contract, same PUCT/noise/terminal semantics (the Rust core is a bit-faithful
port), but the tree walk runs in Rust under ``py.detach`` with the GIL only
re-acquired for the batched NN callback.

Evaluator protocol is UNCHANGED: ``evaluate(boards) -> (logits, values)``.
One callback fires per expansion wave with the wave leaves batched - the
mirror of how ``BatchNNEvaluator`` aggregates. ``batch_size=1`` (the default)
reduces to python-sequential semantics exactly, which is the C6 parity mode.

Known deviation (documented, WP-H closes it): leaf positions cross the bridge
as FENs, so a reconstructed leaf board has NO move stack. Encoder v1 is
unaffected (``encode_batch`` never reads the stack). Encoder v2 leaf rows see
zeroed history planes. The ROOT evaluation uses the fully replayed board with
its stack intact, so root rows are bit-identical to the python path for both
encoders. Keep ``search_backend="python"`` for v2 profiles until WP-H.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import chess
import numpy as np

from .encoder import legal_move_map
from .model import NUM_POLICY_MOVES_V2
from .selfplay import MCTSResult, SearchConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .selfplay import Evaluator

try:  # pragma: no cover - exercised only on machines without the wheel
    import gambit_core as _gambit_core
except ImportError:  # pragma: no cover
    _gambit_core = None

RUST_SEARCH_AVAILABLE = _gambit_core is not None and hasattr(_gambit_core, "search")

log = logging.getLogger(__name__)

__all__ = ["RUST_SEARCH_AVAILABLE", "RustMCTSSearch", "raw_fen"]


def raw_fen(board: chess.Board) -> str:
    """python-chess FEN with the RAW en-passant field re-injected.

    python-chess FEN output drops ep squares no enemy pawn can capture, while
    ``board.ep_square`` (and the Rust position state) carry the raw value.
    Re-injecting keeps the FEN handed to Rust exactly the state python sees -
    the same convention as the differential-fuzz helper in
    ``tests/test_gambit_core.py``.
    """
    parts = board.fen().split()
    if board.ep_square is not None:
        parts[3] = chess.SQUARE_NAMES[board.ep_square]
    return " ".join(parts)


def _history_spec(board: chess.Board) -> tuple[str, list[str]]:
    """(game root FEN, UCI move stack) - the rust board/history contract.

    The root carries the raw ep field; the move stack lets the Rust side
    replay the game so repetition and fifty-move claims match
    ``board.is_game_over(claim_draw=True)``.
    """
    root = board.root()
    moves = [move.uci() for move in board.move_stack]
    return raw_fen(root), moves


def _root_fen_of(start_fen: str, moves: list[str]) -> str:
    """FEN of the position reached by replaying ``moves`` from ``start_fen``.

    Only used to IDENTIFY the root call inside the callback; the board handed
    to the evaluator is the original (stacked) python board, not this replay.
    """
    replay = chess.Board(start_fen)
    for uci in moves:
        replay.push_uci(uci)
    return raw_fen(replay)


class RustMCTSSearch:
    """PUCT MCTS executed by ``gambit_core.search`` (WP-F C3 bridge)."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: SearchConfig | None = None,
        rng: np.random.Generator | None = None,
        batch_size: int = 1,
    ) -> None:
        if config is None:
            config = SearchConfig()
        if not RUST_SEARCH_AVAILABLE:
            raise RuntimeError(
                "rust search backend unavailable: rebuild gambit_core with the C3 "
                "search pyfunction and install the wheel (maturin build --release)"
            )
        if config.search_kind != "puct":
            raise ValueError(
                "rust backend implements the PUCT root policy only, got "
                f"search_kind={config.search_kind!r}"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.evaluator = evaluator
        self.config = config
        self._rng = rng
        self.batch_size = batch_size
        # Per-search root state for the callback (see _eval_callback).
        self._root_fen: str | None = None
        self._root_board: chess.Board | None = None

    def search(self, board: chess.Board, num_simulations: int | None = None) -> MCTSResult:
        """Search ``board``; same result contract as ``MCTSSearch.search``."""
        sims = num_simulations if num_simulations is not None else self.config.num_simulations
        cfg = _gambit_core.SearchConfig(
            cpuct=self.config.cpuct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            root_noise=self.config.root_noise,
            num_simulations=sims,
            batch_size=self.batch_size,
            seed=self._next_seed(),
            forced_playouts=self.config.forced_playouts,
            fp_k=self.config.fp_k,
        )
        start_fen, moves = _history_spec(board)
        # Keep the stacked board for the root evaluation callback (v2 encoder
        # history planes); leaves are reconstructed from FENs.
        self._root_fen = _root_fen_of(start_fen, moves)
        self._root_board = board
        counts, root_value = _gambit_core.search((start_fen, moves), self._eval_callback, cfg)
        counts = np.asarray(counts, dtype=np.float32)
        if counts.shape != (NUM_POLICY_MOVES_V2,):
            raise RuntimeError(
                f"rust search returned visit counts of shape {counts.shape}, "
                f"expected ({NUM_POLICY_MOVES_V2},)"
            )
        return self._to_result(board, counts, float(root_value))

    def _next_seed(self) -> int:
        """Root Dirichlet seed: derived from the run rng when provided."""
        if self._rng is not None:
            return int(self._rng.integers(0, 2**63 - 1))
        return 0

    def _eval_callback(self, fens: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Bridge callback: FENs -> evaluator boards -> (logits, values).

        Exactly one position per call in the default sequential mode, one
        wave batch otherwise. The root evaluation (always a standalone
        single-position call) reuses the original stacked board.
        """
        boards: list[chess.Board]
        if len(fens) == 1 and self._root_fen is not None and fens[0] == self._root_fen:
            boards = [self._root_board]  # type: ignore[list-item]
        else:
            boards = [chess.Board(fen) for fen in fens]
        return self.evaluator.evaluate(boards)

    def _to_result(self, board: chess.Board, counts: np.ndarray, root_value: float) -> MCTSResult:
        """Derive the MCTSResult fields the 2-tuple contract does not carry.

        ``chosen_move`` uses the python tie-break (first child in
        python-chess legal-move order on visit-count ties - ``max`` over the
        insertion-ordered legal map). A zero-count export means the root was
        never searched (terminal or claimable-draw root), matching the python
        ``visited_root=False`` + ``chosen_move=None`` short circuit. (A
        0-simulation search is the one case that misreports as unvisited -
        no pipeline caller issues one.)
        """
        if counts.sum() <= 0.0:
            return MCTSResult(counts, root_value, None, False)
        legal_map = legal_move_map(board)
        best_index = max(legal_map, key=lambda idx: counts[idx])
        return MCTSResult(counts, root_value, legal_map[best_index], True)
