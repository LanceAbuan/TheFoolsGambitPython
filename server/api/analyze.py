"""Position analysis API: evaluate any FEN with the Fool's NN and Stockfish.

``POST /api/analyze`` returns one normalized, client-friendly report for a
position (the pinned analysis contract, §14.0 style). The Fool runs a short
PUCT MCTS at the requested visit budget; Stockfish runs a dedicated, kept-warm
UCI engine at the requested depth. Either engine can be omitted, and either
engine failing degrades to ``{"available": false, "error": ...}`` for that
engine instead of failing the request — the client calls this while the user
edits/plays, so one broken dependency must never blank the panel.

Successful payloads are cached in-process by
``(fen, sorted engines, visits, depth, multipv)`` (LRU 256, 10 min TTL) and
error payloads are never cached, so a debounced client re-analysis of the same
position is essentially free.

Blocking work (torch MCTS, UCI round-trips) always runs in a worker thread.
Two module-level locks keep the box sane: :data:`_FOOL_LOCK` serializes MCTS
searches (one GPU evaluator, one search at a time) and :data:`_SF_LOCK`
serializes the single warm analysis engine.
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import OrderedDict
from typing import Any, Literal

import chess
import chess.engine
from fastapi import APIRouter, HTTPException
from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)

from .. import model_server, stockfish

router = APIRouter()
log = logging.getLogger(__name__)

EngineName = Literal["fool", "stockfish"]
TerminalKind = Literal["checkmate", "stalemate", "insufficient", "seventyfive", "fivefold"]

DEFAULT_ENGINES: tuple[EngineName, ...] = ("fool", "stockfish")
DEFAULT_VISITS = 96
MIN_VISITS = 8
MAX_VISITS = 400
DEFAULT_DEPTH = 16
MIN_DEPTH = 6
MAX_DEPTH = 22
DEFAULT_MULTIPV = 3
MIN_MULTIPV = 1
MAX_MULTIPV = 5

FOOL_LABEL = "The Fool (trained)"
STOCKFISH_LABEL = "Stockfish"
NO_CHAMPION_ERROR = "no trained champion yet"
NO_BINARY_ERROR = "stockfish binary not found"

CACHE_MAX_ENTRIES = 256
CACHE_TTL_S = 600.0  # 10 minutes
MAX_PV_PLIES = 6  # keep the payload small; the first plies carry the story
ANALYSIS_ENGINE_OPTIONS: dict[str, int] = {"Threads": 1, "Hash": 32}  # stay off the box's back
MIN_EXPECTED_WHITE = 0.002
MAX_EXPECTED_WHITE = 0.998

# One search at a time: the champion evaluator is a single model instance.
_FOOL_LOCK = threading.Lock()
# One UCI round-trip at a time on the shared warm analysis engine.
_SF_LOCK = threading.Lock()
_sf_engine: chess.engine.SimpleEngine | None = None

CacheKey = tuple[str, tuple[str, ...], int, int, int]


class _NoChampionError(RuntimeError):
    """No trained champion is available for the Fool engine."""


class _StockfishUnavailableError(RuntimeError):
    """Stockfish cannot be used (missing binary, failed spawn, crash)."""


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AnalyzeIn(BaseModel):
    """One analysis request. Out-of-range visits/depth are clamped, not rejected."""

    fen: str = Field(min_length=1, description="Position to analyze, as a FEN string.")
    engines: list[EngineName] = Field(
        default_factory=lambda: list(DEFAULT_ENGINES),
        min_length=1,
        description="Which engines to run (default: both). Duplicates are ignored.",
    )
    visits: int = Field(
        default=DEFAULT_VISITS,
        description=f"Fool MCTS simulations, clamped to {MIN_VISITS}..{MAX_VISITS}.",
    )
    depth: int = Field(
        default=DEFAULT_DEPTH,
        description=f"Stockfish search depth, clamped to {MIN_DEPTH}..{MAX_DEPTH}.",
    )
    multipv: int = Field(
        default=DEFAULT_MULTIPV,
        ge=MIN_MULTIPV,
        le=MAX_MULTIPV,
        description=f"Top moves reported per engine ({MIN_MULTIPV}..{MAX_MULTIPV}).",
    )

    @field_validator("visits", mode="before")
    @classmethod
    def _clamp_visits(cls, value: Any) -> int:
        """Clamp the Fool visit budget to the contract range."""
        return _clamp_int(value, MIN_VISITS, MAX_VISITS)

    @field_validator("depth", mode="before")
    @classmethod
    def _clamp_depth(cls, value: Any) -> int:
        """Clamp the Stockfish depth to the contract range."""
        return _clamp_int(value, MIN_DEPTH, MAX_DEPTH)

    @field_validator("engines")
    @classmethod
    def _dedupe_engines(cls, value: list[EngineName]) -> list[EngineName]:
        """Drop duplicate engine names while preserving request order."""
        deduped: list[EngineName] = []
        for name in value:
            if name not in deduped:
                deduped.append(name)
        return deduped


class FoolTopMove(BaseModel):
    """One root move from the Fool's MCTS."""

    uci: str = Field(description="Move in UCI notation.")
    san: str = Field(description="Move in SAN notation.")
    visits: int = Field(description="MCTS visit count for this root move.")
    q: float = Field(description="Mover-POV expected value after this move, [-1, 1].")


class FoolAnalysis(BaseModel):
    """The Fool's report. ``available=false`` carries only ``error`` meaningfully."""

    available: bool = Field(description="Whether the Fool produced a real analysis.")
    label: str = Field(default=FOOL_LABEL, description="Display name.")
    checkpoint: str | None = Field(default=None, description="Champion checkpoint tag.")
    elo: float | None = Field(default=None, description="Champion Elo, when known.")
    visits: int | None = Field(default=None, description="MCTS simulations actually run.")
    root_value: float | None = Field(
        default=None, description="Side-to-move POV root value, [-1, 1]."
    )
    score_cp: int | None = Field(default=None, description="White-POV centipawns.")
    mate: int | None = Field(default=None, description="Always null: the Fool has no mate score.")
    expected_white: float | None = Field(default=None, description="White expected score, [0, 1].")
    best_move: str | None = Field(default=None, description="Best move in UCI, null when terminal.")
    best_san: str | None = Field(default=None, description="Best move in SAN.")
    pv: list[str] = Field(default_factory=list, description="Principal variation in SAN (best move only).")
    top: list[FoolTopMove] = Field(default_factory=list, description="Top moves by visit count.")
    elapsed_ms: int | None = Field(default=None, description="Analysis wall time in milliseconds.")
    error: str | None = Field(default=None, description="Failure reason when unavailable.")


class StockfishTopMove(BaseModel):
    """One MultiPV line from Stockfish."""

    uci: str = Field(description="First move of the line, in UCI.")
    san: str | None = Field(default=None, description="First move of the line, in SAN.")
    score_cp: int | None = Field(default=None, description="White-POV centipawns (null on mate).")
    mate: int | None = Field(default=None, description="White-POV mate distance in moves, or null.")
    pv: list[str] = Field(default_factory=list, description="Line continuation in SAN.")


class StockfishAnalysis(BaseModel):
    """Stockfish's report. ``available=false`` carries only ``error`` meaningfully."""

    available: bool = Field(description="Whether Stockfish produced a real analysis.")
    label: str = Field(default=STOCKFISH_LABEL, description="Display name.")
    depth: int | None = Field(default=None, description="Requested search depth.")
    score_cp: int | None = Field(default=None, description="White-POV centipawns (null on mate).")
    mate: int | None = Field(default=None, description="White-POV mate distance in moves, or null.")
    expected_white: float | None = Field(default=None, description="White expected score, [0, 1].")
    best_move: str | None = Field(default=None, description="Best move in UCI, null when terminal.")
    best_san: str | None = Field(default=None, description="Best move in SAN.")
    pv: list[str] = Field(default_factory=list, description="Principal variation in SAN.")
    top: list[StockfishTopMove] = Field(default_factory=list, description="Top MultiPV lines.")
    elapsed_ms: int | None = Field(default=None, description="Analysis wall time in milliseconds.")
    error: str | None = Field(default=None, description="Failure reason when unavailable.")


class AnalyzeEngines(BaseModel):
    """Per-engine reports; only requested engines appear in the response."""

    fool: FoolAnalysis | None = Field(default=None, description="The Fool's analysis, when requested.")
    stockfish: StockfishAnalysis | None = Field(
        default=None, description="Stockfish's analysis, when requested."
    )

    @model_serializer(mode="wrap")
    def _drop_unrequested(
        self,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> dict[str, Any]:
        """Serialize only the engines that were requested (no ``null`` placeholders)."""
        data = handler(self)
        return {name: value for name, value in data.items() if value is not None}


class AnalyzeOut(BaseModel):
    """Normalized analysis payload for one position."""

    fen: str = Field(description="Normalized FEN of the analyzed position.")
    turn: Literal["white", "black"] = Field(description="Side to move.")
    terminal: TerminalKind | None = Field(
        default=None, description="Terminal state, or null when the position is live."
    )
    engines: AnalyzeEngines = Field(description="Per-engine reports (requested engines only).")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class _ResponseCache:
    """Tiny thread-safe LRU + TTL cache for successful analysis payloads."""

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES, ttl: float = CACHE_TTL_S) -> None:
        self._max_entries = max_entries
        self._ttl = ttl
        self._entries: OrderedDict[CacheKey, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: CacheKey) -> dict[str, Any] | None:
        """Return the cached payload for ``key`` or None when absent/expired."""
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, payload = entry
            if expires_at <= now:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return payload

    def put(self, key: CacheKey, payload: dict[str, Any]) -> None:
        """Store ``payload`` for ``key``, dropping the least recently used entries."""
        with self._lock:
            self._entries[key] = (time.monotonic() + self._ttl, payload)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop every cached payload."""
        with self._lock:
            self._entries.clear()


_CACHE = _ResponseCache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_int(value: Any, low: int, high: int) -> int:
    """Coerce ``value`` to int and clamp it into ``[low, high]``."""
    return max(low, min(high, int(value)))


def _elapsed_ms(started: float) -> int:
    """Milliseconds elapsed since a ``time.perf_counter()`` reading."""
    return round((time.perf_counter() - started) * 1000.0)


def _parse_fen(fen: str) -> chess.Board:
    """Parse a FEN into a board; 422 with a readable detail when invalid."""
    try:
        return chess.Board(fen)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid fen: {exc}") from None


def _terminal_kind(board: chess.Board) -> TerminalKind | None:
    """Classify a terminal position, or None when play continues."""
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient"
    if board.is_seventyfive_moves():
        return "seventyfive"
    if board.is_fivefold_repetition():
        return "fivefold"
    return None


def _cp_from_expected(expected_white: float) -> int:
    """White-POV centipawns from a white expected score (pinned logistic inverse).

    ``cp = round(-400 * log10(1/E - 1))`` with E clamped so the log is finite and
    E=0.5 maps to 0.
    """
    expected = min(max(float(expected_white), MIN_EXPECTED_WHITE), MAX_EXPECTED_WHITE)
    return round(-400.0 * math.log10(1.0 / expected - 1.0))


def _expected_from_score(score_cp: int | None, mate: int | None) -> float:
    """White expected score from a white-POV cp / mate score."""
    if mate is not None:
        return 1.0 if mate > 0 else 0.0
    if score_cp is None:
        return 0.5
    return 1.0 / (1.0 + 10.0 ** (-score_cp / 400.0))


def _unavailable_fool(error: str, elapsed_ms: int) -> dict[str, Any]:
    """The Fool's failure shape (full stable keys, error set)."""
    return {
        "available": False,
        "label": FOOL_LABEL,
        "checkpoint": None,
        "elo": None,
        "visits": None,
        "root_value": None,
        "score_cp": None,
        "mate": None,
        "expected_white": None,
        "best_move": None,
        "best_san": None,
        "pv": [],
        "top": [],
        "elapsed_ms": elapsed_ms,
        "error": error,
    }


def _unavailable_stockfish(error: str, depth: int, elapsed_ms: int) -> dict[str, Any]:
    """Stockfish's failure shape (full stable keys, error set)."""
    return {
        "available": False,
        "label": STOCKFISH_LABEL,
        "depth": depth,
        "score_cp": None,
        "mate": None,
        "expected_white": None,
        "best_move": None,
        "best_san": None,
        "pv": [],
        "top": [],
        "elapsed_ms": elapsed_ms,
        "error": error,
    }


def _terminal_entries(board: chess.Board, engines: list[EngineName], depth: int) -> dict[str, Any]:
    """Engine entries for a terminal position: no search, fixed draw/mate values."""
    checkmate = board.is_checkmate()
    # Pinned terminal values: draws are 0.5 for White; a checkmate is won for the
    # side that is NOT to move, so White's expectation is 0 when White is mated.
    expected_white = (0.0 if board.turn == chess.WHITE else 1.0) if checkmate else 0.5
    mate = 0 if checkmate else None
    root_value = -1.0 if checkmate else 0.0  # side-to-move POV
    reports: dict[str, Any] = {}
    for name in engines:
        if name == "fool":
            reports[name] = {
                "available": True,
                "label": FOOL_LABEL,
                "checkpoint": None,
                "elo": None,
                "visits": 0,
                "root_value": root_value,
                "score_cp": None,
                "mate": mate,
                "expected_white": expected_white,
                "best_move": None,
                "best_san": None,
                "pv": [],
                "top": [],
                "elapsed_ms": 0,
                "error": None,
            }
        else:
            reports[name] = {
                "available": True,
                "label": STOCKFISH_LABEL,
                "depth": depth,
                "score_cp": None,
                "mate": mate,
                "expected_white": expected_white,
                "best_move": None,
                "best_san": None,
                "pv": [],
                "top": [],
                "elapsed_ms": 0,
                "error": None,
            }
    return reports


# ---------------------------------------------------------------------------
# Fool engine
# ---------------------------------------------------------------------------


def _fool_analyze_sync(board: chess.Board, visits: int, multipv: int) -> dict[str, Any]:
    """Blocking Fool analysis: one MCTS root, then mover-POV q for each top move.

    Runs on a worker thread under :data:`_FOOL_LOCK`. Values follow the training
    convention: every evaluator value is from the side-to-move's perspective at
    that board, so the mover's expectation after pushing a move is ``-value``.
    """
    # Lazy: importing training pulls torch; API boot and non-NN games never pay it.
    from training.selfplay import MCTSSearch, SearchConfig, encode_move, terminal_value

    with _FOOL_LOCK:
        champion = model_server.current()
        if champion is None:
            raise _NoChampionError(NO_CHAMPION_ERROR)

        search = MCTSSearch(
            champion.evaluator,
            SearchConfig(num_simulations=visits, root_noise=False),
        )
        result = search.search(board)

        legal = list(board.legal_moves)
        visits_by_move = {move: float(result.visit_counts[encode_move(board, move)]) for move in legal}
        ranked = sorted(legal, key=lambda move: visits_by_move[move], reverse=True)[:multipv]

        top: list[dict[str, Any]] = []
        for move in ranked:
            san = board.san(move)
            child = board.copy(stack=False)
            child.push(move)
            if child.is_game_over(claim_draw=True):
                # Terminal children have exact values; never ask the value head.
                q = -terminal_value(child)
            else:
                _logits, values = champion.evaluator.evaluate([child])
                q = -float(values[0])  # child is the opponent's turn: negate
            top.append(
                {
                    "uci": move.uci(),
                    "san": san,
                    "visits": int(visits_by_move[move]),
                    "q": round(q, 4),
                }
            )

        best = result.chosen_move
        if best is None and ranked:
            best = ranked[0]
        best_san = board.san(best) if best is not None else None
        best_uci = best.uci() if best is not None else None

        root_value = float(result.root_value)
        expected_white = (
            (root_value + 1.0) / 2.0 if board.turn == chess.WHITE else (1.0 - root_value) / 2.0
        )

        return {
            "available": True,
            "label": FOOL_LABEL,
            "checkpoint": str(champion.tag),
            "elo": float(champion.elo),
            "visits": visits,
            "root_value": round(root_value, 4),
            "score_cp": _cp_from_expected(expected_white),
            "mate": None,
            "expected_white": round(expected_white, 4),
            "best_move": best_uci,
            "best_san": best_san,
            "pv": [best_san] if best_san else [],
            "top": top,
            "elapsed_ms": None,  # the async wrapper measures wall time
            "error": None,
        }


async def _fool_entry(board: chess.Board, visits: int, multipv: int) -> dict[str, Any]:
    """Fool report for one board; failures become an unavailable entry, never a 500."""
    started = time.perf_counter()
    try:
        payload = await asyncio.to_thread(_fool_analyze_sync, board, visits, multipv)
    except _NoChampionError as exc:
        return _unavailable_fool(str(exc), _elapsed_ms(started))
    except Exception as exc:
        log.exception("fool analysis failed")
        detail = str(exc) or exc.__class__.__name__
        return _unavailable_fool(f"fool analysis failed: {detail}", _elapsed_ms(started))
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


# ---------------------------------------------------------------------------
# Stockfish engine
# ---------------------------------------------------------------------------


def _analysis_engine_locked() -> chess.engine.SimpleEngine:
    """Return the warm analysis engine, spawning it on first use (lock held)."""
    global _sf_engine
    if _sf_engine is not None:
        return _sf_engine
    path = stockfish.resolve_binary()
    if path is None:
        raise _StockfishUnavailableError(NO_BINARY_ERROR)
    proc = chess.engine.SimpleEngine.popen_uci(path, timeout=stockfish.PROCESS_TIMEOUT_S)
    try:
        proc.configure(ANALYSIS_ENGINE_OPTIONS)
    except Exception:
        _close_hard(proc)
        raise
    _sf_engine = proc
    log.info("analysis stockfish started")
    return proc


def _drop_engine_locked() -> None:
    """Close + hard-kill the warm engine after a crash (lock held)."""
    global _sf_engine
    proc, _sf_engine = _sf_engine, None
    if proc is None:
        return
    log.warning("dropping crashed analysis stockfish")
    _close_hard(proc)


def _close_hard(proc: chess.engine.SimpleEngine) -> None:
    """Best-effort close + OS-level kill of a misbehaving engine process."""
    try:
        proc.close()
    except Exception:
        log.debug("analysis stockfish close failed", exc_info=True)
    stockfish._kill_hard(proc)  # shared package helper (same kill policy as game engines)


def _collect_lines(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
    multipv: int,
) -> dict[int, dict[str, Any]]:
    """Run MultiPV analysis and keep the deepest info line per multipv index.

    The pinned contract says "first info line per multipv index"; with a
    streaming ``analysis()`` iterator that literal reading would keep the
    depth-1 lines. The stream re-emits every index at each depth, so the
    *first line at the final depth* is the last line received — which this
    keeps. (Flagged as an interpretation in the task report.)
    """
    lines: dict[int, dict[str, Any]] = {}
    with engine.analysis(board, chess.engine.Limit(depth=depth), multipv=multipv) as analysis:
        for info in analysis:
            score = info.get("score")
            pv = info.get("pv")
            if score is None or not pv:
                continue
            index = int(info.get("multipv", 1))
            if 1 <= index <= multipv:
                lines[index] = info  # later (deeper) lines supersede earlier ones
    return lines


def _pv_sans(board: chess.Board, moves: list[chess.Move]) -> list[str]:
    """Convert a PV move list to SAN, truncated and defensively capped."""
    probe = board.copy(stack=False)
    sans: list[str] = []
    for move in moves[:MAX_PV_PLIES]:
        if move not in probe.legal_moves:
            break
        sans.append(probe.san(move))
        probe.push(move)
    return sans


def _build_stockfish_payload(
    lines: dict[int, dict[str, Any]],
    board: chess.Board,
    depth: int,
    multipv: int,
) -> dict[str, Any]:
    """Shape collected MultiPV lines into the pinned Stockfish report."""
    top: list[dict[str, Any]] = []
    for index in sorted(lines)[:multipv]:
        info = lines[index]
        pov = info["score"].pov(chess.WHITE)
        sans = _pv_sans(board, info["pv"])
        top.append(
            {
                "uci": info["pv"][0].uci(),
                "san": sans[0] if sans else None,
                "score_cp": pov.score(),
                "mate": pov.mate(),
                "pv": sans,
            }
        )
    if not top:
        # Defensive: a live position always yields at least one line, but never
        # let an odd engine reply fail the request.
        return {
            "available": True,
            "label": STOCKFISH_LABEL,
            "depth": depth,
            "score_cp": None,
            "mate": None,
            "expected_white": 0.5,
            "best_move": None,
            "best_san": None,
            "pv": [],
            "top": [],
            "elapsed_ms": None,
            "error": None,
        }
    best = top[0]
    return {
        "available": True,
        "label": STOCKFISH_LABEL,
        "depth": depth,
        "score_cp": best["score_cp"],
        "mate": best["mate"],
        "expected_white": round(_expected_from_score(best["score_cp"], best["mate"]), 4),
        "best_move": best["uci"],
        "best_san": best["san"],
        "pv": best["pv"],
        "top": top,
        "elapsed_ms": None,  # the async wrapper measures wall time
        "error": None,
    }


def _stockfish_analyze_sync(board: chess.Board, depth: int, multipv: int) -> dict[str, Any]:
    """Blocking Stockfish analysis on the warm engine (worker thread, lock held)."""
    with _SF_LOCK:
        try:
            engine = _analysis_engine_locked()
            lines = _collect_lines(engine, board, depth, multipv)
        except _StockfishUnavailableError:
            raise
        except Exception:
            # Dead/hung process: drop it so the next request respawns cleanly.
            _drop_engine_locked()
            raise
    return _build_stockfish_payload(lines, board, depth, multipv)


async def _stockfish_entry(board: chess.Board, depth: int, multipv: int) -> dict[str, Any]:
    """Stockfish report for one board; failures become an unavailable entry."""
    started = time.perf_counter()
    try:
        payload = await asyncio.to_thread(_stockfish_analyze_sync, board, depth, multipv)
    except _StockfishUnavailableError as exc:
        return _unavailable_stockfish(str(exc), depth, _elapsed_ms(started))
    except Exception as exc:
        log.exception("stockfish analysis failed")
        detail = str(exc) or exc.__class__.__name__
        return _unavailable_stockfish(f"stockfish analysis failed: {detail}", depth, _elapsed_ms(started))
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


# ---------------------------------------------------------------------------
# Lifecycle (registered in server.main's lifespan)
# ---------------------------------------------------------------------------


def _pop_engine() -> chess.engine.SimpleEngine | None:
    """Take the warm engine out of the registry (waits for in-flight analysis)."""
    global _sf_engine
    with _SF_LOCK:
        proc, _sf_engine = _sf_engine, None
    return proc


async def shutdown() -> bool:
    """Quit the warm analysis engine; True when one was stopped. Idempotent."""
    proc = await asyncio.to_thread(_pop_engine)
    if proc is None:
        return False
    try:
        await asyncio.wait_for(asyncio.to_thread(proc.quit), stockfish.QUIT_TIMEOUT_S)
    except Exception:
        log.warning("analysis stockfish quit failed; killing", exc_info=True)
        try:
            await asyncio.wait_for(
                asyncio.to_thread(_close_hard, proc), stockfish.QUIT_TIMEOUT_S
            )
        except Exception:
            log.debug("analysis stockfish hard kill failed", exc_info=True)
    return True


def reset_for_tests() -> None:
    """Test isolation: drop the warm engine + cached payloads (not for production)."""
    global _sf_engine
    with _SF_LOCK:
        proc, _sf_engine = _sf_engine, None
    if proc is not None:
        try:
            proc.close()
            stockfish._kill_hard(proc)
        except Exception:
            log.warning("analysis stockfish cleanup failed", exc_info=True)
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=AnalyzeOut)
async def analyze(body: AnalyzeIn) -> dict[str, Any]:
    """Evaluate one FEN with the Fool and/or Stockfish and return a normalized report.

    422 for an invalid FEN; per-engine failures are reported in-band as
    ``available: false`` so one unavailable engine never blanks the payload.
    Successful responses are cached (LRU 256, 10 min); error responses are not.
    """
    board = _parse_fen(body.fen)
    fen = board.fen()
    engines = tuple(sorted(body.engines))  # stable engine order (and cache identity)
    cache_key: CacheKey = (fen, engines, body.visits, body.depth, body.multipv)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        log.debug("analyze cache hit: %s", fen)
        return cached

    terminal = _terminal_kind(board)
    if terminal is not None:
        reports = _terminal_entries(board, list(engines), body.depth)
    else:
        tasks: dict[str, Any] = {}
        if "fool" in engines:
            tasks["fool"] = _fool_entry(board, body.visits, body.multipv)
        if "stockfish" in engines:
            tasks["stockfish"] = _stockfish_entry(board, body.depth, body.multipv)
        results = await asyncio.gather(*tasks.values())
        by_name = dict(zip(tasks.keys(), results))
        reports = {name: by_name[name] for name in engines}

    payload: dict[str, Any] = {
        "fen": fen,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "terminal": terminal,
        "engines": reports,
    }
    if all(bool(entry.get("available")) for entry in reports.values()):
        _CACHE.put(cache_key, payload)
    log.debug(
        "analyze %s engines=%s terminal=%s elapsed_ms=%s",
        fen,
        ",".join(engines),
        terminal,
        {name: entry.get("elapsed_ms") for name, entry in reports.items()},
    )
    return payload
