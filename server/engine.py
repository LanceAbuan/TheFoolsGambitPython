"""Live game sessions: python-chess rules + pluggable opponents.

Opponents: the demo Fool (material-greedy, random tiebreaks) and a UCI
Stockfish subprocess (see :mod:`server.stockfish`, one process per active
game, stopped on finish/resign/shutdown). All HTTP-mapped failures raise
GameError subclasses whose ``status`` maps to an HTTP code.

Every mutating handler (move/undo/resign/opening reply) runs under the
session's asyncio lock; the blocking opponent search runs inside
``asyncio.to_thread`` so a thinking UCI engine never blocks the event loop.
"""
from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import chess
import chess.engine

from . import db, model_server, stockfish

log = logging.getLogger(__name__)

MAX_UNDO_PLIES = 2  # a human undo pops the opponent's reply too
DEFAULT_STRENGTH = "curious"  # fallback for legacy rows predating games.strength
_RESTORE_ORIENTATIONS = frozenset({"white", "black"})
_RESTORE_OPPONENTS = frozenset({"fool", "stockfish", "nn"})
_RESTORE_WINNERS = frozenset({"white", "black", "draw"})


class GameError(Exception):
    """Base class: `status` maps to an HTTP error code."""

    status = 400


class NotFound(GameError):
    status = 404


class IllegalMove(GameError):
    status = 422


class Conflict(GameError):
    """Wrong turn or finished game."""

    status = 409


class EngineUnavailable(GameError):
    """Opponent engine binary missing (e.g. no stockfish installed)."""

    status = 503


@dataclass
class Session:
    id: str
    kind: str  # 'play' | 'selfplay'
    orientation: str  # the human's color: 'white' | 'black'
    opponent: str  # 'fool' | 'stockfish'
    strength: str = "curious"  # wandering | curious | enlightened (stockfish only)
    sans: list[str] = field(default_factory=list)
    ucis: list[str] = field(default_factory=list)
    fens: list[str] = field(default_factory=list)
    board: chess.Board = field(default_factory=chess.Board)
    status: str = "active"
    winner: str | None = None
    reason: str | None = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)  # TTL-reaper clock (FS-9)
    sf: chess.engine.SimpleEngine | None = field(default=None, repr=False, compare=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    @property
    def human_color(self) -> chess.Color:
        return chess.WHITE if self.orientation == "white" else chess.BLACK

    def snapshot(self) -> dict[str, object]:
        last = self.board.move_stack[-1] if self.board.move_stack else None
        return {
            "game_id": self.id,
            "fen": self.board.fen(),
            "moves": list(self.sans),
            "status": self.status,
            "winner": self.winner,
            # F-11 follow-up (additive): why the game ended — "resign",
            # "abandoned" (idle auto-resign), or a chess termination name
            # (e.g. "checkmate"). None while active. Lets the client render
            # "Claimed by time" on reload instead of probing undo's 409.
            "reason": self.reason,
            "in_check": self.board.is_check(),
            "last_move": last.uci() if last is not None else None,
        }

    def record(self) -> list[tuple[str, str, str]]:
        """(san, uci, fen_after) triples for persistence."""
        return list(zip(self.sans, self.ucis, self.fens))

    @property
    def pgn(self) -> str:
        result = {None: "*", "white": "1-0", "black": "0-1", "draw": "1/2-1/2"}[self.winner]
        return build_pgn(self.sans, self.orientation, result)


_sessions: dict[str, Session] = {}
_lock = threading.Lock()

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def build_pgn(sans: list[str], orientation: str, result: str = "*") -> str:
    import chess.pgn

    game = chess.pgn.Game()
    game.headers["Event"] = "The Fool's Gambit"
    game.headers["White"] = "You" if orientation == "white" else "The Fool"
    game.headers["Black"] = "The Fool" if orientation == "white" else "You"
    game.headers["Result"] = result
    node: chess.pgn.GameNode = game
    board = chess.Board()
    for san in sans:
        move = board.parse_san(san)
        node = node.add_variation(move)
        board.push(move)
    return str(game)


def reset_for_tests() -> None:
    with _lock:
        _sessions.clear()


def all_sessions() -> list[Session]:
    with _lock:
        return list(_sessions.values())


def _finish_if_over(session: Session) -> None:
    if not session.board.is_game_over(claim_draw=True):
        return
    outcome = session.board.outcome()
    assert outcome is not None  # is_game_over() guarantees an outcome
    session.status = "finished"
    if outcome.winner is chess.WHITE:
        session.winner, session.reason = "white", outcome.termination.name.lower()
    elif outcome.winner is chess.BLACK:
        session.winner, session.reason = "black", outcome.termination.name.lower()
    else:
        session.winner, session.reason = "draw", outcome.termination.name.lower()


def _fool_move(board: chess.Board) -> chess.Move | None:
    """Demo engine: 1-ply material-greedy with random tiebreaks."""
    legal = list(board.legal_moves)
    if not legal:
        return None
    sign = 1 if board.turn == chess.WHITE else -1

    def score(move: chess.Move) -> float:
        probe = board.copy(stack=False)
        probe.push(move)
        material = sum(
            len(probe.pieces(piece_type, color)) * value * (1 if color == chess.WHITE else -1)
            for piece_type, value in _PIECE_VALUES.items()
            for color in (chess.WHITE, chess.BLACK)
        )
        value = sign * material
        if probe.is_checkmate():
            value += 100.0
        elif probe.is_check():
            value += 0.3
        return value + random.random() * 0.01

    return max(legal, key=score)


def _push(session: Session, move: chess.Move) -> str:
    san = session.board.san(move)
    session.board.push(move)
    session.sans.append(san)
    session.ucis.append(move.uci())
    session.fens.append(session.board.fen())
    session.last_activity = time.time()
    return san


def _opponent_move_sync(session: Session) -> chess.Move | None:
    """Blocking opponent move computation. Runs on a worker thread.

    A crashed/missing Stockfish degrades to the Fool for that move (logged) so
    a live game is never stranded; the dead process is deregistered and the
    next move respawns if the binary is available again.
    """
    if session.board.is_game_over(claim_draw=True):
        return None
    if session.opponent == "fool":
        return _fool_move(session.board)
    if session.opponent == "nn":
        try:
            return model_server.best_move(session)
        except Exception:
            log.exception("nn model failed for game %s; falling back to the Fool", session.id)
            return _fool_move(session.board)
    if session.opponent == "stockfish":
        try:
            return stockfish.best_move(session)
        except Exception:
            log.exception("stockfish failed for game %s; falling back to the Fool", session.id)
            stockfish.drop(session)
            return _fool_move(session.board)
    return None


async def _opponent_turn(session: Session) -> None:
    """Compute + apply the opponent's reply; stop the UCI process when the game ends.

    Caller must hold ``session.lock`` and the game must be active.
    """
    reply = await asyncio.to_thread(_opponent_move_sync, session)
    if reply is not None:
        _push(session, reply)
        _finish_if_over(session)
    if session.status != "active":
        await stockfish.stop(session)


def create_game(color: str, opponent: str, strength: str = "curious") -> Session:
    """Create a session; raises EngineUnavailable (503) for unavailable opponents.

    - ``stockfish``: 503 when no UCI binary resolves.
    - ``nn``: 503 when no trained champion exists in any ``runs/<regime>/``
      ("no trained model yet - run scripts/train.py").

    The opponent's opening reply (human plays Black) is applied by the caller
    via :func:`opening_reply`, keeping the blocking UCI round-trip off the
    event loop and out of this sync function.
    """
    if opponent == "stockfish" and stockfish.resolve_binary() is None:
        raise EngineUnavailable("stockfish binary not found")
    if opponent == "nn" and not model_server.available():
        raise EngineUnavailable("no trained model yet - run scripts/train.py")
    orientation = color if color != "random" else random.choice(["white", "black"])
    session = Session(
        id=uuid.uuid4().hex[:12],
        kind="play",
        orientation=orientation,
        opponent=opponent,
        strength=strength,
    )
    with _lock:
        _sessions[session.id] = session
    return session


async def opening_reply(session: Session) -> str | None:
    """Apply the opponent's opening reply when the human plays Black. Returns its SAN."""
    async with session.lock:
        if session.status != "active" or session.board.turn == session.human_color:
            return None  # human moves first — nothing to open with
        await _opponent_turn(session)
        return session.sans[-1] if session.sans else None


def _apply_human_move(session: Session, from_sq: str, to_sq: str, promotion: str | None) -> str:
    """Validate + apply the human's move. Caller must hold ``session.lock``."""
    if session.status != "active":
        raise Conflict("game is finished")
    if session.board.turn != session.human_color:
        raise Conflict("not your turn")
    uci = from_sq + to_sq + (promotion or "")
    try:
        move = chess.Move.from_uci(uci)
        if move not in session.board.legal_moves:
            raise ValueError(uci)
    except ValueError as exc:
        raise IllegalMove(f"illegal move {uci}") from exc
    san = _push(session, move)
    _finish_if_over(session)
    return san


async def player_move(session: Session, from_sq: str, to_sq: str, promotion: str | None) -> str:
    """Validate + apply the human move, then the opponent's reply. Returns the human's SAN."""
    async with session.lock:
        san = _apply_human_move(session, from_sq, to_sq, promotion)
        await _opponent_turn(session)
        return san


def _pop_plies(session: Session, pops: int) -> None:
    """Pop ``pops`` plies off the board and every parallel move record.

    Caller must hold ``session.lock``. The board's internal stack and the
    sans/ucis/fens records are always mutated together so ``record()`` and a
    rehydrate replay stay consistent with the board.
    """
    for _ in range(max(pops, 0)):
        session.board.pop()
        session.sans.pop()
        session.ucis.pop()
        session.fens.pop()


def _pops_for_human_turn(session: Session) -> int:
    """Ply count to pop so the human is on turn afterwards, capped at MAX_UNDO_PLIES.

    The parity trim keeps the game from ever being stranded before the
    opponent's opening reply (e.g. undoing with only the engine's 1st move on
    the board pops nothing).
    """
    total = len(session.sans)
    pops = min(MAX_UNDO_PLIES, total)
    human_to_move_after = 0 if session.human_color is chess.WHITE else 1
    if (total - pops) % 2 != human_to_move_after:
        pops -= 1
    return max(pops, 0)


def _undo_finished(session: Session) -> None:
    """Take-back after game over: revert the finish and resume as an active game.

    Resignations have no finishing board position — the move list is untouched
    and only the result is cleared (un-resign). Board finishes (mate,
    stalemate, ...) revert the finishing plies with the same MAX_UNDO_PLIES
    parity rule as mid-game undo, so the human is on turn when play resumes.
    Abandoned games raise Conflict: the TTL reaper's retirement is final —
    undo must not resurrect a game the reaper closed precisely because it was
    left idle.

    Caller must hold ``session.lock``. The cleared result state is written
    back by the API layer's persist; the rehydrate layer trusts the stored
    result for finished games, so the persist is what makes an undone game
    stay undone across restarts.
    """
    if session.reason == "abandoned":
        raise Conflict("abandoned games cannot take back")
    if session.reason != "resign":
        _pop_plies(session, _pops_for_human_turn(session))
    session.status = "active"
    session.winner = None
    session.reason = None


async def undo(session: Session) -> None:
    """Pop the human's last move (+ the opponent's reply), landing on the human's turn.

    Active games no-op when there is nothing of the human's to undo yet.
    Finished games take back too (casual play, user story #10): a board finish
    reverts the finishing plies and clears the result, a resignation is
    un-resigned with the move list untouched, and only abandoned (reaped)
    games stay final.
    """
    async with session.lock:
        if session.status != "active":
            _undo_finished(session)
        else:
            if not session.sans or session.board.turn != session.human_color:
                return  # nothing of the human's to undo yet
            _pop_plies(session, _pops_for_human_turn(session))
        session.last_activity = time.time()


async def resign(session: Session) -> None:
    """The human resigns; the UCI opponent process (if any) is stopped."""
    async with session.lock:
        if session.status != "active":
            raise Conflict("game is finished")
        session.status = "finished"
        session.winner = "black" if session.human_color is chess.WHITE else "white"
        session.reason = "resign"
        await stockfish.stop(session)


async def shutdown_engines() -> int:
    """Stop every live stockfish process (app shutdown). Returns the count stopped."""
    stopped = 0
    for session in all_sessions():
        if await stockfish.stop(session):
            stopped += 1
    return stopped


def get_session(gid: str) -> Session:
    """The live session for ``gid``, rehydrating from SQLite on a memory miss.

    Boot (:func:`rehydrate_active`) restores still-active games at startup; any
    other persisted play game — typically a finished archive row — is rebuilt
    here on first access so a server restart never 404s a persisted game. The
    miss path is one small indexed SQLite read, only paid once per game per
    process. Corrupt rows restore as absent (404), never a crash.
    """
    session = _sessions.get(gid)
    if session is not None:
        return session
    source = db.restore_one(gid)
    if source is None:
        raise NotFound(f"game {gid} not found")
    row, ucis = source
    if str(row.get("kind") or "") != "play":
        raise NotFound(f"game {gid} not found")
    restored = _session_from_row(row, ucis)
    if restored is None:
        raise NotFound(f"game {gid} not found")
    with _lock:  # a concurrent miss may have raced us; first insert wins
        existing = _sessions.get(gid)
        if existing is not None:
            return existing
        _sessions[gid] = restored
    return restored


def _parse_iso_ts(value: object) -> float:
    """Epoch seconds from an ISO-8601 ``db._iso`` timestamp; unparseable → now."""
    text = str(value or "")
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    except ValueError:
        log.debug("rehydrate: unparseable timestamp %r; using now", value)
        return time.time()


def _session_from_row(row: dict[str, Any], ucis: list[str]) -> Session | None:
    """Rebuild a Session from its persisted games row + ordered UCIs.

    Corrupt rows (bad enum, malformed or position-illegal stored UCI) return
    ``None`` with a warning — rehydration never crashes boot and never turns a
    request into a 500. The stored result state is trusted for finished games
    (resign/abandon endings are not derivable from the board). The engine
    process (stockfish) is intentionally not resurrected: it respawns lazily on
    the next move via :func:`server.stockfish.ensure`.
    """
    gid = str(row.get("id") or "")
    try:
        orientation = str(row.get("orientation"))
        opponent = str(row.get("opponent"))
        status = str(row.get("status"))
        if orientation not in _RESTORE_ORIENTATIONS:
            raise ValueError(f"bad orientation {orientation!r}")
        if opponent not in _RESTORE_OPPONENTS:
            raise ValueError(f"bad opponent {opponent!r}")
        if status not in ("active", "finished"):
            raise ValueError(f"bad status {status!r}")
        winner = row.get("winner")
        reason = row.get("reason")
        if status == "finished" and winner not in _RESTORE_WINNERS:
            raise ValueError(f"bad winner {winner!r}")
        board = chess.Board()
        sans: list[str] = []
        played: list[str] = []
        fens: list[str] = []
        for raw in ucis:
            move = chess.Move.from_uci(str(raw).strip())  # ValueError on malformed UCI
            if move not in board.legal_moves:
                raise ValueError(f"stored move illegal in position: {raw!r}")
            sans.append(board.san(move))
            board.push(move)
            played.append(move.uci())
            fens.append(board.fen())
    except (TypeError, ValueError) as exc:
        log.warning("rehydrate: skipping corrupt game row %s (%s)", gid or "(no id)", exc)
        return None
    finished = status == "finished"
    return Session(
        id=gid,
        kind=str(row.get("kind") or "play"),
        orientation=orientation,
        opponent=opponent,
        strength=str(row.get("strength") or DEFAULT_STRENGTH),
        sans=sans,
        ucis=played,
        fens=fens,
        board=board,
        status=status,
        winner=winner if finished else None,
        reason=reason if finished else None,
        created_at=_parse_iso_ts(row.get("created_at")),
        last_activity=_parse_iso_ts(row.get("updated_at")),
    )


def rehydrate_active() -> int:
    """Boot-time (lifespan): rebuild every still-active play session from SQLite.

    Games replay their stored UCIs in order; ``last_activity`` comes from the
    persisted ``updated_at`` so rehydrated sessions obey the TTL reaper's
    unchanged semantics (idle before the restart ⇒ reaped after it). Corrupt or
    duplicate rows are skipped with a warning. Returns the count restored. The
    restore layer self-migrates (``games.strength``) so this is order-safe.
    """
    restored = 0
    for gid, (row, ucis) in db.restore_active().items():
        if str(row.get("kind") or "") != "play":
            log.warning("rehydrate: skipping non-play game %s (kind=%r)", gid, row.get("kind"))
            continue
        session = _session_from_row(row, ucis)
        if session is None:
            continue
        with _lock:
            _sessions.setdefault(gid, session)
        restored += 1
    if restored:
        log.info("rehydrated %d active game session(s) from the database", restored)
    return restored
