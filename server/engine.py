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

import chess
import chess.engine

from . import stockfish

log = logging.getLogger(__name__)

MAX_UNDO_PLIES = 2  # a human undo pops the opponent's reply too


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
    if not session.board.is_game_over():
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
    return san


def _opponent_move_sync(session: Session) -> chess.Move | None:
    """Blocking opponent move computation. Runs on a worker thread.

    A crashed/missing Stockfish degrades to the Fool for that move (logged) so
    a live game is never stranded; the dead process is deregistered and the
    next move respawns if the binary is available again.
    """
    if session.board.is_game_over():
        return None
    if session.opponent == "fool":
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
    """Create a session; raises EngineUnavailable (503) for stockfish with no binary.

    The opponent's opening reply (human plays Black) is applied by the caller
    via :func:`opening_reply`, keeping the blocking UCI round-trip off the
    event loop and out of this sync function.
    """
    if opponent == "stockfish" and stockfish.resolve_binary() is None:
        raise EngineUnavailable("stockfish binary not found")
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


async def undo(session: Session) -> None:
    """Pop the human's last move (+ the opponent's reply), landing on the human's turn.

    The pop count is trimmed so the game is never stranded before the
    opponent's opening reply (e.g. undoing with only the engine's 1st move on
    the board is a no-op).
    """
    async with session.lock:
        if session.status != "active":
            raise Conflict("game is finished")
        if not session.sans or session.board.turn != session.human_color:
            return  # nothing of the human's to undo yet
        total = len(session.sans)
        pops = min(MAX_UNDO_PLIES, total)
        human_to_move_after = 0 if session.human_color is chess.WHITE else 1
        if (total - pops) % 2 != human_to_move_after:
            pops -= 1
        for _ in range(max(pops, 0)):
            session.board.pop()
            session.sans.pop()
            session.ucis.pop()
            session.fens.pop()


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
    session = _sessions.get(gid)
    if session is None:
        raise NotFound(f"game {gid} not found")
    return session
