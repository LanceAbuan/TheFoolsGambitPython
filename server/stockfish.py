"""Stockfish UCI opponent: one subprocess per active game.

Binary resolution (docs/UPGRADE_PLAN.md §2): ``FSG_STOCKFISH`` env → repo-local
``tools/stockfish.exe`` (gitignored) → ``PATH``. When nothing resolves, creating
a stockfish game fails with 503 (see :func:`server.engine.create_game`).

Strength presets map the API's ``strength`` string onto Stockfish's Skill Level
(0-20) plus a search-depth cap. Stockfish's low skill levels inject deliberate
move noise, so no extra randomness is layered on top:

===============  ==============  ========  ==================================
API string       Skill Level     depth     character
===============  ==============  ========  ==================================
``wandering``    2               1         barely coherent, near-instant
``curious``      6               4         default; novice club player
``enlightened``  12              12        solid tactician, still beatable
===============  ==============  ========  ==================================

Every process is pinned to one thread and a 16 MB hash so concurrent games
cannot saturate the box. All blocking UCI round-trips (spawn, play, quit) are
invoked from :mod:`server.engine` inside ``asyncio.to_thread`` — never on the
event loop. The ``_processes`` registry is keyed by session id; CPython's GIL
makes its single dict operations atomic, and per-game asyncio locks in
server.engine serialize access to any one process.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import chess
import chess.engine
import psutil

if TYPE_CHECKING:
    from .engine import Session

log = logging.getLogger(__name__)

ENV_VAR = "FSG_STOCKFISH"
TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
PROCESS_TIMEOUT_S = 10.0  # SimpleEngine per-command timeout (spawn, move, quit)
QUIT_TIMEOUT_S = 5.0  # graceful-quit budget before a hard kill
CONFIG_OPTIONS: dict[str, int] = {"Threads": 1, "Hash": 16}  # keep games light
DEFAULT_STRENGTH = "curious"

STRENGTH_PRESETS: dict[str, dict[str, int]] = {
    "wandering": {"skill": 2, "depth": 1},
    "curious": {"skill": 6, "depth": 4},
    "enlightened": {"skill": 12, "depth": 12},
}

_CANDIDATE_NAMES = ("stockfish.exe", "stockfish") if os.name == "nt" else ("stockfish", "stockfish.exe")

_processes: dict[str, chess.engine.SimpleEngine] = {}


def resolve_binary() -> str | None:
    """Locate a Stockfish binary: FSG_STOCKFISH → tools/ → PATH → None.

    A set-but-invalid ``FSG_STOCKFISH`` falls through to the next source so a
    stale env var can never shadow a working local binary.
    """
    env = os.environ.get(ENV_VAR, "").strip()
    if env and Path(env).is_file():
        return env
    for name in _CANDIDATE_NAMES:
        local = TOOLS_DIR / name
        if local.is_file():
            return str(local)
    return shutil.which("stockfish")


def preset(strength: str) -> dict[str, int]:
    """Resolve a strength string to its ``{"skill", "depth"}`` preset (unknown → curious)."""
    if strength not in STRENGTH_PRESETS:
        log.debug("unknown strength %r; using %r", strength, DEFAULT_STRENGTH)
        return STRENGTH_PRESETS[DEFAULT_STRENGTH]
    return STRENGTH_PRESETS[strength]


def live_count() -> int:
    """Number of currently registered (live) stockfish processes."""
    return len(_processes)


def pid_for(session_id: str) -> int | None:
    """OS pid of the session's engine process, or None when none is registered."""
    proc = _processes.get(session_id)
    if proc is None:
        return None
    try:
        pid: int | None = proc.transport.get_pid()
    except AttributeError:  # transport already closed
        return None
    return pid


def ensure(session: Session) -> chess.engine.SimpleEngine:
    """Return the session's engine, spawning + configuring it on first use.

    Blocking (process spawn + UCI handshake) — call from a worker thread.
    Raises :class:`chess.engine.EngineError` when no binary resolves.
    """
    existing = _processes.get(session.id)
    if existing is not None:
        return existing
    path = resolve_binary()
    if path is None:
        raise chess.engine.EngineError("stockfish binary not found")
    proc = chess.engine.SimpleEngine.popen_uci(path, timeout=PROCESS_TIMEOUT_S)
    options = {**CONFIG_OPTIONS, "Skill Level": preset(session.strength)["skill"]}
    proc.configure(options)
    _processes[session.id] = proc
    session.sf = proc
    log.info(
        "stockfish spawned for game %s (pid %s, strength %r)", session.id, pid_for(session.id), session.strength
    )
    return proc


def best_move(session: Session) -> chess.Move | None:
    """Blocking: the engine's move for the side to move (spawns on first use)."""
    board = session.board
    if board.is_game_over():
        return None
    proc = ensure(session)
    limit = chess.engine.Limit(depth=preset(session.strength)["depth"])
    result = proc.play(board, limit, info=chess.engine.INFO_NONE)
    return result.move


def drop(session: Session) -> None:
    """Deregister + best-effort hard-stop the session's engine (crash path)."""
    proc = _processes.pop(session.id, None)
    if proc is None:
        return
    try:
        proc.close()
    except Exception:
        log.debug("stockfish close failed for game %s", session.id, exc_info=True)
    _kill_hard(proc)
    session.sf = None


async def stop(session: Session) -> bool:
    """Gracefully quit the session's engine, hard-killing on refusal. Idempotent.

    Returns True when a registered process was stopped.
    """
    proc = _processes.pop(session.id, None)
    if proc is None:
        return False

    def _quit() -> None:
        proc.quit()  # sends "quit" and waits for the process to exit

    try:
        await asyncio.wait_for(asyncio.to_thread(_quit), QUIT_TIMEOUT_S)
    except TimeoutError:
        log.warning("stockfish quit timed out for game %s; killing", session.id)
        await asyncio.to_thread(_kill_hard, proc)
    except Exception:
        log.exception("stockfish quit failed for game %s; killing", session.id)
        await asyncio.to_thread(_kill_hard, proc)
    session.sf = None
    return True


def reset_for_tests() -> None:
    """Force-stop every registered engine (test isolation only, not for production)."""
    for session_id in list(_processes):
        proc = _processes.pop(session_id)
        try:
            proc.close()
            _kill_hard(proc)
        except Exception:
            log.warning("stockfish cleanup failed for game %s", session_id, exc_info=True)


def _kill_hard(proc: chess.engine.SimpleEngine) -> None:
    """Terminate the engine's OS process without waiting for a graceful exit."""
    try:
        pid = proc.transport.get_pid()
    except AttributeError:  # transport already closed
        return
    if not pid:
        return
    try:
        psutil.Process(pid).kill()
    except psutil.Error as exc:
        log.debug("failed to kill stockfish pid %s: %s", pid, exc)
