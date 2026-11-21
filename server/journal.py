"""Training journal reader (FS-6/FS-6b): parsing + tailing ``data/train_log.jsonl``.

The training pipeline appends structured events, one JSON object per line.
This module is pure file logic — no hub, no event loop — so the WS subscribe
path (:mod:`server.ws`) and the tail task (:mod:`server.telemetry`) share it
without import cycles.

Two event kinds route onto dedicated WS frames (wire schemas documented in
:mod:`server.telemetry`); everything else flows as a plain ``log`` frame:

    {"kind": "live", "game": 3, "ply": 17, "move": "e2e4",
     "fen": "rnbq... w KQkq - 3", "run_id": "2026...", "ts": 1795245000.0}

    {"kind": "game_end", "game": 3, "result": "1-0", "plies": 84,
     "ts": 1795245010.0}

Parsing is defensive against writer-side drift: ``kind`` matches
case-insensitively, common field-name aliases are accepted (``game_id`` for
``game``, ``uci``/``move_uci`` for ``move``, ``position``/``board`` for
``fen``, ``timestamp`` for ``ts``, ...), numerics arrive coerced, and a
``live``/``game_end`` line without a routable ``game`` degrades to a log
frame rather than being dropped. Unknown kinds render as log frames — never
dropped, never fatal.

The tail cursor (byte offset of the next unread line) lives here so both the
loop and the subscribe path can advance it; :func:`seek_to_now` is the
FS-6b stale-replay guard (a fresh subscriber never receives pre-subscribe
position frames).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import config

log = logging.getLogger(__name__)

BACKLOG_LINES = 50  # sent on train-topic subscribe

LIVE_FRAME = "live"
GAME_END_FRAME = "game_end"
STREAM_KINDS = frozenset({LIVE_FRAME, GAME_END_FRAME})

# Field-name aliases tolerated on journal lines (the writer's exact names may shift).
_GAME_KEYS: tuple[str, ...] = ("game", "game_id", "game_index")
_PLY_KEYS: tuple[str, ...] = ("ply", "move_number", "move_no")
_MOVE_KEYS: tuple[str, ...] = ("move", "uci", "move_uci")
_FEN_KEYS: tuple[str, ...] = ("fen", "position", "board")
_RESULT_KEYS: tuple[str, ...] = ("result", "outcome")
_PLIES_KEYS: tuple[str, ...] = ("plies", "ply_count", "num_plies", "moves")
_RUN_KEYS: tuple[str, ...] = ("run_id", "run")
_TS_KEYS: tuple[str, ...] = ("ts", "timestamp", "time")

_cursor = 0  # byte offset of the next unread journal line (event-loop use only)


def _decode(raw: str | bytes) -> str:
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw


def _first(data: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """First non-None value among alias ``keys`` (writer-shape tolerance)."""
    for key in keys:
        if data.get(key) is not None:
            return data[key]
    return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


def parse_event(raw: str | bytes) -> dict[str, Any] | None:
    """The line's raw JSON object; None when blank, unparseable, or not an object."""
    text = _decode(raw).strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def parse_line(raw: str | bytes) -> dict[str, str] | None:
    """Map one journal line onto the pinned ``{level, message}`` log payload.

    Blank lines yield None; unparseable lines pass through verbatim at info.
    """
    text = _decode(raw).strip()
    if not text:
        return None
    data = parse_event(text)
    if data is None:
        return {"level": "info", "message": text}
    message = data.get("message")
    if message is None:
        message = json.dumps(data, sort_keys=True)
    return {"level": str(data.get("level", "info")), "message": str(message)}


def classify_line(raw: str | bytes) -> tuple[str, dict[str, Any]] | None:
    """Route one journal line: ``("live"|"game_end", payload)`` or ``("log", payload)``.

    None only for blank lines. See the module docstring for the tolerance
    rules; a live/game_end line lacking a routable ``game`` falls back to a
    log frame so nothing is silently lost.
    """
    text = _decode(raw).strip()
    if not text:
        return None
    data = parse_event(text)
    if data is None:
        return ("log", {"level": "info", "message": text})
    kind = str(data.get("kind", "")).strip().lower()
    game = _first(data, _GAME_KEYS)
    if kind in STREAM_KINDS and game is not None:
        ts = _as_float(_first(data, _TS_KEYS, 0.0))
        if kind == LIVE_FRAME:
            return (
                LIVE_FRAME,
                {
                    "game": _as_int(game),
                    "ply": _as_int(_first(data, _PLY_KEYS, 0)),
                    "move": _as_str(_first(data, _MOVE_KEYS, "")),
                    "fen": _as_str(_first(data, _FEN_KEYS, "")),
                    "run_id": _as_str(_first(data, _RUN_KEYS, "")),
                    "ts": ts,
                },
            )
        return (
            GAME_END_FRAME,
            {
                "game": _as_int(game),
                "result": _as_str(_first(data, _RESULT_KEYS, "")),
                "plies": _as_int(_first(data, _PLIES_KEYS, 0)),
                "ts": ts,
            },
        )
    payload = parse_line(text)
    return ("log", payload) if payload is not None else None


def backlog(limit: int = BACKLOG_LINES, path: Path | None = None) -> list[dict[str, str]]:
    """Last ``limit`` journal lines as LOG payloads, oldest first.

    FS-6b: ``live``/``game_end`` lines are EXCLUDED — old positions are noise;
    a client joining mid-stream gets live frames from subscription time
    onward. The window is the last ``limit`` lines, minus position frames, so
    a dense live stream can legitimately produce an empty backlog (the
    metrics snapshot + live tail cover the client).
    """
    target = path or config.TRAIN_LOG_PATH
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    frames: list[dict[str, str]] = []
    for line in lines[-limit:]:
        classified = classify_line(line)
        if classified is not None and classified[0] == "log":
            frames.append(classified[1])
    return frames


def delta(path: Path, offset: int) -> tuple[int, list[tuple[str, dict[str, Any]]]]:
    """Classified complete lines appended since ``offset``: ``(new_offset, events)``.

    A trailing partial line (writer mid-write) is left for the next call; a
    truncated/rotated file restarts the tail from the top.
    """
    try:
        current = path.stat().st_size
    except OSError:
        return offset, []
    if current < offset:
        offset = 0  # truncated or rotated: re-read from the top
    if current == offset:
        return offset, []
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
    except OSError:
        log.exception("journal read failed at offset %d", offset)
        return offset, []
    cut = chunk.rfind(b"\n")
    if cut < 0:
        return offset, []  # no complete line yet
    events = [c for raw in chunk[:cut].split(b"\n") if (c := classify_line(raw)) is not None]
    return offset + cut + 1, events


def size(path: Path | None = None) -> int:
    """Current journal size in bytes (0 when absent)."""
    target = path or config.TRAIN_LOG_PATH
    try:
        return target.stat().st_size
    except OSError:
        return 0


def reset_cursor() -> None:
    """Re-anchor the tail cursor to EOF (tail-loop start)."""
    global _cursor
    _cursor = size()


def seek_to_now() -> None:
    """Advance the cursor to EOF — the FS-6b stale-replay guard.

    Called on train-topic subscribe (before the subscriber is registered):
    lines appended while nobody watched are never replayed as live frames.
    Recent aggregate history still arrives via :func:`backlog`.
    """
    global _cursor
    _cursor = max(_cursor, size())


def delta_from_cursor(path: Path | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Classified events since the shared cursor; advances it."""
    global _cursor
    _cursor, events = delta(path or config.TRAIN_LOG_PATH, _cursor)
    return events
