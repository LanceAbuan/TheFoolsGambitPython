"""Training journal reader (FS-6): parsing + tailing ``data/train_log.jsonl``.

The training pipeline (FS-6 writer half) appends structured events, one JSON
object per line. This module is pure file logic — no hub, no event loop — so
both the WS subscribe path (:mod:`server.ws`) and the 2-second tail task
(:mod:`server.telemetry`) can share it without import cycles.

Line → payload mapping follows the §14.0 log frame (``{level, message}``):
a JSON object uses its ``level`` (default ``"info"``) and ``message`` fields
(objects without ``message`` are rendered as compact JSON); anything
non-JSON is passed through as an info-level message.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

BACKLOG_LINES = 50  # sent on train-topic subscribe


def parse_line(raw: str | bytes) -> dict[str, str] | None:
    """Map one journal line onto the pinned ``{level, message}`` payload.

    Blank lines yield None; unparseable lines pass through verbatim at info.
    """
    text = (raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw).strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return {"level": "info", "message": text}
    if not isinstance(data, dict):
        return {"level": "info", "message": text}
    message = data.get("message")
    if message is None:
        message = json.dumps(data, sort_keys=True)
    return {"level": str(data.get("level", "info")), "message": str(message)}


def backlog(limit: int = BACKLOG_LINES, path: Path | None = None) -> list[dict[str, str]]:
    """Last ``limit`` parsed journal lines, oldest first; empty when no file."""
    target = path or config.TRAIN_LOG_PATH
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    payloads = [payload for line in lines if (payload := parse_line(line)) is not None]
    return payloads[-limit:]


def delta(path: Path, offset: int) -> tuple[int, list[dict[str, str]]]:
    """Complete lines appended since ``offset``: returns ``(new_offset, payloads)``.

    A trailing partial line (writer mid-write) is left for the next call; a
    truncated/rotated file restarts the tail from the top.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return offset, []
    if size < offset:
        offset = 0  # truncated or rotated: re-read from the top
    if size == offset:
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
    payloads = [p for raw in chunk[:cut].split(b"\n") if (p := parse_line(raw)) is not None]
    return offset + cut + 1, payloads


def size(path: Path | None = None) -> int:
    """Current journal size in bytes (0 when absent) — the tail's initial offset."""
    target = path or config.TRAIN_LOG_PATH
    try:
        return target.stat().st_size
    except OSError:
        return 0
