"""Trainer-process liveness — the fact behind ``gate_in_progress``.

Journal freshness cannot answer "is the trainer running?". Measured behavior of
the production trainer (2026-11-21, run ``second-light``): ``data/train_log.jsonl``
streams ~2.4 lines/s through self-play and the short (~2 s) training phase,
then goes **completely silent for the whole arena gate** — gates are not
broadcast move by move — and resumes with the ``iteration`` line when the gate
result lands. The gap between the checkpoint line before a gate and the
iteration line after it is 40-80 minutes, i.e. journal silence during exactly
the window it used to be treated as liveness evidence for.

So liveness is read from the OS process table instead: is there a process whose
cmdline runs ``scripts/train.py``? That answer is what
:func:`server.api.train._gate_hint` builds on.

Scan cost + caching: a full cmdline pass costs ~1.3 s on the production box
under trainer load (~360 processes, dominated by per-process handle opens;
psutil returns ``None`` for access-denied junk, it does not raise). The metrics
feed polls every 10 s per subscribed client plus REST traffic, so the boolean is
cached for :data:`CACHE_TTL_SECONDS` on a monotonic clock — "is the trainer
running" tolerates 30 s of staleness and the scan must never be per-request.
30 s keeps the cost at ~4 % of one core instead of ~13 % on a box that is
already training, while the hint's resolution is minutes.
The scan never runs directly on the event loop: REST ``/api/metrics`` is a sync
endpoint (threadpool) and the WS/telemetry feeds call ``metrics_snapshot`` via
``asyncio.to_thread`` (:mod:`server.telemetry`, :mod:`server.ws`).

Failure posture: a failed scan is False, never an exception — "we could not
look" must not fabricate a running trainer, and the caller's hint simply stays
honest about the facts it does have.

Test seams: :func:`reset_cache` drops the cached answer, and both
:data:`_cmdlines` (process enumeration) and :data:`_now` (clock) are module
attributes tests replace so no test ever depends on real local processes.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import psutil

log = logging.getLogger(__name__)

TRAINER_SCRIPT = "scripts/train.py"  # matched after backslash normalization
CACHE_TTL_SECONDS = 30.0

# Injectable clock (tests replace this; production code never reassigns it).
_now: Callable[[], float] = time.monotonic

_cache_lock = threading.Lock()
_cache_at: float | None = None  # _now() of the last completed scan; None = empty
_cache_alive: bool = False


def _normalize(cmdline: str) -> str:
    """One cmdline string in match form: forward slashes, lowercased."""
    return cmdline.replace("\\", "/").lower()


def _read_cmdlines() -> list[str]:
    """Every visible process's cmdline, space-joined (raw; matching normalizes).

    Never raises: processes vanish and system processes deny access mid-scan,
    and those entries are simply skipped. A failure of the enumeration itself
    yields the lines gathered so far (usually none).
    """
    lines: list[str] = []
    try:
        for proc in psutil.process_iter(["cmdline"]):
            try:
                parts = (proc.info or {}).get("cmdline") or []
            except (psutil.Error, OSError):  # pragma: no cover - teardown race
                continue
            lines.append(" ".join(str(part) for part in parts))
    except (psutil.Error, OSError):
        log.warning("trainer liveness: process scan failed", exc_info=True)
    return lines


_cmdlines: Callable[[], list[str]] = _read_cmdlines


def _scan_processes(run_id: str | None) -> bool:
    """True when a ``scripts/train.py`` process is running (never raises).

    ``run_id`` scopes the match to a trainer whose cmdline carries that run id
    (``--run-id <id>``; a bare substring is enough, so ``--run-id=id`` matches
    too). When no process carries the id — a trainer launched without the flag
    generates its own timestamp id, so the journal's run id may not appear in
    any cmdline — the plain ``scripts/train.py`` match still answers, because a
    false "not running" is precisely the defect this probe exists to fix. The
    unscoped fallback is logged at debug; with the single-trainer deployment it
    is not observable, and a false "running" would require a second, orphaned
    trainer for a different run.
    """
    try:
        cmdlines = _cmdlines()
    except (psutil.Error, OSError):
        log.warning("trainer liveness: process scan failed", exc_info=True)
        return False
    normalized = [_normalize(line) for line in cmdlines]
    trainers = [line for line in normalized if TRAINER_SCRIPT in line]
    if not trainers:
        return False
    if run_id:
        scoped = _normalize(run_id)
        if any(scoped in line for line in trainers):
            return True
        log.debug("trainer alive but run id %r absent from its cmdline", run_id)
    return True


def trainer_alive(run_id: str | None = None) -> bool:
    """Whether a trainer process is running, cached for ``CACHE_TTL_SECONDS``.

    The cached answer is shared across callers and run ids: a run id that
    appears/disappears is picked up on the next TTL expiry (≤30 s), which is
    far finer than the 40-80 min gate windows this feeds.
    """
    global _cache_at, _cache_alive
    now = _now()
    with _cache_lock:
        if _cache_at is not None and (now - _cache_at) < CACHE_TTL_SECONDS:
            return _cache_alive
        _cache_alive = _scan_processes(run_id)
        _cache_at = now
        return _cache_alive


def reset_cache() -> None:
    """Drop the cached answer (tests; the next call rescans immediately)."""
    global _cache_at, _cache_alive
    with _cache_lock:
        _cache_at = None
        _cache_alive = False
