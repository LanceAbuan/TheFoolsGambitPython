"""Neural opponent "nn" (FS-1): singleton champion loader + MCTS reply path.

Champion resolution: the newest valid ``runs/<regime>/champion.json`` pointer
across all regimes wins (mtime order — a fresh training run in any regime
promotes its champion to the play server). The pointer's ``checkpoint_path``
is resolved against the repo root when relative (the training pipeline writes
relative paths). The ``.pt`` checkpoint carries the embedded
:class:`training.model.ModelConfig` + state dict, so a champion trained with
any preset loads without extra wiring.

Reload: the pointer's mtime is checked on every move request; a newer champion
swaps in atomically under the module lock (the training pipeline replaces the
pointer file in place). A failed reload keeps the previous champion serving.

Device: CUDA when available, else CPU. The blocking MCTS search runs inside
``asyncio.to_thread`` under the session's lock in :mod:`server.engine`,
mirroring the Stockfish path; a crashed/missing model degrades to the Fool
for that move (never strands a live game).

Torch/training imports are lazy so API boot stays fast and no NN game pays
the import unless a model is actually used.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chess

from . import config

if TYPE_CHECKING:
    from .engine import Session

log = logging.getLogger(__name__)

CHAMPION_FILENAME = "champion.json"
DEFAULT_STRENGTH = "curious"

# FS-1 sims ladder, mirroring the Stockfish skill presets (wandering/curious/enlightened).
NN_SIMS: dict[str, int] = {"wandering": 32, "curious": 96, "enlightened": 256}


@dataclass(frozen=True)
class Champion:
    """A loaded champion: model + evaluator + provenance (immutable snapshot)."""

    model: Any  # training.model.GambitNet (loosely typed: torch stays a lazy import)
    evaluator: Any  # training.selfplay.BatchNNEvaluator
    device: Any  # torch.device
    tag: str
    elo: float
    regime: str
    mtime: float  # champion.json mtime this load is keyed on


def _resolve_ckpt_path(raw: object) -> Path | None:
    """Checkpoint path from a pointer's ``checkpoint_path`` (relative → repo root)."""
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = config.ROOT / path
    return path if path.is_file() else None


def _load_champion(pointer: Path, mtime: float) -> Champion:
    """Build a :class:`Champion` from a champion.json pointer (blocking I/O + torch)."""
    import torch

    from training.model import GambitNet, ModelConfig
    from training.selfplay import BatchNNEvaluator

    data = json.loads(pointer.read_text(encoding="utf-8"))
    ckpt = _resolve_ckpt_path(data.get("checkpoint_path", ""))
    if ckpt is None:
        raise FileNotFoundError(f"champion pointer {pointer} has no readable checkpoint")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_config = ModelConfig.from_dict(payload["config"])  # config embedded in the .pt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GambitNet(model_config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return Champion(
        model=model,
        evaluator=BatchNNEvaluator(model, device),
        device=device,
        tag=str(payload.get("tag", data.get("tag", "unknown"))),
        elo=float(data.get("elo", 800.0)),
        regime=pointer.parent.name,
        mtime=mtime,
    )


class ModelServer:
    """Thread-safe singleton holding at most one loaded champion."""

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._champion: Champion | None = None

    def resolve_champion(self) -> tuple[Path, float] | None:
        """(champion.json, mtime) of the newest valid pointer; None when absent.

        Valid = readable JSON whose ``checkpoint_path`` resolves to a file.
        """
        best: tuple[Path, float] | None = None
        for pointer in config.RUNS_DIR.glob(f"*/{CHAMPION_FILENAME}"):
            try:
                data = json.loads(pointer.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if _resolve_ckpt_path(data.get("checkpoint_path") if isinstance(data, dict) else None) is None:
                continue
            try:
                mtime = pointer.stat().st_mtime
            except OSError:
                continue
            if best is None or mtime > best[1]:
                best = (pointer, mtime)
        return best

    def available(self) -> bool:
        """Whether a champion exists on disk (gates ``create_game(opponent="nn")``)."""
        return self.resolve_champion() is not None

    @property
    def loaded(self) -> Champion | None:
        """The in-memory champion (None before first use / after shutdown)."""
        return self._champion

    def current(self) -> Champion | None:
        """Current champion, (re)loading when champion.json's mtime changed.

        Thread-safe: loads/swap happen under the module lock; a vanished or
        broken pointer keeps the previously loaded champion serving.
        """
        with self._mutex:
            pointer = self.resolve_champion()
            if pointer is None:
                return self._champion  # pointer removed: keep serving the in-memory model
            path, mtime = pointer
            if self._champion is not None and self._champion.mtime == mtime:
                return self._champion
            try:
                champion = _load_champion(path, mtime)
            except Exception:
                log.exception("failed to load champion from %s; keeping previous", path)
                return self._champion
            self._champion = champion
            log.info(
                "champion loaded: regime=%s tag=%s elo=%.1f device=%s",
                champion.regime, champion.tag, champion.elo, champion.device,
            )
            return champion

    def best_move(self, session: Session) -> chess.Move | None:
        """Blocking NN reply for the session's side to move. Runs on a worker thread.

        Raises ``RuntimeError`` when no champion can be loaded — the engine
        caller degrades that move to the Fool (mirrors the Stockfish crash path).
        """
        board = session.board
        if board.is_game_over():
            return None
        champion = self.current()
        if champion is None:
            raise RuntimeError("no trained model available")
        from training.selfplay import MCTSSearch, SearchConfig

        sims = NN_SIMS.get(session.strength, NN_SIMS[DEFAULT_STRENGTH])
        search = MCTSSearch(champion.evaluator, SearchConfig(num_simulations=sims, root_noise=False))
        return search.search(board).chosen_move

    def shutdown(self) -> None:
        """Drop the loaded model and free device memory (app shutdown / tests)."""
        with self._mutex:
            champion, self._champion = self._champion, None
        if champion is None:
            return
        try:
            import torch

            device_type = champion.device.type
            del champion.model, champion.evaluator
            if device_type == "cuda":
                torch.cuda.empty_cache()
            log.info("champion model released (device %s)", device_type)
        except Exception:
            log.debug("champion cleanup failed", exc_info=True)


_singleton = ModelServer()


def resolve_champion() -> tuple[Path, float] | None:
    """Singleton: newest valid champion pointer (see :meth:`ModelServer.resolve_champion`)."""
    return _singleton.resolve_champion()


def available() -> bool:
    """Singleton: whether any champion exists on disk."""
    return _singleton.available()


def loaded() -> Champion | None:
    """Singleton: the in-memory champion, if any."""
    return _singleton.loaded


def best_move(session: Session) -> chess.Move | None:
    """Singleton: blocking NN reply (see :meth:`ModelServer.best_move`)."""
    return _singleton.best_move(session)


async def warmup_async() -> None:
    """Startup pre-warm: load the champion off the event loop (no-op without one).

    Keeps the first NN move inside the < 2s gate — otherwise torch/CUDA init
    + model load land inside the user's first move (~3s cold on this box).
    """
    try:
        await asyncio.to_thread(_singleton.current)
    except Exception:  # pre-warm is best-effort; per-move fallback exists
        log.exception("champion pre-warm failed")


def shutdown() -> None:
    """Singleton: release the model + device memory."""
    _singleton.shutdown()


def reset_for_tests() -> None:
    """Test isolation: release any in-memory champion."""
    _singleton.shutdown()
