"""Resumable runs (FS-5) + warm-start resolution (FS-3 hybrid/finetune).

Run state — everything needed to rebuild a run EXACTLY after a kill:
model weights, optimizer state, replay-buffer contents, RNG states (torch,
numpy global, and the run's play generator), the iteration counter, and the
champion provenance. Persisted via `torch.save` under
``runs/<regime>/<run-id>/state/state-<steps>.pt`` at every checkpoint
(`training.pipeline` owns the cadence) and restored by
`load_latest_run_state`.

Warm-start resolution — `RegimeConfig.warmstart_from` accepts:
  - a regime name ("sf-distill")  -> runs/<regime>/champion.json pointer
  - "champion"                    -> the newest champion across all regimes
  - a filesystem path to a .pt    -> loaded directly
  - an HF repo id ("user/repo")   -> downloaded from the Hub (best-effort)
Anything unresolvable logs a warning and returns None: a missing warm start
must degrade to a fresh run, never crash one.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from .regimes import REGIME_NAMES, RunPaths

log = logging.getLogger(__name__)

STATE_PREFIX = "state-"


@dataclass
class RunState:
    """The restorable slice of one run's progress (FS-5)."""

    iterations_completed: int
    steps: int
    games_played: int
    champion_tag: str | None
    champion_elo: float
    profile_signature: dict[str, object]


def profile_signature(profile: object) -> dict[str, object]:
    """Serializable profile snapshot saved with every state (mismatch guard)."""
    from .pipeline import TrainingProfile  # local import avoids a cycle

    assert isinstance(profile, TrainingProfile)
    return {
        "name": profile.name,
        "regime": profile.regime,
        "model": profile.model.to_dict(),
        "iterations": profile.iterations,
        "games_per_iteration": profile.games_per_iteration,
        "max_plies": profile.max_plies,
        "sims": profile.sims,
        "train_steps": profile.train_steps,
        "batch_size": profile.batch_size,
        "min_examples": profile.min_examples,
        "buffer_size": profile.buffer_size,
        "parallel_games": profile.parallel_games,
        "ladder": {
            "markers_minutes": list(profile.ladder.markers_minutes),
            "repeat_every_minutes": profile.ladder.repeat_every_minutes,
            "promote_win_rate": profile.ladder.promote_win_rate,
            "arena_games": profile.ladder.arena.games,
        },
    }


def validate_profile_signature(
    saved: dict[str, object], current: dict[str, object]
) -> None:
    """Refuse to resume a run whose identity-critical knobs changed."""
    strict_keys = ("regime", "model", "buffer_size", "games_per_iteration", "batch_size")
    for key in strict_keys:
        if saved.get(key) != current.get(key):
            raise ValueError(
                f"cannot resume run: profile '{key}' changed "
                f"({saved.get(key)!r} -> {current.get(key)!r})"
            )
    drifted = [key for key in saved if key not in strict_keys and saved.get(key) != current.get(key)]
    if drifted:
        log.warning("[RESUME] profile drift on non-strict keys: %s", ", ".join(sorted(drifted)))


class ReplayBufferLike(Protocol):
    """Structural slice of the legacy ``TrainingBuffer`` that FS-5 persists.

    Declared here so resume never imports the legacy trainer module (and its
    heavyweight module side effects) just to read the replay deque.
    """

    buffer: deque[dict[str, Any]]

    def add(self, examples: object) -> None:
        """Append examples (a list or generator) to the replay deque."""
        ...  # pragma: no cover


def save_run_state(
    paths: RunPaths,
    *,
    iterations_completed: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBufferLike,
    play_rng: np.random.Generator,
    steps: int,
    games_played: int,
    champion_tag: str | None,
    champion_elo: float,
    profile: object,
) -> Path:
    """Persist everything needed to rebuild the run exactly (FS-5)."""
    examples = list(buffer.buffer)
    payload = {
        "iterations_completed": iterations_completed,
        "steps": steps,
        "games_played": games_played,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "buffer_examples": examples,
        "buffer_max_size": buffer.buffer.maxlen,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "play_rng_state": play_rng.bit_generator.state,
        "champion_tag": champion_tag,
        "champion_elo": champion_elo,
        "profile_signature": profile_signature(profile),
        "regime": paths.regime,
        "run_id": paths.run_id,
        "ts": time.time(),
    }
    target = paths.state / f"{STATE_PREFIX}{steps:06d}.pt"
    torch.save(payload, target)
    log.info(
        "[RESUME] state saved: %s (iter %d, steps %d, buffer %d)",
        target,
        iterations_completed,
        steps,
        len(examples),
    )
    return target


def load_latest_run_state(paths: RunPaths) -> dict[str, object] | None:
    """Newest persisted state for this run; None when the run has no state yet."""
    candidates = sorted(paths.state.glob(f"{STATE_PREFIX}*.pt"))
    if not candidates:
        return None
    latest = candidates[-1]  # zero-padded step stamp sorts chronologically
    try:
        state = torch.load(latest, map_location="cpu", weights_only=False)
    except Exception:  # torn/corrupt state files degrade to a fresh start
        log.exception("[RESUME] state file %s unreadable; starting fresh", latest)
        return None
    log.info("[RESUME] loaded state %s", latest)
    if isinstance(state, dict):
        return state
    return None


def restore_run_state(
    state: dict[str, object],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBufferLike,
    play_rng: np.random.Generator,
) -> RunState:
    """Rebuild model/optimizer/buffer/RNGs in place; returns the progress counters."""
    model.load_state_dict(state["model_state"])  # type: ignore[arg-type]
    optimizer.load_state_dict(state["optimizer_state"])  # type: ignore[arg-type]
    examples = state.get("buffer_examples", [])
    buffer.add(examples)
    torch.set_rng_state(state["torch_rng_state"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy_rng_state"])  # type: ignore[call-overload]
    play_rng.bit_generator.state = state["play_rng_state"]  # type: ignore[assignment]
    run_state = RunState(
        iterations_completed=int(state["iterations_completed"]),  # type: ignore[arg-type]
        steps=int(state["steps"]),  # type: ignore[arg-type]
        games_played=int(state["games_played"]),  # type: ignore[arg-type]
        champion_tag=state.get("champion_tag"),  # type: ignore[arg-type]
        champion_elo=float(state.get("champion_elo", 800.0)),  # type: ignore[arg-type]
        profile_signature=dict(state.get("profile_signature", {})),  # type: ignore[arg-type]
    )
    log.info(
        "[RESUME] restored: iterations=%d steps=%d games=%d buffer=%d champion=%s",
        run_state.iterations_completed,
        run_state.steps,
        run_state.games_played,
        len(examples),
        run_state.champion_tag,
    )
    return run_state


# ---------------------------------------------------------------------------
# Warm-start resolution (FS-3 hybrid/finetune)
# ---------------------------------------------------------------------------


def _champion_pointer(base_dir: Path, regime: str) -> Path | None:
    pointer = base_dir / regime / "champion.json"
    return pointer if pointer.is_file() else None


def _newest_champion_pointer(base_dir: Path) -> Path | None:
    """The most recently updated champion.json across every regime."""
    best: tuple[float, Path] | None = None
    for regime in REGIME_NAMES:
        pointer = _champion_pointer(base_dir, regime)
        if pointer is None:
            continue
        try:
            updated = float(json.loads(pointer.read_text(encoding="utf-8")).get("updated_at", 0.0))
        except (OSError, ValueError):
            continue
        if best is None or updated > best[0]:
            best = (updated, pointer)
    return best[1] if best is not None else None


def _checkpoint_from_pointer(pointer: Path) -> Path | None:
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    checkpoint = Path(str(data.get("checkpoint_path", "")))
    return checkpoint if checkpoint.is_file() else None


def _download_hf_checkpoint(repo_id: str) -> Path | None:
    """Best-effort Hub download of a regime repo's checkpoint.pt."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.warning("[WARMSTART] huggingface_hub not installed; cannot fetch %s", repo_id)
        return None
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename="checkpoint.pt"))
    except Exception as exc:  # noqa: BLE001 - Hub misses degrade to fresh init
        log.warning("[WARMSTART] HF download from %s failed: %s", repo_id, exc)
        return None


def resolve_warmstart(spec: str, base_dir: Path | str = "runs") -> dict[str, torch.Tensor] | None:
    """`warmstart_from` spec -> a model state_dict; None degrades to fresh init."""
    base = Path(base_dir)
    checkpoint: Path | None = None
    if spec.startswith("hf:") or "/" in spec:
        checkpoint = _download_hf_checkpoint(spec.removeprefix("hf:"))
    elif spec == "champion":
        pointer = _newest_champion_pointer(base)
        checkpoint = _checkpoint_from_pointer(pointer) if pointer else None
    elif spec in REGIME_NAMES:
        pointer = _champion_pointer(base, spec)
        checkpoint = _checkpoint_from_pointer(pointer) if pointer else None
    else:
        candidate = Path(spec)
        checkpoint = candidate if candidate.is_file() else None

    if checkpoint is None:
        log.warning("[WARMSTART] spec '%s' resolved to nothing; training from scratch", spec)
        return None
    try:
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - corrupt/partial checkpoints degrade to fresh init
        log.warning("[WARMSTART] checkpoint %s unreadable; training from scratch", checkpoint)
        return None
    state = data.get("model_state") if isinstance(data, dict) else None
    if state is None:
        log.warning("[WARMSTART] checkpoint %s has no model_state", checkpoint)
        return None
    log.info("[WARMSTART] loaded weights from %s (spec '%s')", checkpoint, spec)
    return state  # type: ignore[return-value]
