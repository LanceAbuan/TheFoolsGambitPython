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

State-file format (audit F-2): the replay buffer persists as an fp16 npz blob
(`buffer_npz`) instead of a pickled per-example list -- planes are 0/1 and
value targets are game outcomes, both exact in fp16; policy targets round at
~1e-3 relative (see _BUFFER_FP16). Legacy state files (pickled
`buffer_examples`) still restore, and only the newest STATE_FILES_TO_KEEP
state files are kept on disk.
"""
from __future__ import annotations

import io
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
# Audit F-2: the replay buffer dominates state-file size (~26 KB/example as a
# pickled list). Tensors persist as an fp16 npz blob instead: planes are 0/1
# and value targets are game outcomes, both exact in fp16; policy targets take
# ~1e-3 relative rounding, which is acceptable for resume fidelity. Flip to
# False for exact-fp32 state files (roughly 2x larger).
_BUFFER_FP16 = True
# Audit F-2: state files previously accumulated unbounded (one full-buffer
# dump per checkpoint); keep only the newest few.
STATE_FILES_TO_KEEP = 2


@dataclass
class RunState:
    """The restorable slice of one run's progress (FS-5)."""

    iterations_completed: int
    steps: int
    games_played: int
    champion_tag: str | None
    champion_elo: float
    profile_signature: dict[str, object]
    # U9: the champion lineage's Glicko-2 record {r, rd, sigma}; None on
    # state files written before U9 (restore falls back to the ledger
    # replay, then the default prior — never a crash).
    champion_glicko: dict[str, float] | None = None


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


def _buffer_npz_bytes(examples: list[dict[str, Any]]) -> bytes:
    """Encode replay examples as an in-memory npz (planes, policies, values)."""
    dtype: type = np.float16 if _BUFFER_FP16 else np.float32
    if examples:
        planes = np.stack([np.asarray(ex["board_tensor"], dtype=np.float32) for ex in examples])
        policies = np.stack([np.asarray(ex["policy"], dtype=np.float32) for ex in examples])
    else:
        planes = np.zeros((0, 0, 0, 0), dtype=np.float32)
        policies = np.zeros((0, 0), dtype=np.float32)
    values = np.array([float(ex["value"]) for ex in examples], dtype=np.float32)
    handle = io.BytesIO()
    np.savez(handle, planes=planes.astype(dtype), policies=policies.astype(dtype), values=values)
    return handle.getvalue()


def _examples_from_npz(data: bytes) -> list[dict[str, Any]]:
    """Decode an npz blob back into replay-example dicts (fp16 -> fp32)."""
    with np.load(io.BytesIO(data)) as npz:
        planes = npz["planes"].astype(np.float32)
        policies = npz["policies"].astype(np.float32)
        values = npz["values"].astype(np.float32)
    return [
        {"board_tensor": planes[i], "policy": policies[i], "value": float(values[i])}
        for i in range(values.shape[0])
    ]


def _prune_state_files(paths: RunPaths, keep: int = STATE_FILES_TO_KEEP) -> list[Path]:
    """Delete all but the newest `keep` state files (zero-padded = chronological)."""
    states = sorted(paths.state.glob(f"{STATE_PREFIX}*.pt"))
    stale = states[:-keep] if len(states) > keep else []
    for path in stale:
        path.unlink(missing_ok=True)
    if stale:
        log.info("[RESUME] pruned %d old state file(s); keeping newest %d", len(stale), keep)
    return stale


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
    champion_glicko: dict[str, float] | None = None,
) -> Path:
    """Persist everything needed to rebuild the run exactly (FS-5).

    Buffer tensors go in as an fp16 npz blob (see _BUFFER_FP16): the pickled
    per-example list this replaced cost ~26 KB/example and made night-profile
    state files multi-GB. Older state files are pruned to the newest
    STATE_FILES_TO_KEEP. `champion_glicko` (U9) is additive: state files
    from before it existed simply lack the key, and restore reads it with a
    default so a crash-restart across the upgrade stays resume-exact.
    """
    examples = list(buffer.buffer)
    payload = {
        "iterations_completed": iterations_completed,
        "steps": steps,
        "games_played": games_played,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "buffer_npz": _buffer_npz_bytes(examples),
        "buffer_count": len(examples),
        "buffer_max_size": buffer.buffer.maxlen,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "play_rng_state": play_rng.bit_generator.state,
        "champion_tag": champion_tag,
        "champion_elo": champion_elo,
        "champion_glicko": champion_glicko,
        "profile_signature": profile_signature(profile),
        "regime": paths.regime,
        "run_id": paths.run_id,
        "ts": time.time(),
    }
    target = paths.state / f"{STATE_PREFIX}{steps:06d}.pt"
    torch.save(payload, target)
    _prune_state_files(paths)
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
    if state.get("buffer_npz") is not None:
        examples = _examples_from_npz(bytes(state["buffer_npz"]))
    else:  # legacy pre-F-2 state files carried a pickled example list
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
        champion_glicko=state.get("champion_glicko") if isinstance(state.get("champion_glicko"), dict) else None,
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
