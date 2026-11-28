"""WP-C training pipeline v2 — the self-play -> optimize -> gate loop.

Orchestrates (all CPU-safe; CUDA used when available):
  1. regime config + lineage (`runs/<regime>/<run-id>/`, §5.2)
  2. data collection via the regime's provider (FS-3) — parallel games feed the
     queued evaluator (`training.evaluator`, FS-4) when parallel_games > 1
  3. replay-buffer sampling + optimization (reuses legacy `TrainingBuffer`)
  4. wall-clock checkpoint ladder + arena gating (§5.1, `ladder.py`)
  5. arena ledger JSONL + metrics feed (`ledger.py`)
  6. resumable state at every checkpoint (FS-5, `resume.py`) + per-regime HF
     upload (`hf.py`) + the FS-6 training journal (`journal.py`)

`scripts/train.py` is a thin CLI over `run_training`; tests import it directly.
Profiles are dataclasses — `smoke` finishes in well under two minutes on CPU,
`night` is the real run the orchestrator starts.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .encoder import ENCODER_VERSION, NUM_PLANES, NUM_PLANES_V2
from .ema import EmaWeights
from .tablebase import TablebaseCorrector
from .evaluator import QueuedNNEvaluator
from .glicko import GlickoRating, expected_games_to_accept, match_honesty
from .hf import upload_checkpoint
from .journal import TrainingJournal
from .ladder import (
    Arena,
    LadderConfig,
    MatchResult,
    PromotionDecision,
    decide_promotion,
    due_snapshot_minutes,
    update_elo,
    update_match_pair,
)
from .ledger import ArenaLedger, LedgerEntry, MetricsWriter, recompute_elos, recompute_glickos
from .eval_set import EVAL_SET_FILENAME, load_eval_set, score
from .model import GambitNet, ModelConfig
from .openings import gate_opening_seed
from .regimes import RegimeConfig, RunPaths, build_data_provider, create_run, get_regime
from .resume import (
    load_latest_run_state,
    profile_signature,
    resolve_warmstart,
    restore_run_state,
    save_run_state,
    validate_profile_signature,
)
from .rust_workers import (
    collect_from_spool,
    export_model_to_onnx,
    prune_exported_models,
    prune_processed_games,
    purge_stale_games,
    record_to_selfplay_result,
)
from .selfplay import (
    ArenaConfig,
    BatchNNEvaluator,
    Evaluator,
    SearchConfig,
    SelfPlayConfig,
    SelfPlayResult,
    StreamContext,
)
from .trainer import TrainingBuffer, wdl_targets

log = logging.getLogger(__name__)
# Cached frozen eval-set tensors; scoring reloads only when the path changes.
_eval_cache: dict[str, object] = {}

CHAMPION_FILENAME = "champion.json"


@dataclass(frozen=True)
class LRStepSchedule:
    """Step learning-rate decay on ABSOLUTE iteration milestones (WP-A).

    The LR for iteration `i` is ``base_lr * gamma ** (#milestones <= i)``.
    Because the rung depends only on the absolute iteration number — never on
    steps-since-start — the schedule is stable under `--resume` from any
    iteration: a run resumed at iteration 350 lands on exactly the rung it
    would have been on had it never stopped.

    `base_lr` is passed per call (the pipeline feeds the regime's
    `learning_rate`) so one schedule instance stays regime-agnostic.
    """

    milestones: tuple[int, ...]
    gamma: float = 0.5

    def __post_init__(self) -> None:
        if self.gamma <= 0 or self.gamma > 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if any(m < 0 for m in self.milestones) or list(self.milestones) != sorted(set(self.milestones)):
            raise ValueError("milestones must be non-negative, strictly increasing integers")

    def lr_at(self, iteration: int, base_lr: float) -> float:
        """LR for `iteration`: one gamma decay per milestone already reached."""
        decayed = sum(1 for milestone in self.milestones if iteration >= milestone)
        return base_lr * (self.gamma**decayed)

    def to_dict(self) -> dict[str, object]:
        return {"milestones": list(self.milestones), "gamma": self.gamma}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> LRStepSchedule:
        return cls(
            milestones=tuple(int(m) for m in data["milestones"]),  # type: ignore[arg-type]
            gamma=float(data["gamma"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class TrainingProfile:
    """One runnable slice of training. `smoke` for CI/sanity, `night` for real."""

    name: str
    model: ModelConfig
    regime: str = "scratch"
    iterations: int = 2
    games_per_iteration: int = 2
    max_plies: int = 8
    sims: int = 2
    train_steps: int = 2
    batch_size: int = 8
    min_examples: int = 4
    # 100k examples of 104x8x8 fp32 planes = ~2.7 GB in RAM plus ~1.9 GB for
    # policies. With the 4 GB Windows pagefile and the CUDA fleet that is the
    # difference between training and `buffer.sample()` dying on a 1.3 MB
    # allocation (2026-11-27 crash loop). 40k keeps ~2.5x its own batch size of
    # fresh rows and fits the commit budget; `buffer_size` is now a drift-warn
    # resume key (a memory knob, not run identity).
    buffer_size: int = 40_000
    ladder: LadderConfig = field(default_factory=LadderConfig)
    seed: int | None = 0
    snapshot_every_iteration: bool = False  # smoke: gate every iteration
    # Durability, independent of the gate: save a checkpoint + resumable state
    # every N iterations even when the ladder snapshot is not due. Checkpoints
    # used to exist ONLY on the gate cadence (90 min here), so a crash between
    # gates lost everything since the last snapshot - unacceptable at 14
    # iterations/hr. 0 disables the extra writes.
    checkpoint_every_iterations: int = 0
    parallel_games: int | None = None  # None -> RegimeConfig.parallel_games
    stream_games: bool | None = None  # None -> RegimeConfig.stream_moves (FS-6b)
    # WP-A: optional step-decay LR schedule (milestones on ABSOLUTE iterations).
    # None -> the regime's constant `learning_rate` exactly as before, so every
    # pre-existing profile (smoke, night) keeps its behavior byte-identical.
    lr_schedule: LRStepSchedule | None = None
    # WP-F C5: which MCTS engine runs self-play/arena searches. The default
    # stays "python" (MCTSSearch) until the C6 gates justify flipping it;
    # "rust" routes through training.rust_search.RustMCTSSearch.
    search_backend: str = "python"
    # WP-H H5: self-play game source. "python" (default - every pre-H4
    # profile and the live run keep byte-identical behavior) collects
    # games in-process via the regime provider; "rust_workers" drains a
    # spool directory of games played by external
    # gambit_core.selfplay_game worker processes (training.rust_workers).
    # The default stays "python" until the H6 gates justify a flip.
    selfplay_backend: str = "python"
    # WP-H H4: worker spool directory override. None -> "<run root>/spool"
    # (per-run lineage dir; workers get the same path from the launcher).
    spool_dir: str | None = None
    # WP-H H4: how long one iteration blocks for worker games before
    # proceeding with a shortfall (the min_examples guard skips training
    # until the buffer is viable). Not part of the resume signature.
    spool_timeout_s: float = 600.0
    # Frozen eval-set scoring cadence (0 = off). See training/eval_set.py: the
    # live-buffer metrics are not comparable across iterations, this is.
    eval_every: int = 0
    eval_set_path: str | None = None
    gate_every_iterations: int = 0
    # WP4: exponential moving average of the weights (0 = disabled). The
    # EMA is exported to the workers and used as the gate candidate once
    # iteration >= ema_min_iteration (early EMA still carries weak weights).
    ema_decay: float = 0.0
    ema_min_iteration: int = 2000
    # WP10: optimizer package (all defaults = legacy behavior).
    optimizer_kind: str = "adam"   # "adam" (legacy) | "adamw"
    weight_decay: float = 0.0
    grad_clip: float = 0.0          # 0 = off
    lr_warmup_steps: int = 0        # linear warmup from ~0 over N steps
    lr_cosine: bool = False         # cosine decay base_lr -> floor over the run
    lr_floor_ratio: float = 0.01
    base_lr: float | None = None    # override the regime learning rate
    # WP8: directory with Syzygy 3-4-5 WDL tables. When set, <=5-man rows
    # get perfect tablebase value targets. None = disabled.
    tb_path: str | None = None

    def __post_init__(self) -> None:
        """WP-H H5: validate the backend selector at construction time."""
        if self.optimizer_kind not in ("adam", "adamw"):
            raise ValueError(
                f"optimizer_kind must be 'adam' or 'adamw', got {self.optimizer_kind!r}"
            )
        if self.selfplay_backend not in ("python", "rust_workers"):
            raise ValueError(
                f"selfplay_backend must be 'python' or 'rust_workers', "
                f"got {self.selfplay_backend!r}"
            )


SMOKE_PROFILE = TrainingProfile(
    name="smoke",
    model=ModelConfig(in_channels=NUM_PLANES, num_blocks=2, filters=32),
    regime="scratch",
    iterations=2,
    games_per_iteration=2,
    max_plies=8,
    sims=2,
    train_steps=2,
    batch_size=8,
    min_examples=4,
    ladder=LadderConfig(arena=ArenaConfig(num_simulations=2, max_plies=8, games=2)),
    seed=0,
    snapshot_every_iteration=True,
)

NIGHT_PROFILE = TrainingProfile(
    name="night",
    model=ModelConfig(in_channels=NUM_PLANES, num_blocks=10, filters=128, value_channels=32, value_fc=512),
    regime="scratch",
    iterations=1000,
    games_per_iteration=32,
    max_plies=200,
    sims=400,
    train_steps=64,
    batch_size=256,
    min_examples=64,
    ladder=LadderConfig(arena=ArenaConfig(num_simulations=128, max_plies=200, games=16)),
    seed=None,
    parallel_games=16,
)

# WP-A: throughput knobs on top of `night` — the live run's profile is NOT
# edited in place (its on-disk state must stay resumable byte-identically).
# Changes: parallel_games 16->32 (exactly at MAX_PARALLEL_GAMES), sims
# 400->600, train_steps 64->96, batch_size 256->512, min_examples 64->128,
# plus the step-decay LR schedule (x0.5 at iterations 200/400/600/800).
# The arena stays at the section 5.1 reduced settings (128 sims / 16 games).
#
# WP-D (AMENDED — B3 capacity bump DEFERRED, orchestrator decision): the v2
# net stays 10 blocks x 128 filters. Residual conv shapes depend on the
# filter count ([128,128,3,3] vs [160,160,3,3]), so a 12x160 tower could not
# accept the v1 tower's weights and the warm-start would be impossible; at
# ~750 absolute Elo the transferable knowledge outweighs fresh capacity.
# Capacity can rise at a future generation once the loop is proven. What v2
# changes: the 104-channel encoder (WP-B) and the W/D/L value head (WP-C).
NIGHT_V2_PROFILE = replace(
    NIGHT_PROFILE,
    name="night-v2",
    model=ModelConfig(
        in_channels=NUM_PLANES_V2,
        num_blocks=10,
        filters=128,
        value_channels=32,
        value_fc=512,
        value_head="wdl",
    ),
    parallel_games=32,
    sims=600,
    # GPU utilisation fix (2026-11-24): a 64-game iteration costs ~25 min of
    # fleet time but only ~11 s of GPU (96 steps x 512). 288 was tried and
    # REVERTED the same evening: with the fleet at ~600 games/hr an iteration
    # only drains the half-batch minimum (~34 games), so 288 steps on 34 fresh
    # games is ~12 passes over them while replaying a saturated 100k buffer -
    # policy_ce climbed 1.08 -> 1.95 and loss 1.46 -> 2.05 over 25 iterations.
    # 96 steps keeps the data:update ratio that was stable all evening; the GPU
    # now has headroom for a bigger NETWORK (WP11) rather than more passes over
    # the same rows.
    train_steps=96,
    # 2026-11-24 rebalance: the fleet produces ~150 games/hr and the drain is
    # newest-first, so a 64-game iteration consumes fresh data twice as fast
    # and trains on more positions per cycle.
    games_per_iteration=64,
    batch_size=512,
    min_examples=128,
    # Sequential search (--search-batch 1, E1 verdict) halves the fleet's game
    # rate, so a 10-minute drain only fills ~11-20 of the 32 games an iteration
    # wants and trains on a starved batch. Wait up to 30 minutes instead.
    spool_timeout_s=1800.0,
    # Frozen eval set (training/eval_set.py): the comparable progress series.
    # 0 disables; path None -> <run root>/eval_set.npz.
    eval_every=1,
    eval_set_path=None,
    # Gate cadence in ITERATIONS (0 = wall-clock markers only). Restart-safe,
    # unlike ladder.markers_minutes which measures process uptime.
    gate_every_iterations=20,
    # LR (2026-11-27): restored to 3e-4 after the bench/head-to-head session.
    # The earlier drift investigation (1e-3 -> 3e-4 -> 1e-4, effective 2.5e-5)
    # showed the CE climb was NOT learning-rate driven, so the floor was too
    # conservative: at 2.5e-5 the net was annealing, not learning. 3e-4 with the
    # step schedule (halving at 200/400/600/800) is the defensible default.
    base_lr=3e-4,
    # Durability (2026-11-24): the ladder gate is the only thing that used to
    # write a checkpoint, and it runs every 90 min - at ~14 iterations/hr that
    # is ~20 iterations of unsaved progress. A plain checkpoint+state every 4
    # iterations (~17 min) bounds the loss without running the gate.
    checkpoint_every_iterations=4,
    lr_schedule=LRStepSchedule(milestones=(200, 400, 600, 800), gamma=0.5),
    # WP-F C5: the flag is wired end-to-end; the default stays python
    # (flipping is a separate, measured decision after the C6 gates).
    search_backend="python",
    # WP-H flip (2026-11-24, user go): self-play games come from the Rust
    # worker fleet via the spool (2.6x games/hr measured); the arena/gate
    # stays on the python search path per the flip procedure.
    selfplay_backend="rust_workers",
    # Gate throughput (2026-11-24): the 16 title games are independent -
    # run them 8-way concurrently, on the RUST search backend (arena leaf
    # evaluation is history-free exactly like the python path, so fidelity is
    # unchanged; ~10x faster), and only every ~90 min (the champion rarely
    # changes within one iteration - gating every iteration was 60-90% of the
    # cycle). Decision math is unchanged.
    ladder=replace(
        NIGHT_PROFILE.ladder,
        # Gate (2026-11-27): re-enabled after an end-to-end verification run.
        # 32 games, 96 sims, parallel=8, NO SPRT early stop (SPRT silently
        # forces Arena.match onto the sequential path, which turned every gate
        # into ~2 h). Verified with scripts/verify_gate.py: 32 games complete in
        # ~16 min and the match returns a real MatchResult.
        markers_minutes=(180.0,),
        repeat_every_minutes=180.0,
        arena=replace(
            NIGHT_PROFILE.ladder.arena,
            games=32,
            num_simulations=96,
            parallel=8,
            sprt_early_stop=False,
            search_backend="rust",
        ),
    ),
)


@dataclass
class RunSummary:
    """What one `run_training` invocation produced."""

    run_id: str
    regime: str
    root: Path
    checkpoints: list[str] = field(default_factory=list)
    champion: str | None = None
    elo: float | None = None
    games_played: int = 0
    arena_games: int = 0
    steps: int = 0
    final_loss: float | None = None
    elapsed_seconds: float = 0.0
    games_per_hour: float | None = None
    iterations_completed: int = 0
    resumed: bool = False


def _utc_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


# WP-B: which encoder layout a model consumes, by its input plane count. The
# stamp is DERIVED from model.config.in_channels (never hardcoded per run) so
# the current 29-plane runs stamp encoder_version=1 honestly and a future v2
# profile stamps 2 automatically. NUM_PLANES_V2 (=104) and ENCODER_VERSION
# (=2) are re-exported here from training.encoder — the plan's constants.
_ENCODER_VERSION_BY_PLANES: dict[int, int] = {NUM_PLANES: 1, NUM_PLANES_V2: ENCODER_VERSION}


def encoder_version_for(in_channels: int) -> int:
    """Encoder layout version for a model with `in_channels` input planes (WP-B)."""
    try:
        return _ENCODER_VERSION_BY_PLANES[int(in_channels)]
    except KeyError:
        raise ValueError(
            f"unknown encoder plane count {in_channels}; expected one of "
            f"{sorted(_ENCODER_VERSION_BY_PLANES)}"
        ) from None


def _save_checkpoint(
    model: GambitNet,
    path: Path,
    tag: str,
    regime: str,
    run_id: str,
    step: int,
    games: int,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model.config.to_dict(),
            "tag": tag,
            "regime": regime,
            "run_id": run_id,
            "step": step,
            "games": games,
            # WP-B: encoder-layout stamp — derived from the model's input
            # plane count (29 -> v1, 104 -> v2) so loaders can branch without
            # guessing (WP-D warm-start reads this).
            "in_channels": int(model.config.in_channels),
            "encoder_version": encoder_version_for(model.config.in_channels),
            # WP-C: value-head layout stamp ("scalar" | "wdl") — the loader
            # rebuilds the matching head from the checkpoint's config dict.
            "value_head": model.config.value_head,
            "ts": time.time(),
        },
        path,
    )
    log.info("[CKPT] saved %s", path)


def _load_champion(
    checkpoint_path: Path, device: torch.device
) -> tuple[GambitNet, str, float] | None:
    """Load a champion checkpoint file; None when absent or incompatible.

    WP-C: the head layout branches on the checkpoint's config — a `wdl`
    checkpoint rebuilds with 3 value logits, a pre-WP-C `scalar` checkpoint
    (whose config dict lacks the `value_head` key) loads exactly as before
    via the `ModelConfig.value_head` default.
    """
    try:
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - unreadable champion checkpoints gate as "no champion"
        log.warning("[CHAMPION] could not load %s; treating as no champion", checkpoint_path)
        return None
    config = ModelConfig.from_dict(data["config"])
    model = GambitNet(config).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    log.info("[CHAMPION] loaded %s (value_head=%s)", checkpoint_path.name, config.value_head)
    return model, str(data.get("tag", "unknown")), float(data.get("elo", 800.0))


def _champion_snapshot(
    checkpoint_path: Path, device: torch.device, live_model: GambitNet
) -> GambitNet:
    """Materialize the champion as a FROZEN model (audit F-1).

    Reloads the checkpoint that was just saved -- the same loading path
    champion.json serving uses -- so every subsequent gate plays an independent
    snapshot whose weights cannot drift with training. If the file is
    unreadable (should not happen immediately after a successful save) a deep
    copy is used instead. Aliasing the live model is never acceptable: an
    aliased champion keeps learning in place and then plays ITSELF in the
    arena, pinning every gate at 0.5 and making promotion impossible.
    """
    loaded = _load_champion(checkpoint_path, device)
    if loaded is None:
        log.warning(
            "[GATE] checkpoint %s unreadable; snapshotting the live model via deep copy",
            checkpoint_path,
        )
        frozen = copy.deepcopy(live_model)
    else:
        frozen = loaded[0]
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def _load_champion_pointer(champion_file: Path) -> tuple[Path, float, dict[str, float] | None]:
    """Read champion.json -> (checkpoint path, elo, glicko dict | None).

    The glicko key is U9-additive; pointers written before it existed load
    as (path, elo, None) and the caller falls back to other seeds.
    """
    try:
        data = json.loads(champion_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    path = Path(str(data.get("checkpoint_path", "")))
    if not path.exists():
        log.warning("[CHAMPION] pointer %s missing checkpoint %s", champion_file, path)
        return None
    glicko = data.get("glicko")
    return path, float(data.get("elo", 800.0)), glicko if isinstance(glicko, dict) else None


def _apply_warmstart(
    model: GambitNet, regime: RegimeConfig, base_dir: Path
) -> bool:
    """FS-3: hybrid/finetune weight init. False = degraded to fresh init."""
    if not regime.warmstart_from:
        return False
    weights = resolve_warmstart(regime.warmstart_from, base_dir)
    if weights is None:
        return False
    try:
        model.load_state_dict(weights)
    except (RuntimeError, KeyError) as exc:
        log.warning(
            "[WARMSTART] weights from '%s' do not fit this architecture (%s); fresh init",
            regime.warmstart_from,
            exc,
        )
        return False
    log.info("[WARMSTART] regime '%s' initialized from '%s'", regime.name, regime.warmstart_from)
    return True


def _apply_init_checkpoint(model: GambitNet, path: Path) -> None:
    """WP-D: seed a FRESH run's model weights from an explicit checkpoint.

    Raises on an unreadable, architecture-incompatible or config-mismatched
    file: an explicit seed that silently degraded to random init would restart
    training from scratch without anyone noticing — the opposite of what the
    warm-start exists for. The optimizer is intentionally untouched (fresh
    Adam); v1 checkpoints carry no optimizer state and the restart starts a
    new Adam on the transferred weights (noted in the artifact metadata).
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    state = data.get("model_state") if isinstance(data, dict) else None
    if state is None:
        raise ValueError(f"init checkpoint {path} carries no model_state")
    saved_config = data.get("config")
    if isinstance(saved_config, dict):
        saved_in = int(saved_config.get("in_channels", -1))
        if saved_in != model.config.in_channels:
            raise ValueError(
                f"init checkpoint {path} is {saved_in}-plane; this run needs "
                f"{model.config.in_channels} — build the matching artifact "
                "with scripts/warmstart_v2.py"
            )
    model.load_state_dict(state)
    log.info(
        "[INIT] seeded model weights from %s (tag=%s, value_head=%s)",
        path,
        data.get("tag", "unknown"),
        model.config.value_head,
    )


def _apply_lr_schedule(
    optimizer: torch.optim.Optimizer,
    schedule: LRStepSchedule | None,
    iteration: int,
    base_lr: float,
) -> float:
    """Set the optimizer LR for `iteration`; returns the LR in effect (WP-A).

    With a schedule, the LR is a pure function of the ABSOLUTE iteration
    number (`LRStepSchedule.lr_at`), so `--resume` from any iteration lands on
    the correct rung. With no schedule (every pre-existing profile) the
    optimizer is left exactly as constructed — constant regime LR, untouched.
    """
    if schedule is None:
        if not optimizer.param_groups:
            return base_lr
        return float(optimizer.param_groups[0]["lr"])
    lr = schedule.lr_at(iteration, base_lr)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _iteration_lr(
    profile: "TrainingProfile",
    steps_done: int,
    iteration: int,
    base_lr: float,
    total_iterations: int,
) -> float:
    """WP10: warmup -> (cosine | step-decay | constant), resume-stable."""
    base = profile.base_lr if profile.base_lr is not None else base_lr
    if profile.lr_warmup_steps > 0 and steps_done < profile.lr_warmup_steps:
        return base * max(steps_done + 1, 1) / profile.lr_warmup_steps
    if profile.lr_cosine:
        floor = base * profile.lr_floor_ratio
        progress = min(max(iteration / max(total_iterations - 1, 1), 0.0), 1.0)
        return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * progress))
    if profile.lr_schedule is not None:
        return profile.lr_schedule.lr_at(iteration, base)
    return base


def run_training(
    profile: TrainingProfile,
    run_id: str | None = None,
    base_dir: Path | str = "runs",
    data_dir: Path | str = "data",
    resume: str | bool = False,
    search_kind: str | None = None,
    playout_randomization: bool | None = None,
    init_checkpoint: str | Path | None = None,
) -> RunSummary:
    """Run the full loop: collect -> optimize -> ladder -> arena -> ledger.

    `resume=True` resumes `run_id`; `resume="<run-id>"` resumes that run
    directly (FS-5): model weights, optimizer, replay buffer, RNG streams and
    the iteration counter are rebuilt from the run's persisted state.
    `init_checkpoint` (WP-D) seeds a FRESH run's weights from an explicit
    checkpoint file (e.g. the WP-D warm-start artifact); it is ignored on
    resume (a resumed run rebuilds its own weights) and raises on an
    incompatible file.
    """
    started = time.monotonic()
    if profile.seed is not None:
        torch.manual_seed(profile.seed)
        np.random.seed(profile.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regime: RegimeConfig = get_regime(profile.regime)
    # U1/U2 CLI overrides: explicit flags beat the regime defaults so a regime
    # stays a swappable config rather than a hardcode (section 5.2).
    regime_overrides: dict[str, object] = {}
    if search_kind is not None:
        regime_overrides["search_kind"] = search_kind
    if playout_randomization is not None:
        regime_overrides["playout_randomization"] = playout_randomization
    if regime_overrides:
        regime = replace(regime, **regime_overrides)
    is_resume = bool(resume)
    resume_id = resume if isinstance(resume, str) else None
    resolved_run_id = resume_id or run_id or _utc_run_id()
    paths: RunPaths = create_run(regime, resolved_run_id, base_dir)
    ledger = ArenaLedger(Path(data_dir) / "arena.jsonl")
    metrics = MetricsWriter(Path(data_dir) / "metrics.json")
    journal = TrainingJournal(Path(data_dir) / "train_log.jsonl")
    play_rng = (
        np.random.default_rng(profile.seed) if profile.seed is not None else np.random.default_rng()
    )

    model = GambitNet(profile.model).to(device)
    initial_lr = profile.base_lr if profile.base_lr is not None else regime.learning_rate
    if profile.optimizer_kind == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=initial_lr,
            weight_decay=profile.weight_decay,
            fused=(device.type == "cuda"),
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    # WP4: EMA shadow (created empty; state restored from the resume sidecar).
    ema = EmaWeights(model, profile.ema_decay) if profile.ema_decay > 0 else None
    tb_corrector = TablebaseCorrector(profile.tb_path) if profile.tb_path else None
    if ema is not None:
        sidecars = sorted(paths.state.glob("ema-*.pt"))
        if sidecars:
            try:
                ema.load_state_dict(torch.load(sidecars[-1], map_location="cpu", weights_only=True))
                log.info("[EMA] restored %s", sidecars[-1].name)
            except Exception as exc:  # noqa: BLE001 - sidecar must not kill the run
                log.warning("[EMA] sidecar restore failed: %s", exc)
    buffer: TrainingBuffer = TrainingBuffer(max_size=profile.buffer_size)
    # WP-B: encoder-layout stamp for every journal record this run writes.
    encoder_version = encoder_version_for(model.config.in_channels)

    champion_file = Path(base_dir) / regime.name / CHAMPION_FILENAME
    champion_tag: str | None = None
    champion_elo = profile.ladder.base_elo
    # U9: the champion lineage's Glicko-2 record (r, RD, sigma). Seeded from
    # (in order) the resumed run state, the ledger replay for the champion
    # tag, the champion.json pointer, then the default prior. Kept separate
    # from champion_elo: the Elo series keeps its exact old meaning.
    champion_glicko: GlickoRating | None = None

    # --- FS-5 resume: exact rebuild from persisted state -------------------
    state = load_latest_run_state(paths) if is_resume else None
    if state is not None:
        if init_checkpoint is not None:
            log.warning("[INIT] --init-checkpoint ignored on a resumed run (state rebuild wins)")
        saved_signature = state.get("profile_signature")
        if isinstance(saved_signature, dict):
            validate_profile_signature(saved_signature, profile_signature(profile))
        restored = restore_run_state(
            state, model=model, optimizer=optimizer, buffer=buffer, play_rng=play_rng
        )
        iteration_start = restored.iterations_completed
        champion_tag = restored.champion_tag
        champion_elo = restored.champion_elo
        if restored.champion_glicko is not None:
            champion_glicko = GlickoRating.from_dict(restored.champion_glicko)
            log.info(
                "[RESUME] champion glicko record r=%.1f rd=%.1f",
                champion_glicko.r,
                champion_glicko.rd,
            )
        log.info(
            "[RESUME] run %s continues at iteration %d (steps %d, games %d)",
            resolved_run_id,
            iteration_start,
            restored.steps,
            restored.games_played,
        )
    else:
        iteration_start = 0
        if is_resume:
            log.warning(
                "[RESUME] no persisted state for run %s — starting from iteration 0", resolved_run_id
            )
        if init_checkpoint is not None:
            # WP-D: the explicit seed wins over any regime warm-start spec —
            # a named artifact is the more specific instruction.
            _apply_init_checkpoint(model, Path(init_checkpoint))
        else:
            _apply_warmstart(model, regime, Path(base_dir))

    # --- reigning champion (regime-level pointer, §5.1) ---------------------
    # A resumed run first tries ITS OWN checkpoint for the restored champion
    # tag (exact rebuild), then falls back to the regime pointer.
    champion_model: GambitNet | None = None
    # FB-2: the champion's tag and its checkpoint path are resolved TOGETHER
    # (one checkpoint object) and carried as a pair everywhere, so the
    # champion.json writer can never pair one checkpoint's tag with another's
    # path — the observed corruption (`runs/scratch/champion.json` exhibited
    # tag ckpt-001536-0768 with a path pointing at a different file).
    champion_ckpt_path: Path | None = None
    pointer = _load_champion_pointer(champion_file)
    if champion_tag is not None:
        own_checkpoint = paths.checkpoints / f"{champion_tag}.pt"
        loaded: tuple[GambitNet, str, float] | None = None
        loaded_from: Path | None = None
        if own_checkpoint.exists():
            loaded = _load_champion(own_checkpoint, device)
            loaded_from = own_checkpoint
        if loaded is None and pointer is not None:
            loaded = _load_champion(pointer[0], device)
            loaded_from = pointer[0]
        if loaded is not None:
            champion_model = loaded[0]
            champion_ckpt_path = loaded_from
            log.info(
                "[CHAMPION] resumed champion %s (elo %.1f)", champion_tag, champion_elo
            )
    elif pointer is not None:
        loaded = _load_champion(pointer[0], device)
        if loaded is not None:
            # The tag comes FROM the loaded checkpoint object (not the stale
            # pointer's tag field), so (tag, path) is one consistent pair.
            champion_model, champion_tag, champion_elo = loaded[0], loaded[1], pointer[1]
            champion_ckpt_path = pointer[0]
            log.info("[CHAMPION] reigning champion %s (elo %.1f)", champion_tag, champion_elo)

    # --- U9: champion Glicko bootstrap (resume-state -> ledger -> pointer) --
    # A state file written before U9 carries no glicko key; the ledger replay
    # reconstructs the champion lineage's record from its recorded gates so a
    # crash-restart across the upgrade keeps a continuous rating series.
    if champion_glicko is None and champion_tag is not None:
        champion_glicko = recompute_glickos(ledger.read()).get(champion_tag)
        if champion_glicko is not None:
            log.info(
                "[GLICKO] champion %s record rebuilt from ledger: r=%.1f rd=%.1f",
                champion_tag,
                champion_glicko.r,
                champion_glicko.rd,
            )
    if champion_glicko is None and pointer is not None and pointer[2] is not None:
        champion_glicko = GlickoRating.from_dict(pointer[2])
    if champion_glicko is None and champion_tag is not None:
        champion_glicko = GlickoRating()
        log.info("[GLICKO] champion %s seeded at the default prior (r=1500, rd=350)", champion_tag)

    summary = RunSummary(
        run_id=resolved_run_id,
        regime=regime.name,
        root=paths.root,
        steps=0 if state is None else int(state.get("steps", 0) or 0),
        games_played=0 if state is None else int(state.get("games_played", 0) or 0),
        resumed=state is not None,
    )
    last_snapshot_minute = 0.0
    last_iteration_lr = -1.0  # WP-A: forces the first iteration's LR log line
    arena_games_total = 0
    last_loss: float | None = None
    last_policy_loss: float | None = None
    last_value_loss: float | None = None
    last_match: MatchResult | None = None
    last_decision: PromotionDecision | None = None
    collect_seconds_total = 0.0
    session_games = 0  # games collected in THIS invocation (resume-aware g/h)

    effective_parallel = (
        profile.parallel_games if profile.parallel_games is not None else regime.parallel_games
    )
    effective_stream = (
        profile.stream_games if profile.stream_games is not None else regime.stream_moves
    )
    # FS-6b: game ids are unique across the WHOLE run — iteration i owns the
    # slot block [i*N, (i+1)*N), built fresh per loop pass below, so journal
    # events from different iterations (and across a resume) never interleave
    # in one slot on the client's board grid.
    if effective_stream:
        log.info("[STREAM] live move streaming on (journal %s)", journal.path)
    # WP-H H5: "rust_workers" swaps the GAME SOURCE only (a spool of
    # worker-produced JSONL games); buffer/training/gates/journal are
    # backend-agnostic. With the default "python" nothing below changes.
    rust_backend = profile.selfplay_backend == "rust_workers"
    onnx_dir = paths.root / "onnx"
    spool_dir = Path(profile.spool_dir) if profile.spool_dir else paths.root / "spool"
    current_weight_version = -1  # last exported worker model version
    version_stride = max(profile.train_steps, 1)  # one iteration's version stride
    if rust_backend:
        log.info("[SELFPLAY] backend=rust_workers: spool=%s models=%s", spool_dir, onnx_dir)
    queued_evaluator: QueuedNNEvaluator | None = None
    provider = build_data_provider(regime)
    try:
        if rust_backend:
            pass  # collection needs no in-process NN evaluator (workers run ort)
        elif effective_parallel > 1:
            queued_evaluator = QueuedNNEvaluator(model, device)
            log.info("[SELFPLAY] parallel_games=%d via the queued evaluator", effective_parallel)

        for iteration in range(iteration_start, profile.iterations):
            # --- collect (provider-driven, §5.2) ----------------------------
            model.eval()
            if rust_backend:
                # WP-H H4: games come from external rust worker processes;
                # collection itself needs no in-process NN evaluator.
                evaluator = None
            elif queued_evaluator is not None:
                evaluator: Evaluator = queued_evaluator
            else:
                evaluator = BatchNNEvaluator(model, device)
            selfplay_config = SelfPlayConfig(
                search=SearchConfig(
                    num_simulations=profile.sims,
                    search_kind=regime.search_kind,
                    gumbel_candidates=regime.gumbel_candidates,
                    search_backend=profile.search_backend,
                ),
                max_plies=profile.max_plies,
                temperature_plies=profile.max_plies,  # sample throughout a smoke game
                playout_randomization=regime.playout_randomization,
                playout_cap_full_prob=regime.playout_cap_full_prob,
                playout_cap_fast_fraction=regime.playout_cap_fast_fraction,
                playout_cap_start_ply=regime.playout_cap_start_ply,
                # WP-D: recorded example planes match the model's encoder —
                # 104-plane v2 models train on move-stack-history encodings.
                encoder_version=encoder_version,
            )
            elapsed_minutes = (time.monotonic() - started) / 60.0
            progress = (iteration + 1) / max(profile.iterations, 1)
            collect_started = time.monotonic()
            stream_context = StreamContext(
                journal=journal,
                run_id=resolved_run_id,
                regime=regime.name,
                enabled=effective_stream,
                game_offset=iteration * profile.games_per_iteration,
            )
            games_this_iter: list[SelfPlayResult]
            if rust_backend:
                # WP-H H4: block for worker games (up to spool_timeout_s);
                # a shortfall proceeds with fewer games - the min_examples
                # guard below skips training until the buffer is viable.
                # Freshness: newest games claimed first and nothing older than
                # 3 ITERATIONS is ever read (A86); stale files are only
                # deleted under disk pressure (purge_stale_games).
                staleness_horizon = (
                    max(current_weight_version - 3 * version_stride, 0)
                    if current_weight_version >= 0
                    else None
                )
                if staleness_horizon is not None:
                    purge_stale_games(spool_dir, staleness_horizon)
                # Retention: at GPU-speed self-play the claimed archive grows
                # ~60 GB/day; keep a 20k-game / 48h window instead.
                prune_processed_games(spool_dir)
                # Racing guard: an iteration that finds almost no fresh games
                # must NOT train - it would burn 288 steps on stale replay and
                # export a version the fleet has not even played yet. But the
                # wait is bounded by HALF a batch, not a full one: `want` is the
                # minimum, so a fast fleet is never held back for 10 minutes
                # waiting to top up to `games_per_iteration` (collect blocks
                # until `want` is reached or its timeout expires).
                min_fresh = max(profile.games_per_iteration // 2, 1)
                spool_records = collect_from_spool(
                    spool_dir,
                    min_fresh,
                    timeout_s=profile.spool_timeout_s,
                    poll_s=1.0,
                    min_weight_version=staleness_horizon,
                )
                # Top up with whatever else is already waiting - never wait for
                # it (a short non-blocking-ish poll keeps iterations tight).
                if len(spool_records) >= min_fresh:
                    spool_records.extend(
                        collect_from_spool(
                            spool_dir,
                            profile.games_per_iteration - len(spool_records),
                            timeout_s=min(5.0, profile.spool_timeout_s),
                            poll_s=0.5,
                            min_weight_version=staleness_horizon,
                        )
                    )
                else:
                    log.warning(
                        "[RUST-SPOOL] starved: only %d/%d fresh games after "
                        "waiting (trainer v%d) - training on the shortfall",
                        len(spool_records),
                        profile.games_per_iteration,
                        current_weight_version,
                    )
                versions = sorted({r["weight_version"] for r in spool_records})
                if versions and versions != [current_weight_version]:
                    log.info(
                        "[RUST-SPOOL] games tagged weight_version %s (trainer at v%d)",
                        versions,
                        current_weight_version,
                    )
                games_this_iter = [
                    record_to_selfplay_result(r, tb=tb_corrector) for r in spool_records
                ]
            else:
                games_this_iter = provider.collect(
                    evaluator,
                    profile.games_per_iteration,
                    selfplay_config,
                    parallel_games=effective_parallel,
                    elapsed_minutes=elapsed_minutes,
                    progress=progress,
                    rng=play_rng,  # FS-5: the run's play stream, restored on resume
                    stream=stream_context,  # FS-6b: live per-move journal events
                )
            collect_seconds = time.monotonic() - collect_started
            collect_seconds_total += collect_seconds
            for result in games_this_iter:
                buffer.add(
                    {
                        "board_tensor": ex.planes,
                        "policy": ex.policy,
                        "value": ex.value,
                        "policy_mask": getattr(ex, "policy_mask", 1.0),
                    }
                    for ex in result.examples
                )
                summary.games_played += 1
                session_games += 1
            games_per_hour_measured = (
                len(games_this_iter) / collect_seconds * 3600.0 if collect_seconds > 0 else None
            )
            log.info(
                "[SELFPLAY] iter %d: %d games via %s, buffer=%d, results=%s",
                iteration,
                len(games_this_iter),
                provider.__class__.__name__,
                len(buffer),
                [r.result for r in games_this_iter],
            )

            # --- optimize ---------------------------------------------------
            # WP-A: per-iteration LR (constant when no schedule is configured).
            iteration_lr = _iteration_lr(
                profile, summary.steps, iteration, regime.learning_rate, profile.iterations
            )
            for group in optimizer.param_groups:
                group["lr"] = iteration_lr
            if iteration_lr != last_iteration_lr:
                log.info(
                    "[TRAIN] iteration %d: lr %.4g%s",
                    iteration,
                    iteration_lr,
                    " (schedule rung)" if profile.lr_schedule is not None else "",
                )
                last_iteration_lr = iteration_lr
            model.train()
            for _step in range(profile.train_steps):
                batch = buffer.sample(profile.batch_size)
                if batch is None or len(batch[0]) < profile.min_examples:
                    log.info("[TRAIN] buffer too small (%s), skipping step", len(buffer))
                    break
                tensors, policies, values, masks = batch
                tensors = tensors.to(device)
                policies = policies.to(device)
                values = values.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                pred_policy, pred_value = model(tensors)
                log_probs = torch.log_softmax(pred_policy, dim=-1)
                # WP2: PCR fast rows carry an all-zero policy and mask 0.0 —
                # they train the value head only.
                per_row_policy = -(policies * log_probs).sum(dim=-1)
                policy_loss = (per_row_policy * masks).sum() / masks.sum().clamp(min=1.0)
                if profile.model.value_head == "wdl":
                    # WP-C: cross-entropy on the one-hot game result (W/D/L
                    # from the mover's perspective); the MSE term is dropped
                    # for this head (policy CE + value CE + nothing else).
                    value_loss = torch.nn.functional.cross_entropy(
                        pred_value, wdl_targets(values)
                    )
                else:
                    value_loss = torch.nn.functional.mse_loss(pred_value, values)
                loss = policy_loss + value_loss
                loss.backward()
                if profile.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), profile.grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                summary.steps += 1
                last_loss = float(loss.item())
                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
            if last_loss is not None:
                summary.final_loss = last_loss
                log.info("[TRAIN] iter %d: loss %.4f", iteration, last_loss)

            # WP5: training-health metrics on a fresh sampled batch:
            # policy CE, policy entropy, P UL (CE vs uniform) and value
            # accuracy. Cheap (one forward, no grads) and journaled so
            # value-head overfitting / entropy collapse are visible early.
            # NOTE: this batch is drawn from the LIVE buffer, so the number is
            # not comparable across iterations (that is why it swings 0.8-3.7).
            # The FROZEN eval-set scores below are the comparable series.
            health: dict[str, object] | None = None
            holdout = buffer.sample(256)
            if holdout is not None and len(holdout[0]) >= 4:
                h_t, h_p, h_v, h_m = holdout
                model.eval()
                with torch.no_grad():
                    out_p, out_v = model(h_t.to(device))
                    lp = torch.log_softmax(out_p, dim=-1)
                    ce_rows = -(h_p.to(device) * lp).sum(dim=-1)
                    mask = h_m.to(device)
                    policy_ce = float((ce_rows * mask).sum() / mask.sum().clamp(min=1.0))
                    probs = lp.exp()
                    entropy = float(-(probs * lp).sum(dim=-1).mean())
                    p_ul = float(-lp.mean())
                    if profile.model.value_head == "wdl":
                        tgt = wdl_targets(h_v.to(device)).argmax(dim=-1)
                        value_acc = float((out_v.argmax(dim=-1) == tgt).float().mean())
                    else:
                        value_acc = None  # scalar head: no WDL accuracy
                model.train()
                health = {
                    "policy_ce": policy_ce,
                    "policy_entropy": entropy,
                    "p_ul": p_ul,
                    "value_acc": value_acc,
                }
                journal.append({"kind": "metrics", "iteration": iteration, **health})
                log.info(
                    "[METRICS] iter %d: policy_ce %.3f entropy %.2f p_ul %.3f value_acc %.2f",
                    iteration, policy_ce, entropy, p_ul, value_acc,
                )

            # Frozen eval-set scores: the COMPARABLE progress series. Built once
            # (training/eval_set.py) and scored against identical positions every
            # eval_every iterations, with proper scoring rules instead of argmax
            # accuracy (research A88).
            if profile.eval_every > 0 and (iteration + 1) % profile.eval_every == 0:
                try:
                    evalset_path = (
                        Path(profile.eval_set_path)
                        if profile.eval_set_path
                        else paths.root / EVAL_SET_FILENAME
                    )
                    if evalset_path.is_file():
                        if _eval_cache.get("path") != str(evalset_path):
                            _eval_cache["data"] = load_eval_set(evalset_path)
                            _eval_cache["path"] = str(evalset_path)
                        scores = score(
                            model,
                            _eval_cache["data"],
                            device,
                            value_head=profile.model.value_head,
                        )
                        journal.append({"kind": "eval", "iteration": iteration, **scores})
                        log.info(
                            "[EVAL] iter %d: policy_ce %.3f top1 %.3f value_ce %.3f "
                            "brier %.3f mean_p_true %.3f score_mae %.3f (n=%d)",
                            iteration,
                            scores["eval_policy_ce"],
                            scores["eval_policy_top1"],
                            scores["eval_value_ce"],
                            scores["eval_value_brier"],
                            scores["eval_mean_p_true"],
                            scores["eval_score_mae"],
                            int(scores["eval_n"]),
                        )
                    elif iteration == 0:
                        log.info(
                            "[EVAL] no frozen eval set at %s; build one with "
                            "`python -m training.eval_set build`",
                            evalset_path,
                        )
                except Exception:  # noqa: BLE001 - never kill a run over metrics
                    log.exception("[EVAL] scoring failed; continuing without it")

            # WP4: one EMA eval copy per iteration - the export, the
            # checkpoint and the gate candidate all use it once the EMA
            # warmup window has passed (None -> live weights).
            eval_model = None
            if ema is not None and iteration >= profile.ema_min_iteration:
                eval_model = copy.deepcopy(model)
                ema.apply_to(eval_model)
                eval_model.eval()

            if rust_backend:
                # WP-H H4: publish the current weights to the worker fleet
                # every iteration (atomic ONNX file + latest.json pointer;
                # workers hot-reload BETWEEN games only). Failures log and
                # never kill the run - workers keep the last good model.
                try:
                    prev_weight_version = current_weight_version
                    pointer = export_model_to_onnx(
                        eval_model if eval_model is not None else model,
                        onnx_dir,
                        weight_version=max(summary.steps, current_weight_version + 1),
                    )
                    current_weight_version = pointer.weight_version
                    # Keep the export directory bounded (one file per iteration,
                    # never pruned: 944 files / 14.6 GB by 2026-11-28).
                    prune_exported_models(onnx_dir)
                    # One iteration's version stride (weight_version follows
                    # cumulative steps, so the stride is ~train_steps). The
                    # staleness horizon is expressed in ITERATIONS, not raw
                    # version units.
                    if prev_weight_version >= 0:
                        version_stride = max(current_weight_version - prev_weight_version, 1)
                    else:
                        version_stride = max(profile.train_steps, 1)
                    log.info(
                        "[RUST-EXPORT] v%d -> %s (%.1f MB)",
                        pointer.weight_version,
                        pointer.path.name,
                        pointer.path.stat().st_size / 1e6,
                    )
                except Exception as exc:  # noqa: BLE001 - export must not kill training
                    log.error("[RUST-EXPORT] iteration %d export failed: %s", iteration, exc)

            # --- FS-6 journal: one event per iteration ----------------------
            journal.append(
                {
                    "kind": "iteration",
                    "iteration": iteration,
                    "loss": last_loss,
                    "policy_loss": last_policy_loss,
                    "value_loss": last_value_loss,
                    "elo": champion_elo,
                    "arena_result": None,
                    "games": len(games_this_iter),
                    "buffer_size": len(buffer),
                    "games_per_hour": round(games_per_hour_measured, 1)
                    if games_per_hour_measured is not None
                    else None,
                    "parallel_games": effective_parallel,
                    "lr": iteration_lr,
                    "encoder_version": encoder_version,
                    "regime": regime.name,
                    "run_id": resolved_run_id,
                }
            )

            # --- ladder + arena gate (§5.1) ---------------------------------
            model.eval()
            elapsed_minutes = (time.monotonic() - started) / 60.0
            # Gate cadence is measured in ITERATIONS, not process-uptime: the
            # wall-clock marker restarts at 0 on every resume, and with the
            # watchdog/restart cadence the 180-minute marker was unreachable, so
            # the ladder silently never ran (arena.jsonl frozen for days while
            # the trainer advanced 30k games). Iteration numbers are persisted,
            # so this survives restarts.
            due = due_snapshot_minutes(profile.ladder, elapsed_minutes, last_snapshot_minute)
            if (
                profile.gate_every_iterations > 0
                and (iteration + 1) % profile.gate_every_iterations == 0
            ):
                due = True
            final_iteration = iteration == profile.iterations - 1
            if not due and not final_iteration and not profile.snapshot_every_iteration:
                # Durability checkpoint: no gate, no ledger entry - just the
                # weights + resumable state, so a crash costs at most N
                # iterations instead of a whole snapshot interval.
                every = profile.checkpoint_every_iterations
                if every > 0 and (iteration + 1) % every == 0:
                    tag = f"ckpt-{summary.steps:06d}-{summary.games_played:04d}"
                    ckpt_path = paths.checkpoints / f"{tag}.pt"
                    _save_checkpoint(
                        eval_model if eval_model is not None else model,
                        ckpt_path,
                        tag,
                        regime.name,
                        resolved_run_id,
                        summary.steps,
                        summary.games_played,
                    )
                    summary.checkpoints.append(tag)
                    save_run_state(
                        paths,
                        iterations_completed=iteration + 1,
                        model=model,
                        optimizer=optimizer,
                        buffer=buffer,
                        play_rng=play_rng,
                        steps=summary.steps,
                        games_played=summary.games_played,
                        champion_tag=champion_tag,
                        champion_elo=champion_elo,
                        profile=profile,
                        champion_glicko=(
                            champion_glicko.to_dict() if champion_glicko is not None else None
                        ),
                    )
                    log.info(
                        "[CKPT] durability checkpoint %s (every %d iterations)",
                        tag,
                        every,
                    )
                continue
            last_snapshot_minute = max([elapsed_minutes, *due])

            tag = f"ckpt-{summary.steps:06d}-{summary.games_played:04d}"
            ckpt_path = paths.checkpoints / f"{tag}.pt"
            _save_checkpoint(
                eval_model if eval_model is not None else model,
                ckpt_path, tag, regime.name, resolved_run_id, summary.steps, summary.games_played,
            )
            summary.checkpoints.append(tag)
            if ema is not None:
                ema_dir = paths.state
                torch.save(ema.state_dict(), ema_dir / f"ema-{summary.steps:06d}.pt")
                old_emas = sorted(ema_dir.glob("ema-*.pt"))[:-2]
                for stale in old_emas:
                    stale.unlink(missing_ok=True)

            # F-5 (audit): each checkpoint is a FRESH candidate with no rating
            # history. base_elo is only the provisional rating inside the Elo
            # expectation term; a candidate that does not promote gets NO rating
            # claim in the ledger (elo=None), and a promoted candidate INHERITS
            # the champion lineage's post-match rating instead of resetting the
            # run's rating to a one-match read from base. champion_elo -- seeded
            # from the reigning pointer / resume state, moved by the match delta
            # at every gate -- is the continuous, honest rating series the
            # journal and metrics feed expose.
            provisional_elo = profile.ladder.base_elo
            # Bound for the HF-publish metadata below; the baseline branch and
            # the promote branch set it to True (both seat a new champion).
            promote = False
            opponent_for_event = champion_tag or "unknown"
            if champion_model is None:
                # First checkpoint becomes champion by default; recorded as baseline.
                first_glicko = champion_glicko or GlickoRating()
                ledger.append(
                    LedgerEntry(
                        run_id=resolved_run_id,
                        regime=regime.name,
                        checkpoint=tag,
                        opponent="none",
                        games=0,
                        wins=0,
                        draws=0,
                        elo=provisional_elo,
                        glicko_r=first_glicko.r,
                        glicko_rd=first_glicko.rd,
                        kind="baseline",
                        promoted=True,
                    )
                )
                # F-1 (audit): the champion is a FROZEN snapshot reloaded from
                # the just-saved checkpoint. Aliasing the live model made every
                # later gate play the model against itself (0.5 forever).
                champion_model = _champion_snapshot(ckpt_path, device, model)
                champion_tag, champion_elo = tag, provisional_elo
                champion_ckpt_path = ckpt_path  # FB-2: tag/path as one pair
                champion_glicko = first_glicko
                promote = True
                log.info("[GATE] %s becomes the first champion (baseline elo %.1f)", tag, provisional_elo)
                last_match = None
                last_decision = None
                opponent_for_event = "none"
            else:
                # U9: LadderConfig is the single source of truth for the SPRT
                # knobs; they are copied onto the match config so `Arena.match`
                # can stop sequentially (gate economics — decisive matches
                # cost fewer games; at 16-game gates this is a provable no-op).
                # FB-1: every gate event seeds a FRESH opening set from
                # f(run_seed, gate_index) — gate_index is the ABSOLUTE
                # iteration number, so a resumed run reproduces the opening
                # set of the gate it stopped at. Openings are mirrored across
                # color pairs and logged in the gate record below.
                gate_seed = gate_opening_seed(profile.seed, iteration)
                arena_config = replace(
                    profile.ladder.arena,
                    # SPRT early stop forces SEQUENTIAL play (match() switches
                    # to the sequential path whenever it is enabled), which made
                    # every gate 16 x ~8 min = ~2 h and starved training for
                    # half of each 90-min cycle. It is a provable no-op at the
                    # default 16-game size anyway, so it now follows the profile
                    # (night-v2: False -> the 8-way parallel path).
                    sprt_early_stop=profile.ladder.arena.sprt_early_stop,
                    sprt_elo0=profile.ladder.sprt_elo0,
                    sprt_elo1=profile.ladder.sprt_elo1,
                    sprt_alpha=profile.ladder.sprt_alpha,
                    sprt_beta=profile.ladder.sprt_beta,
                    opening_seed=gate_seed,
                )
                if profile.ladder.screen_games > 0:
                    arena_config = replace(arena_config, games=profile.ladder.screen_games)
                match: MatchResult = _arena_match(
                    eval_model if eval_model is not None else model,
                    champion_model, device, arena_config, tag, champion_tag or "unknown"
                )
                # WP6 stage 2: confirmation on FRESH openings (a different
                # seed namespace) once the screen clears its threshold.
                confirm_match = None
                if (
                    profile.ladder.confirm_games > 0
                    and match.games > 0
                    and match.win_rate >= profile.ladder.screen_threshold
                ):
                    confirm_config = replace(
                        arena_config,
                        games=profile.ladder.confirm_games,
                        opening_seed=gate_seed + 10_000_000,
                    )
                    confirm_match = _arena_match(
                        eval_model if eval_model is not None else model,
                        champion_model, device, confirm_config, tag,
                        champion_tag or "unknown",
                    )
                    arena_games_total += confirm_match.games
                    log.info(
                        "[GATE] %s confirmation: %dW-%dD-%dL of %d (fresh openings)",
                        tag, confirm_match.wins, confirm_match.draws,
                        confirm_match.losses, confirm_match.games,
                    )
                last_match = match
                arena_games_total += match.games
                _, champion_elo = update_elo(
                    provisional_elo, champion_elo, match, profile.ladder.k_factor
                )
                # U9: both parties take a Glicko-2 update from the same games —
                # the candidate keeps its own record even when it does not
                # promote (previously candidates had no rating record at all).
                candidate_glicko = GlickoRating()
                candidate_glicko, champion_glicko = update_match_pair(
                    candidate_glicko, champion_glicko or GlickoRating(), match
                )
                decision = decide_promotion(match, candidate_glicko, profile.ladder)
                last_decision = decision
                promote = decision.promote
                if confirm_match is not None:
                    # WP6: combine both stages and demand a Wilson lower
                    # bound above the configured floor.
                    from .ladder import wilson_lower_bound

                    games = match.games + confirm_match.games
                    wins = match.wins + confirm_match.wins
                    draws = match.draws + confirm_match.draws
                    score = (wins + 0.5 * draws) / games if games else 0.0
                    lb = wilson_lower_bound(score, games)
                    promote = (
                        score >= profile.ladder.promote_win_rate
                        and lb > profile.ladder.confirm_lower_bound
                    )
                    log.info(
                        "[GATE] %s combined score %.3f over %d games, Wilson LB %.3f -> %s",
                        tag, score, games, lb, "promote" if promote else "hold",
                    )
                ledger.append(
                    LedgerEntry(
                        run_id=resolved_run_id,
                        regime=regime.name,
                        checkpoint=tag,
                        opponent=champion_tag or "unknown",
                        games=match.games,
                        wins=match.wins,
                        draws=match.draws,
                        elo=champion_elo if promote else None,
                        glicko_r=candidate_glicko.r,
                        glicko_rd=candidate_glicko.rd,
                        losses=match.losses,
                        promoted=promote,
                    )
                )
                if promote:
                    # F-1 (audit): freeze the promoted candidate from its
                    # just-saved checkpoint; never alias the live model.
                    champion_model = _champion_snapshot(ckpt_path, device, model)
                    champion_tag = tag
                    champion_ckpt_path = ckpt_path  # FB-2: tag/path as one pair
                    log.info(
                        "[GATE] %s promoted (%s: score %.3f, llr %.3f in [%.3f, %.3f], rd %.1f)",
                        tag,
                        decision.reason,
                        match.win_rate,
                        decision.llr,
                        decision.lower,
                        decision.upper,
                        candidate_glicko.rd,
                    )
                else:
                    log.info(
                        "[GATE] champion %s holds (%s: score %.3f vs %.2f, llr %.3f, rd %.1f)",
                        champion_tag,
                        decision.reason,
                        match.win_rate,
                        profile.ladder.promote_win_rate,
                        decision.llr,
                        candidate_glicko.rd,
                    )

            # --- champion + metrics feed -------------------------------------
            # FB-2: tag and path come from the SAME resolved checkpoint object
            # (`champion_ckpt_path` moves with `champion_tag` at every
            # baseline/promote/hold), and the writer asserts
            # `tag == Path(path).stem` before touching champion.json — a
            # champion that HOLDS keeps its own checkpoint's pointer, never
            # the losing candidate's path under the champion's tag (the
            # observed corruption).
            assert champion_ckpt_path is not None, "champion resolved without its checkpoint path"
            _save_champion(champion_file, champion_model, champion_tag or tag, champion_elo,
                           resolved_run_id, champion_ckpt_path, champion_glicko)
            ledger_rows = ledger.read()
            elo_map = recompute_elos(ledger_rows, profile.ladder.base_elo, profile.ladder.k_factor)
            glicko_map = recompute_glickos(ledger_rows)
            summary.arena_games = arena_games_total
            summary.champion = champion_tag
            summary.elo = champion_elo
            metrics.update(
                elo=summary.elo,
                win_rate=last_match.win_rate if last_match is not None else None,
                games_played=summary.games_played + arena_games_total,
                arena_games=arena_games_total,
                loss=summary.final_loss,
                champion=champion_tag,
                regime=regime.name,
                checkpoints={k: round(v, 1) for k, v in elo_map.items()},
                # U9: per-checkpoint Glicko records + the champion's own record
                # + the last gate's SPRT state, for the Watch/metrics feed.
                glicko={
                    k: {"r": round(v.r, 1), "rd": round(v.rd, 1)} for k, v in glicko_map.items()
                },
                champion_glicko=champion_glicko.to_dict() if champion_glicko is not None else None,
                sprt=last_decision.to_dict() if last_decision is not None else None,
                parallel_games=effective_parallel,
            )

            # --- FS-6 journal: one event per arena gate ----------------------
            # Arena events carry the full required key set too (loss family
            # reflects the iteration that produced this checkpoint). U9 adds
            # the SPRT state (LLR + bounds + verdict), both parties' Glicko
            # records, and the match-size honesty report so the feed can show
            # what the gate size can actually claim.
            candidate_glicko_after = (
                glicko_map.get(tag, GlickoRating()) if last_match is not None else (champion_glicko or GlickoRating())
            )
            champion_glicko_after = champion_glicko or GlickoRating()
            honesty = (
                match_honesty(candidate_glicko_after, champion_glicko_after, last_match.games)
                if last_match is not None
                else None
            )
            journal.append(
                {
                    "kind": "arena",
                    "iteration": iteration,
                    "checkpoint": tag,
                    "opponent": opponent_for_event,
                    "games": last_match.games if last_match is not None else 0,
                    "wins": last_match.wins if last_match is not None else 0,
                    "draws": last_match.draws if last_match is not None else 0,
                    "losses": last_match.losses if last_match is not None else 0,
                    # FB-1: the gate record carries the gate event identity
                    # (absolute iteration) and every opening FEN actually
                    # played, so any gate is reproducible from
                    # (run_seed, gate_index) -> sample_gate_openings.
                    "gate_index": iteration,
                    "openings": list(last_match.openings) if last_match is not None else [],
                    "arena_result": {
                        "win_rate": last_match.win_rate,
                        "promoted": promote,
                        "decision": last_decision.to_dict() if last_decision is not None else None,
                    }
                    if last_match is not None
                    else None,
                    "sprt": last_decision.to_dict() if last_decision is not None else None,
                    "glicko": {
                        "candidate_r": round(candidate_glicko_after.r, 2),
                        "candidate_rd": round(candidate_glicko_after.rd, 2),
                        "champion_r": round(champion_glicko_after.r, 2),
                        "champion_rd": round(champion_glicko_after.rd, 2),
                    },
                    "honesty": honesty.to_dict() if honesty is not None else None,
                    "sprt_games_to_accept": (
                        expected_games_to_accept(
                            last_match.win_rate,
                            last_match.draws / last_match.games if last_match.games else 0.0,
                            elo0=profile.ladder.sprt_elo0,
                            elo1=profile.ladder.sprt_elo1,
                            alpha=profile.ladder.sprt_alpha,
                            beta=profile.ladder.sprt_beta,
                        )
                        if last_match is not None
                        else None
                    ),
                    "elo": champion_elo,
                    "loss": last_loss,
                    "policy_loss": last_policy_loss,
                    "value_loss": last_value_loss,
                    "encoder_version": encoder_version,
                    "regime": regime.name,
                    "run_id": resolved_run_id,
                }
            )

            # --- FS-5: persist resumable state at every checkpoint ------------
            save_run_state(
                paths,
                iterations_completed=iteration + 1,
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                play_rng=play_rng,
                steps=summary.steps,
                games_played=summary.games_played,
                champion_tag=champion_tag,
                champion_elo=champion_elo,
                profile=profile,
                champion_glicko=champion_glicko.to_dict() if champion_glicko is not None else None,
            )

            # --- FS-5: per-regime HF publish (no-op without HF_TOKEN) --------
            if regime.hf_repo:
                upload_checkpoint(
                    regime.hf_repo,
                    ckpt_path,
                    metadata={
                        "regime": regime.name,
                        "run_id": resolved_run_id,
                        "tag": tag,
                        "steps": summary.steps,
                        "games_played": summary.games_played,
                        "elo": champion_elo if promote else None,
                        "config_hash": regime.to_dict(),
                    },
                )
    finally:
        if queued_evaluator is not None:
            queued_evaluator.close()
        provider.close()
        journal.close()  # FS-6b: release the persistent stream handle

    summary.elapsed_seconds = time.monotonic() - started
    summary.iterations_completed = profile.iterations - iteration_start
    if collect_seconds_total > 0 and session_games > 0:
        summary.games_per_hour = session_games / collect_seconds_total * 3600.0
    log.info(
        "[RUN] %s/%s done in %.1fs: checkpoints=%s champion=%s elo=%s games/h=%.1f",
        regime.name,
        resolved_run_id,
        summary.elapsed_seconds,
        summary.checkpoints,
        summary.champion,
        summary.elo,
        summary.games_per_hour or 0.0,
    )
    return summary


def _arena_match(
    candidate: GambitNet,
    champion: GambitNet,
    device: torch.device,
    arena_config: ArenaConfig,
    candidate_tag: str,
    champion_tag: str,
) -> MatchResult:
    """Candidate vs champion, both colors, reduced sims (§5.1)."""
    arena = Arena(
        BatchNNEvaluator(candidate, device),
        BatchNNEvaluator(champion, device),
        arena_config,
    )
    return arena.match(candidate_tag, champion_tag)


def _save_champion(
    champion_file: Path,
    model: GambitNet,
    tag: str,
    elo: float,
    run_id: str,
    checkpoint_path: Path,
    glicko: GlickoRating | None = None,
) -> None:
    """champion.json is a POINTER to the champion checkpoint (.pt holds weights).

    U9: carries the champion lineage's Glicko record additively (`glicko`
    key) so a fresh run seeding from the pointer continues the rating
    series; older pointers without the key are read fine.

    FB-2: `tag` and `checkpoint_path` must be one resolved checkpoint object.
    The writer asserts ``tag == Path(checkpoint_path).stem`` BEFORE writing —
    a divergent pair (champion's tag under a different checkpoint's path, the
    corruption observed on runs/scratch/champion.json) raises instead of
    poisoning every downstream reader (resume champion load, warm start,
    bench scripts).
    """
    resolved = Path(checkpoint_path)
    if tag != resolved.stem:
        raise ValueError(
            "champion pointer integrity (FB-2): refusing to write "
            f"{champion_file} — tag {tag!r} does not match checkpoint "
            f"filename {resolved.name!r}; tag and path must come from the "
            "same resolved checkpoint object"
        )
    champion_file.parent.mkdir(parents=True, exist_ok=True)
    champion_file.write_text(
        json.dumps(
            {
                "tag": tag,
                "checkpoint_path": str(checkpoint_path),
                "elo": elo,
                "glicko": glicko.to_dict() if glicko is not None else None,
                "run_id": run_id,
                "model_config": model.config.to_dict(),
                "updated_at": time.time(),
            }
        ),
        encoding="utf-8",
    )
