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
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .encoder import ENCODER_VERSION, NUM_PLANES, NUM_PLANES_V2
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
from .model import GambitNet, ModelConfig
from .regimes import RegimeConfig, RunPaths, build_data_provider, create_run, get_regime
from .resume import (
    load_latest_run_state,
    profile_signature,
    resolve_warmstart,
    restore_run_state,
    save_run_state,
    validate_profile_signature,
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
    buffer_size: int = 100_000
    ladder: LadderConfig = field(default_factory=LadderConfig)
    seed: int | None = 0
    snapshot_every_iteration: bool = False  # smoke: gate every iteration
    parallel_games: int | None = None  # None -> RegimeConfig.parallel_games
    stream_games: bool | None = None  # None -> RegimeConfig.stream_moves (FS-6b)
    # WP-A: optional step-decay LR schedule (milestones on ABSOLUTE iterations).
    # None -> the regime's constant `learning_rate` exactly as before, so every
    # pre-existing profile (smoke, night) keeps its behavior byte-identical.
    lr_schedule: LRStepSchedule | None = None


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
NIGHT_V2_PROFILE = replace(
    NIGHT_PROFILE,
    name="night-v2",
    parallel_games=32,
    sims=600,
    train_steps=96,
    batch_size=512,
    min_examples=128,
    lr_schedule=LRStepSchedule(milestones=(200, 400, 600, 800), gamma=0.5),
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


def run_training(
    profile: TrainingProfile,
    run_id: str | None = None,
    base_dir: Path | str = "runs",
    data_dir: Path | str = "data",
    resume: str | bool = False,
    search_kind: str | None = None,
    playout_randomization: bool | None = None,
) -> RunSummary:
    """Run the full loop: collect -> optimize -> ladder -> arena -> ledger.

    `resume=True` resumes `run_id`; `resume="<run-id>"` resumes that run
    directly (FS-5): model weights, optimizer, replay buffer, RNG streams and
    the iteration counter are rebuilt from the run's persisted state.
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
    optimizer = torch.optim.Adam(model.parameters(), lr=regime.learning_rate)
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
        _apply_warmstart(model, regime, Path(base_dir))

    # --- reigning champion (regime-level pointer, §5.1) ---------------------
    # A resumed run first tries ITS OWN checkpoint for the restored champion
    # tag (exact rebuild), then falls back to the regime pointer.
    champion_model: GambitNet | None = None
    pointer = _load_champion_pointer(champion_file)
    if champion_tag is not None:
        own_checkpoint = paths.checkpoints / f"{champion_tag}.pt"
        loaded = _load_champion(own_checkpoint, device) if own_checkpoint.exists() else None
        if loaded is None and pointer is not None:
            loaded = _load_champion(pointer[0], device)
        if loaded is not None:
            champion_model = loaded[0]
            log.info(
                "[CHAMPION] resumed champion %s (elo %.1f)", champion_tag, champion_elo
            )
    elif pointer is not None:
        loaded = _load_champion(pointer[0], device)
        if loaded is not None:
            champion_model, champion_tag, champion_elo = loaded[0], loaded[1], pointer[1]
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
    queued_evaluator: QueuedNNEvaluator | None = None
    provider = build_data_provider(regime)
    try:
        if effective_parallel > 1:
            queued_evaluator = QueuedNNEvaluator(model, device)
            log.info("[SELFPLAY] parallel_games=%d via the queued evaluator", effective_parallel)

        for iteration in range(iteration_start, profile.iterations):
            # --- collect (provider-driven, §5.2) ----------------------------
            model.eval()
            if queued_evaluator is not None:
                evaluator: Evaluator = queued_evaluator
            else:
                evaluator = BatchNNEvaluator(model, device)
            selfplay_config = SelfPlayConfig(
                search=SearchConfig(
                    num_simulations=profile.sims,
                    search_kind=regime.search_kind,
                    gumbel_candidates=regime.gumbel_candidates,
                ),
                max_plies=profile.max_plies,
                temperature_plies=profile.max_plies,  # sample throughout a smoke game
                playout_randomization=regime.playout_randomization,
                playout_cap_full_prob=regime.playout_cap_full_prob,
                playout_cap_fast_fraction=regime.playout_cap_fast_fraction,
                playout_cap_start_ply=regime.playout_cap_start_ply,
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
            games_this_iter: list[SelfPlayResult] = provider.collect(
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
                    {"board_tensor": ex.planes, "policy": ex.policy, "value": ex.value}
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
            iteration_lr = _apply_lr_schedule(
                optimizer, profile.lr_schedule, iteration, regime.learning_rate
            )
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
                tensors, policies, values = batch
                tensors = tensors.to(device)
                policies = policies.to(device)
                values = values.to(device)
                optimizer.zero_grad()
                pred_policy, pred_value = model(tensors)
                log_probs = torch.log_softmax(pred_policy, dim=-1)
                policy_loss = -(policies * log_probs).sum(dim=-1).mean()
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
                optimizer.step()
                summary.steps += 1
                last_loss = float(loss.item())
                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
            if last_loss is not None:
                summary.final_loss = last_loss
                log.info("[TRAIN] iter %d: loss %.4f", iteration, last_loss)

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
            due = due_snapshot_minutes(profile.ladder, elapsed_minutes, last_snapshot_minute)
            final_iteration = iteration == profile.iterations - 1
            if not due and not final_iteration and not profile.snapshot_every_iteration:
                continue
            last_snapshot_minute = max([elapsed_minutes, *due])

            tag = f"ckpt-{summary.steps:06d}-{summary.games_played:04d}"
            ckpt_path = paths.checkpoints / f"{tag}.pt"
            _save_checkpoint(model, ckpt_path, tag, regime.name, resolved_run_id, summary.steps, summary.games_played)
            summary.checkpoints.append(tag)

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
                arena_config = replace(
                    profile.ladder.arena,
                    sprt_early_stop=True,
                    sprt_elo0=profile.ladder.sprt_elo0,
                    sprt_elo1=profile.ladder.sprt_elo1,
                    sprt_alpha=profile.ladder.sprt_alpha,
                    sprt_beta=profile.ladder.sprt_beta,
                )
                match: MatchResult = _arena_match(
                    model, champion_model, device, arena_config, tag, champion_tag or "unknown"
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
            _save_champion(champion_file, champion_model, champion_tag or tag, champion_elo,
                           resolved_run_id, ckpt_path, champion_glicko)
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
    """
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
