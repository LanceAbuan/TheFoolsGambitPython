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

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .encoder import NUM_PLANES
from .evaluator import QueuedNNEvaluator
from .hf import upload_checkpoint
from .journal import TrainingJournal
from .ladder import (
    Arena,
    LadderConfig,
    MatchResult,
    due_snapshot_minutes,
    should_promote,
    update_elo,
)
from .ledger import ArenaLedger, LedgerEntry, MetricsWriter, recompute_elos
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
from .trainer import TrainingBuffer

log = logging.getLogger(__name__)

CHAMPION_FILENAME = "champion.json"


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
            "ts": time.time(),
        },
        path,
    )
    log.info("[CKPT] saved %s", path)


def _load_champion(
    checkpoint_path: Path, device: torch.device
) -> tuple[GambitNet, str, float] | None:
    """Load a champion checkpoint file; None when absent or incompatible."""
    try:
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - unreadable champion checkpoints gate as "no champion"
        log.warning("[CHAMPION] could not load %s; treating as no champion", checkpoint_path)
        return None
    config = ModelConfig.from_dict(data["config"])
    model = GambitNet(config).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, str(data.get("tag", "unknown")), float(data.get("elo", 800.0))


def _load_champion_pointer(champion_file: Path) -> tuple[Path, float] | None:
    """Read champion.json -> (checkpoint path, elo); None when absent/broken."""
    try:
        data = json.loads(champion_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    path = Path(str(data.get("checkpoint_path", "")))
    if not path.exists():
        log.warning("[CHAMPION] pointer %s missing checkpoint %s", champion_file, path)
        return None
    return path, float(data.get("elo", 800.0))


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


def run_training(
    profile: TrainingProfile,
    run_id: str | None = None,
    base_dir: Path | str = "runs",
    data_dir: Path | str = "data",
    resume: str | bool = False,
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

    champion_file = Path(base_dir) / regime.name / CHAMPION_FILENAME
    champion_tag: str | None = None
    champion_elo = profile.ladder.base_elo

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

    summary = RunSummary(
        run_id=resolved_run_id,
        regime=regime.name,
        root=paths.root,
        steps=0 if state is None else int(state.get("steps", 0) or 0),
        games_played=0 if state is None else int(state.get("games_played", 0) or 0),
        resumed=state is not None,
    )
    last_snapshot_minute = 0.0
    arena_games_total = 0
    last_loss: float | None = None
    last_policy_loss: float | None = None
    last_value_loss: float | None = None
    last_match: MatchResult | None = None
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
                search=SearchConfig(num_simulations=profile.sims),
                max_plies=profile.max_plies,
                temperature_plies=profile.max_plies,  # sample throughout a smoke game
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

            candidate_elo = profile.ladder.base_elo
            opponent_for_event = champion_tag or "unknown"
            if champion_model is None:
                # First checkpoint becomes champion by default; recorded as baseline.
                ledger.append(
                    LedgerEntry(
                        run_id=resolved_run_id,
                        regime=regime.name,
                        checkpoint=tag,
                        opponent="none",
                        games=0,
                        wins=0,
                        draws=0,
                        elo=candidate_elo,
                        kind="baseline",
                        promoted=True,
                    )
                )
                champion_model, champion_tag, champion_elo = model, tag, candidate_elo
                log.info("[GATE] %s becomes the first champion (baseline elo %.1f)", tag, candidate_elo)
                last_match = None
                opponent_for_event = "none"
            else:
                match: MatchResult = _arena_match(
                    model, champion_model, device, profile.ladder.arena, tag, champion_tag or "unknown"
                )
                last_match = match
                arena_games_total += match.games
                candidate_elo, champion_elo = update_elo(
                    profile.ladder.base_elo, champion_elo, match, profile.ladder.k_factor
                )
                promote = should_promote(match, profile.ladder.promote_win_rate)
                ledger.append(
                    LedgerEntry(
                        run_id=resolved_run_id,
                        regime=regime.name,
                        checkpoint=tag,
                        opponent=champion_tag or "unknown",
                        games=match.games,
                        wins=match.wins,
                        draws=match.draws,
                        elo=candidate_elo,
                        losses=match.losses,
                        promoted=promote,
                    )
                )
                if promote:
                    champion_model, champion_tag, champion_elo = model, tag, candidate_elo
                    log.info("[GATE] %s promoted (score %.3f >= %.2f)",
                             tag, match.win_rate, profile.ladder.promote_win_rate)
                else:
                    log.info("[GATE] champion %s holds (score %.3f < %.2f)",
                             champion_tag, match.win_rate, profile.ladder.promote_win_rate)

            # --- champion + metrics feed -------------------------------------
            _save_champion(champion_file, champion_model, champion_tag or tag, champion_elo,
                           resolved_run_id, ckpt_path)
            elo_map = recompute_elos(ledger.read(), profile.ladder.base_elo, profile.ladder.k_factor)
            summary.arena_games = arena_games_total
            summary.champion = champion_tag
            summary.elo = elo_map.get(champion_tag or "", champion_elo)
            metrics.update(
                elo=summary.elo,
                win_rate=last_match.win_rate if last_match is not None else None,
                games_played=summary.games_played + arena_games_total,
                arena_games=arena_games_total,
                loss=summary.final_loss,
                champion=champion_tag,
                regime=regime.name,
                checkpoints={k: round(v, 1) for k, v in elo_map.items()},
                parallel_games=effective_parallel,
            )

            # --- FS-6 journal: one event per arena gate ----------------------
            # Arena events carry the full required key set too (loss family
            # reflects the iteration that produced this checkpoint).
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
                        "promoted": last_match is not None and should_promote(
                            last_match, profile.ladder.promote_win_rate
                        ),
                    }
                    if last_match is not None
                    else None,
                    "elo": candidate_elo,
                    "loss": last_loss,
                    "policy_loss": last_policy_loss,
                    "value_loss": last_value_loss,
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
                        "elo": candidate_elo,
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
    champion_file: Path, model: GambitNet, tag: str, elo: float, run_id: str, checkpoint_path: Path
) -> None:
    """champion.json is a POINTER to the champion checkpoint (.pt holds weights)."""
    champion_file.parent.mkdir(parents=True, exist_ok=True)
    champion_file.write_text(
        json.dumps(
            {
                "tag": tag,
                "checkpoint_path": str(checkpoint_path),
                "elo": elo,
                "run_id": run_id,
                "model_config": model.config.to_dict(),
                "updated_at": time.time(),
            }
        ),
        encoding="utf-8",
    )
