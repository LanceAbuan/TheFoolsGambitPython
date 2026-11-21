"""WP-C training pipeline v2 — the self-play -> optimize -> gate loop.

Orchestrates (all CPU-safe; CUDA used when available):
  1. regime config + lineage (`runs/<regime>/<run-id>/`, §5.2)
  2. self-play games with the §10-correct MCTS (`selfplay.py`)
  3. replay-buffer sampling + optimization (reuses legacy `TrainingBuffer`)
  4. wall-clock checkpoint ladder + arena gating (§5.1, `ladder.py`)
  5. arena ledger JSONL + metrics feed (`ledger.py`)

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
from .regimes import RegimeConfig, RunPaths, create_run, get_regime
from .selfplay import (
    ArenaConfig,
    BatchNNEvaluator,
    SearchConfig,
    SelfPlayConfig,
    SelfPlayResult,
    self_play_game,
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
    except (OSError, ValueError, KeyError):
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


def run_training(
    profile: TrainingProfile,
    run_id: str | None = None,
    base_dir: Path | str = "runs",
    data_dir: Path | str = "data",
) -> RunSummary:
    """Run the full loop: self-play -> optimize -> ladder -> arena -> ledger."""
    started = time.monotonic()
    if profile.seed is not None:
        torch.manual_seed(profile.seed)
        np.random.seed(profile.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regime: RegimeConfig = get_regime(profile.regime)
    resolved_run_id = run_id or _utc_run_id()
    paths: RunPaths = create_run(regime, resolved_run_id, base_dir)
    ledger = ArenaLedger(Path(data_dir) / "arena.jsonl")
    metrics = MetricsWriter(Path(data_dir) / "metrics.json")

    model = GambitNet(profile.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=regime.learning_rate)
    buffer: TrainingBuffer = TrainingBuffer(max_size=profile.buffer_size)

    champion_file = Path(base_dir) / regime.name / CHAMPION_FILENAME
    pointer = _load_champion_pointer(champion_file)
    champion: tuple[GambitNet, str, float] | None = None
    if pointer is not None:
        champion = _load_champion(pointer[0], device)
        if champion is not None:
            champion = (champion[0], champion[1], pointer[1])
    champion_tag: str | None = None
    champion_elo = profile.ladder.base_elo
    if champion is not None:
        champion_model, champion_tag, champion_elo = champion
        log.info("[CHAMPION] reigning champion %s (elo %.1f)", champion_tag, champion_elo)
    else:
        champion_model = None

    summary = RunSummary(run_id=resolved_run_id, regime=regime.name, root=paths.root)
    last_snapshot_minute = 0.0
    arena_games_total = 0
    last_loss: float | None = None
    last_match: MatchResult | None = None

    for iteration in range(profile.iterations):
        # --- self-play -----------------------------------------------------
        model.eval()
        evaluator = BatchNNEvaluator(model, device)
        selfplay_config = SelfPlayConfig(
            search=SearchConfig(num_simulations=profile.sims),
            max_plies=profile.max_plies,
            temperature_plies=profile.max_plies,  # sample throughout a smoke game
        )
        games_this_iter: list[SelfPlayResult] = []
        for _game in range(profile.games_per_iteration):
            result = self_play_game(evaluator, selfplay_config)
            buffer.add(
                {"board_tensor": ex.planes, "policy": ex.policy, "value": ex.value}
                for ex in result.examples
            )
            games_this_iter.append(result)
            summary.games_played += 1
        log.info(
            "[SELFPLAY] iter %d: %d games, buffer=%d, results=%s",
            iteration,
            len(games_this_iter),
            len(buffer),
            [r.result for r in games_this_iter],
        )

        # --- optimize ------------------------------------------------------
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
        if last_loss is not None:
            summary.final_loss = last_loss
            log.info("[TRAIN] iter %d: loss %.4f", iteration, last_loss)

        # --- ladder + arena gate (§5.1) ------------------------------------
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

        # --- champion + metrics feed ---------------------------------------
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
        )

    summary.elapsed_seconds = time.monotonic() - started
    log.info(
        "[RUN] %s/%s done in %.1fs: checkpoints=%s champion=%s elo=%s",
        regime.name,
        resolved_run_id,
        summary.elapsed_seconds,
        summary.checkpoints,
        summary.champion,
        summary.elo,
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
