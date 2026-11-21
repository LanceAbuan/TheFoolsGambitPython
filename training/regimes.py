"""Training regimes — the §5.2 experiment matrix as swappable dataclass configs.

A regime is a NAMED training recipe, never hardcoded into the pipeline: every
run records its regime, every regime owns its model lineage under
``runs/<regime>/<run-id>/``, and data providers are selected from config.

=================  ==========================================================
`scratch`          pure self-play (AlphaZero) — canonical, highest ceiling
`sf-games`         self-play + games vs Stockfish on a fixed skill ladder
`sf-distill`       policy <- SF move choice, value <- SF eval on SP positions
`hybrid`           sf-distill warm start -> pure self-play
`finetune`         continue from a champion or a public HF net
=================  ==========================================================

Slice 1 tonight: all five configs + lineage paths exist and are constructible;
only `scratch` has a working data provider. The sf-*/finetune providers raise
NotImplementedError with clear messages until slice 2 wires the data hooks.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .selfplay import Evaluator, SelfPlayConfig, SelfPlayResult, self_play_game

log = logging.getLogger(__name__)

REGIME_NAMES: tuple[str, ...] = ("scratch", "sf-games", "sf-distill", "hybrid", "finetune")


@dataclass(frozen=True)
class RegimeConfig:
    """One named training recipe (§5.2). Immutable + hashable for provenance."""

    name: str
    description: str
    data_source: str  # 'selfplay' | 'sf-games' | 'sf-distill' | 'hybrid' | 'finetune'
    num_simulations: int
    games_per_iteration: int
    train_steps_per_iteration: int
    batch_size: int
    learning_rate: float
    warmstart_from: str | None = None  # champion tag or HF repo id, regime-dependent

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


REGIMES: dict[str, RegimeConfig] = {
    "scratch": RegimeConfig(
        name="scratch",
        description="Pure AlphaZero self-play from random init (§5.2 canonical run)",
        data_source="selfplay",
        num_simulations=400,
        games_per_iteration=32,
        train_steps_per_iteration=64,
        batch_size=256,
        learning_rate=1e-3,
    ),
    "sf-games": RegimeConfig(
        name="sf-games",
        description="Self-play + games vs Stockfish on a fixed skill ladder (§5.2)",
        data_source="sf-games",
        num_simulations=400,
        games_per_iteration=32,
        train_steps_per_iteration=64,
        batch_size=256,
        learning_rate=1e-3,
    ),
    "sf-distill": RegimeConfig(
        name="sf-distill",
        description="Policy <- Stockfish moves, value <- Stockfish evals on self-play positions",
        data_source="sf-distill",
        num_simulations=200,
        games_per_iteration=64,
        train_steps_per_iteration=128,
        batch_size=256,
        learning_rate=1e-3,
    ),
    "hybrid": RegimeConfig(
        name="hybrid",
        description="sf-distill warm start -> pure self-play (§5.2 Elo-per-hour hypothesis)",
        data_source="hybrid",
        num_simulations=400,
        games_per_iteration=32,
        train_steps_per_iteration=64,
        batch_size=256,
        learning_rate=1e-3,
        warmstart_from="sf-distill",
    ),
    "finetune": RegimeConfig(
        name="finetune",
        description="Continue from a champion or seed from a public HF net (transfer test)",
        data_source="finetune",
        num_simulations=400,
        games_per_iteration=32,
        train_steps_per_iteration=64,
        batch_size=256,
        learning_rate=3e-4,
        warmstart_from="champion",
    ),
}


def get_regime(name: str) -> RegimeConfig:
    """Fetch a regime by name; unknown names list the valid choices."""
    try:
        return REGIMES[name]
    except KeyError:
        raise ValueError(f"unknown regime '{name}'; valid regimes: {', '.join(REGIME_NAMES)}") from None


def config_hash(config: RegimeConfig) -> str:
    """Stable content hash of the regime config (run provenance, §5.2)."""
    payload = json.dumps(config.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class RunPaths:
    """Lineage layout for one run: runs/<regime>/<run-id>/. (§5.2)"""

    regime: str
    run_id: str
    root: Path
    checkpoints: Path
    games: Path
    meta_path: Path


def create_run(config: RegimeConfig, run_id: str, base_dir: Path | str = "runs") -> RunPaths:
    """Create the on-disk lineage for a run and stamp its metadata file."""
    root = Path(base_dir) / config.name / run_id
    checkpoints = root / "checkpoints"
    games = root / "games"
    checkpoints.mkdir(parents=True, exist_ok=True)
    games.mkdir(parents=True, exist_ok=True)
    meta_path = root / "run.json"
    meta = {
        "regime": config.name,
        "run_id": run_id,
        "config_hash": config_hash(config),
        "config": config.to_dict(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("[REGIME] run lineage created: %s (hash %s)", root, meta["config_hash"])
    return RunPaths(
        regime=config.name,
        run_id=run_id,
        root=root,
        checkpoints=checkpoints,
        games=games,
        meta_path=meta_path,
    )


class DataProvider(Protocol):
    """Source of labelled training examples for a regime."""

    def collect(
        self, evaluator: Evaluator, num_games: int, config: SelfPlayConfig
    ) -> list[SelfPlayResult]:
        ...  # pragma: no cover


class SelfPlayProvider:
    """`scratch` regime data: pure AlphaZero self-play (works tonight)."""

    def collect(
        self, evaluator: Evaluator, num_games: int, config: SelfPlayConfig
    ) -> list[SelfPlayResult]:
        return [self_play_game(evaluator, config) for _ in range(num_games)]


class StubDataProvider:
    """Placeholder provider for regimes whose data hooks land in slice 2."""

    def __init__(self, regime_name: str, data_source: str) -> None:
        self.regime_name = regime_name
        self.data_source = data_source

    def collect(
        self, evaluator: Evaluator, num_games: int, config: SelfPlayConfig
    ) -> list[SelfPlayResult]:
        raise NotImplementedError(
            f"regime '{self.regime_name}' (data_source='{self.data_source}') needs the "
            "WP-C slice 2 data hooks: Stockfish data plumbing (sf-games / sf-distill), "
            "warm-start loading (hybrid / finetune). Configs and lineage paths already exist."
        )


def build_data_provider(config: RegimeConfig) -> DataProvider:
    """Regime -> data provider. Never hardcodes a regime into the pipeline."""
    if config.data_source == "selfplay":
        return SelfPlayProvider()
    return StubDataProvider(config.name, config.data_source)


def all_regime_names() -> Sequence[str]:
    return REGIME_NAMES
