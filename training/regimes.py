"""Training regimes — the §5.2 experiment matrix as swappable dataclass configs.

A regime is a NAMED training recipe, never hardcoded into the pipeline: every
run records its regime, every regime owns its model lineage under
``runs/<regime>/<run-id>/``, and data providers are selected from config.

=================  =========================================================
`scratch`          pure self-play (AlphaZero) — canonical, highest ceiling
`sf-games`         self-play + games vs Stockfish on a skill ladder (2->12)
`sf-distill`       policy <- SF move one-hot, value <- SF eval, on SP positions
`hybrid`           sf-distill warm start (N minutes) -> pure self-play chain
`finetune`         continue from a champion or a public HF net
=================  =========================================================

Slice 2 (FS-3): all five providers return examples in the shared buffer
format. `training.sf_data` holds the Stockfish game generators; every provider
degrades to pure self-play when no Stockfish binary resolves so the pipeline
runs anywhere. Parallelism (`parallel_games`) feeds the queued evaluator
(`training.evaluator`, FS-4) when > 1.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from .selfplay import Evaluator, SelfPlayConfig, SelfPlayResult, parallel_self_play
from .sf_data import (
    SFEngine,
    alternate_colors,
    create_stockfish,
    distill_game,
    play_nn_vs_sf,
    sf_skill_for_progress,
    skill_share_plan,
)

log = logging.getLogger(__name__)

REGIME_NAMES: tuple[str, ...] = ("scratch", "sf-games", "sf-distill", "hybrid", "finetune")

MIN_PARALLEL_GAMES = 1
MAX_PARALLEL_GAMES = 32


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
    warmstart_from: str | None = None  # champion tag, regime name, path, or HF repo id
    # FS-4: concurrent self-play games feeding the queued evaluator (1-32).
    parallel_games: int = 8
    # sf-games: fraction of each iteration's games played vs Stockfish, and the
    # skill ladder (SF "Skill Level" 0-20) scaled linearly by run progress.
    sf_games_share: float = 0.5
    sf_skill_start: int = 2
    sf_skill_end: int = 12
    sf_depth: int = 10
    # hybrid: minutes of sf-distill before the chain switches to pure self-play.
    warmstart_minutes: float = 30.0
    # FS-5: per-regime HF repo ("<owner>/gambit-<regime>", §5.2 provenance).
    hf_repo: str | None = None

    def __post_init__(self) -> None:
        if not MIN_PARALLEL_GAMES <= self.parallel_games <= MAX_PARALLEL_GAMES:
            raise ValueError(
                f"parallel_games must be in [{MIN_PARALLEL_GAMES}, {MAX_PARALLEL_GAMES}], "
                f"got {self.parallel_games}"
            )
        if not 0.0 <= self.sf_games_share <= 1.0:
            raise ValueError(f"sf_games_share must be in [0, 1], got {self.sf_games_share}")
        if self.warmstart_minutes < 0:
            raise ValueError("warmstart_minutes must be >= 0")
        if not 0 <= self.sf_skill_start <= 20 or not 0 <= self.sf_skill_end <= 20:
            raise ValueError("sf_skill_start/sf_skill_end must be SF Skill Levels in [0, 20]")

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
        parallel_games=16,
        hf_repo="LanceAbuan/gambit-scratch",
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
        parallel_games=8,
        sf_games_share=0.5,
        sf_skill_start=2,
        sf_skill_end=12,
        hf_repo="LanceAbuan/gambit-sf-games",
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
        parallel_games=8,
        hf_repo="LanceAbuan/gambit-sf-distill",
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
        parallel_games=16,
        warmstart_minutes=30.0,
        hf_repo="LanceAbuan/gambit-hybrid",
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
        parallel_games=16,
        hf_repo="LanceAbuan/gambit-finetune",
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

    @property
    def state(self) -> Path:
        """FS-5 resumability state directory."""
        return self.root / "state"


def create_run(config: RegimeConfig, run_id: str, base_dir: Path | str = "runs") -> RunPaths:
    """Create the on-disk lineage for a run and stamp its metadata file.

    Idempotent for an existing run id (resume): the original ``created_at``
    survives and a config-hash mismatch is logged, never silently accepted.
    """
    root = Path(base_dir) / config.name / run_id
    checkpoints = root / "checkpoints"
    games = root / "games"
    state = root / "state"
    checkpoints.mkdir(parents=True, exist_ok=True)
    games.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    meta_path = root / "run.json"
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
        if existing.get("config_hash") != config_hash(config):
            log.warning(
                "[REGIME] run %s resumed with a different config hash (%s != %s)",
                run_id,
                config_hash(config),
                existing.get("config_hash"),
            )
        log.info("[REGIME] resuming existing lineage: %s", root)
        return RunPaths(
            regime=config.name,
            run_id=run_id,
            root=root,
            checkpoints=checkpoints,
            games=games,
            meta_path=meta_path,
        )
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


# ---------------------------------------------------------------------------
# Data providers (FS-3). Every provider returns SelfPlayResult games in the
# shared buffer format; keyword args carry the run context the pipeline knows:
#   parallel_games  — concurrent games feeding the queued evaluator (FS-4)
#   elapsed_minutes — wall-clock minutes since the run started (hybrid chain)
#   progress        — run completion in [0, 1] (sf-games skill ladder)
# ---------------------------------------------------------------------------


class DataProvider(Protocol):
    """Source of labelled training examples for a regime."""

    def collect(
        self,
        evaluator: Evaluator,
        num_games: int,
        config: SelfPlayConfig,
        *,
        parallel_games: int = 1,
        elapsed_minutes: float = 0.0,
        progress: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[SelfPlayResult]:
        ...  # pragma: no cover

    def close(self) -> None:
        ...  # pragma: no cover


def _game_rngs(rng: np.random.Generator | None, count: int) -> list[np.random.Generator]:
    """Per-game generators: spawned children of the run stream (or entropy)."""
    if rng is not None:
        return list(rng.spawn(count))
    return [_np_random_generator() for _ in range(count)]


# Engine factory: skill -> engine (None = unavailable). Injectable for tests.
EngineFactory = Callable[[int | None], SFEngine | None]


def _default_engine_factory(depth: int) -> EngineFactory:
    def factory(skill: int | None) -> SFEngine | None:
        return create_stockfish(depth=depth, skill=skill)

    return factory


def _map_games(
    play_one: Callable[[int], SelfPlayResult],
    num_games: int,
    parallel_games: int,
) -> list[SelfPlayResult]:
    """Run game closures across threads; results keep game order."""
    if parallel_games <= 1 or num_games <= 1:
        return [play_one(index) for index in range(num_games)]
    results: list[SelfPlayResult | None] = [None] * num_games

    def _run(index: int) -> None:
        try:
            results[index] = play_one(index)
        except Exception:
            log.exception("[PROVIDER] game %d failed and was skipped", index)

    with ThreadPoolExecutor(max_workers=min(parallel_games, num_games), thread_name_prefix="games") as pool:
        futures = [pool.submit(_run, index) for index in range(num_games)]
        for future in futures:
            future.result()
    missing = [index for index, result in enumerate(results) if result is None]
    if missing:
        log.warning("[PROVIDER] %d/%d games failed and were skipped", len(missing), num_games)
    return [result for result in results if result is not None]


class SelfPlayProvider:
    """`scratch` regime data: pure AlphaZero self-play."""

    def collect(
        self,
        evaluator: Evaluator,
        num_games: int,
        config: SelfPlayConfig,
        *,
        parallel_games: int = 1,
        elapsed_minutes: float = 0.0,
        progress: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[SelfPlayResult]:
        del elapsed_minutes, progress  # pure self-play has no schedule hooks
        return parallel_self_play(
            evaluator, num_games, config, parallel_games=parallel_games, rng=rng
        )

    def close(self) -> None:
        return None


class _StockfishBackedProvider:
    """Shared engine lifecycle for providers that talk to Stockfish."""

    def __init__(self, regime_name: str, engine_factory: EngineFactory | None, depth: int) -> None:
        self.regime_name = regime_name
        self._engine_factory = engine_factory if engine_factory is not None else _default_engine_factory(depth)
        self._engine: SFEngine | None = None
        self._engine_skill: int | None = None
        self._engine_lock = threading.Lock()

    def _engine_for(self, skill: int | None) -> SFEngine | None:
        """Lazily open (or re-skill) the shared engine; None when unavailable."""
        with self._engine_lock:
            if self._engine is not None and self._engine_skill == skill:
                return self._engine
            if self._engine is not None and skill is not None and self._engine_skill is not None:
                # Skill moves monotonically within a collect; a fresh engine per
                # skill step is wasteful, but correctness-first: rebuild only on
                # an actual skill change.
                self._close_engine_locked()
            if self._engine is None:
                self._engine = self._engine_factory(skill)
                self._engine_skill = skill if self._engine is not None else None
                if self._engine is not None:
                    log.info("[PROVIDER] %s engine ready (skill=%s)", self.regime_name, skill)
            return self._engine

    def _close_engine_locked(self) -> None:
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:  # engine cleanup must never crash training
                log.debug("[PROVIDER] engine close failed", exc_info=True)
            self._engine = None
            self._engine_skill = None

    def close(self) -> None:
        with self._engine_lock:
            self._close_engine_locked()


class SFGamesProvider(_StockfishBackedProvider):
    """`sf-games`: self-play mixed with games vs the Stockfish skill ladder.

    Roughly `sf_games_share` of each collect's games pit the NN (MCTS) against
    Stockfish at ``skill = ladder(progress)`` (2 -> 12), alternating colors;
    the rest are pure self-play. Examples come from the self-play games and
    the NN plies of the SF games (same buffer format).
    """

    def __init__(self, regime: RegimeConfig, engine_factory: EngineFactory | None = None) -> None:
        super().__init__(regime.name, engine_factory, regime.sf_depth)
        self.regime = regime

    def collect(
        self,
        evaluator: Evaluator,
        num_games: int,
        config: SelfPlayConfig,
        *,
        parallel_games: int = 1,
        elapsed_minutes: float = 0.0,
        progress: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[SelfPlayResult]:
        del elapsed_minutes  # unused: the skill ladder runs on `progress`
        skill = sf_skill_for_progress(self.regime.sf_skill_start, self.regime.sf_skill_end, progress)
        num_sf_games = skill_share_plan(num_games, self.regime.sf_games_share)
        results: list[SelfPlayResult] = []

        self_play_results = parallel_self_play(
            evaluator,
            num_games - num_sf_games,
            config,
            parallel_games=parallel_games,
            rng=rng,
        )
        results.extend(self_play_results)

        if num_sf_games > 0:
            engine = self._engine_for(skill)
            if engine is None:
                log.warning(
                    "[PROVIDER] %s: no Stockfish — %d SF games degrade to self-play",
                    self.regime_name,
                    num_sf_games,
                )
                sf_results: list[SelfPlayResult] = parallel_self_play(
                    evaluator, num_sf_games, config, parallel_games=parallel_games, rng=rng
                )
            else:
                sf_game_rngs = _game_rngs(rng, num_sf_games)
                sf_results = [
                    play_nn_vs_sf(
                        evaluator,
                        engine,
                        alternate_colors(sf_index),
                        config,
                        skill=skill,
                        rng=sf_game_rngs[sf_index],
                    )
                    for sf_index in range(num_sf_games)
                ]
            results.extend(sf_results)
        log.info(
            "[PROVIDER] %s: %d games (%d vs SF skill %d, %d self-play)",
            self.regime_name,
            len(results),
            num_sf_games,
            skill,
            len(results) - num_sf_games,
        )
        return results


class SFDistillProvider(_StockfishBackedProvider):
    """`sf-distill`: self-play positions labelled by the Stockfish teacher.

    Policy target = SF's chosen move as a one-hot over the 4672 encoder
    indices; value target = SF's evaluation of the position from the mover's
    perspective (see `training.sf_data.distill_game`).
    """

    def __init__(self, regime: RegimeConfig, engine_factory: EngineFactory | None = None) -> None:
        super().__init__(regime.name, engine_factory, regime.sf_depth)
        self.regime = regime

    def collect(
        self,
        evaluator: Evaluator,
        num_games: int,
        config: SelfPlayConfig,
        *,
        parallel_games: int = 1,
        elapsed_minutes: float = 0.0,
        progress: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[SelfPlayResult]:
        del elapsed_minutes, progress  # distill labels do not depend on run progress
        engine = self._engine_for(None)  # full strength for teaching moves
        if engine is None:
            log.warning(
                "[PROVIDER] %s: no Stockfish — distillation degrades to pure self-play",
                self.regime_name,
            )
            return parallel_self_play(
                evaluator, num_games, config, parallel_games=parallel_games, rng=rng
            )
        game_rngs = _game_rngs(rng, num_games)
        return _map_games(
            lambda index: distill_game(evaluator, engine, config, rng=game_rngs[index]),
            num_games,
            parallel_games,
        )


class HybridProvider(_StockfishBackedProvider):
    """`hybrid`: sf-distill warm start for N minutes, then pure self-play.

    The chain length is config (`warmstart_minutes`); the pipeline passes
    elapsed wall-clock minutes and this provider picks the phase. The WEIGHTS
    half of the warm start (loading the sf-distill champion) is the pipeline's
    job (`training.resume.resolve_warmstart`).
    """

    def __init__(self, regime: RegimeConfig, engine_factory: EngineFactory | None = None) -> None:
        super().__init__(regime.name, engine_factory, regime.sf_depth)
        self.regime = regime
        self._distill = SFDistillProvider(regime, engine_factory)
        self._selfplay = SelfPlayProvider()

    def collect(
        self,
        evaluator: Evaluator,
        num_games: int,
        config: SelfPlayConfig,
        *,
        parallel_games: int = 1,
        elapsed_minutes: float = 0.0,
        progress: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[SelfPlayResult]:
        if elapsed_minutes < self.regime.warmstart_minutes:
            log.info(
                "[PROVIDER] hybrid distill phase (%.1f < %.1f min)",
                elapsed_minutes,
                self.regime.warmstart_minutes,
            )
            return self._distill.collect(
                evaluator, num_games, config,
                parallel_games=parallel_games, elapsed_minutes=elapsed_minutes,
                progress=progress, rng=rng,
            )
        log.info(
            "[PROVIDER] hybrid self-play phase (>= %.1f min)",
            self.regime.warmstart_minutes,
        )
        return self._selfplay.collect(
            evaluator, num_games, config,
            parallel_games=parallel_games, elapsed_minutes=elapsed_minutes,
            progress=progress, rng=rng,
        )

    def close(self) -> None:
        self._distill.close()
        self._selfplay.close()


class FinetuneProvider:
    """`finetune`: self-play data on top of warm-started weights.

    The weight init (another regime's champion or an HF repo id) happens once
    at run start in the pipeline; the data stream itself is self-play, so
    transfer-vs-fresh measures cleanly with everything else held fixed.
    """

    def collect(
        self,
        evaluator: Evaluator,
        num_games: int,
        config: SelfPlayConfig,
        *,
        parallel_games: int = 1,
        elapsed_minutes: float = 0.0,
        progress: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[SelfPlayResult]:
        del elapsed_minutes, progress  # data stream is plain self-play
        return parallel_self_play(
            evaluator, num_games, config, parallel_games=parallel_games, rng=rng
        )

    def close(self) -> None:
        return None


class StubDataProvider:
    """Placeholder provider for regimes whose data hooks are not wired."""

    def __init__(self, regime_name: str, data_source: str) -> None:
        self.regime_name = regime_name
        self.data_source = data_source

    def collect(
        self,
        evaluator: Evaluator,
        num_games: int,
        config: SelfPlayConfig,
        *,
        parallel_games: int = 1,
        elapsed_minutes: float = 0.0,
        progress: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[SelfPlayResult]:
        raise NotImplementedError(
            f"regime '{self.regime_name}' (data_source='{self.data_source}') has no data "
            "provider wired; known sources: selfplay, sf-games, sf-distill, hybrid, finetune"
        )

    def close(self) -> None:
        return None


_PROVIDER_BY_SOURCE: dict[str, Callable[[RegimeConfig], DataProvider]] = {
    "selfplay": lambda config: SelfPlayProvider(),  # type: ignore[dict-item,misc]
    "sf-games": SFGamesProvider,
    "sf-distill": SFDistillProvider,
    "hybrid": HybridProvider,
    "finetune": lambda config: FinetuneProvider(),  # type: ignore[dict-item,misc]
}


def build_data_provider(config: RegimeConfig) -> DataProvider:
    """Regime -> data provider. Never hardcodes a regime into the pipeline."""
    factory = _PROVIDER_BY_SOURCE.get(config.data_source)
    if factory is None:
        return StubDataProvider(config.name, config.data_source)
    return factory(config)


def _np_random_generator() -> np.random.Generator:
    """Fresh entropy-backed generator for SF-driven games."""
    return np.random.default_rng()


def all_regime_names() -> Sequence[str]:
    return REGIME_NAMES
