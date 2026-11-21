"""Stockfish data plumbing for the sf-* regimes (plan §5.2, FS-3).

Two game generators share the self-play labeling conventions:

- :func:`play_nn_vs_sf` — `sf-games`: the NN (MCTS) plays full games against
  Stockfish on a skill ladder (skill 2 -> 12 scaled by run progress). Training
  examples are recorded for NN plies only (policy = MCTS visit counts, value =
  game outcome from the mover's perspective) — the SF plies are opponent
  pressure, not supervision.
- :func:`distill_game` — `sf-distill`: positions come from self-play play (the
  NN's own MCTS drives the moves), but the TEACHER labels every position:
  policy target = Stockfish's chosen move as a one-hot over the 4672 policy
  indices (via `legal_move_map`/`encode_move`), value target = Stockfish's
  evaluation of the position normalized to [-1, 1] from the mover's
  perspective. Where the engine cannot answer (dead process, game-over edge)
  the label falls back to the self-play label for that ply so data keeps
  flowing instead of crashing the run.

Every function degrades gracefully when no Stockfish binary exists: the
providers in `training.regimes` catch the resolution failure and fall back to
pure self-play so CI (no engine, no GPU) stays green.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import chess
import numpy as np

from .encoder import encode_move, encode_state, legal_move_map
from .selfplay import (
    Evaluator,
    MCTSSearch,
    SelfPlayConfig,
    SelfPlayResult,
    TrainingExample,
    build_pgn,
    finalize_game_examples,
    game_outcome,
    select_move_from_visits,
)
from .stockfish_engine import StockfishPlayer, resolve_stockfish_path

log = logging.getLogger(__name__)

# Centipawn normalization shared with the legacy wrapper (cp/2000 -> [-1, 1]).
SF_EVAL_CP_SCALE = 2000.0


class SFEngine(Protocol):
    """The slice of `StockfishPlayer` the data generators rely on."""

    def get_move(self, board: chess.Board, depth: int | None = None) -> str | None:
        ...  # pragma: no cover

    def get_evaluation(self, board: chess.Board) -> int:
        ...  # pragma: no cover

    def close(self) -> None:
        ...  # pragma: no cover


def sf_skill_for_progress(skill_start: int, skill_end: int, progress: float) -> int:
    """Linear skill ladder 2 -> 12 scaled by run progress in [0, 1]."""
    clamped = min(max(progress, 0.0), 1.0)
    return round(skill_start + (skill_end - skill_start) * clamped)


def create_stockfish(depth: int = 10, skill: int | None = None) -> StockfishPlayer | None:
    """Open a Stockfish player for data generation; None when no binary resolves."""
    if resolve_stockfish_path() is None:
        log.warning("[SF-DATA] no Stockfish binary found — sf-* data degrades to self-play")
        return None
    try:
        return StockfishPlayer(depth=depth, skill=skill)
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        log.warning("[SF-DATA] Stockfish unavailable (%s) — degrading to self-play", exc)
        return None


def _normalized_eval_from_mover_perspective(engine: SFEngine, board: chess.Board) -> float:
    """SF eval of `board` in [-1, 1] from the side-to-move's perspective."""
    cp = engine.get_evaluation(board)  # white-positive centipawns (legacy wrapper)
    value = max(-1.0, min(1.0, cp / SF_EVAL_CP_SCALE))
    return value if board.turn == chess.WHITE else -value


def play_nn_vs_sf(
    nn_evaluator: Evaluator,
    engine: SFEngine,
    nn_color: chess.Color,
    config: SelfPlayConfig,
    skill: int,
    rng: np.random.Generator | None = None,
) -> SelfPlayResult:
    """One NN-vs-Stockfish training game; examples recorded for NN plies only."""
    search = MCTSSearch(nn_evaluator, config.search)
    board = chess.Board()
    prev_board: chess.Board | None = None
    pending: list[tuple[np.ndarray, np.ndarray, chess.Color]] = []
    sans: list[str] = []
    resigned_side: str | None = None
    streak = 0

    for ply in range(config.max_plies):
        if board.is_game_over(claim_draw=True):
            break

        if board.turn == nn_color:
            result = search.search(board)
            legal_map = legal_move_map(board)
            total = result.visit_counts.sum()
            policy = result.visit_counts / total if total > 0 else np.zeros_like(result.visit_counts)
            pending.append((encode_state(board, prev_board), policy.copy(), board.turn))
            temperature = config.temperature if ply < config.temperature_plies else 0.0
            move = select_move_from_visits(result.visit_counts, legal_map, temperature, rng)
            # Resignation (§10 convention): consecutive sub-threshold evals past
            # the minimum ply, default off via SelfPlayConfig.
            if (
                config.resign_enabled
                and (ply + 1) >= config.resign_min_ply
                and result.root_value < config.resign_threshold
            ):
                streak += 1
                if streak >= config.resign_consecutive:
                    resigned_side = 'white' if board.turn == chess.WHITE else 'black'
                    break
            else:
                streak = 0
        else:
            uci = engine.get_move(board)
            move: chess.Move | None = None
            if uci:
                candidate = chess.Move.from_uci(uci)
                if candidate in board.legal_moves:
                    move = candidate
            if move is None:
                # Engine unavailable for this ply — keep the game alive with the
                # NN's own search so the training data never stalls.
                log.debug("[SF-DATA] engine gave no move at ply %d; NN substitutes", ply)
                result = search.search(board)
                legal_map = legal_move_map(board)
                move = select_move_from_visits(result.visit_counts, legal_map, config.temperature, rng)

        sans.append(board.san(move))
        prev_board = board.copy()
        board.push(move)

    outcome, result_str = game_outcome(board, resigned_side)
    examples = finalize_game_examples(pending, outcome)
    return SelfPlayResult(
        examples=examples,
        moves=sans,
        pgn=build_pgn(sans, result_str),
        result=result_str,
        ply_count=len(sans),
        resigned_side=resigned_side,
        opponent=f"stockfish-skill-{skill}",
    )


def distill_game(
    nn_evaluator: Evaluator,
    engine: SFEngine,
    config: SelfPlayConfig,
    rng: np.random.Generator | None = None,
) -> SelfPlayResult:
    """Self-play positions, Stockfish teacher labels (`sf-distill` regime).

    The NN's MCTS chooses the moves (on-policy positions), while each recorded
    ply is labelled by the teacher: policy = one-hot at the encoder index of
    SF's chosen move, value = SF's evaluation from the mover's perspective.
    Terminal positions keep the exact game outcome (SF eval is meaningless
    there). If the engine fails for one ply, that ply falls back to the
    self-play label (visit counts + search value) so the run continues.
    """
    search = MCTSSearch(nn_evaluator, config.search)
    board = chess.Board()
    prev_board: chess.Board | None = None
    pending: list[tuple[np.ndarray, np.ndarray, chess.Color, float | None]] = []
    sans: list[str] = []
    teacher_plies = 0
    fallback_plies = 0

    for ply in range(config.max_plies):
        if board.is_game_over(claim_draw=True):
            break

        result = search.search(board)
        legal_map = legal_move_map(board)
        total = result.visit_counts.sum()
        visit_policy = result.visit_counts / total if total > 0 else np.zeros_like(result.visit_counts)

        teacher_policy: np.ndarray | None = None
        teacher_value: float | None = None
        if not board.is_game_over(claim_draw=True):
            uci = engine.get_move(board)
            if uci:
                sf_move = chess.Move.from_uci(uci)
                if sf_move in legal_map.values():
                    index = encode_move(board, sf_move)
                    one_hot = np.zeros(visit_policy.shape[0], dtype=np.float32)
                    one_hot[index] = 1.0
                    teacher_policy = one_hot
                    teacher_value = _normalized_eval_from_mover_perspective(engine, board)

        if teacher_policy is not None and teacher_value is not None:
            policy = teacher_policy
            value_override: float | None = teacher_value
            teacher_plies += 1
        else:
            policy = visit_policy
            value_override = None  # fall back to the game outcome label
            fallback_plies += 1

        pending.append((encode_state(board, prev_board), policy, board.turn, value_override))

        temperature = config.temperature if ply < config.temperature_plies else 0.0
        move = select_move_from_visits(result.visit_counts, legal_map, temperature, rng)
        sans.append(board.san(move))
        prev_board = board.copy()
        board.push(move)

    outcome, result_str = game_outcome(board)
    examples = []
    for planes, policy, color, value_override in pending:
        outcome_label = outcome if color == chess.WHITE else -outcome
        examples.append(
            TrainingExample(
                planes=planes,
                policy=policy,
                value=value_override if value_override is not None else outcome_label,
            )
        )

    log.debug(
        "[SF-DATA] distill game: %d teacher plies, %d fallback plies, result %s",
        teacher_plies,
        fallback_plies,
        result_str,
    )
    return SelfPlayResult(
        examples=examples,
        moves=sans,
        pgn=build_pgn(sans, result_str),
        result=result_str,
        ply_count=len(sans),
        opponent="stockfish-distill",
    )


def skill_share_plan(num_games: int, sf_games_share: float) -> int:
    """How many of `num_games` are Stockfish games for a given share."""
    if num_games <= 0:
        return 0
    if sf_games_share <= 0.0:
        return 0
    if sf_games_share >= 1.0:
        return num_games
    return max(1, round(num_games * sf_games_share))


def alternate_colors(game_index: int) -> chess.Color:
    """Even SF games as White, odd as Black — balanced pressure over a run."""
    return chess.WHITE if game_index % 2 == 0 else chess.BLACK


def evaluate_positions_with_sf(
    engine: SFEngine,
    boards: Sequence[chess.Board],
) -> list[float]:
    """Batch helper: normalized mover-perspective evals for a list of positions."""
    return [_normalized_eval_from_mover_perspective(engine, board) for board in boards]
