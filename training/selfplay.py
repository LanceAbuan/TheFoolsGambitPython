"""Self-play engine using MCTS with neural network guidance.

Stockfish-guided rollouts: at leaf nodes, MCTS blends Stockfish's centipawn
evaluation (depth 10) with the NN's own value head. The NN still generates
the policy priors for move selection — Stockfish only provides a stronger
value signal at evaluation time.

Blend ratio: 0.6 Stockfish / 0.4 NN to keep the network's own judgment
in the loop and prevent overfitting to engine scores. Gaussian noise
(sigma=0.1) is added to Stockfish evals to avoid memorization.
"""
import logging
import math
import queue
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

import chess
import chess.pgn
import numpy as np
import torch

from .encoder import (
    encode_batch,
    encode_batch_v2,
    encode_move,
    encode_state,
    encode_state_v2,
    encoder_version_for_model,
    legal_move_map,
)
from .fastpath import fastpath_enabled
from .model import NUM_POLICY_MOVES_V2, value_head_scalar
from .tensorize import NUM_POSSIBLE_MOVES, board_to_tensor, move_to_idx

if TYPE_CHECKING:
    from .rust_search import RustMCTSSearch

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

SF_LEAF_BLEND = 0.6
SF_LEAF_DEPTH = 10
SF_EVAL_NOISE_SIGMA = 0.1
MCTS_EVAL_CACHE_MAX = 500  # LRU cap on eval cache

RESIGN_THRESHOLD = -0.9  # §10: 3 consecutive evals below -0.9 (was instant-resign at -0.8)
RESIGN_CONSECUTIVE = 3   # consecutive sub-threshold evals required before resigning
RESIGN_MIN_PLY = 40      # resignation only allowed past this ply
NUM_MCTS_THREADS = 8  # parallel threads per MCTS search


class BatchEvaluator:
    """Handles batching of neural network forward passes to maximize GPU throughput.

    Multiple callers submit board tensors via ``get_eval``; the worker thread
    collects them into batches and runs a single GPU forward pass.  Callers
    block on a ``threading.Event`` (no busy-wait, no GIL contention).
    """
    def __init__(self, model, batch_size=512, max_wait_time=0.010):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.device = next(model.parameters()).device
        self.queue = queue.Queue()
        self.running = True
        self.is_batching = False
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        while self.running:
            # Block until at least one item arrives
            try:
                first = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = [first[0]]
            results = [first[1]]

            # Try to collect more items up to batch_size or max_wait_time
            start_time = time.time()
            while len(batch) < self.batch_size and (time.time() - start_time) < self.max_wait_time:
                try:
                    req = self.queue.get(timeout=0.001)
                    batch.append(req[0])
                    results.append(req[1])
                except queue.Empty:
                    continue

            self.is_batching = True
            batch_start = time.time()

            # Process with OOM retry: split batch in half on failure
            self._process_batch(batch, results)

            duration = time.time() - batch_start
            log.debug(f'[BatchEvaluator] Processed batch of {len(batch)} in {duration:.4f}s')
            self.is_batching = False

    def _process_batch(self, batch, results):
        """Process a batch, retrying with half-size splits on OOM."""
        try:
            input_tensor = torch.stack([torch.as_tensor(b, device=self.device) for b in batch])
            with torch.inference_mode():
                policy_logits, values = self.model(input_tensor)
            # WP-C: a wdl head returns (B, 3) logits; derive the scalar the
            # search consumes (p_win - p_loss). Scalar heads pass through.
            scalar_values = value_head_scalar(values)

            for i, res_container in enumerate(results):
                res_container['policy'] = policy_logits[i].detach().cpu().numpy()
                res_container['value'] = scalar_values[i].detach().item()
                res_container['event'].set()

        except RuntimeError as e:
            if 'cuda' in str(e).lower() or 'alloc' in str(e).lower():
                if len(batch) > 1:
                    mid = len(batch) // 2
                    log.warning(f'[BatchEvaluator] GPU OOM (batch={len(batch)}), splitting...')
                    torch.cuda.empty_cache()
                    self._process_batch(batch[:mid], results[:mid])
                    self._process_batch(batch[mid:], results[mid:])
                else:
                    log.warning('[BatchEvaluator] GPU OOM on single item, returning zeros')
                    res = results[0]
                    res['policy'] = np.zeros(NUM_POSSIBLE_MOVES, dtype=np.float32)
                    res['value'] = 0.0
                    res['event'].set()
            else:
                log.error(f'[BatchEvaluator] RuntimeError: {e}')
                for res_container in results:
                    res_container['policy'] = np.zeros(NUM_POSSIBLE_MOVES, dtype=np.float32)
                    res_container['value'] = 0.0
                    res_container['event'].set()
        except Exception as e:  # noqa: BLE001 - legacy defensive fallback: never kill the worker thread
            log.error(f'[BatchEvaluator] ERROR during forward pass: {e}')
            for res_container in results:
                res_container['policy'] = np.zeros(NUM_POSSIBLE_MOVES, dtype=np.float32)
                res_container['value'] = 0.0
                res_container['event'].set()

    def get_eval(self, board_tensor):
        event = threading.Event()
        res_container = {'policy': np.zeros(NUM_POSSIBLE_MOVES, dtype=np.float32), 'value': 0.0, 'event': event}
        self.queue.put((board_tensor, res_container))
        event.wait()  # blocks until worker sets the event — no GIL spinning
        return res_container['policy'], res_container['value']


class MCTS:
    def __init__(self, model, stockfish=None, cpuct=1.0, noise_epsilon=0.25, noise_alpha=0.03, batch_size=512, use_stockfish=True, evaluator=None):
        self.model = model
        self.stockfish = stockfish
        self.cpuct = cpuct
        self.noise_epsilon = noise_epsilon
        self.noise_alpha = noise_alpha
        self._eval_cache = {}  # FEN -> value cache (LRU capped)
        self._cache_lock = threading.Lock()
        self._device = next(model.parameters()).device
        self._tree_lock = threading.Lock()  # protects tree mutations during parallel sims
        # Use a shared evaluator if provided, otherwise create a private one
        self.evaluator = evaluator if evaluator is not None else BatchEvaluator(model, batch_size=batch_size)
        self.use_stockfish = use_stockfish

    def search(self, board, num_simulations=200):
        """Run MCTS search with parallel threads. Returns (visit_counts, nn_value).

        nn_value is the NN's value estimate for the current position from
        the root node, reused by the caller for resignation checks.
        """
        t0 = time.time()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return np.zeros(NUM_POSSIBLE_MOVES), 0.0

        if len(legal_moves) == 1:
            vc = np.zeros(NUM_POSSIBLE_MOVES)
            vc[move_to_idx(legal_moves[0])] = 1.0
            return vc, 0.0

        root = self._build_root(board, legal_moves)
        nn_value = root.get('nn_value', 0.0)
        t1 = time.time()
        log.info(f'  [MCTS] Root built in {t1-t0:.3f}s, children={len(root["children"])}')

        # Parallel MCTS: split simulations across NUM_MCTS_THREADS threads
        sims_per_thread = max(1, num_simulations // NUM_MCTS_THREADS)

        def _worker(sims):
            for _ in range(sims):
                self._simulate(root, board.copy(), max_depth=15)

        with ThreadPoolExecutor(max_workers=NUM_MCTS_THREADS) as executor:
            futures = []
            for i in range(NUM_MCTS_THREADS):
                n = sims_per_thread if i < num_simulations // sims_per_thread else max(0, num_simulations - len(futures) * sims_per_thread)
                if n > 0:
                    futures.append(executor.submit(_worker, n))
            for f in futures:
                f.result()

        visit_counts = np.zeros(NUM_POSSIBLE_MOVES)
        for child in root['children']:
            if child['move'] is not None:
                visit_counts[move_to_idx(child['move'])] = child['visit_count']

        t2 = time.time()
        avg_sim_time = (t2 - t1) / num_simulations
        log.info(f'  [MCTS] {num_simulations} sims in {t2-t1:.3f}s (avg={avg_sim_time:.4f}s/sim), threads={NUM_MCTS_THREADS}, visits={visit_counts.sum():.0f}')
        return visit_counts, nn_value

    def _build_root(self, board, legal_moves):
        board_tensor = board_to_tensor(board)
        legal_mask = np.zeros(NUM_POSSIBLE_MOVES, dtype=np.float32)
        for m in legal_moves:
            legal_mask[move_to_idx(m)] = 1.0
        
        policy_logits, nn_value = self.evaluator.get_eval(board_tensor)
        
        mask = legal_mask
        policy_logits = policy_logits + (1 - mask) * -1e10
        policy_probs = np.exp(policy_logits - np.max(policy_logits))
        policy_probs = policy_probs / policy_probs.sum()
        
        noise = np.random.dirichlet([self.noise_alpha] * len(legal_moves))
        legal_indices = [move_to_idx(m) for m in legal_moves]
        noisy_prior = np.zeros(NUM_POSSIBLE_MOVES)
        for i, idx in enumerate(legal_indices):
            noisy_prior[idx] = self.noise_epsilon * noise[i] + (1 - self.noise_epsilon) * policy_probs[idx]
        
        children = []
        for move in legal_moves:
            children.append({
                'move': move,
                'value': 0.0,
                'visit_count': 0,
                'prior': noisy_prior[move_to_idx(move)],
                'children': [],
                'expanded': False
            })
        
        return {
            'move': None,
            'value': 0.0,
            'visit_count': 0,
            'prior': 0.0,
            'children': children,
            'expanded': True,
            'nn_value': nn_value,
        }

    def _simulate(self, root, board, max_depth=15):
        node = root
        path = []
        depth = 0

        # --- Selection & Expansion (lock: tree mutations) ---
        with self._tree_lock:
            while node['children'] and depth < max_depth:
                selected = self._select_child(node)
                path.append((node, selected))
                board.push(selected['move'])
                if not selected['expanded']:
                    self._expand(selected, board)
                node = selected
                depth += 1

        # --- Evaluation (no lock: NN/Stockfish can run freely) ---
        value = self._evaluate(board)

        # --- Backpropagation (lock: tree mutations) ---
        with self._tree_lock:
            for parent, child in reversed(path):
                child['visit_count'] += 1
                child['value'] = (child['value'] * (child['visit_count'] - 1) + value) / child['visit_count']
                value = -value
            root['visit_count'] += 1

    def _select_child(self, node):
        best = None
        best_score = -float('inf')

        for child in node['children']:
            if child['visit_count'] == 0:
                return child

            exploit = child['value']
            explore = self.cpuct * child['prior'] * np.sqrt(node['visit_count']) / (1 + child['visit_count'])
            score = exploit + explore

            if score > best_score:
                best_score = score
                best = child

        return best

    def _expand(self, node, board):
        if board.is_game_over():
            node['expanded'] = True
            return
        
        legal = list(board.legal_moves)
        if not legal:
            node['expanded'] = True
            return
        
        board_tensor = board_to_tensor(board)
        legal_mask = np.zeros(NUM_POSSIBLE_MOVES, dtype=np.float32)
        for m in legal:
            legal_mask[move_to_idx(m)] = 1.0
        
        policy_logits, _ = self.evaluator.get_eval(board_tensor)
        
        mask = legal_mask
        policy_logits = policy_logits + (1 - mask) * -1e10
        policy_probs = np.exp(policy_logits - np.max(policy_logits))
        policy_probs = policy_probs / policy_probs.sum()
        
        for move in legal:
            node['children'].append({
                'move': move,
                'value': 0.0,
                'visit_count': 0,
                'prior': policy_probs[move_to_idx(move)],
                'children': [],
                'expanded': False
            })
        
        node['expanded'] = True

    def _evaluate(self, board):
        if board.is_game_over():
            if board.is_checkmate():
                # §10 sign-bug fix: the side to move is MATED -> -1.0 from the
                # side-to-move perspective. The legacy code returned +1.0 for
                # the mated side, poisoning mate search and training labels.
                return -1.0
            return 0.0

        nn_value = self._nn_evaluate(board)

        if self.use_stockfish and self.stockfish:
            # Use a timeout lock — if the engine is busy (e.g., during
            # sf.play_game()), fall back to NN-only to prevent deadlock.
            acquired = self.stockfish._lock.acquire(timeout=0.2)
            if not acquired:
                # Stockfish locked by another thread — use NN only
                return nn_value
            try:
                self.stockfish._engine.set_fen_position(board.fen())
                info = self.stockfish._engine.get_evaluation()
                sf_cp = 0
                if info['type'] == 'cp':
                    sf_cp = info['value']
                elif info['type'] == 'mate':
                    sf_cp = info['value'] * 10000
                if board.turn == chess.BLACK:
                    sf_cp = -sf_cp
                sf_norm = max(-1.0, min(1.0, sf_cp / 2000.0))
                noise = np.random.normal(0, SF_EVAL_NOISE_SIGMA)
                sf_noisy = max(-1.0, min(1.0, sf_norm + noise))
                return SF_LEAF_BLEND * sf_noisy + (1 - SF_LEAF_BLEND) * nn_value
            except Exception:  # noqa: BLE001 - legacy fallback: dead engine must not crash search
                # Engine dead or unresponsive — fall back to NN
                return nn_value
            finally:
                self.stockfish._lock.release()

        return nn_value

    def _nn_evaluate(self, board):
        fen = board.board_fen()
        with self._cache_lock:
            if fen in self._eval_cache:
                return self._eval_cache[fen]
        board_tensor = board_to_tensor(board)
        _, value = self.evaluator.get_eval(board_tensor)

        # LRU cap: evict oldest entry when cache is full
        with self._cache_lock:
            if len(self._eval_cache) >= MCTS_EVAL_CACHE_MAX:
                self._eval_cache.pop(next(iter(self._eval_cache)))
            self._eval_cache[fen] = value
        return value


class SelfPlayGame:
    def __init__(self, model, num_mcts_simulations=400, max_moves=200, stockfish=None, batch_size=2048, use_stockfish=True, evaluator=None, resign_enabled=False):
        self.model = model
        self.num_mcts_simulations = num_mcts_simulations
        self.max_moves = max_moves
        self.mcts = MCTS(model, stockfish=stockfish, batch_size=batch_size, use_stockfish=use_stockfish, evaluator=evaluator)
        self.use_stockfish = use_stockfish
        # §10 resign fix: a single sub-threshold eval must never end a game.
        # Resignation is DEFAULT OFF for self-play and, when enabled, requires
        # RESIGN_CONSECUTIVE consecutive evals below RESIGN_THRESHOLD past ply
        # RESIGN_MIN_PLY.
        self.resign_enabled = resign_enabled
        self.resigned_side = None
        self._resign_streak = 0

    def play(self, on_move=None):
        """Play a complete self-play game.

        Args:
            on_move: callback(moves_list, san_move) called after each move

        Returns dict with:
            - 'examples': list of training examples
            - 'moves': list of SAN move strings
            - 'pgn': full game PGN
            - 'result': game result string (1-0, 0-1, 1/2-1/2)
        """
        board = chess.Board()
        examples = []
        move_sans = []
        self.resigned_side = None
        self._resign_streak = 0
        log.info(f'[GAME] Starting self-play, sims={self.num_mcts_simulations}')
        try:
            for i in range(self.max_moves):
                log.info(f'[GAME] Move {i+1}, turn={"W" if board.turn else "B"}')

                # Check for game-over conditions (checkmate, stalemate, draw, insufficient material)
                if board.is_game_over():
                    log.info(f'[GAME] Game over at move {i}: {board.result()}')
                    break

                t_move = time.time()
                visit_counts, nn_value = self.mcts.search(board, self.num_mcts_simulations)
                log.info(f'[GAME] Search done in {time.time()-t_move:.1f}s, visits={visit_counts.sum():.0f}')

                total_visits = visit_counts.sum()
                policy = visit_counts / total_visits if total_visits > 0 else visit_counts

                # Resignation (§10 fix): requires resign_enabled AND
                # RESIGN_CONSECUTIVE consecutive sub-threshold evals past ply
                # RESIGN_MIN_PLY. A single eval can no longer end a game.
                if self.resign_enabled and (i + 1) >= RESIGN_MIN_PLY and nn_value < RESIGN_THRESHOLD:
                    self._resign_streak += 1
                    if self._resign_streak >= RESIGN_CONSECUTIVE:
                        self.resigned_side = 'white' if board.turn == chess.WHITE else 'black'
                        log.info(
                            f'[GAME] Resigning at move {i + 1} '
                            f'({self._resign_streak} consecutive evals < {RESIGN_THRESHOLD})'
                        )
                        break
                else:
                    self._resign_streak = 0

                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break

                probs = visit_counts[[move_to_idx(m) for m in legal_moves]]
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(legal_moves)) / len(legal_moves)

                move_idx = np.random.choice(len(legal_moves), p=probs)
                move = legal_moves[move_idx]

                san_move = board.san(move)
                examples.append({
                    'board_tensor': board_to_tensor(board),
                    'policy': policy,
                    'turn': board.turn,
                    'san': san_move,
                })
                log.info(f'[GAME] Played {san_move}')
                board.push(move)
                move_sans.append(san_move)
                if on_move:
                    on_move(list(move_sans))

        except Exception as e:  # noqa: BLE001 - legacy: a game error must not kill the trainer thread
            log.error(f'[GAME] ERROR: {e}')
            import traceback
            traceback.print_exc()

        outcome = self._get_outcome(board)
        if self.resigned_side is not None:
            # A resignation is a win for the opponent, not a draw.
            outcome = -1.0 if self.resigned_side == 'white' else 1.0

        training_data = []
        for ex in examples:
            if ex['turn'] == chess.WHITE:
                value = outcome
            else:
                value = -outcome

            training_data.append({
                'board_tensor': ex['board_tensor'],
                'policy': ex['policy'],
                'value': value,
                'san': ex.get('san', ''),
            })

        # Build PGN
        game = chess.pgn.Game()
        b = chess.Board()
        current = game
        for san in move_sans:
            try:
                move = b.parse_san(san)
                current = current.add_variation(move)
                b.push(move)
            except Exception:  # noqa: BLE001 - legacy: tolerate PGN reparse drift
                break
        pgn = game.accept(chess.pgn.StringExporter())

        if board.is_checkmate():
            result = '1-0' if board.turn == chess.BLACK else '0-1'
        elif self.resigned_side == 'white':
            result = '0-1'
        elif self.resigned_side == 'black':
            result = '1-0'
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.can_claim_draw():
            result = '1/2-1/2'
        else:
            # Game ended at max_moves without resolution — still a draw
            result = '1/2-1/2'

        return {
            'examples': training_data,
            'moves': move_sans,
            'pgn': pgn,
            'result': result,
        }

    def _get_outcome(self, board):
        if board.is_checkmate():
            return 1.0 if board.turn == chess.BLACK else -1.0
        if board.is_insufficient_material() or board.is_fivefold_repetition() or board.can_claim_draw():
            return 0.0
        return 0.0


class FastBoard(chess.Board):
    """python-chess Board whose repetition CLAIMS avoid push/pop replays (M3).

    Profiling the self-play hot loop (tmp_m3 profiles, 2026-11-21) showed
    ``is_game_over(claim_draw=True)`` is the single largest per-simulation
    sink: ``can_claim_threefold_repetition()`` pops the entire move stack,
    re-keys every position, pushes everything back, then probes every legal
    move with push/pop - O(ply + moves) python-chess mutations per call, twice
    per simulation (``MCTSSearch._simulate`` + ``_evaluate_and_expand``), and
    the cost grows with game length.

    This subclass overrides exactly three methods to walk the existing
    ``_stack`` snapshots instead (they carry every transposition-key
    component), and inherits everything else unchanged - movegen, SAN, PGN,
    FEN, ``outcome()`` ordering, ``copy()``. The overrides are gated behind
    GAMBIT_CORE_FASTPATH (see training/fastpath.py) and differential-tested
    against the plain Board in tests/test_gambit_m3_fastpath.py.

    Requires python-chess's private ``_BoardState``/``_stack`` layout
    (python-chess 1.11.2, pinned via uv.lock).
    """

    @staticmethod
    def _state_clean_castling_rights(state: "chess._BoardState") -> chess.Bitboard:  # type: ignore[name-defined]
        """python-chess ``clean_castling_rights()`` for an EMPTY-stack board.

        Used only for the game-root snapshot (index 0): python-chess evaluates
        the root position with an empty move stack, where
        ``clean_castling_rights()`` runs the full rook/king filter instead of
        trusting ``castling_rights``. Logic copied from python-chess 1.11.2
        (standard chess, chess960=False).
        """
        castling = state.castling_rights & state.rooks
        white = castling & chess.BB_RANK_1 & state.occupied_w
        black = castling & chess.BB_RANK_8 & state.occupied_b
        white &= chess.BB_A1 | chess.BB_H1
        black &= chess.BB_A8 | chess.BB_H8
        if not state.occupied_w & state.kings & ~state.promoted & chess.BB_E1:
            white = 0
        if not state.occupied_b & state.kings & ~state.promoted & chess.BB_E8:
            black = 0
        return white | black

    def _state_transposition_key(self, state: "chess._BoardState", *, root: bool = False) -> tuple:  # type: ignore[name-defined]
        """Transposition key of a HISTORICAL stack snapshot (no push/pop).

        Mirrors ``Board._transposition_key``: once a board has a move stack,
        python-chess's ``clean_castling_rights()`` returns the raw rights
        (they were filtered at set-up time), so ``state.castling_rights`` is
        the key component - EXCEPT for the game-root snapshot, which
        python-chess evaluates on an empty-stack board (full cleaning, see
        ``_state_clean_castling_rights``). The en-passant component keeps the
        legal-ep filter, computed by momentarily restoring the snapshot (ep
        squares are rare - at most one position in the window has one).
        """
        if state.ep_square is None:
            ep_component = None
        else:
            live = chess._BoardState(self)
            state.restore(self)
            try:
                ep_component = state.ep_square if self.has_legal_en_passant() else None
            finally:
                live.restore(self)
        rights = self._state_clean_castling_rights(state) if root else state.castling_rights
        return (
            state.pawns,
            state.knights,
            state.bishops,
            state.rooks,
            state.queens,
            state.kings,
            state.occupied_w,
            state.occupied_b,
            state.turn,
            rights,
            ep_component,
        )

    def _state_is_irreversible(self, state: "chess._BoardState", move: chess.Move, *, root: bool = False) -> bool:  # type: ignore[name-defined]
        """``Board.is_irreversible(move)`` evaluated on a historical snapshot.

        ``is_zeroing`` and ``_reduces_castling_rights`` only read bitboards
        the snapshot already carries; ``root=True`` (game-root snapshot)
        applies the empty-stack rights cleaning like python-chess does. The ep
        cession term uses the same restore-trick as the key.
        """
        touched = chess.BB_SQUARES[move.from_square] ^ chess.BB_SQUARES[move.to_square]
        their_occupied = state.occupied_w if state.turn == chess.BLACK else state.occupied_b
        if touched & state.pawns or touched & their_occupied:
            return True  # is_zeroing: pawn move or capture
        cr = self._state_clean_castling_rights(state) if root else state.castling_rights
        if touched & cr:
            return True
        if cr & chess.BB_RANK_1 and touched & state.kings & state.occupied_w & ~state.promoted:
            return True
        if cr & chess.BB_RANK_8 and touched & state.kings & state.occupied_b & ~state.promoted:
            return True
        if state.ep_square is not None:
            live = chess._BoardState(self)
            state.restore(self)
            try:
                if self.has_legal_en_passant():
                    return True
            finally:
                live.restore(self)
        return False

    def _replay_window_keys(self) -> list[tuple]:
        """Transposition keys of the current reversible window, newest first.

        Equivalent to ``can_claim_threefold_repetition``'s switchyard replay
        (pop until an irreversible move), evaluated on the ``_stack``
        snapshots without a single push/pop. Only valid on stacked boards -
        callers fall back to the inherited implementation otherwise.
        """
        keys: list[tuple] = []
        states = self._stack
        last = len(states) - 1
        for index, (state, move) in enumerate(zip(reversed(states), reversed(self.move_stack))):
            # Reversed walk: index len-1 is the game ROOT (python-chess
            # evaluates it on an empty-stack board).
            is_root = index == last
            if self._state_is_irreversible(state, move, root=is_root):
                break
            keys.append(self._state_transposition_key(state, root=is_root))
        return keys

    def can_claim_threefold_repetition(self) -> bool:
        """Bit-exact, replay-free port of ``Board.can_claim_threefold_repetition``."""
        if self.chess960 or not self._stack:
            return super().can_claim_threefold_repetition()
        current_key = self._transposition_key()
        counts: dict[tuple, int] = {current_key: 1}
        for key in self._replay_window_keys():
            counts[key] = counts.get(key, 0) + 1
        if counts[current_key] >= 3:
            return True
        # The next legal move is a threefold repetition (same probe, push/pop).
        for move in self.generate_legal_moves():
            self.push(move)
            try:
                if counts.get(self._transposition_key(), 0) >= 2:
                    return True
            finally:
                self.pop()
        return False

    def is_repetition(self, count: int = 3) -> bool:
        """Bit-exact, replay-free port of ``Board.is_repetition`` (stacked boards)."""
        if self.chess960 or not self._stack or count <= 1:
            return super().is_repetition(count)
        # Cheap occupancy pre-filter, then the key-counting window walk.
        maybe = 1
        occupied = self.occupied
        for state in reversed(self._stack):
            if state.occupied == occupied:
                maybe += 1
                if maybe >= count:
                    break
        if maybe < count:
            return False
        current_key = self._transposition_key()
        seen = 1
        states = self._stack
        last = len(states) - 1
        for index, (state, move) in enumerate(zip(reversed(states), reversed(self.move_stack))):
            is_root = index == last
            if self._state_is_irreversible(state, move, root=is_root):
                break
            if self._state_transposition_key(state, root=is_root) == current_key:
                seen += 1
                if seen >= count:
                    return True
        return False


# ---------------------------------------------------------------------------
# Search v2 (WP-C): correct PUCT MCTS over the 73x64 policy encoding.
#
# Value convention throughout: every value is from the perspective of the side
# to move at that node. Terminal checkmate is -1.0 for the mated side-to-move
# (§10 sign bug — regression-tested), and child exploitation in PUCT is
# -child.value_mean (the child's mover is the parent mover's opponent).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchConfig:
    """PUCT / noise / sampling knobs for one search."""

    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    root_noise: bool = True
    num_simulations: int = 64
    # U1: root-search policy. "puct" is the canonical AlphaZero root (argmax
    # visit counts). "gumbel" is Danihelka et al., ICLR 2022 (Alg. 2): gumbel
    # top-m candidates + sequential halving at the root, PUCT below the root,
    # Eq. 11 improved policy as the training target. Default = flag-off.
    search_kind: Literal["puct", "gumbel"] = "puct"
    gumbel_candidates: int = 16   # m in Alg. 2 (paper: 16)
    gumbel_c_visit: float = 50.0  # c_visit in Eq. 8 (paper chess setting)
    gumbel_c_scale: float = 1.0   # c_scale in Eq. 8 (paper chess setting)
    # WP-F C5: which engine runs the tree search. "python" = MCTSSearch
    # (default - every pre-C5 construction is unchanged); "rust" = the
    # gambit_core PUCT search via training.rust_search.RustMCTSSearch. The
    # default stays "python" until the C6 gates justify flipping it.
    search_backend: Literal["python", "rust"] = "python"
    # WP1 (KataGo): forced playouts + policy target pruning at the root.
    # Off = canonical AlphaZero (previous behavior bit-identical).
    forced_playouts: bool = False
    fp_k: float = 2.0

    def __post_init__(self) -> None:
        if self.search_backend not in ("python", "rust"):
            raise ValueError(
                f"search_backend must be 'python' or 'rust', got {self.search_backend!r}"
            )
        if self.search_backend == "rust" and self.search_kind != "puct":
            raise ValueError(
                "search_backend='rust' implements the PUCT root policy only; "
                f"got search_kind={self.search_kind!r} (gumbel stays python-only)"
            )


@dataclass(frozen=True)
class MCTSResult:
    """Outcome of one root search. Visit counts are 4672-length (encoder idx)."""

    visit_counts: np.ndarray
    root_value: float
    chosen_move: chess.Move | None
    visited_root: bool = True
    # U1: gumbel searches also return the Eq. 11 improved policy over the
    # 4672 encoding (zeros off-policy, sums to 1). None for puct searches so
    # the flag-off self-play path is bit-identical to the pre-U1 pipeline.
    improved_policy: np.ndarray | None = None


class Evaluator(Protocol):
    """Batched position evaluator.

    Implementations return, for each board: raw policy logits over the 4672
    encoder indices (unmasked) and a scalar value from the side-to-move's
    perspective.
    """

    def evaluate(self, boards: Sequence[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        ...  # pragma: no cover


class EventSink(Protocol):
    """Structural slice of ``TrainingJournal`` the game generators write to.

    Declared here (not imported) so selfplay stays decoupled from the journal
    module and tests can inject fakes.
    """

    def append(self, event: Mapping[str, Any]) -> None:
        """Append one structured event (thread-safe)."""
        ...  # pragma: no cover


@dataclass(frozen=True)
class GameStream:
    """Per-game live-stream writer (FS-6b).

    One ``live`` event per ply (move + resulting FEN) and one ``game_end``
    when the game finishes, so the server can push them over WS and the
    client can render/clear a board slot. Appends go through the journal's
    lock, which serializes concurrent game threads into whole lines.
    """

    journal: EventSink
    run_id: str
    regime: str
    game: int

    def move(self, ply: int, move_uci: str, fen: str) -> None:
        """Stream one played move (ply is 1-based, fen is AFTER the move)."""
        self.journal.append(
            {
                "kind": "live",
                "run_id": self.run_id,
                "regime": self.regime,
                "game": self.game,
                "ply": ply,
                "move": move_uci,
                "fen": fen,
                "ts": time.time(),
            }
        )

    def end(self, result: str, plies: int) -> None:
        """Stream game completion so the UI can clear the slot."""
        self.journal.append(
            {
                "kind": "game_end",
                "run_id": self.run_id,
                "regime": self.regime,
                "game": self.game,
                "result": result,
                "plies": plies,
                "ts": time.time(),
            }
        )


@dataclass(frozen=True)
class StreamContext:
    """Run-level live-streaming config handed down through providers (FS-6b).

    ``game_offset`` lets a provider run several game batches in one iteration
    (e.g. sf-games: self-play games 0..k-1, then SF games k..k+m-1) without
    colliding board slots in the client grid.
    """

    journal: EventSink | None = None
    run_id: str = ""
    regime: str = ""
    enabled: bool = True
    game_offset: int = 0

    def for_game(self, game_index: int) -> GameStream | None:
        """Per-game writer; None when streaming is off (zero events)."""
        if self.journal is None or not self.enabled:
            return None
        return GameStream(self.journal, self.run_id, self.regime, self.game_offset + game_index)

    def shifted(self, count: int) -> "StreamContext":
        """A context whose game indices continue after `count` earlier games."""
        return replace(self, game_offset=self.game_offset + count)


class BatchNNEvaluator:
    """Synchronous micro-batched NN evaluator (CPU or CUDA).

    Slice 2 replaces this with the queued NNEvaluator service (§5 lever 2);
    the Evaluator interface stays the same.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device | None = None) -> None:
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        # WP-D: the encoder layout follows the model's input planes — a
        # 104-plane (encoder v2) model is fed `encode_batch_v2`, everything
        # else keeps the v1 `encode_batch`.
        self._encoder_version = encoder_version_for_model(model)

    def evaluate(self, boards: Sequence[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        if not boards:
            return np.zeros((0, NUM_POLICY_MOVES_V2), dtype=np.float32), np.zeros(0, dtype=np.float32)
        batch = self._encode(boards)
        with torch.inference_mode():
            logits, values = self.model(torch.from_numpy(batch).to(self.device))
        # WP-C: wdl heads return (B, 3) logits — collapse to the scalar the
        # search consumes (p_win - p_loss, side-to-move perspective). The
        # Evaluator protocol (and thus MCTSSearch) is unchanged.
        scalar_values = value_head_scalar(values)
        return (
            logits.detach().cpu().numpy().astype(np.float32),
            scalar_values.detach().cpu().numpy().astype(np.float32),
        )

    def _encode(self, boards: Sequence[chess.Board]) -> np.ndarray:
        """WP-D: encoder layout follows the model's input plane count."""
        if self._encoder_version == 2:
            return encode_batch_v2(boards)
        return encode_batch(boards)


def build_search(
    evaluator: Evaluator,
    config: SearchConfig,
    rng: np.random.Generator | None = None,
) -> "MCTSSearch | RustMCTSSearch":
    """WP-F C5: construct the search backend a SearchConfig selects.

    'python' returns MCTSSearch unchanged (every pre-C5 call site is
    bit-identical); 'rust' returns the gambit_core-backed
    RustMCTSSearch, imported lazily so python-only environments never need
    the wheel. Gumbel configs reject the rust backend at SearchConfig level.
    """
    if config.search_backend == "rust":
        from .rust_search import RustMCTSSearch  # deferred: the wheel is optional

        return RustMCTSSearch(evaluator, config, rng=rng)
    return MCTSSearch(evaluator, config, rng=rng)


def terminal_value(board: chess.Board) -> float:
    """Value of a TERMINAL position from the side-to-move's perspective.

    §10 sign-bug fix: checkmate means the side to move is mated -> -1.0. The
    legacy code returned +1.0 for the mated side, poisoning mate search and
    training labels. Every non-checkmate terminal is a draw -> 0.0.
    """
    if board.is_checkmate():
        return -1.0
    return 0.0


@dataclass
class _Node:
    move: chess.Move | None
    prior: float
    children: list["_Node"] = field(default_factory=list)
    visit_count: int = 0
    value_sum: float = 0.0
    terminal: bool = False
    terminal_value: float = 0.0
    expanded: bool = False

    @property
    def value_mean(self) -> float:
        """Mean value from THIS node's side-to-move perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTSSearch:
    """PUCT MCTS with Dirichlet root noise and correct terminal signs.

    Sequential tonight (WP-C slice 1); the tree/evaluator split keeps the door
    open for batched parallel sims in slice 2. A new instance per search root
    is not required — the tree is rebuilt per `search` call.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        config: SearchConfig = SearchConfig(),
        rng: np.random.Generator | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config
        # U1: only gumbel draws consume this generator; the PUCT path and the
        # global-rng Dirichlet draw are untouched so seeded runs reproduce.
        self._rng = rng

    def search(self, board: chess.Board, num_simulations: int | None = None) -> MCTSResult:
        """Search `board`; returns visit counts, root value, and chosen move."""
        sims = num_simulations if num_simulations is not None else self.config.num_simulations
        legal = list(board.legal_moves)
        if not legal:
            return MCTSResult(np.zeros(NUM_POLICY_MOVES_V2, dtype=np.float32), terminal_value(board), None, False)

        if self.config.search_kind == "gumbel":
            return self._search_gumbel(board, sims)

        root = _Node(move=None, prior=0.0)
        self._evaluate_and_expand(root, board, add_root_noise=self.config.root_noise)
        if root.terminal:
            return MCTSResult(np.zeros(NUM_POLICY_MOVES_V2, dtype=np.float32), root.terminal_value, None, False)

        if len(legal) == 1:
            # Forced move: no search needed; value is the root eval.
            visits = np.zeros(NUM_POLICY_MOVES_V2, dtype=np.float32)
            visits[encode_move(board, legal[0])] = float(sims)
            return MCTSResult(visits, root.value_mean, legal[0], True)

        for _ in range(sims):
            self._simulate(root, board.copy())

        visit_counts = np.zeros(NUM_POLICY_MOVES_V2, dtype=np.float32)
        for child in root.children:
            visit_counts[encode_move(board, child.move)] = float(child.visit_count)

        # WP1 (KataGo): policy target pruning — subtract the forced-playout
        # share from non-best root children (never below 1; drop at <=1).
        if self.config.forced_playouts and root.children:
            best_child = max(root.children, key=lambda c: c.visit_count)
            total = float(max(root.visit_count, 1))
            for child in root.children:
                if child is best_child:
                    continue
                quota = math.sqrt(
                    self.config.fp_k * min(1.0, max(0.0, child.prior)) * total
                )
                excess = min(quota, float(child.visit_count))
                kept = child.visit_count - excess
                visit_counts[encode_move(board, child.move)] = 0.0 if kept <= 1.0 else kept

        best = max(root.children, key=lambda c: c.visit_count)
        return MCTSResult(visit_counts, root.value_mean, best.move, True)

    # -- internals ---------------------------------------------------------

    def _simulate(
        self, root: _Node, board: chess.Board, force_root_child: _Node | None = None
    ) -> None:
        node = root
        path: list[_Node] = [root]
        while node.expanded and node.children and not node.terminal:
            if force_root_child is not None and node is root:
                node = force_root_child  # U1 gumbel: descend a halving candidate
            elif node is root and self.config.forced_playouts:
                node = self._select_child_forced(node)
            else:
                node = self._select_child(node)
            board.push(node.move)
            path.append(node)

        # Leaf evaluation (value from the leaf's side-to-move perspective).
        if node.terminal:
            value = node.terminal_value
        elif board.is_game_over(claim_draw=True):
            node.terminal = True
            node.terminal_value = terminal_value(board)
            node.expanded = True
            value = node.terminal_value
        else:
            value, _ = self._evaluate_and_expand(node, board, add_root_noise=False)

        # Backpropagate with sign flip at every ply (negamax convention).
        value_back = value
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value_back
            value_back = -value_back

    def _select_child_forced(self, node: _Node) -> _Node:
        """Root selection with KataGo forced playouts (WP1).

        Any root child below its quota `sqrt(k * P * sum(N))` is forced so a
        noise move gets a minimum share of visits; among forced children the
        largest deficit wins. Plain PUCT everywhere else (and below the root).
        """
        total = float(max(node.visit_count, 1))
        forced: _Node | None = None
        forced_deficit = -1.0
        best: _Node | None = None
        best_score = -float("inf")
        for child in node.children:
            visits = float(child.visit_count)
            quota = math.sqrt(self.config.fp_k * min(1.0, max(0.0, child.prior)) * total)
            if visits < quota:
                deficit = quota - visits
                if deficit > forced_deficit:
                    forced, forced_deficit = child, deficit
                continue
            exploit = -child.value_mean
            explore = self.config.cpuct * child.prior * float(np.sqrt(total)) / (1.0 + visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        if forced is not None:
            return forced
        assert best is not None
        return best

    def _select_child(self, node: _Node) -> _Node:
        """PUCT selection: Q(child, parent persp) + cpuct * P * sqrt(N)/(1+n)."""
        sqrt_parent = float(np.sqrt(max(node.visit_count, 1)))
        best: _Node | None = None
        best_score = -float("inf")
        for child in node.children:
            exploit = -child.value_mean  # child value is from the opponent's perspective
            explore = self.config.cpuct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        assert best is not None, "select_child called on a childless node"
        return best

    def _evaluate_and_expand(self, node: _Node, board: chess.Board, add_root_noise: bool) -> tuple[float, bool]:
        """Expand `node` with NN priors. Returns (value, is_terminal)."""
        if board.is_game_over(claim_draw=True):
            node.terminal = True
            node.terminal_value = terminal_value(board)
            node.expanded = True
            return node.terminal_value, True

        logits, values = self.evaluator.evaluate([board])
        value = float(values[0])
        legal_map = legal_move_map(board)
        legal_indices = list(legal_map.keys())
        raw = logits[0][legal_indices].astype(np.float64)
        raw -= raw.max()
        probs = np.exp(raw)
        probs /= probs.sum()

        priors = probs
        if add_root_noise and len(legal_indices) > 1:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal_indices))
            priors = (
                (1.0 - self.config.dirichlet_epsilon) * probs
                + self.config.dirichlet_epsilon * noise
            )

        for index, prior in zip(legal_indices, priors):
            node.children.append(_Node(move=legal_map[index], prior=float(prior)))
        node.expanded = True
        return value, False

    # -- U1: Gumbel AlphaZero root search -----------------------------------

    def _search_gumbel(self, board: chess.Board, sims: int) -> MCTSResult:
        """Gumbel AlphaZero root search (Danihelka et al., ICLR 2022, Alg. 2).

        Root: sample m = min(gumbel_candidates, n_legal) candidates with the
        Gumbel-Top-k trick on (g + prior_logits), distribute `sims` over them
        with sequential halving (round-robin per phase, halve by the Eq. 8
        score), then act with A_{n+1} = argmax over the surviving candidates
        of g(a) + logits(a) + Psi(completedQ(a)). Below the root the tree is
        ordinary PUCT (the paper's Gumbel AlphaZero variant; the Full
        variant's Eq. 14 interior selection is a documented non-goal). NN
        accounting matches the PUCT path: 1 root eval + `sims` leaf evals.
        """
        cfg = self.config
        root = _Node(move=None, prior=0.0)
        root_value, _ = self._evaluate_and_expand(root, board, add_root_noise=cfg.root_noise)
        if root.terminal:
            return MCTSResult(
                np.zeros(NUM_POLICY_MOVES_V2, dtype=np.float32), root.terminal_value, None, False
            )

        legal_children: list[_Node] = list(root.children)
        if len(legal_children) == 1:
            # Forced move: no search needed; mirror the PUCT early return.
            visit_counts = np.zeros(NUM_POLICY_MOVES_V2, dtype=np.float32)
            visit_counts[encode_move(board, legal_children[0].move)] = float(sims)
            one_hot = np.zeros(NUM_POLICY_MOVES_V2, dtype=np.float32)
            one_hot[encode_move(board, legal_children[0].move)] = 1.0
            return MCTSResult(visit_counts, root.value_mean, legal_children[0].move, True, one_hot)

        rng = self._rng if self._rng is not None else np.random.default_rng()
        # Prior logits (post-noise when root_noise is on - the mctx pattern).
        priors = np.array([max(child.prior, 1e-9) for child in legal_children], dtype=np.float64)
        logits = np.log(priors)
        logits -= logits.max()
        gumbel = rng.gumbel(size=len(legal_children))

        m = min(cfg.gumbel_candidates, len(legal_children))
        top = np.argsort(gumbel + logits)[::-1][:m]
        cand: list[_Node] = [legal_children[i] for i in top]
        cand_g = gumbel[top]

        visits = np.zeros(m, dtype=np.int64)
        budget = sims
        survivors = list(range(m))  # candidate order == (g + logits) descending
        while budget > 0 and len(survivors) > 1:
            if len(survivors) == 2:
                per_action = max(1, budget // 2)
            else:
                phases_left = max(1, math.ceil(math.log2(len(survivors))))
                per_action = max(1, budget // (phases_left * len(survivors)))
            exhausted = False
            for _round in range(per_action):
                for idx in survivors:
                    if budget <= 0:
                        exhausted = True
                        break
                    self._simulate(root, board.copy(), force_root_child=cand[idx])
                    budget -= 1
                    visits[idx] += 1
                if exhausted:
                    break
            if exhausted or len(survivors) == 2:
                break
            scores = self._gumbel_scores(cand_g, cand, visits, root_value)
            keep = max(2, len(survivors) // 2)
            ranked = sorted(survivors, key=lambda i: scores[i], reverse=True)
            survivors = sorted(ranked[:keep])

        final_scores = self._gumbel_scores(cand_g, cand, visits, root_value)
        best_idx = max(survivors, key=lambda i: (float(final_scores[i]), int(visits[i])))
        best_move = cand[best_idx].move
        assert best_move is not None, "gumbel halving lost its candidates"

        visit_counts = np.zeros(NUM_POLICY_MOVES_V2, dtype=np.float32)
        for child in legal_children:
            visit_counts[encode_move(board, child.move)] = float(child.visit_count)

        improved = self._gumbel_improved_policy(
            board, legal_children, root_value, int(visits.max())
        )
        return MCTSResult(visit_counts, root.value_mean, best_move, True, improved)

    def _gumbel_scores(
        self,
        cand_g: np.ndarray,
        cand: list[_Node],
        visits: np.ndarray,
        root_value: float,
    ) -> np.ndarray:
        """Eq. 8: g(a) + logits(a) + Psi(completedQ(a)) per candidate.

        Psi(q) = (c_visit + max_b N(b)) * c_scale * q01 with q01 = (q + 1) / 2
        mapping the [-1, 1] value range onto [0, 1] (mctx convention; paper
        chess values c_visit=50, c_scale=1.0).
        """
        logits = np.log(np.array([max(c.prior, 1e-9) for c in cand], dtype=np.float64))
        completed = self._completed_q(cand, root_value)
        q01 = (completed + 1.0) / 2.0
        max_n = int(visits.max()) if visits.size else 0
        scale = (self.config.gumbel_c_visit + max_n) * self.config.gumbel_c_scale
        return cand_g + logits + scale * q01

    def _completed_q(self, children: list[_Node], root_value: float) -> np.ndarray:
        """Paper Eq. 10 completed Q from the ROOT's perspective.

        Visited actions use their empirical mean (child values are stored from
        the child's own side to move, hence the negation); unvisited actions
        are imputed with the mixed value of paper Appendix D (mctx
        qtransform_completed_by_mix_value): (raw_value + sum_N * weighted_q)
        / (sum_N + 1) with weighted_q the prior-weighted mean over visited
        actions.
        """
        q = np.zeros(len(children), dtype=np.float64)
        unvisited: list[int] = []
        sum_n = 0
        weighted = 0.0
        sum_probs = 0.0
        for i, child in enumerate(children):
            if child.visit_count > 0:
                q[i] = -child.value_mean
                prior = max(child.prior, 1e-9)
                sum_n += child.visit_count
                weighted += prior * q[i]
                sum_probs += prior
            else:
                unvisited.append(i)
        weighted_q = weighted / sum_probs if sum_probs > 0 else 0.0
        mixed = (root_value + sum_n * weighted_q) / (sum_n + 1)
        for i in unvisited:
            q[i] = mixed
        return q

    def _gumbel_improved_policy(
        self,
        board: chess.Board,
        legal_children: list[_Node],
        root_value: float,
        max_n: int,
    ) -> np.ndarray:
        """Eq. 11 improved policy: softmax(logits + Psi(completedQ)).

        Over ALL legal moves (unvisited ones get zero advantage via the mixed
        value), scattered into the 4672 encoding. Note: no gumbel values in
        the target - the paper and mctx both use the prior logits (the task
        brief's 'gumbel logits' wording is a documented deviation).
        """
        logits = np.log(np.array([max(c.prior, 1e-9) for c in legal_children], dtype=np.float64))
        completed = self._completed_q(legal_children, root_value)
        q01 = (completed + 1.0) / 2.0
        scale = (self.config.gumbel_c_visit + max_n) * self.config.gumbel_c_scale
        scores = logits + scale * q01
        scores -= scores.max()
        weights = np.exp(scores)
        probs = weights / weights.sum()
        policy = np.zeros(NUM_POLICY_MOVES_V2, dtype=np.float32)
        for child, prob in zip(legal_children, probs):
            policy[encode_move(board, child.move)] = float(prob)
        return policy


def select_move_from_visits(
    visit_counts: np.ndarray,
    legal_map: dict[int, chess.Move],
    temperature: float,
    rng: np.random.Generator | None = None,
) -> chess.Move:
    """Sample a move from the visit distribution at temperature `tau`.

    tau -> 0 degenerates to argmax; tau = 1 samples proportionally to visits.
    """
    if temperature <= 1e-6:
        best_index = max(legal_map, key=lambda idx: visit_counts[idx])
        return legal_map[best_index]
    indices = list(legal_map.keys())
    counts = np.array([visit_counts[idx] for idx in indices], dtype=np.float64)
    powered = counts ** (1.0 / temperature)
    total = powered.sum()
    if total <= 0:
        return legal_map[indices[0]]
    probs = powered / total
    generator = rng if rng is not None else np.random.default_rng()
    chosen = int(generator.choice(len(probs), p=probs))
    return legal_map[indices[chosen]]


@dataclass(frozen=True)
class SelfPlayConfig:
    """Driver knobs for one self-play game."""

    search: SearchConfig = SearchConfig()
    max_plies: int = 200
    temperature: float = 1.0
    temperature_plies: int = 30  # tau > 0 for the first N plies, argmax after
    # §10 resign fix: 3 consecutive evals below -0.9 past ply 40, DEFAULT OFF.
    resign_enabled: bool = False
    resign_threshold: float = -0.9
    resign_consecutive: int = 3
    resign_min_ply: int = 40
    # U2: KataGo playout cap randomization (Wu 2020, sec 3.1). Default OFF -
    # with the flag off, self_play_game consumes its rng exactly as before and
    # games are bit-identical to the pre-U2 pipeline (golden-tested in
    # tests/test_training_playout_cap.py).
    #
    # U1xU2 interaction (documented restriction, NOT an accident): the cap's
    # semantics - which turns are searched fully, which are capped, which are
    # recorded - presume PUCT VISIT-COUNT targets (KataGo sec 3.1). With
    # search_kind="gumbel" a fast ply would record the Eq. 11 improved policy
    # of a reduced-budget search: a mathematically well-defined but completely
    # unvalidated target construction (bias/variance unmeasured, no oracle).
    # Enabling both flags therefore raises ValueError - U2 is restricted to
    # the PUCT path until the gumbel-target analogue is designed and gated on
    # its own (one change at a time).
    playout_randomization: bool = False
    playout_cap_full_prob: float = 0.25      # p: share of turns searched fully (recorded)
    playout_cap_fast_fraction: float = 0.25  # fast cap n = fraction * num_simulations
    playout_cap_fast_min: int = 2            # ...but never below this many sims
    playout_cap_start_ply: int = 10          # plies before this always search fully
    # Paper: ONLY full-search turns are recorded for training. Flip this on to
    # also record fast turns (their own visit distribution as the target).
    # PUCT-only: see the U1xU2 restriction above.
    playout_cap_record_fast: bool = False
    # WP-D: encoder layout for the recorded example planes — 1 (v1, 29
    # planes, `encode_state` + prev_board) or 2 (v2, 104 planes,
    # `encode_state_v2` with move-stack history). Default 1 keeps every
    # pre-WP-D profile bit-identical.
    encoder_version: int = 1

    def __post_init__(self) -> None:
        if self.encoder_version not in (1, 2):
            raise ValueError(f"encoder_version must be 1 or 2, got {self.encoder_version}")
        if self.playout_randomization and self.search.search_kind != "puct":
            raise ValueError(
                "playout cap randomization applies to the PUCT search path only "
                f"(got search_kind={self.search.search_kind!r}): KataGo cap semantics "
                "presume PUCT visit-count targets, and the gumbel improved-policy "
                "analogue is unvalidated"
            )


@dataclass(frozen=True)
class TrainingExample:
    """One (planes, policy, value) training tuple; value from mover's persp."""

    planes: np.ndarray  # (29, 8, 8)
    policy: np.ndarray  # (4672,) visit distribution
    value: float
    # WP2: 1.0 = contribute to the policy loss; 0.0 = value-only row (PCR fast
    # search). Additive default keeps every pre-WP2 construction unchanged.
    policy_mask: float = 1.0


@dataclass(frozen=True)
class SelfPlayResult:
    examples: list[TrainingExample]
    moves: list[str]
    pgn: str
    result: str  # '1-0' | '0-1' | '1/2-1/2'
    ply_count: int
    resigned_side: str | None = None
    opponent: str | None = None  # None for pure self-play; e.g. 'stockfish-skill-4'


def game_outcome(board: chess.Board, resigned_side: str | None = None) -> tuple[float, str]:
    """Final (outcome, result string) — outcome is WHITE's perspective.

    Shared by `self_play_game` and the sf-* data providers so every game
    scores identically (§10-correct mate sign, resign = loss for the resigner,
    ply-capped games score as draws).
    """
    if resigned_side is not None:
        outcome = -1.0 if resigned_side == 'white' else 1.0  # white's perspective
        result_str = '0-1' if resigned_side == 'white' else '1-0'
        return outcome, result_str
    if board.is_checkmate():
        # The mated side is the side to move; white won iff black is mated.
        outcome = 1.0 if board.turn == chess.BLACK else -1.0  # white's perspective
        result_str = '1-0' if board.turn == chess.BLACK else '0-1'
        return outcome, result_str
    result_str = board.result(claim_draw=True)
    if result_str == '*':
        result_str = '1/2-1/2'  # hit the ply cap — score as a draw
    return 0.0, result_str


def build_pgn(sans: Sequence[str], result_str: str) -> str:
    """Render a SAN move list as a minimal PGN string with the result header."""
    game = chess.pgn.Game()
    node = game
    replay = chess.Board()
    for san in sans:
        move = replay.parse_san(san)
        node = node.add_variation(move)
        replay.push(move)
    game.headers["Result"] = result_str
    return str(game.accept(chess.pgn.StringExporter(headers=None, variations=False, comments=False)))


def finalize_game_examples(
    pending: Sequence[tuple[np.ndarray, np.ndarray, chess.Color]],
    outcome: float,
) -> list[TrainingExample]:
    """Turn recorded (planes, policy, mover color) plies into value-labelled examples."""
    return [
        TrainingExample(planes=planes, policy=pol, value=outcome if color == chess.WHITE else -outcome)
        for planes, pol, color in pending
    ]


def self_play_game(
    evaluator: Evaluator,
    config: SelfPlayConfig = SelfPlayConfig(),
    rng: np.random.Generator | None = None,
    stream: GameStream | None = None,
) -> SelfPlayResult:
    """Play one self-play game and return labelled training examples.

    Labels follow AlphaZero: policy target = normalized visit counts, value
    target = game outcome from the mover's perspective. Terminal values inside
    the search use the §10-correct side-to-move sign. With `stream`, every ply
    is journaled as a ``live`` event and the finish as ``game_end`` (FS-6b).

    U2 note: playout cap randomization is PUCT-only (config-validated); with
    search_kind="gumbel" it raises ValueError rather than silently changing
    which target construction the cheap-search plies would record.
    """
    search = build_search(evaluator, config.search, rng=rng)
    # U2: cap_rng is the only new rng consumer and exists only when the flag
    # is on, so flag-off games draw the rng exactly as before (bit-identical).
    cap_rng: np.random.Generator | None = None
    if config.playout_randomization:
        cap_rng = rng if rng is not None else np.random.default_rng()
    fast_search: MCTSSearch | RustMCTSSearch | None = None
    # M3: FastBoard makes repetition CLAIMS cheap inside the search's
    # is_game_over(claim_draw=True) calls. Gated; plain Board keeps the
    # pre-M3 behavior bit-for-bit.
    board: chess.Board = FastBoard() if fastpath_enabled() else chess.Board()
    prev_board: chess.Board | None = None
    pending: list[tuple[np.ndarray, np.ndarray, chess.Color]] = []
    sans: list[str] = []
    resigned_side: str | None = None
    streak = 0

    for ply in range(config.max_plies):
        if board.is_game_over(claim_draw=True):
            break

        # U2: a proportion p of turns gets the full (recorded) search; the
        # rest get a fast noise-off search, not recorded (paper sec 3.1)
        # unless playout_cap_record_fast is set. The move is always played
        # from whichever search ran.
        full_search = True
        if cap_rng is not None and ply >= config.playout_cap_start_ply:
            full_search = bool(cap_rng.random() < config.playout_cap_full_prob)
        if full_search:
            result = search.search(board)
        else:
            if fast_search is None:
                fast_n = max(
                    config.playout_cap_fast_min,
                    round(config.search.num_simulations * config.playout_cap_fast_fraction),
                )
                fast_search = build_search(
                    evaluator,
                    replace(config.search, num_simulations=fast_n, root_noise=False),
                    rng=rng,
                )
            result = fast_search.search(board)

        legal_map = legal_move_map(board)
        # U1: gumbel results carry the Eq. 11 improved policy; it replaces the
        # visit distribution as the training target and the sampling basis.
        basis = result.improved_policy if result.improved_policy is not None else result.visit_counts
        total = basis.sum()
        policy = basis / total if total > 0 else np.zeros_like(basis)
        if full_search or config.playout_cap_record_fast:
            if config.encoder_version == 2:
                # WP-D: v2 examples carry the 104-plane encoding with the
                # move-stack history — the identical derivation the evaluator
                # used for the search-time eval of this same position.
                pending.append((encode_state_v2(board), policy.copy(), board.turn))
            else:
                pending.append((encode_state(board, prev_board), policy.copy(), board.turn))

        temperature = config.temperature if ply < config.temperature_plies else 0.0
        if result.improved_policy is not None and temperature <= 1e-6:
            gumbel_move = result.chosen_move
            assert gumbel_move is not None, "gumbel search returned no action on a live position"
            move = gumbel_move  # A_{n+1}: the gumbel action (paper sec 3)
        else:
            move = select_move_from_visits(basis, legal_map, temperature, rng)

        sans.append(board.san(move))
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

        prev_board = board.copy()
        board.push(move)
        if stream is not None:
            stream.move(len(sans), move.uci(), board.fen())

    outcome, result_str = game_outcome(board, resigned_side)

    examples = finalize_game_examples(pending, outcome)
    pgn = build_pgn(sans, result_str)
    if stream is not None:
        stream.end(result_str, len(sans))

    return SelfPlayResult(
        examples=examples,
        moves=sans,
        pgn=pgn,
        result=result_str,
        ply_count=len(sans),
        resigned_side=resigned_side,
    )


def parallel_self_play(
    evaluator: Evaluator,
    num_games: int,
    config: SelfPlayConfig = SelfPlayConfig(),
    parallel_games: int = 1,
    rng: np.random.Generator | None = None,
    stream: StreamContext | None = None,
) -> list[SelfPlayResult]:
    """Play `num_games` self-play games, up to `parallel_games` concurrently.

    With `parallel_games <= 1` this is the plain sequential loop. Above 1 the
    games share ONE evaluator — the queued batch service (`training.evaluator`)
    merges their per-position requests into single forwards. Results keep game
    order regardless of completion order. A game that crashes is logged and
    skipped; if every game fails the error propagates.

    `stream` (FS-6b) journals a ``live`` event per ply and one ``game_end``
    per finished game; game indices are `stream.game_offset` + local index.
    """
    if num_games < 0:
        raise ValueError("num_games must be >= 0")
    if parallel_games < 1:
        raise ValueError("parallel_games must be >= 1")
    if parallel_games <= 1 or num_games <= 1:
        return [
            self_play_game(
                evaluator,
                config,
                rng,
                stream=stream.for_game(index) if stream is not None else None,
            )
            for index in range(num_games)
        ]

    game_rngs: list[np.random.Generator | None]
    if rng is not None:
        game_rngs = rng.spawn(num_games)  # deterministic per-game children
    else:
        game_rngs = [None] * num_games

    results: list[SelfPlayResult | None] = [None] * num_games
    failures: list[tuple[int, BaseException]] = []

    def _play(index: int) -> None:
        try:
            results[index] = self_play_game(
                evaluator,
                config,
                game_rngs[index],
                stream=stream.for_game(index) if stream is not None else None,
            )
        except Exception as exc:
            failures.append((index, exc))
            log.exception("[SELFPLAY] parallel game %d failed", index)

    workers = min(parallel_games, num_games)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="selfplay") as pool:
        futures = [pool.submit(_play, index) for index in range(num_games)]
        for future in futures:
            future.result()

    missing = [index for index, result in enumerate(results) if result is None]
    if len(missing) == num_games:
        raise RuntimeError(f"all {num_games} parallel self-play games failed") from (
            failures[0][1] if failures else None
        )
    if missing:
        log.warning("[SELFPLAY] %d/%d parallel games failed and were skipped", len(missing), num_games)
    return [result for result in results if result is not None]


@dataclass(frozen=True)
class ArenaConfig:
    """Match settings for §5.1 arena gating (reduced sims, both colors).

    The `sprt_*` fields are the transport for U9 sequential early-stopping:
    the pipeline copies the gate's LadderConfig SPRT knobs into them via
    `replace()` (LadderConfig stays the single source of truth) and
    `Arena.match` stops the match at an even game count the moment the LLR
    crosses a bound. Disabled by default: at the default 16-game gates the
    bounds are unreachable (max |LLR| ~ 16 x 0.074 = 1.19 < ln 19 = 2.944),
    so enabling it is behavior-preserving tonight and caps the cost of any
    future larger match sizes.
    """

    num_simulations: int = 128  # reduced vs training's 400-800
    max_plies: int = 200
    games: int = 16  # forced even below, so colors are balanced
    # Gate throughput: games played concurrently. 1 = the original sequential
    # match. Games are independent (per-game openings, balanced colors), and
    # the evaluators are stateless per call, so parallel > 1 only yields
    # games in completion order — the decision math is unchanged.
    parallel: int = 1
    # Search backend for the gate games: "python" (legacy MCTSSearch) or
    # "rust" (gambit_core, ~10x faster). Rust leaf evaluation is history-free
    # exactly like the python arena path, so gate fidelity is unchanged.
    search_backend: str = "python"
    sprt_early_stop: bool = False
    sprt_elo0: float = 0.0
    sprt_elo1: float = 10.0
    sprt_alpha: float = 0.05
    sprt_beta: float = 0.05
    # FB-1: per-gate-event opening randomization. None (default) keeps the
    # pre-FB-1 behavior bit-identical — every game from the initial position.
    # When set, the arena samples a fresh seeded opening set (one FEN per
    # game, mirrored across color pairs) and `MatchResult.openings` records
    # the played FENs for the gate record.
    opening_seed: int | None = None


def play_match_game(
    white_evaluator: Evaluator,
    black_evaluator: Evaluator,
    num_simulations: int,
    max_plies: int = 200,
    white_config: SearchConfig | None = None,
    black_config: SearchConfig | None = None,
    rng: np.random.Generator | None = None,
    opening_fen: str | None = None,
) -> tuple[str, str]:
    """Play one arena game; returns (result, termination).

    Arena play is deterministic-strength: no root noise, argmax moves.
    termination is 'game-over' or 'max-plies' (scored as a draw).
    white_config/black_config default to the canonical PUCT arena search
    (section 5.1 semantics unchanged); U1 ablations pass explicit per-side
    SearchConfig objects (e.g. gumbel) instead. `opening_fen` (FB-1) starts
    the game from a seeded opening position instead of the initial position.
    """
    white_search = build_search(
        white_evaluator,
        white_config
        if white_config is not None
        else SearchConfig(root_noise=False, num_simulations=num_simulations),
        rng=rng,
    )
    black_search = build_search(
        black_evaluator,
        black_config
        if black_config is not None
        else SearchConfig(root_noise=False, num_simulations=num_simulations),
        rng=rng,
    )
    if opening_fen is not None:
        board: chess.Board = (
            FastBoard(fen=opening_fen) if fastpath_enabled() else chess.Board(fen=opening_fen)
        )
    else:
        board = FastBoard() if fastpath_enabled() else chess.Board()
    termination = "max-plies"
    for _ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            termination = "game-over"
            break
        searcher = white_search if board.turn == chess.WHITE else black_search
        outcome = searcher.search(board)
        legal_map = legal_move_map(board)
        assert outcome.chosen_move is not None, "search returned no move on a live position"
        if outcome.chosen_move not in legal_map.values():  # defensive: never push an illegal move
            raise AssertionError(f"search chose illegal move {outcome.chosen_move}")
        board.push(outcome.chosen_move)
    result = board.result(claim_draw=True)
    if result == "*":
        result = "1/2-1/2"  # capped games score as draws (arena aggregation)
    return result, termination
