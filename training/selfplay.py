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
import queue
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

import chess
import chess.pgn
import numpy as np
import torch

from .encoder import encode_batch, encode_move, encode_state, legal_move_map
from .model import NUM_POLICY_MOVES_V2
from .tensorize import NUM_POSSIBLE_MOVES, board_to_tensor, move_to_idx

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

            for i, res_container in enumerate(results):
                res_container['policy'] = policy_logits[i].detach().cpu().numpy()
                if values.dim() == 2:
                    res_container['value'] = values[i].detach().item()
                else:
                    res_container['value'] = values[i].detach().item()
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


@dataclass(frozen=True)
class MCTSResult:
    """Outcome of one root search. Visit counts are 4672-length (encoder idx)."""

    visit_counts: np.ndarray
    root_value: float
    chosen_move: chess.Move | None
    visited_root: bool = True


class Evaluator(Protocol):
    """Batched position evaluator.

    Implementations return, for each board: raw policy logits over the 4672
    encoder indices (unmasked) and a scalar value from the side-to-move's
    perspective.
    """

    def evaluate(self, boards: Sequence[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        ...  # pragma: no cover


class BatchNNEvaluator:
    """Synchronous micro-batched NN evaluator (CPU or CUDA).

    Slice 2 replaces this with the queued NNEvaluator service (§5 lever 2);
    the Evaluator interface stays the same.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device | None = None) -> None:
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device

    def evaluate(self, boards: Sequence[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        if not boards:
            return np.zeros((0, NUM_POLICY_MOVES_V2), dtype=np.float32), np.zeros(0, dtype=np.float32)
        batch = torch.from_numpy(encode_batch(boards)).to(self.device)
        with torch.inference_mode():
            logits, values = self.model(batch)
        return (
            logits.detach().cpu().numpy().astype(np.float32),
            values.detach().cpu().numpy().astype(np.float32),
        )


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

    def __init__(self, evaluator: Evaluator, config: SearchConfig = SearchConfig()) -> None:
        self.evaluator = evaluator
        self.config = config

    def search(self, board: chess.Board, num_simulations: int | None = None) -> MCTSResult:
        """Search `board`; returns visit counts, root value, and chosen move."""
        sims = num_simulations if num_simulations is not None else self.config.num_simulations
        legal = list(board.legal_moves)
        if not legal:
            return MCTSResult(np.zeros(NUM_POLICY_MOVES_V2, dtype=np.float32), terminal_value(board), None, False)

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

        best = max(root.children, key=lambda c: c.visit_count)
        return MCTSResult(visit_counts, root.value_mean, best.move, True)

    # -- internals ---------------------------------------------------------

    def _simulate(self, root: _Node, board: chess.Board) -> None:
        node = root
        path: list[_Node] = [root]
        while node.expanded and node.children and not node.terminal:
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


@dataclass(frozen=True)
class TrainingExample:
    """One (planes, policy, value) training tuple; value from mover's persp."""

    planes: np.ndarray  # (29, 8, 8)
    policy: np.ndarray  # (4672,) visit distribution
    value: float


@dataclass(frozen=True)
class SelfPlayResult:
    examples: list[TrainingExample]
    moves: list[str]
    pgn: str
    result: str  # '1-0' | '0-1' | '1/2-1/2'
    ply_count: int
    resigned_side: str | None = None


def self_play_game(
    evaluator: Evaluator,
    config: SelfPlayConfig = SelfPlayConfig(),
    rng: np.random.Generator | None = None,
) -> SelfPlayResult:
    """Play one self-play game and return labelled training examples.

    Labels follow AlphaZero: policy target = normalized visit counts, value
    target = game outcome from the mover's perspective. Terminal values inside
    the search use the §10-correct side-to-move sign.
    """
    search = MCTSSearch(evaluator, config.search)
    board = chess.Board()
    prev_board: chess.Board | None = None
    pending: list[tuple[np.ndarray, np.ndarray, chess.Color]] = []
    sans: list[str] = []
    resigned_side: str | None = None
    streak = 0

    for ply in range(config.max_plies):
        if board.is_game_over(claim_draw=True):
            break

        result = search.search(board)
        legal_map = legal_move_map(board)
        total = result.visit_counts.sum()
        policy = result.visit_counts / total if total > 0 else np.zeros_like(result.visit_counts)
        pending.append((encode_state(board, prev_board), policy.copy(), board.turn))

        temperature = config.temperature if ply < config.temperature_plies else 0.0
        move = select_move_from_visits(result.visit_counts, legal_map, temperature, rng)

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

    if resigned_side is not None:
        outcome = -1.0 if resigned_side == 'white' else 1.0  # white's perspective
        result_str = '0-1' if resigned_side == 'white' else '1-0'
    elif board.is_checkmate():
        # The mated side is the side to move; white won iff black is mated.
        outcome = 1.0 if board.turn == chess.BLACK else -1.0  # white's perspective
        result_str = '1-0' if board.turn == chess.BLACK else '0-1'
    else:
        outcome = 0.0
        result_str = board.result(claim_draw=True)
        if result_str == '*':
            result_str = '1/2-1/2'  # hit the ply cap — score as a draw

    examples = [
        TrainingExample(planes=planes, policy=pol, value=outcome if color == chess.WHITE else -outcome)
        for planes, pol, color in pending
    ]

    game = chess.pgn.Game()
    node = game
    replay = chess.Board()
    for san in sans:
        move = replay.parse_san(san)
        node = node.add_variation(move)
        replay.push(move)
    game.headers["Result"] = result_str
    pgn = str(game.accept(chess.pgn.StringExporter(headers=None, variations=False, comments=False)))

    return SelfPlayResult(
        examples=examples,
        moves=sans,
        pgn=pgn,
        result=result_str,
        ply_count=len(sans),
        resigned_side=resigned_side,
    )


@dataclass(frozen=True)
class ArenaConfig:
    """Match settings for §5.1 arena gating (reduced sims, both colors)."""

    num_simulations: int = 128  # reduced vs training's 400-800
    max_plies: int = 200
    games: int = 16  # forced even below, so colors are balanced


def play_match_game(
    white_evaluator: Evaluator,
    black_evaluator: Evaluator,
    num_simulations: int,
    max_plies: int = 200,
) -> tuple[str, str]:
    """Play one arena game; returns (result, termination).

    Arena play is deterministic-strength: no root noise, argmax moves.
    termination is 'game-over' or 'max-plies' (scored as a draw).
    """
    white_search = MCTSSearch(white_evaluator, SearchConfig(root_noise=False, num_simulations=num_simulations))
    black_search = MCTSSearch(black_evaluator, SearchConfig(root_noise=False, num_simulations=num_simulations))
    board = chess.Board()
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
