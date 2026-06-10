"""Self-play engine using MCTS with neural network guidance.

Stockfish-guided rollouts: at leaf nodes, MCTS blends Stockfish's centipawn
evaluation (depth 10) with the NN's own value head. The NN still generates
the policy priors for move selection — Stockfish only provides a stronger
value signal at evaluation time.

Blend ratio: 0.6 Stockfish / 0.4 NN to keep the network's own judgment
in the loop and prevent overfitting to engine scores. Gaussian noise
(sigma=0.1) is added to Stockfish evals to avoid memorization.
"""
import chess
import chess.pgn
import numpy as np
import threading
import queue
import time
import torch
from .tensorize import board_to_tensor, move_to_idx, NUM_POSSIBLE_MOVES
from .model import ChessNet
import logging


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

SF_LEAF_BLEND = 0.6
SF_LEAF_DEPTH = 10
SF_EVAL_NOISE_SIGMA = 0.1
MCTS_EVAL_CACHE_MAX = 500  # LRU cap on eval cache

RESIGN_THRESHOLD = -0.8  # NN value below this → resign


class BatchEvaluator:
    """Handles batching of neural network forward passes to maximize GPU throughput."""
    def __init__(self, model, batch_size=256, max_wait_time=0.05):
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
            batch = []
            results = []
            
            start_time = time.time()
            while len(batch) < self.batch_size and (time.time() - start_time) < self.max_wait_time:
                try:
                    req = self.queue.get(timeout=0.001)
                    batch.append(req[0])
                    results.append(req[1])
                except queue.Empty:
                    continue
            
            if not batch:
                time.sleep(0.001)
                continue
            
            self.is_batching = True
            try:
                # batch is a list of numpy arrays of shape (NUM_POSSIBLE_MOVES,)
                input_tensor = torch.stack([torch.as_tensor(b, device=self.device) for b in batch])
                policy_logits, values = self.model(input_tensor)
                
                # values is [batch_size, 1] or [batch_size]
                for i, res_container in enumerate(results):
                    res_container['policy'] = policy_logits[i].detach().cpu().numpy()
                    if values.dim() == 2:
                        res_container['value'] = values[i].detach().item()
                    else:
                        res_container['value'] = values[i].detach().item()

                    res_container['done'] = True
            finally:
                self.is_batching = False

    def get_eval(self, board_tensor):
        res_container = {'policy': None, 'value': None, 'done': False}
        self.queue.put((board_tensor, res_container))
        
        while not res_container['done']:
            time.sleep(0.0001)
            
        return res_container['policy'], res_container['value']


class MCTS:
    def __init__(self, model, stockfish=None, cpuct=1.0, noise_epsilon=0.25, noise_alpha=0.03, batch_size=256, use_stockfish=True):
        self.model = model
        self.stockfish = stockfish
        self.cpuct = cpuct
        self.noise_epsilon = noise_epsilon
        self.noise_alpha = noise_alpha
        self._eval_cache = {}  # FEN -> value cache (LRU capped)
        self._device = next(model.parameters()).device
        self.evaluator = BatchEvaluator(model, batch_size=batch_size)
        self.use_stockfish = use_stockfish

    def search(self, board, num_simulations=800):
        """Run MCTS search. Returns (visit_counts, nn_value) tuple.

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
        for _ in range(num_simulations):
            board_copy = board.copy()
            self._simulate(root, board_copy, max_depth=15)

        visit_counts = np.zeros(NUM_POSSIBLE_MOVES)
        for child in root['children']:
            if child['move'] is not None:
                visit_counts[move_to_idx(child['move'])] = child['visit_count']

        t2 = time.time()
        log.info(f'  [MCTS] {num_simulations} sims in {t2-t1:.3f}s, visits={visit_counts.sum():.0f}')
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

        while node['children'] and depth < max_depth:
            selected = self._select_child(node)
            path.append((node, selected))

            board.push(selected['move'])

            if not selected['expanded']:
                self._expand(selected, board)

            node = selected
            depth += 1

        value = self._evaluate(board)

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
                return 1.0 if board.turn == chess.BLACK else -1.0
            return 0.0

        nn_value = self._nn_evaluate(board)

        if self.use_stockfish and self.stockfish:
            # Use a timeout lock — if the engine is busy (e.g., during
            # sf.play_game()), fall back to NN-only to prevent deadlock.
            import threading
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
            except Exception:
                # Engine dead or unresponsive — fall back to NN
                return nn_value
            finally:
                self.stockfish._lock.release()

        return nn_value

    def _nn_evaluate(self, board):
        fen = board.fen()
        if fen in self._eval_cache:
            return self._eval_cache[fen]
        board_tensor = board_to_tensor(board)
        _, value = self.evaluator.get_eval(board_tensor)
        
        # LRU cap: evict oldest entry when cache is full
        if len(self._eval_cache) >= MCTS_EVAL_CACHE_MAX:
            self._eval_cache.pop(next(iter(self._eval_cache)))
        self._eval_cache[fen] = value
        return value



class SelfPlayGame:
    def __init__(self, model, num_mcts_simulations=800, max_moves=200, stockfish=None, batch_size=256, use_stockfish=True):
        self.model = model
        self.num_mcts_simulations = num_mcts_simulations
        self.max_moves = max_moves
        self.mcts = MCTS(model, stockfish=stockfish, batch_size=batch_size, use_stockfish=use_stockfish)
        self.use_stockfish = use_stockfish

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

                # Resignation: if the NN thinks the position is clearly lost, end the game
                # nn_value is already computed by _build_root() and returned by search()
                
                if nn_value < RESIGN_THRESHOLD:
                    log.info(f'[GAME] Resigning at move {i+1} (value={nn_value:.3f} < {RESIGN_THRESHOLD})')
                    break

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
        except Exception as e:
            log.error(f'[GAME] ERROR: {e}')
            import traceback
            traceback.print_exc()

        outcome = self._get_outcome(board)

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
            except:
                break
        pgn = game.accept(chess.pgn.StringExporter())

        if board.is_checkmate():
            result = '1-0' if board.turn == chess.BLACK else '0-1'
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
