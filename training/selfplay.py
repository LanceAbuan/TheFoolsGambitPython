"""Self-play engine using MCTS with neural network guidance.

The self-play loop:
1. Run MCTS search from current position using the neural net for evaluation
2. Select move based on visit counts (with exploration noise for root)
3. Store (board_tensor, policy) tuple during the game
4. After game end, assign outcome value to all stored positions
5. Return training examples with game outcome as target value

MCTS parameters follow AlphaZero style:
- CPuct: exploration constant
- num_mcts_simulations: simulations per move
- noise_epsilon: Dirichlet noise strength for root policy
- noise_alpha: Dirichlet concentration parameter
"""
import chess
import numpy as np
import torch
from .tensorize import board_to_tensor, move_to_idx, NUM_POSSIBLE_MOVES
from .model import ChessNet


class MCTS:
    """Monte Carlo Tree Search with neural network evaluation.
    
    Uses a simplified UCT-style search suitable for batch self-play.
    """
    
    def __init__(self, model, cpuct=1.0, noise_epsilon=0.25, noise_alpha=0.03):
        self.model = model
        self.cpuct = cpuct
        self.noise_epsilon = noise_epsilon
        self.noise_alpha = noise_alpha
    
    def search(self, board, num_simulations=800):
        """Run MCTS search and return visit distribution over legal moves.
        
        Args:
            board: chess.Board instance
            num_simulations: number of MCTS simulations to run
        
        Returns:
            numpy.ndarray: visit counts for each possible move (shape: 4096)
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return np.zeros(NUM_POSSIBLE_MOVES)
        
        if len(legal_moves) == 1:
            vc = np.zeros(NUM_POSSIBLE_MOVES)
            vc[move_to_idx(legal_moves[0])] = 1.0
            return vc
        
        root = self._build_root(board, legal_moves)
        
        for _ in range(num_simulations):
            board_copy = chess.Board(board.fen())
            self._simulate(root, board_copy)
        
        visit_counts = np.zeros(NUM_POSSIBLE_MOVES)
        for child in root:
            if child['move'] is not None:
                visit_counts[move_to_idx(child['move'])] = child['visit_count']
        
        return visit_counts
    
    def _build_root(self, board, legal_moves):
        """Build root node with neural net prior and Dirichlet noise."""
        board_tensor = board_to_tensor(board)
        legal_mask = np.zeros(NUM_POSSIBLE_MOVES, dtype=np.float32)
        for m in legal_moves:
            legal_mask[move_to_idx(m)] = 1.0
        
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(board_tensor).unsqueeze(0)
            policy_logits, _ = self.model(x)
            policy_logits = policy_logits.squeeze(0).numpy()
        
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
            'expanded': True
        }
    
    def _simulate(self, root, board):
        """Run a single MCTS simulation."""
        node = root
        path = []
        
        while node['children']:
            selected = self._select_child(node)
            path.append((node, selected))
            
            if not selected['expanded']:
                self._expand(selected, board)
            
            board.push(selected['move'])
            node = selected
        
        value = self._evaluate(board)
        
        for parent, child in reversed(path):
            child['visit_count'] += 1
            child['value'] = (child['value'] * (child['visit_count'] - 1) + value) / child['visit_count']
            value = -value
        
        root['visit_count'] += 1
    
    def _select_child(self, node):
        """Select child using PUCT formula."""
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
        """Expand node by generating children."""
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
        
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(board_tensor).unsqueeze(0)
            policy_logits, _ = self.model(x)
            policy_logits = policy_logits.squeeze(0).numpy()
        
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
        """Evaluate leaf position with neural network."""
        if board.is_game_over():
            if board.is_checkmate():
                return 1.0 if board.turn == chess.BLACK else -1.0
            return 0.0
        
        board_tensor = board_to_tensor(board)
        
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(board_tensor).unsqueeze(0)
            _, value = self.model(x)
            return float(value.squeeze())


class SelfPlayGame:
    """Plays a complete game using MCTS + neural net, collecting training data.
    
    Each move generates a training example:
    - board_tensor: position representation
    - policy_target: normalized visit counts from MCTS
    - value_target: game outcome (+1/-1/0) from winner's perspective
    """
    
    def __init__(self, model, num_mcts_simulations=800, max_moves=200):
        self.model = model
        self.num_mcts_simulations = num_mcts_simulations
        self.max_moves = max_moves
        self.mcts = MCTS(model)
    
    def play(self):
        """Play a complete self-play game.
        
        Returns:
            list of dicts: training examples with keys:
                - board_tensor: (16, 8, 8) numpy array
                - policy: (4096,) numpy array of move probabilities
                - value: float in {-1, 0, 1}
        """
        board = chess.Board()
        examples = []
        
        for _ in range(self.max_moves):
            board_tensor = board_to_tensor(board)
            visit_counts = self.mcts.search(board, self.num_mcts_simulations)
            
            total_visits = visit_counts.sum()
            policy = visit_counts / total_visits if total_visits > 0 else visit_counts
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            probs = visit_counts[[move_to_idx(m) for m in legal_moves]]
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(legal_moves)) / len(legal_moves)
            
            move_idx = np.random.choice(len(legal_moves), p=probs)
            move = legal_moves[move_idx]
            
            examples.append({
                'board_tensor': board_tensor,
                'policy': policy,
                'turn': board.turn
            })
            
            board.push(move)
        
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
                'value': value
            })
        
        return training_data
    
    def _get_outcome(self, board):
        """Get game outcome from white's perspective."""
        if board.is_checkmate():
            return 1.0 if board.turn == chess.BLACK else -1.0
        if board.is_draw():
            return 0.0
        return 0.0
