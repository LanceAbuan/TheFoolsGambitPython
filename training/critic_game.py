"""Critic-guided RL: NN picks its own moves, Stockfish evaluates every legal move.

The NN is free to choose any legal move. Before each turn, Stockfish evaluates
every legal move and produces a weighted policy target — moves leading to better
positions get higher target probability. The NN's value head learns to predict
position quality from Stockfish's evals.

This lets the NN develop its own style while still being guided toward good moves.
"""
import chess
import numpy as np
import time
import torch

from .tensorize import board_to_tensor, move_to_idx, NUM_POSSIBLE_MOVES


class CriticGame:
    """Play games where NN chooses moves, Stockfish acts as position critic."""

    def __init__(self, model, stockfish, temperature=0.3, max_moves=200):
        self.model = model
        # Always use own Stockfish to avoid thread contention
        from .stockfish_engine import StockfishPlayer, STOCKFISH_PATH
        self.stockfish = StockfishPlayer(depth=15, threads=2, hash_mb=128)
        self.temperature = temperature
        self.max_moves = min(max_moves, 40)

    def close(self):
        self.stockfish.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def play(self, on_move=None):
        """Play one complete game. NN plays both sides, Stockfish critiques.

        Returns dict with examples, moves, pgn, result.
        """
        board = chess.Board()
        examples = []
        move_sans = []
        t0 = time.time()

        for i in range(self.max_moves):
            if board.is_game_over():
                break

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            current_eval = self.stockfish.get_evaluation(board)
            eval_map = self.stockfish.evaluate_legal_moves_batch(board, depth=12)
            move_evals = []
            for m in board.legal_moves:
                cp = eval_map.get(m.uci(), 0)
                move_evals.append((m, board.san(m), cp))

            # --- Step 3: Build weighted policy target from evals ---
            eval_values = np.array([mv[2] for mv in move_evals], dtype=np.float32)
            # Shift to positive range, softmax to get weights
            eval_shifted = eval_values - np.min(eval_values) + 1e-10
            weights = eval_shifted / eval_shifted.sum()

            target_policy = np.zeros(NUM_POSSIBLE_MOVES, dtype=np.float32)
            for idx, mv in enumerate(move_evals):
                target_policy[move_to_idx(mv[0])] = weights[idx]

            # --- Step 4: NN picks a move (own policy + small critic bias) ---
            chosen_move, chosen_san = self._nn_pick_move(board, legal_moves, move_evals)
            if chosen_move is None:
                break

            # --- Step 5: Record training example ---
            value = np.clip(current_eval / 2000.0, -1.0, 1.0)

            # Find eval of the chosen move
            chosen_eval = 0
            for mv, san, cp in move_evals:
                if mv == chosen_move:
                    chosen_eval = cp
                    break

            examples.append({
                'board_tensor': board_to_tensor(board),
                'policy': target_policy,
                'value': value,
                'san': chosen_san,
                'chosen_eval': chosen_eval,
            })

            board.push(chosen_move)
            move_sans.append(chosen_san)
            print(f'[CRITIC] move {i+1}: {chosen_san} (eval={current_eval}cp)', flush=True)
            if on_move:
                on_move(list(move_sans))

        result = self._game_result(board)
        elapsed = time.time() - t0
        print(f'[CRITIC] {elapsed:.1f}s | {len(move_sans)} moves | result={result}', flush=True)

        return {
            'examples': examples,
            'moves': move_sans,
            'pgn': self._build_pgn(move_sans),
            'result': result,
        }

    def _nn_pick_move(self, board, legal_moves, move_evals):
        """Combine NN policy with Stockfish critic bias, then sample."""
        device = next(self.model.parameters()).device
        self.model.eval()

        # NN logits for current position
        board_tensor = board_to_tensor(board)
        with torch.no_grad():
            x = torch.FloatTensor(board_tensor).unsqueeze(0).to(device)
            logits, _ = self.model(x)
            logits = logits.squeeze(0).cpu().numpy()

        # Add critic bias: scale evals to logits-space nudge
        # 100cp difference = 0.1 logit boost (subtle guidance, not forcing)
        critic_bias = np.zeros(NUM_POSSIBLE_MOVES, dtype=np.float32)
        for mv, san, cp in move_evals:
            critic_bias[move_to_idx(mv)] = cp * 0.001

        logits = logits + critic_bias

        # Mask illegal moves
        logits_masked = np.full(NUM_POSSIBLE_MOVES, -1e10, dtype=np.float32)
        for mv, san, cp in move_evals:
            logits_masked[move_to_idx(mv)] = logits[move_to_idx(mv)]

        # Temperature scaling
        logits_masked /= max(self.temperature, 1e-10)

        # Softmax
        exp_logits = np.exp(logits_masked - np.max(logits_masked))
        probs = exp_logits / exp_logits.sum()

        # Sample
        chosen_idx = np.random.choice(len(probs), p=probs)
        for mv, san, cp in move_evals:
            if move_to_idx(mv) == chosen_idx:
                return mv, san

        # Fallback to random legal move
        mv, san, _ = legal_moves[0], board.san(legal_moves[0]), 0
        idx = np.random.randint(len(legal_moves))
        return legal_moves[idx], board.san(legal_moves[idx])

    def _game_result(self, board):
        if board.is_checkmate():
            return '1-0' if board.turn == chess.BLACK else '0-1'
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return '1/2-1/2'
        return '1/2-1/2'

    def _build_pgn(self, move_sans):
        import chess.pgn
        game = chess.pgn.Game()
        board = chess.Board()
        current = game
        for san in move_sans:
            try:
                move = board.parse_san(san)
                current = current.add_variation(move)
                board.push(move)
            except Exception:
                break
        return str(game)
