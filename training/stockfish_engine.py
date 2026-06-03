"""Stockfish engine wrapper for training and evaluation.

Provides Stockfish as a stronger opponent and position evaluator
for the neural network training pipeline.

Usage:
    from training.stockfish_engine import StockfishPlayer
    sf = StockfishPlayer()
    move = sf.get_move(board, depth=20)
    eval_cp = sf.get_evaluation(board)
"""
import os
import chess
from stockfish import Stockfish as SF

STOCKFISH_PATH = os.environ.get('STOCKFISH_PATH', '/home/lance/.local/bin/stockfish')
SF_DEPTH = 15
SF_THREADS = 4
SF_HASH = 256


class StockfishPlayer:
    def __init__(self, depth=SF_DEPTH, threads=SF_THREADS, hash_mb=SF_HASH):
        self.engine = SF(
            path=STOCKFISH_PATH,
            depth=depth,
            parameters={
                "Threads": threads,
                "Hash": hash_mb,
            }
        )

    def set_position(self, fen):
        self.engine.set_position(fen.split(' '))

    def get_move(self, board, depth=None):
        """Get Stockfish's best move as UCI string."""
        self.engine.set_position(board.fen().split(' '))
        if depth:
            self.engine.set_option("Depth", str(depth))
        return self.engine.get_best_move()

    def get_best_move_san(self, board):
        """Get Stockfish's best move as SAN string."""
        uci = self.get_move(board)
        if uci == '0-1' or uci == '1-0' or uci is None:
            return None
        move = chess.Move.from_uci(uci)
        return board.san(move)

    def get_evaluation(self, board):
        """Get Stockfish's evaluation in centipawns (positive = white advantage)."""
        self.engine.set_position(board.fen().split(' '))
        info = self.engine.get_board_evaluation()
        if info['type'] == 'cp':
            return info['value']
        elif info['type'] == 'mate':
            return info['value'] * 10000
        return 0

    def evaluate_move(self, board, move):
        """Evaluate the position AFTER making a move. Returns centipawn score."""
        board.push(move)
        eval_ = self.get_evaluation(board)
        board.pop()
        return eval_

    def evaluate_legal_moves(self, board):
        """Evaluate every legal move. Returns list of (move, san, eval_cp)."""
        results = []
        for m in board.legal_moves:
            cp = self.evaluate_move(board, m)
            results.append((m, board.san(m), cp))
        return results

    def get_top_moves(self, board, num_moves=5):
        """Get Stockfish's top N moves with evaluations."""
        self.engine.set_position(board.fen().split(' '))
        self.engine.set_option("MultiPV", num_moves)
        best = self.engine.get_board_evaluation()
        moves = []
        for i in range(num_moves):
            uci = self.engine.get_best_move()
            move = chess.Move.from_uci(uci)
            san = board.san(move)
            eval_cp = self.get_evaluation(board)
            moves.append({
                'uci': uci,
                'san': san,
                'evaluation': eval_cp,
                'depth': self.engine.get_depth(),
            })
        self.engine.set_option("MultiPV", 1)
        return moves

    def play_game(self, opponent_move_fn=None, max_moves=200):
        """Play a complete game, optionally with a custom opponent.

        Args:
            opponent_move_fn: callable(board) -> uci move for black

        Returns:
            dict with 'moves', 'pgn', 'result'
        """
        board = chess.Board()
        move_sans = []

        for _ in range(max_moves):
            if board.turn == chess.WHITE:
                uci = self.get_move(board)
            else:
                if opponent_move_fn:
                    uci = opponent_move_fn(board)
                else:
                    uci = self.get_move(board)

            if uci in ('0-1', '1-0', 'resign', None):
                break

            try:
                move = chess.Move.from_uci(uci)
                san = board.san(move)
                board.push(move)
                move_sans.append(san)
            except:
                break

        # Build PGN
        board2 = chess.Board()
        for san in move_sans:
            try:
                board2.push_san(san)
            except:
                pass
        game = chess.Game.from_board(board2)
        pgn = game.pgn()

        if board.is_checkmate():
            result = '1-0' if board.turn == chess.BLACK else '0-1'
        elif board.is_insufficient_material() or board.is_fivefold_repetition() or board.can_claim_draw():
            result = '1/2-1/2'
        else:
            result = '1/2-1/2'

        return {
            'moves': move_sans,
            'pgn': pgn,
            'result': result,
            'final_fen': board.fen(),
        }
