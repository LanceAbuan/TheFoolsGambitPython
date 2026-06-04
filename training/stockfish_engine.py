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
import time
import chess
from stockfish import Stockfish as SF

STOCKFISH_PATH = os.environ.get('STOCKFISH_PATH', '/home/lance/.local/bin/stockfish')
SF_DEPTH = 12
SF_THREADS = 1
SF_HASH = 128


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

    def close(self):
        try:
            self.engine.close()
        except Exception:
            pass

    def get_move(self, board, depth=None):
        """Get Stockfish's best move as UCI string."""
        try:
            self.engine.set_fen_position(board.fen())
            if depth:
                self.engine.set_depth(depth)
            return self.engine.get_best_move()
        except Exception:
            return None

    def get_best_move_san(self, board):
        """Get Stockfish's best move as SAN string."""
        uci = self.get_move(board)
        if uci == '0-1' or uci == '1-0' or uci is None:
            return None
        move = chess.Move.from_uci(uci)
        return board.san(move)

    def get_evaluation(self, board):
        """Get Stockfish's evaluation in centipawns (positive = white advantage)."""
        try:
            self.engine.set_fen_position(board.fen())
            info = self.engine.get_evaluation()
            val = 0
            if info['type'] == 'cp':
                val = info['value']
            elif info['type'] == 'mate':
                val = info['value'] * 10000
            # Stockfish returns eval from side-to-move perspective; flip for Black's turn
            if board.turn == chess.BLACK:
                val = -val
            return val
        except Exception:
            pass
        return 0

    def get_evaluation_normalized(self, board):
        """Get Stockfish eval normalized to [-1, 1] range for NN training."""
        cp = self.get_evaluation(board)
        return max(-1.0, min(1.0, cp / 2000.0))

    def evaluate_move(self, board, move):
        """Evaluate the position AFTER making a move. Returns centipawn score."""
        board.push(move)
        eval_ = self.get_evaluation(board)
        board.pop()
        return eval_

    def evaluate_legal_moves(self, board, depth=None):
        """Evaluate every legal move. Returns list of (move, san, eval_cp)."""
        results = []
        for m in board.legal_moves:
            cp = self.evaluate_move(board, m)
            results.append((m, board.san(m), cp))
        return results

    @staticmethod
    def evaluate_legal_moves_batch(board, depth=10):
        """Evaluate all legal moves using a dedicated Stockfish subprocess.
        
        Uses its own process to avoid threading issues with shared engine.
        Returns dict mapping UCI move string to centipawn evaluation.
        """
        try:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return {}
            
            tmp_sf = SF(
                path=STOCKFISH_PATH,
                depth=depth,
                parameters={"Threads": 1, "Hash": 128}
            )
            
            eval_map = {}
            for m in legal_moves:
                try:
                    board.push(m)
                    tmp_sf.set_fen_position(board.fen())
                    info = tmp_sf.get_evaluation()
                    cp = 0
                    if info['type'] == 'cp':
                        cp = info['value']
                    elif info['type'] == 'mate':
                        cp = info['value'] * 10000
                    # Flip to White's perspective when it's Black's turn
                    if board.turn == chess.BLACK:
                        cp = -cp
                    eval_map[m.uci()] = cp
                    board.pop()
                except Exception:
                    try:
                        board.pop()
                    except Exception:
                        pass
                    eval_map[m.uci()] = 0
            
            try:
                tmp_sf.close()
            except Exception:
                pass
            
            return eval_map
        except Exception as e:
            print(f'[SF-BATCH] Error: {e}', flush=True)
            return {}

    def get_move_quality(self, board, move):
        """Get quality score for a specific move relative to best move.
        
        Returns:
            - 'evaluation': centipawn eval after the move
            - 'best_eval': best move evaluation
            - 'diff': how many centipawns worse than best (0 = blunder-free)
            - 'accuracy': float 0-1, how good the move is
        """
        board_after = board.copy()
        board_after.push(move)
        eval_after = self.get_evaluation(board_after)
        
        best_uci = self.get_move(board)
        if best_uci in ('0-1', '1-0', 'resign', None):
            return {
                'evaluation': eval_after,
                'best_eval': eval_after,
                'diff': 0,
                'accuracy': 1.0,
            }
        
        best_move = chess.Move.from_uci(best_uci)
        board_after_best = board.copy()
        board_after_best.push(best_move)
        best_eval = self.get_evaluation(board_after_best)
        
        diff = best_eval - eval_after if board.turn == chess.WHITE else eval_after - best_eval
        accuracy = max(0.0, 1.0 - abs(diff) / 1000.0)
        
        return {
            'evaluation': eval_after,
            'best_eval': best_eval,
            'diff': diff,
            'accuracy': accuracy,
        }

    def get_top_moves(self, board, num_moves=5):
        """Get Stockfish's top N moves with evaluations."""
        try:
            self.engine.set_fen_position(board.fen())
            self.engine.update_engine_parameters({"MultiPV": num_moves})
            best = self.engine.get_evaluation()
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
            self.engine.update_engine_parameters({"MultiPV": 1})
            return moves
        except Exception:
            return []

    def analyze_position(self, board):
        """Full position analysis for UI display.
        
        Returns dict with eval, top moves, and move evaluations.
        """
        try:
            pos_eval = self.get_evaluation(board)
            top_moves = self.get_top_moves(board, num_moves=3)
            
            legal_moves = list(board.legal_moves)
            eval_map = self.evaluate_legal_moves_batch(board, depth=12)
            
            move_analysis = []
            for m in legal_moves:
                uci = m.uci()
                cp = eval_map.get(uci, 0)
                move_analysis.append({
                    'san': board.san(m),
                    'uci': uci,
                    'evaluation': cp,
                })
            
            move_analysis.sort(key=lambda x: x['evaluation'], reverse=True)
            
            return {
                'evaluation': pos_eval,
                'evaluation_normalized': max(-1.0, min(1.0, pos_eval / 2000.0)),
                'depth': self.engine.get_depth(),
                'top_moves': top_moves,
                'move_analysis': move_analysis[:10],
                'num_legal_moves': len(legal_moves),
            }
        except Exception:
            return {'evaluation': 0, 'evaluation_normalized': 0, 'top_moves': [], 'move_analysis': [], 'num_legal_moves': 0}

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
