"""Stockfish engine wrapper for training and evaluation.

Provides Stockfish as a stronger opponent and position evaluator
for the neural network training pipeline.

IMPORTANT: Only ONE instance should exist. Use the shared instance
from training.server.get_stockfish() — do NOT create new instances.

THREAD SAFETY: All public methods acquire _lock before touching the engine.
This prevents race conditions when the training thread, MCTS, and HTTP
threads all share a single Stockfish subprocess.

CRASH RECOVERY: If the engine dies, methods auto-restart the subprocess
and retry once before returning a safe fallback.
"""
import os
import sys
import time
import threading
import chess
from stockfish import Stockfish as SF

STOCKFISH_PATH = os.environ.get('STOCKFISH_PATH', '/home/lance/.local/bin/stockfish')
SF_DEPTH = 12
SF_THREADS = 2
SF_HASH = 256
_MAX_RESTART = 1  # How many times to auto-restart a dead engine per call


class StockfishPlayer:
    def __init__(self, depth=SF_DEPTH, threads=SF_THREADS, hash_mb=SF_HASH):
        self._lock = threading.RLock()
        self._depth = depth
        self._threads = threads
        self._hash_mb = hash_mb
        self._engine = self._create_engine()
        self._alive = True

    # ---- internal helpers ----

    def _create_engine(self):
        """Create a new Stockfish engine subprocess."""
        return SF(
            path=STOCKFISH_PATH,
            depth=self._depth,
            parameters={
                "Threads": self._threads,
                "Hash": self._hash_mb,
                "MultiPV": 1,
            },
        )

    @property
    def engine(self):
        """Expose engine for compatibility — always acquire _lock before use."""
        return self._engine

    def _is_alive(self):
        """Probe whether the engine subprocess is still responsive."""
        try:
            self._engine.get_evaluation()
            return True
        except Exception:
            return False

    def _restart(self):
        """Kill the dead subprocess and spawn a fresh one."""
        try:
            self._engine.close()
        except Exception:
            pass
        self._engine = self._create_engine()
        self._alive = True
        print('[SF] Engine restarted after crash', flush=True)

    def _ensure_multipv_1(self):
        """Reset MultiPV to 1 — critical cleanup after batch operations."""
        try:
            self._engine.update_engine_parameters({"MultiPV": 1})
        except Exception:
            pass

    def _safe_call(self, fn, *args, **kwargs):
        """Call *fn* under _lock with automatic crash-recovery and timeout."""
        for attempt in range(_MAX_RESTART + 1):
            acquired = self._lock.acquire(timeout=2.0)
            if not acquired:
                raise TimeoutError("Failed to acquire Stockfish lock within 2 seconds")
            try:
                return fn(*args, **kwargs)
            except Exception:
                if attempt < _MAX_RESTART:
                    self._restart()
                else:
                    raise
            finally:
                self._lock.release()

    # ---- public API (thread-safe, crash-safe) ----

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Safely close the Stockfish engine subprocess."""
        with self._lock:
            try:
                self._ensure_multipv_1()
                self._engine.close()
                self._alive = False
            except Exception:
                pass

    def get_move(self, board, depth=None):
        """Get Stockfish's best move as UCI string."""
        def _inner():
            self._engine.set_fen_position(board.fen())
            if depth:
                self._engine.set_depth(depth)
            return self._engine.get_best_move()
        try:
            return self._safe_call(_inner)
        except Exception:
            return None

    def get_best_move_san(self, board):
        """Get Stockfish's best move as SAN string."""
        uci = self.get_move(board)
        if uci in ('0-1', '1-0', None):
            return None
        move = chess.Move.from_uci(uci)
        return board.san(move)

    def get_evaluation(self, board):
        """Get Stockfish's evaluation in centipawns (positive = white advantage)."""
        def _inner():
            self._engine.set_fen_position(board.fen())
            info = self._engine.get_evaluation()
            val = 0
            if info['type'] == 'cp':
                val = info['value']
            elif info['type'] == 'mate':
                val = info['value'] * 10000
            # Stockfish returns eval from side-to-move perspective; flip for Black
            if board.turn == chess.BLACK:
                val = -val
            return val
        try:
            return self._safe_call(_inner)
        except Exception:
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

    def evaluate_legal_moves_batch(self, board, depth=12):
        """Evaluate all legal moves using MultiPV for speed.

        Uses Stockfish's MultiPV analysis to evaluate multiple moves
        in a single search, much faster than evaluating one by one.
        Returns dict mapping UCI move string to centipawn evaluation.
        """
        def _inner():
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return {}

            num_moves = len(legal_moves)
            self._engine.set_fen_position(board.fen())
            if depth:
                self._engine.set_depth(depth)

            max_multipv = min(num_moves, 50)
            self._engine.update_engine_parameters({"MultiPV": max_multipv})

            eval_map = {}
            try:
                for _ in range(num_moves):
                    try:
                        uci = self._engine.get_best_move()
                        if uci in ('0-1', '1-0', 'resign', None, ''):
                            break
                        info = self._engine.get_evaluation()
                        cp = 0
                        if info['type'] == 'cp':
                            cp = info['value']
                        elif info['type'] == 'mate':
                            cp = info['value'] * 10000
                        eval_map[uci] = cp
                    except Exception:
                        break
            finally:
                self._ensure_multipv_1()

            for m in legal_moves:
                uci = m.uci()
                if uci not in eval_map:
                    eval_map[uci] = 0
            return eval_map

        try:
            return self._safe_call(_inner)
        except Exception:
            return {}

    def get_move_quality(self, board, move):
        """Get quality score for a specific move relative to best move."""
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

    def _parse_best_move_eval(self, uci):
        """Extract centipawn evaluation from a UCI best move string."""
        try:
            # Example: "e2e4(cp 15)" or "e2e4(mate 3)"
            if '(' in uci and ')' in uci:
                content = uci[uci.find('('):uci.find(')')+1]
                if 'cp' in content:
                    return int(content.split('cp')[1].split(')')[0].strip())
                elif 'mate' in content:
                    return int(content.split('mate')[1].split(')')[0].strip()) * 10000
        except Exception:
            pass
        return 0

    def get_top_moves(self, board, num_moves=5):
        """Get Stockfish's top N moves with evaluations."""
        def _inner():
            legal_count = len(list(board.legal_moves))
            if legal_count == 0:
                return []
            
            actual_num = min(num_moves, legal_count)
            self._engine.set_fen_position(board.fen())
            self._engine.update_engine_parameters({"MultiPV": actual_num})
            
            moves = []
            try:
                for i in range(actual_num):
                    try:
                        uci = self._engine.get_best_move()
                        if uci in ('0-1', '1-0', 'resign', None, ''):
                            break
                        move = chess.Move.from_uci(uci)
                        san = board.san(move)
                        
                        # Parse evaluation from UCI string
                        eval_cp = self._parse_best_move_eval(uci)
                        
                        # If it's White's turn, the position after the move is Black's turn.
                        # In our class's get_evaluation, if it's Black's turn, we flip.
                        # The UCI string evaluation is for the side-to-move.
                        # If White moves, it's Black's turn, so the UCI eval is from Black's perspective.
                        # We want to flip it to White's perspective.
                        if board.turn == chess.WHITE:
                            eval_cp = -eval_cp
                        
                        moves.append({
                            'uci': uci,
                            'san': san,
                            'evaluation': eval_cp,
                            'depth': self._engine.get_depth() if hasattr(self._engine, 'get_depth') else self._depth,
                            })
                    except Exception:
                        break
            finally:
                self._ensure_multipv_1()
            return moves
        
        try:
            return self._safe_call(_inner)
        except Exception:
            return []

    def analyze_position(self, board):
        """Full position analysis for UI display."""
        def _inner():
            pos_eval = self._engine.get_evaluation()
            val = 0
            if pos_eval['type'] == 'cp':
                val = pos_eval['value']
            elif pos_eval['type'] == 'mate':
                val = pos_eval['value'] * 10000
            if board.turn == chess.BLACK:
                val = -val
            
            top_moves = self.get_top_moves(board, num_moves=5)
            
            legal_moves = list(board.legal_moves)
            eval_map = self.evaluate_legal_moves_batch(board, depth=self._depth)
            
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
                'evaluation': val,
                'evaluation_normalized': max(-1.0, min(1.0, val / 2000.0)),
                'depth': self._engine.get_depth() if hasattr(self._engine, 'get_depth') else self._depth,
                'top_moves': top_moves,
                'move_analysis': move_analysis[:10],
                'num_legal_moves': len(legal_moves),
            }
        
        try:
            return self._safe_call(_inner)
        except Exception:
            return {
                'evaluation': 0,
                'evaluation_normalized': 0,
                'top_moves': [],
                'move_analysis': [],
                'num_legal_moves': 0,
            }

    def play_game(self, opponent_move_fn=None, max_moves=200):
        """Play a complete game, optionally with a custom opponent."""
        board = chess.Board()
        move_sans = []
        
        for _ in range(max_moves):
            if board.is_game_over():
                break
            
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
            except Exception:
                break
        
        # Build PGN
        board2 = chess.Board()
        for san in move_sans:
            try:
                board2.push_san(san)
            except Exception:
                pass
        game = chess.pgn.Game()
        current = game
        for san in move_sans:
            try:
                m = board2.parse_san(san)
                current = current.add_variation(m)
                board2.push(m)
            except Exception:
                pass
        pgn = str(game)
        
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