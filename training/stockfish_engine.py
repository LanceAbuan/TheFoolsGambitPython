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
import signal
from stockfish import Stockfish as SF

STOCKFISH_PATH = os.environ.get('STOCKFISH_PATH', '/home/lance/.local/bin/stockfish')
SF_DEPTH = 10
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
        # Incremental position cache — keeps SF's TT warm across consecutive moves
        self._cached_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        self._cached_board = chess.Board(self._cached_fen)

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
        def _inner():
            try:
                self._engine.get_evaluation()
                return True
            except Exception:
                return False
        try:
            return self._safe_call(_inner)
        except Exception:
            return False

    def _restart(self):
        """Kill the dead subprocess and spawn a fresh one."""
        with self._lock:
            try:
                # Use a timeout for close() to prevent hanging if the engine is already stuck
                self._call_with_timeout(self._engine.close, restart_on_timeout=False)
            except Exception:
                pass
            self._engine = self._create_engine()
            self._alive = True
            self._cached_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
            self._cached_board = chess.Board(self._cached_fen)
            print('[SF] Engine restarted after crash', flush=True)

    def _sync_engine_to_fen(self, fen):
        """Sync the engine to the target FEN, reusing its TT when possible.

        If the target FEN is reachable by pushing moves from the cached
        position, incrementally play those moves so the transposition
        table stays warm.  Otherwise, do a hard set_fen_position reset.
        """
        if fen == self._cached_fen:
            return  # already at target

        # Quick sanity check
        try:
            target_board = chess.Board(fen)
        except Exception:
            self._engine.set_fen_position(fen)
            self._cached_fen = fen
            self._cached_board = chess.Board(fen)
            return

        # Same side to move — check if positions are equivalent
        if target_board.turn == self._cached_board.turn:
            try:
                if self._cached_board.board_fen() == target_board.board_fen() and \
                   self._cached_board.castling_rights == target_board.castling_rights:
                    self._cached_fen = fen
                    self._cached_board = target_board
                    return
            except Exception:
                pass

        # Incremental: try playing 1-2 moves forward from cached position.
        # This is the common case during live games (opponent just moved,
        # or we jumped 1-2 moves forward in history).
        try:
            temp = self._cached_board.copy()
            moves_to_play = []
            for _ in range(2):
                if temp.board_fen() == target_board.board_fen() and temp.turn == target_board.turn:
                    break
                found = None
                for m in temp.legal_moves:
                    temp.push(m)
                    if temp.board_fen() == target_board.board_fen():
                        found = m
                        break
                    temp.pop()
                if found is None:
                    break
                moves_to_play.append(found)
                temp.push(found)

            if moves_to_play and temp.board_fen() == target_board.board_fen() and \
               temp.turn == target_board.turn:
                for m in moves_to_play:
                    self._engine.set_fen_position(m.uci())
                self._cached_fen = fen
                self._cached_board = target_board
                return
        except Exception:
            pass

        # Fallback: hard reset
        self._engine.set_fen_position(fen)
        self._cached_fen = fen
        self._cached_board = target_board

    def _ensure_multipv_1(self):
        """Reset MultiPV to 1 — critical cleanup after batch operations."""
        with self._lock:
            try:
                self._engine.update_engine_parameters({"MultiPV": 1})
            except Exception:
                pass

    def _safe_call(self, fn, *args, **kwargs):
        """Call *fn* under _lock with automatic crash-recovery and timeout."""
        for attempt in range(_MAX_RESTART + 1):
            acquired = self._lock.acquire(timeout=30.0)
            if not acquired:
                print(f'[SF] TIMEOUT: Failed to acquire Stockfish lock within 30 seconds', flush=True)
                raise TimeoutError("Failed to acquire Stockfish lock within 30 seconds")
            try:
                res = fn(*args, **kwargs)
                return res
            except Exception as e:
                print(f'[SF] ERROR in _safe_call: {e}', flush=True)
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
            self._sync_engine_to_fen(board.fen())
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
            self._sync_engine_to_fen(board.fen())
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

    def evaluate_legal_moves_batch(self, board, depth=11):
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
            self._sync_engine_to_fen(board.fen())
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
        """Get Stockfish's top N moves with evaluations.

        Uses the built-in get_top_moves() from the stockfish package, which
        correctly handles MultiPV internally. The custom loop that called
        get_best_move() in a loop was broken because get_best_move() always
        returns only the #1 move regardless of MultiPV.
        """
        def _inner():
            legal_count = len(list(board.legal_moves))
            if legal_count == 0:
                return []

            actual_num = min(num_moves, legal_count)
            self._sync_engine_to_fen(board.fen())
            self._engine.update_engine_parameters({"MultiPV": actual_num})

            try:
                # Built-in method that correctly returns MultiPV results
                raw_moves = self._engine.get_top_moves(actual_num)
            finally:
                self._ensure_multipv_1()

            moves = []
            for m in raw_moves:
                uci = m['Move']
                if uci in ('0-1', '1-0', 'resign', None, ''):
                    continue
                move = chess.Move.from_uci(uci)
                san = board.san(move)
                eval_cp = m['Centipawn'] if m['Centipawn'] is not None else 0
                if m['Mate'] is not None:
                    eval_cp = m['Mate'] if m['Mate'] > 0 else m['Mate']

                # Flip to White's perspective: Centipawn is from side-to-move's POV
                if board.turn == chess.WHITE:
                    eval_cp = -eval_cp

                moves.append({
                    'uci': uci,
                    'san': san,
                    'evaluation': eval_cp,
                    'depth': self._engine.get_depth() if hasattr(self._engine, 'get_depth') else self._depth,
                })
            return moves

        try:
            return self._safe_call(_inner)
        except Exception:
            return []

    def analyze_position(self, board):
        """Full position analysis for UI display.

        Single search: uses evaluate_legal_moves_batch for all moves,
        then picks top_moves from the same results. No redundant searches.
        """
        def _inner():
            legal_moves = list(board.legal_moves)
            eval_map = self.evaluate_legal_moves_batch(board, depth=self._depth)

            # Position eval = best move eval (from the single search)
            best_cp = max(eval_map.values()) if eval_map else 0
            val = best_cp if board.turn == chess.WHITE else -best_cp

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

            # Top moves come from the same eval_map — no second search needed
            top_moves = []
            seen = set()
            for entry in move_analysis[:5]:
                san = entry['san']
                if san not in seen:
                    seen.add(san)
                    top_moves.append({
                        'san': san,
                        'uci': entry['uci'],
                        'evaluation': entry['evaluation'],
                    })

            return {
                'evaluation': val,
                'evaluation_normalized': max(-1.0, min(1.0, val / 2000.0)),
                'depth': self._depth,
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