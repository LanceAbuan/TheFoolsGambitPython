"""Persistent event log and metric history storage.

Stores events and metrics to disk so new clients can see historical data
and sparklines have real data points.
"""
import json
import os
import time
import threading
import logging
from collections import deque
from typing import List, Dict, Any, Optional

log = logging.getLogger(__name__)

# ---- CONFIG ----
EVENTS_FILE = "data/events.json"
METRICS_FILE = "data/metrics.json"
OPENINGS_FILE = "data/openings.json"
IMPROVEMENTS_FILE = "data/improvements.json"
EVAL_HISTORY_FILE = "data/eval_history.json"
MAX_EVENTS = 2000          # Keep last N events on disk
MAX_METRIC_POINTS = 100    # Keep last N metric data points for sparklines
MAX_OPENING_ENTRIES = 200  # Keep last N games for opening stats
MAX_IMPROVEMENTS = 50      # Keep last N improvement records
MAX_EVAL_HISTORY = 500     # Keep last N per-move evaluations
SAVE_INTERVAL = 30.0       # Seconds between disk writes
DATA_DIR = "data"

# ---- OPENING DETECTION ----
# Maps the first N half-moves (UCI format) to opening names.
# This covers the most common openings encountered in training.
OPENING_BOOK = {
    # King's Pawn
    ("e2e4",): "King's Pawn Opening",
    ("e2e4", "e7e5"): "Open Game",
    ("e2e4", "e7e5", "g1f3"): "King's Knight Opening",
    ("e2e4", "e7e5", "g1f3", "b8c6"): "Open Game (Nc3)",
    ("e2e4", "e7e5", "f2f4"): "King's Gambit",
    ("e2e4", "c7c5"): "Sicilian Defense",
    ("e2e4", "c7c5", "g1f3"): "Sicilian Defense, Open",
    ("e2e4", "c7c5", "g1f3", "d7d6"): "Sicilian Defense, Najdorf",
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4"): "Sicilian, Najdorf (Open)",
    ("e2e4", "c7c5", "g1f3", "b8c6"): "Sicilian Defense, Classical",
    ("e2e4", "c7c5", "g1f3", "e7e6"): "Sicilian Defense, French Variation",
    ("e2e4", "c7c5", "c2c3"): "Sicilian Defense, Alapin",
    ("e2e4", "c7c5", "b1c3"): "Sicilian Defense, Closed",
    ("e2e4", "e7e6"): "French Defense",
    ("e2e4", "e7e6", "d2d4"): "French Defense, Main Line",
    ("e2e4", "e7e6", "d2d4", "d7d5"): "French Defense, Advance/Classical",
    ("e2e4", "c7c6"): "Caro-Kann Defense",
    ("e2e4", "c7c6", "d2d4", "d7d5"): "Caro-Kann, Main Line",
    ("e2e4", "d7d5"): "Scandinavian Defense",
    ("e2e4", "g8f6"): "Alekhine's Defense",
    ("e2e4", "b7b6"): "Owen's Defense",
    ("e2e4", "f7f5"): "Bird's Opening (From Gambit)",
    ("e2e4", "d7d6"): "Pirc Defense",
    ("e2e4", "g7g6"): "Modern Defense",
    # Queen's Pawn
    ("d2d4",): "Queen's Pawn Opening",
    ("d2d4", "d7d5"): "Queen's Gambit",
    ("d2d4", "d7d5", "c2c4"): "Queen's Gambit",
    ("d2d4", "d7d5", "c2c4", "e7e6"): "Queen's Gambit Declined",
    ("d2d4", "d7d5", "c2c4", "c7c6"): "Slav Defense",
    ("d2d4", "d7d5", "c2c4", "d5c4"): "Queen's Gambit Accepted",
    ("d2d4", "g8f6"): "Indian Defense",
    ("d2d4", "g8f6", "c2c4"): "Indian, Main Line",
    ("d2d4", "g8f6", "c2c4", "e7e6"): "Nimzo/Queen's Indian",
    ("d2d4", "g8f6", "c2c4", "g7g6"): "King's Indian Defense",
    ("d2d4", "g8f6", "c2c4", "c7c5"): "Benoni Defense",
    ("d2d4", "f7f5"): "Dutch Defense",
    ("d2d4", "b8c6"): "Nimzowitsch Defense",
    # Flank Openings
    ("c2c4"): "English Opening",
    ("c2c4", "e7e5"): "English, Reversed Sicilian",
    ("c2c4", "c7c5"): "English, Symmetrical",
    ("c2c4", "g8f6"): "English, Indian",
    ("g1f3"): "Reti Opening",
    ("g1f3", "d7d5"): "Reti, Main Line",
    ("b2b3"): "Larsen's Opening",
    ("f2f4"): "Bird's Opening",
}


def detect_opening(moves: list) -> dict:
    """Detect opening name from a list of SAN moves.

    Returns dict with 'name' and 'move_count' (how many moves are book).
    """
    if not moves:
        return {"name": "Unknown", "move_count": 0}

    # Convert SAN to approximate UCI for lookup
    # We use a simplified approach: just match move count
    # For full accuracy you'd need a chess.Board to convert SAN -> UCI
    # But since we only need the first few moves, we can use move count
    for length in range(min(len(moves), 10), 0, -1):
        # Try all opening book entries that match this many moves
        for pattern, name in OPENING_BOOK.items():
            if len(pattern) == length:
                # We can't do exact UCI matching without a board,
                # so we use move count as a heuristic
                pass

    # Simpler approach: use chess.Board to convert and lookup
    try:
        import chess
        board = chess.Board()
        uci_moves = []
        for san in moves[:10]:
            try:
                move = board.push_san(san)
                uci_moves.append(move.uci())
            except Exception:
                break

        # Check longest match first
        for length in range(min(len(uci_moves), 8), 0, -1):
            pattern = tuple(uci_moves[:length])
            if pattern in OPENING_BOOK:
                return {"name": OPENING_BOOK[pattern], "move_count": length}

        return {"name": "Unknown Opening", "move_count": 0}
    except Exception:
        return {"name": "Unknown Opening", "move_count": 0}


class EvalHistory:
    """Tracks per-move evaluation history for the current game."""

    def __init__(self, filepath: str = EVAL_HISTORY_FILE, max_entries: int = MAX_EVAL_HISTORY):
        self.filepath = filepath
        self.max_entries = max_entries
        self._current_game: List[Dict[str, Any]] = []
        self._all_games: List[List[Dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._last_save = 0.0
        self._load()

    def _load(self):
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                self._all_games = data.get('games', [])
        except Exception as e:
            log.error(f'[EVAL_HISTORY] Error loading: {e}')

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump({'games': self._all_games[-50:]}, f)
            self._last_save = time.time()
        except Exception as e:
            log.error(f'[EVAL_HISTORY] Error saving: {e}')

    def add_point(self, move_num: int, san: str, eval_cp: int, eval_norm: float, quality: str = 'ok'):
        """Record an evaluation for a specific move."""
        with self._lock:
            self._current_game.append({
                'move_num': move_num,
                'san': san,
                'eval_cp': eval_cp,
                'eval_norm': eval_norm,
                'quality': quality,
            })
            # Throttled save
            now = time.time()
            if now - self._last_save >= SAVE_INTERVAL:
                self._save()

    def end_game(self):
        """Mark current game as finished and start tracking a new one."""
        with self._lock:
            if self._current_game:
                self._all_games.append(self._current_game)
                if len(self._all_games) > 50:
                    self._all_games = self._all_games[-50:]
            self._current_game = []
            self._save()

    def get_current_game(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._current_game.copy()

    def get_last_game(self) -> List[Dict[str, Any]]:
        with self._lock:
            if self._all_games:
                return self._all_games[-1].copy()
            return self._current_game.copy()


class OpeningTracker:
    """Tracks opening frequency across games."""

    def __init__(self, filepath: str = OPENINGS_FILE):
        self.filepath = filepath
        self._openings: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._last_save = 0.0
        self._load()

    def _load(self):
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    self._openings = json.load(f)
        except Exception as e:
            log.error(f'[OPENINGS] Error loading: {e}')

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump(self._openings, f)
            self._last_save = time.time()
        except Exception as e:
            log.error(f'[OPENINGS] Error saving: {e}')

    def record_game(self, moves: list, result: str, is_white_win: bool):
        """Record the opening used in a completed game."""
        opening = detect_opening(moves)
        name = opening['name']
        with self._lock:
            if name not in self._openings:
                self._openings[name] = {'count': 0, 'white_wins': 0, 'draws': 0, 'black_wins': 0}
            self._openings[name]['count'] += 1
            if result == '1/2-1/2':
                self._openings[name]['draws'] += 1
            elif result == '1-0':
                self._openings[name]['white_wins'] += 1
            else:
                self._openings[name]['black_wins'] += 1

            # Throttled save
            now = time.time()
            if now - self._last_save >= SAVE_INTERVAL:
                self._save()

    def get_top_openings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most frequently played openings with stats."""
        with self._lock:
            items = []
            for name, data in self._openings.items():
                total = data['count']
                if total == 0:
                    continue
                nn_win_rate = round((data['white_wins'] + data['black_wins']) / total * 100, 1)
                items.append({
                    'name': name,
                    'count': total,
                    'nn_win_rate': nn_win_rate,
                    'white_wins': data['white_wins'],
                    'draws': data['draws'],
                    'black_wins': data['black_wins'],
                })
            items.sort(key=lambda x: x['count'], reverse=True)
            return items[:limit]


class ImprovementTracker:
    """Tracks improvements over time (ELO, win rate changes)."""

    def __init__(self, filepath: str = IMPROVEMENTS_FILE):
        self.filepath = filepath
        self._improvements: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_save = 0.0
        self._load()

    def _load(self):
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    self._improvements = json.load(f)
        except Exception as e:
            log.error(f'[IMPROVEMENTS] Error loading: {e}')

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump(self._improvements[-MAX_IMPROVEMENTS:], f)
            self._last_save = time.time()
        except Exception as e:
            log.error(f'[IMPROVEMENTS] Error saving: {e}')

    def record(self, improvement: Dict[str, Any]):
        """Record an improvement event."""
        improvement['timestamp'] = time.time()
        with self._lock:
            self._improvements.append(improvement)
            if len(self._improvements) > MAX_IMPROVEMENTS:
                self._improvements = self._improvements[-MAX_IMPROVEMENTS:]
            now = time.time()
            if now - self._last_save >= SAVE_INTERVAL:
                self._save()

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return self._improvements[-limit:][::-1]  # newest first

    def check_and_record(self, prev_elo: float, curr_elo: float, prev_win_rate: float, curr_win_rate: float, prev_loss: float, curr_loss: float):
        """Compare previous and current metrics, record any improvements."""
        improvements = []
        if curr_elo > prev_elo:
            delta = curr_elo - prev_elo
            improvements.append({
                'text': f'+{delta} ELO vs Stockfish',
                'type': 'elo',
                'delta': delta,
            })
        if curr_win_rate > prev_win_rate:
            delta = curr_win_rate - prev_win_rate
            improvements.append({
                'text': f'+{delta:.1f}% Win Rate vs Last Checkpoint',
                'type': 'win_rate',
                'delta': delta,
            })
        if prev_loss > 0 and curr_loss < prev_loss:
            delta = prev_loss - curr_loss
            improvements.append({
                'text': f'-{delta:.3f} Policy Loss',
                'type': 'loss',
                'delta': delta,
            })
        for imp in improvements:
            self.record(imp)


class EventLog:
    """Persistent event log that survives server restarts."""

    def __init__(self, filepath: str = EVENTS_FILE, max_events: int = MAX_EVENTS):
        self.filepath = filepath
        self.max_events = max_events
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_save = 0.0
        self._load()

    def _load(self):
        """Load events from disk."""
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    self._events = json.load(f)
                log.info(f'[EVENTS] Loaded {len(self._events)} events from disk')
            else:
                self._events = []
        except Exception as e:
            log.error(f'[EVENTS] Error loading events: {e}')
            self._events = []

    def _save(self):
        """Save events to disk."""
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump(self._events[-self.max_events:], f)
            self._last_save = time.time()
        except Exception as e:
            log.error(f'[EVENTS] Error saving events: {e}')

    def add(self, event: Dict[str, Any]):
        """Add an event to the log and optionally persist to disk."""
        with self._lock:
            self._events.append(event)
            # Trim old events
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]

            # Throttled disk write
            now = time.time()
            if now - self._last_save >= SAVE_INTERVAL:
                self._save()

    def get_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all events (newest last)."""
        with self._lock:
            events = self._events.copy()
        if limit:
            events = events[-limit:]
        return events

    def flush(self):
        """Force save to disk."""
        with self._lock:
            self._save()


class MetricHistory:
    """Tracks historical metric values for sparkline charts."""

    def __init__(self, filepath: str = METRICS_FILE, max_points: int = MAX_METRIC_POINTS):
        self.filepath = filepath
        self.max_points = max_points
        self._history: Dict[str, List[Dict[str, Any]]] = {
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'elo': [],
            'buffer_size': [],
            'games_played': [],
        }
        self._lock = threading.Lock()
        self._last_save = 0.0
        self._load()

    def _load(self):
        """Load metric history from disk."""
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                self._history.update(data)
                log.info(f'[METRICS] Loaded metric history from disk')
            else:
                # Initialize empty
                pass
        except Exception as e:
            log.error(f'[METRICS] Error loading metrics: {e}')

    def _save(self):
        """Save metric history to disk."""
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump(self._history, f)
            self._last_save = time.time()
        except Exception as e:
            log.error(f'[METRICS] Error saving metrics: {e}')

    def record(self, metrics: Dict[str, Any]):
        """Record a set of metrics with a timestamp."""
        timestamp = time.time()
        with self._lock:
            for key, value in metrics.items():
                if key in self._history and value is not None:
                    try:
                        self._history[key].append({
                            't': timestamp,
                            'v': float(value),
                        })
                        # Trim old points
                        if len(self._history[key]) > self.max_points:
                            self._history[key] = self._history[key][-self.max_points:]
                    except (TypeError, ValueError):
                        pass

            # Throttled disk write
            now = time.time()
            if now - self._last_save >= SAVE_INTERVAL:
                self._save()

    def get_history(self, key: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get history for a specific metric."""
        with self._lock:
            data = self._history.get(key, []).copy()
        if limit:
            data = data[-limit:]
        return data

    def get_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metric histories."""
        with self._lock:
            return {k: v.copy() for k, v in self._history.items()}

    def flush(self):
        """Force save to disk."""
        with self._lock:
            self._save()


def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage (CPU, RAM, GPU if available)."""
    import psutil

    resources = {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'cpu_count': psutil.cpu_count(),
        'ram': {
            'total_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'used_gb': round(psutil.virtual_memory().used / (1024**3), 1),
            'percent': psutil.virtual_memory().percent,
        },
    }

    # Try to get GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            gpu_mem_used = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            resources['gpu'] = {
                'name': gpu_name,
                'memory_total_gb': round(gpu_mem_total, 1),
                'memory_used_gb': round(gpu_mem_used, 1),
                'memory_reserved_gb': round(gpu_mem_reserved, 1),
                'memory_percent': round(gpu_mem_used / gpu_mem_total * 100, 1) if gpu_mem_total > 0 else 0,
            }
    except Exception:
        pass

    # Try nvidia-smi for GPU utilization
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            gpu_util = float(result.stdout.strip())
            if 'gpu' not in resources:
                resources['gpu'] = {}
            resources['gpu']['utilization_percent'] = gpu_util
    except Exception:
        pass

    return resources
