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
MAX_EVENTS = 2000          # Keep last N events on disk
MAX_METRIC_POINTS = 100    # Keep last N metric data points for sparklines
SAVE_INTERVAL = 30.0       # Seconds between disk writes
DATA_DIR = "data"


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
