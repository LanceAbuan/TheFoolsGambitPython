"""Flask API for training management, live SSE streaming, and Stockfish integration."""
import logging
import os
import sys
import json
import time
import threading
import queue
import chess
from datetime import datetime
from collections import OrderedDict
from huggingface_hub import HfApi, login
from flask import Blueprint, jsonify, request, Response

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
torch.backends.cudnn.benchmark = True

from .trainer import Trainer, LOCAL_MODEL_DIR
from .tensorize import board_to_tensor
from .stockfish_engine import StockfishPlayer
from .critic_game import CriticGame

training_bp = Blueprint('training', __name__)

trainer = None
training_thread = None
_lock = threading.Lock()
recent_games = []
sse_event_queue = queue.Queue(maxsize=5000)

# ---- BROADCAST MANAGER ----
class StreamManager:
    """Manages SSE connections and broadcasts events from a queue."""
    def __init__(self, source_queue: queue.Queue):
        self.source_queue = source_queue
        self._client_queues = []
        self.lock = threading.Lock()
        # Start the background broadcaster
        threading.Thread(target=self._broadcast_loop, daemon=True).start()

    def _broadcast_loop(self):
        """Continuously pulls from source queue and pushes to all connected client queues.

        Events may contain a special ``_sse_event`` key that maps to an SSE
        ``event:`` field.  If present, the broadcast formats the message as::

            event: <_sse_event>
            data: <json-without-_sse_event>

        Otherwise it falls back to a plain ``data:`` message.
        """
        while True:
            try:
                event = self.source_queue.get(timeout=1.0)
                # Extract optional SSE event name
                sse_name = event.pop('_sse_event', None)
                # Also strip internal-only keys before serialising
                event.pop('type', None)
                data = json.dumps(event)
                if sse_name:
                    msg = f"event: {sse_name}\ndata: {data}\n\n"
                else:
                    msg = f"data: {data}\n\n"
                with self.lock:
                    dead = []
                    for cq in self._client_queues:
                        try:
                            cq.put_nowait(msg)
                        except queue.Full:
                            dead.append(cq)
                    for dq in dead:
                        self._client_queues.remove(dq)
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f'[STREAM] Broadcast error: {e}')

    def subscribe(self):
        """Return a per-client SSE generator for the Flask route."""
        client_queue = queue.Queue(maxsize=500)
        with self.lock:
            self._client_queues.append(client_queue)

        def generate():
            try:
                while True:
                    try:
                        msg = client_queue.get(timeout=15.0)
                        yield msg.encode('utf-8')
                    except queue.Empty:
                        # Send SSE keepalive comment so the connection stays open
                        yield b": keepalive\n\n"
            except GeneratorExit:
                with self.lock:
                    if client_queue in self._client_queues:
                        self._client_queues.remove(client_queue)
        return generate()

stream_manager = StreamManager(sse_event_queue)

# ---- GAME STATE ----
current_game_moves = []  # legacy: still used for main game status endpoint
current_game_status = "idle"

NUM_GAMES = 10  # 1 main + 9 side games
# Per-game state
game_moves = [[] for _ in range(NUM_GAMES)]
game_fens = ['' for _ in range(NUM_GAMES)]
game_statuses = ['idle' for _ in range(NUM_GAMES)]
game_locks = [threading.Lock() for _ in range(NUM_GAMES)]

# Lock-free snapshots for HTTP status endpoint (written by side games after releasing locks)
side_game_snapshots = [{'status': 'idle', 'moves': []} for _ in range(NUM_GAMES)]

# Per-game SF instances for side games (main game still uses shared _stockfish_instance)
_side_game_sfs = [None] * NUM_GAMES  # index 0 unused; 1-9 for side games

# ---- SINGLE SHARED STOCKFISH (main game) ----
_stockfish_instance = None
_stockfish_lock = threading.Lock()

# Side game execution state
_side_game_running = [False] * NUM_GAMES
_side_game_threads = [None] * NUM_GAMES
_side_game_event_queue = None  # threading.Queue, created in start_side_games
_side_game_event_consumer_thread = None

# ---- EVAL CACHE ----
# LRU cache: FEN -> {eval_cp, eval_norm, top_moves, move_analysis, timestamp}
eval_cache = OrderedDict()
eval_cache_lock = threading.Lock()
EVAL_CACHE_MAX = 2000
EVAL_CACHE_TTL = 600  # seconds

# ---- PERSISTENT STORAGE ----
CACHE_FILE = "eval_cache.json"
_LAST_CACHE_SAVE = 0.0
_CACHE_SAVE_INTERVAL = 30.0  # seconds between disk writes

def load_persistent_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            log.error(f'[CACHE] Error loading cache: {e}')
    return {}

def save_persistent_cache(cache):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        log.error(f'[CACHE] Error saving cache: {e}')

# Initialize persistent cache
persistent_cache = load_persistent_cache()

# ---- Default play mode ----
play_mode = 'critic'

def _classify_move_quality(diff_cp):
    """Classify move quality by centipawn difference from best."""
    if diff_cp <= 5:
        return 'best'
    if diff_cp <= 15:
        return 'good'
    if diff_cp <= 50:
        return 'ok'
    if diff_cp <= 200:
        return 'bad'
    return 'blunder'

def compute_eval_with_stockfish(board):
    """Compute Stockfish evaluation and top 5 moves for a board position.
    Uses the shared Stockfish instance — no subprocess spawning."""
    sf = get_stockfish()
    if sf is None:
        return {
            'eval_cp': 0,
            'eval_norm': 0.0,
            'top_moves': [],
            'move_analysis': [],
        }
    try:
        top_moves = sf.get_top_moves(board, num_moves=5)
        pos_eval = sf.get_evaluation(board)
        eval_norm = max(-1.0, min(1.0, pos_eval / 2000.0))

        # Build move analysis (top 5 best moves)
        best_cp = top_moves[0]['evaluation'] if top_moves else 0
        move_analysis = []
        for m in top_moves:
            diff = best_cp - m['evaluation'] if best_cp >= 0 else m['evaluation'] - best_cp
            diff = abs(diff)
            quality = _classify_move_quality(diff)
            move_analysis.append({
                'san': m['san'],
                'uci': m['uci'],
                'evaluation': m['evaluation'],
                'quality': quality,
                'diff': diff,
            })

        return {
            'eval_cp': pos_eval,
            'eval_norm': eval_norm,
            'top_moves': top_moves,
            'move_analysis': move_analysis,
            'depth': sf.engine.get_depth() if hasattr(sf.engine, 'get_depth') else 12,
        }
    except Exception as e:
        log.error(f'[EVAL] Stockfish eval error: {e}')
        return {
            'eval_cp': 0,
            'eval_norm': 0.0,
            'top_moves': [],
            'move_analysis': [],
        }

def get_cached_eval(fen):
    """Get cached evaluation for a FEN, or None if expired/missing."""
    with eval_cache_lock:
        entry = eval_cache.get(fen)
        if entry and (time.time() - entry['timestamp']) < EVAL_CACHE_TTL:
            return entry

        # Check persistent cache
        if fen in persistent_cache:
            # Note: We don't have a timestamp for persistent entries,
            # so we treat them as valid but could implement a TTL if needed.
            return persistent_cache[fen]

        return None

def set_cached_eval(fen, eval_data):
    """Cache an evaluation result. LRU eviction when full. Throttles disk writes."""
    global _LAST_CACHE_SAVE
    eval_data['timestamp'] = time.time()
    with eval_cache_lock:
        if fen in eval_cache:
            eval_cache.move_to_end(fen)
        else:
            if len(eval_cache) >= EVAL_CACHE_MAX:
                eval_cache.popitem(last=False)
        eval_cache[fen] = eval_data

    # Update persistent cache (in-memory always)
    persistent_cache[fen] = eval_data
    # Throttle disk writes to every _CACHE_SAVE_INTERVAL seconds
    now = time.time()
    if now - _LAST_CACHE_SAVE >= _CACHE_SAVE_INTERVAL:
        save_persistent_cache(persistent_cache)
        _LAST_CACHE_SAVE = now

def get_or_compute_eval(board):
    """Get evaluation from cache or compute it. Always returns fresh data."""
    fen = board.fen()
    cached = get_cached_eval(fen)
    if cached:
        return cached
    result = compute_eval_with_stockfish(board)
    set_cached_eval(fen, result)
    return result

def get_trainer():
    global trainer
    if trainer is None:
        sf = get_stockfish()
        trainer = Trainer(stockfish=sf)
    return trainer

def get_stockfish():
    """Get the single shared Stockfish instance (lazy init)."""
    global _stockfish_instance
    if _stockfish_instance is None:
        with _stockfish_lock:
            if _stockfish_instance is None:
                _stockfish_instance = StockfishPlayer(depth=10, threads=2, hash_mb=256)
                log.info('[SF] Main Stockfish instance created')
    return _stockfish_instance

def _eager_init():
    """Initialize the trainer and all side games on boot."""
    global trainer
    try:
        sf = get_stockfish()
        trainer = Trainer(stockfish=sf)
        log.info('[BOOT] Trainer initialized eagerly')

        # Side game Stockfish instances (None for NN-only)
        for i in range(1, NUM_GAMES):
            _side_game_sfs[i] = None
            log.info(f'[SF] Side game {i} configured for NN-only')

    except Exception as e:
        log.error(f'[BOOT] Eager init failed: {e}')
        import traceback
        traceback.print_exc()

def _side_game_worker(gid, model, num_mcts_simulations, shared_evaluator, event_queue):
    """Entry point for a side-game thread.

    Runs in a daemon thread with its own MCTS tree, sharing a single
    BatchEvaluator (and therefore a single GPU model) with all other
    side games.  The shared evaluator batches NN forward passes across
    all games for maximum GPU throughput.

    Args:
        gid: game index (1-9)
        model: shared ChessNet model on GPU (read-only during inference)
        num_mcts_simulations: MCTS simulations per move
        shared_evaluator: shared BatchEvaluator for all side games
        event_queue: threading.Queue to send events back to main process
    """
    from .selfplay import SelfPlayGame

    log.info(f'[SIDE-GAME {gid}] Thread started')

    # Each game gets its own SelfPlayGame (own MCTS tree) but shares the evaluator
    sp = SelfPlayGame(
        model,
        num_mcts_simulations=num_mcts_simulations,
        max_moves=200,
        stockfish=None,
        use_stockfish=False,
        evaluator=shared_evaluator,
    )

    while True:
        try:
            board = chess.Board()
            log.info(f'[SIDE-GAME {gid}] New game started')

            # Notify frontend that a new game started
            event_queue.put({
                "type": "game_start",
                "game_id": gid,
                "timestamp": time.time(),
            })

            def on_move_callback(moves):
                event_queue.put({
                    "type": "move",
                    "game_id": gid,
                    "moves": moves,
                    "status": "playing",
                    "timestamp": time.time(),
                })

            game_data = sp.play(on_move=on_move_callback)

            # Add a small delay so the game is watchable on the stream


            event_queue.put({
                "type": "finished",
                "game_id": gid,
                "moves": game_data['moves'],
                "result": game_data['result'],
                "status": "finished",
                "timestamp": time.time(),
            })
            log.info(f'[SIDE-GAME {gid}] Game finished: {game_data["result"]}')
            time.sleep(1)

        except Exception as e:
            log.error(f'[SIDE-GAME {gid}] Thread error: {e}')
            import traceback
            traceback.print_exc()
            time.sleep(5)


def _side_game_event_consumer():
    """Background thread that reads side-game events and feeds the SSE queue."""
    while True:
        try:
            event = _side_game_event_queue.get(timeout=1.0)
            etype = event.get("type")
            gid = event["game_id"]

            if etype == "game_start":
                side_game_snapshots[gid] = {"status": "playing", "moves": []}
                try:
                    sse_event_queue.put_nowait({
                        "_sse_event": "game_start",
                        "game_id": gid,
                        "timestamp": event.get("timestamp"),
                    })
                except queue.Full:
                    log.warning('[SSE] Queue full, dropping game_start event')

            elif etype == "move":
                moves = event["moves"]
                side_game_snapshots[gid] = {"status": "playing", "moves": moves}
                # Send as SSE 'game_progress' event
                try:
                    sse_event_queue.put_nowait({
                        "_sse_event": "game_progress",
                        "game_id": gid,
                        "moves": moves,
                        "status": "playing",
                        "timestamp": event.get("timestamp"),
                    })
                except queue.Full:
                    log.warning('[SSE] Queue full, dropping move event')

            elif etype == "finished":
                moves = event["moves"]
                side_game_snapshots[gid] = {"status": "finished", "moves": moves}
                recent_games.append({
                    "result": event["result"],
                    "moves": event["moves"][:20],
                    "timestamp": event["timestamp"],
                    "game_num": gid + 1,
                    "mode": "critic" if event.get("status") == "finished" else "self-play",
                })
                if len(recent_games) > 100:
                    recent_games.pop(0)
                # Send as SSE 'game_progress' event
                try:
                    sse_event_queue.put_nowait({
                        "_sse_event": "game_progress",
                        "game_id": gid,
                        "moves": moves,
                        "result": event["result"],
                        "status": "finished",
                        "timestamp": event.get("timestamp"),
                    })
                except queue.Full:
                    log.warning('[SSE] Queue full, dropping finished event')

        except Exception:
            time.sleep(0.1)


def start_side_games(trainer):
    """Start all side game workers (indices 1-9).

    Creates one shared model and one shared BatchEvaluator on GPU.  Each
    side game runs in its own daemon thread with its own MCTS tree, but
    all games share the same BatchEvaluator — enabling batched GPU
    inference across all games simultaneously.
    """
    from .selfplay import BatchEvaluator

    global _side_game_event_queue, _side_game_event_consumer_thread

    # Shared event queue for all side games → SSE
    _side_game_event_queue = queue.Queue()
    _side_game_event_consumer_thread = threading.Thread(
        target=_side_game_event_consumer, daemon=True
    )
    _side_game_event_consumer_thread.start()

    # One model on GPU, one shared BatchEvaluator for all side games
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = trainer.model  # already on GPU from Trainer init
    shared_evaluator = BatchEvaluator(model, batch_size=512, max_wait_time=0.010)

    num_sims = 400  # per side game (user requested)

    for gid in range(1, NUM_GAMES):
        if _side_game_threads[gid] and _side_game_threads[gid].is_alive():
            continue

        t = threading.Thread(
            target=_side_game_worker,
            args=(gid, model, num_sims, shared_evaluator, _side_game_event_queue),
            daemon=True,
        )
        t.start()
        _side_game_threads[gid] = t
        log.info(f'[SIDE-GAME] Started thread {gid} (sims={num_sims})')

@training_bp.route('/api/train/status')
def train_status():
    """Return training status and live game state."""
    t = get_trainer()
    status = t.get_status()

    # Include all active side games
    side_games_status = []
    for gid in range(1, NUM_GAMES):
        snap = side_game_snapshots[gid]
        if snap['status'] == 'playing':
            side_games_status.append({
                'game_id': gid,
                'status': snap['status'],
                'moves': snap['moves'],
            })

    status['side_games'] = side_games_status
    status['current_game'] = {
        'status': current_game_status,
        'moves': current_game_moves,
    }

    status['recent_games'] = recent_games[-20:]
    return jsonify(status)


@training_bp.route('/api/train/start', methods=['POST'])
def train_start():
    """Start the training loop in a background thread."""
    global training_thread
    t = get_trainer()

    data = request.get_json(silent=True) or {}
    games_per_cycle = data.get('games_per_cycle', 10)
    steps_per_cycle = data.get('steps_per_cycle', 20)
    mcts_sims = data.get('mcts_simulations', 400)
    use_stockfish = data.get('use_stockfish', True)

    if training_thread and training_thread.is_alive():
        return jsonify({"status": "already running"})

    # Reconfigure MCTS simulations if the selfplay engine supports it
    try:
        t.selfplay.num_mcts_simulations = mcts_sims
    except AttributeError:
        pass

    def _training_loop():
        global training_thread
        t.running = True
        try:
            while t.running:
                # Play games
                for _ in range(games_per_cycle):
                    if not t.running:
                        break
                    t.status = "critic" if use_stockfish else "self-play"
                    # Notify frontend that a new main game is starting
                    try:
                        sse_event_queue.put_nowait({
                            "_sse_event": "game_start",
                            "game_id": 0,
                            "timestamp": time.time(),
                        })
                    except queue.Full:
                        pass

                    global current_game_status, current_game_moves
                    current_game_status = "playing"

                    def on_move_callback(moves):
                        try:
                            sse_event_queue.put_nowait({
                                "_sse_event": "game_progress",
                                "game_id": 0,
                                "moves": moves,
                                "status": "playing",
                                "timestamp": time.time(),
                            })
                        except queue.Full:
                            pass

                    game_data = t.play_game(on_move=on_move_callback)
                    result = game_data.get('result', '*')
                    moves = game_data.get('moves', [])

                    current_game_moves = moves
                    recent_games.append({
                        'result': result,
                        'moves': moves[:20],
                        'timestamp': time.time(),
                        'game_num': 1,
                        'mode': 'critic' if use_stockfish else 'self-play',
                    })
                    if len(recent_games) > 100:
                        recent_games.pop(0)
                    # Push game-finished event to SSE
                    try:
                        sse_event_queue.put_nowait({
                            "_sse_event": "game_progress",
                            "game_id": 0,
                            "moves": moves,
                            "result": result,
                            "status": "finished",
                            "timestamp": time.time(),
                        })
                        sse_event_queue.put_nowait({
                            "_sse_event": "status_update",
                            "data": {
                                "status": "playing",
                                "games_played": t.games_played,
                                "result": result,
                                "buffer_size": len(t.buffer),
                                "loss": t.loss,
                                "step": t.step,
                                "policy_loss": t.policy_loss,
                                "value_loss": t.value_loss,
                                "estimated_elo": t.estimate_elo()
                            },
                        })
                    except queue.Full:
                        log.warning('[SSE] Queue full, dropping training events')









                # Train
                for _ in range(steps_per_cycle):
                    if not t.running:
                        break
                    result = t.train_step()
                    if result:
                        try:
                            sse_event_queue.put_nowait({
                                "_sse_event": "status_update",
                                "data": {
                                    "status": "training",
                                    "step": result['step'],
                                    "loss": result['loss'],
                                    "buffer_size": result.get('buffer_size', 0),
                                },
                            })
                        except queue.Full:
                            pass

                t.save_checkpoint()
                t.status = "idle"
                try:
                    sse_event_queue.put_nowait({
                        "_sse_event": "status_update",
                        "data": {
                            "status": "idle",
                            "games_played": t.games_played,
                            "step": t.step,
                        },
                    })
                except queue.Full:
                    pass
        except Exception as e:
            log.error(f'[TRAINING] Loop error: {e}')
            t.status = "error"
        finally:
            t.running = False

    training_thread = threading.Thread(target=_training_loop, daemon=True)
    training_thread.start()
    log.info('[TRAINING] Training loop started')
    return jsonify({"status": "started"})


@training_bp.route('/api/train/stop', methods=['POST'])
def train_stop():
    """Stop the training loop."""
    t = get_trainer()
    t.running = False
    t.status = "stopped"
    log.info('[TRAINING] Training loop stopped')
    return jsonify({"status": "stopped"})


@training_bp.route('/api/train/stream')
def train_stream():
    """SSE stream for real-time game updates."""
    return Response(stream_manager.subscribe(), mimetype='text/event-stream')

@training_bp.route('/api/train/evaluate', methods=['POST'])
def train_evaluate():
    """Quick Stockfish evaluation for the current position (used by frontend)."""
    data = request.get_json(silent=True) or {}
    fen = data.get('fen', chess.STARTING_FEN)

    board = chess.Board(fen)
    if not board.is_valid():
        return jsonify({"error": "Invalid FEN"}), 400

    result = get_or_compute_eval(board)
    return jsonify({
        'stockfish': {
            'centipawns': result.get('eval_cp', 0),
            'depth': result.get('depth', 10),
        },
        'top_moves': result.get('top_moves', []),
        'cached': 'timestamp' in result,
    })

@training_bp.route('/api/train/analyze', methods=['POST'])
def train_analyze():
    """Full Stockfish analysis — served from cache or shared Stockfish (NO temp instance)."""
    data = request.get_json(silent=True) or {}
    fen = data.get('fen', chess.STARTING_FEN)

    board = chess.Board(fen)
    if not board.is_valid():
        return jsonify({"error": "Invalid FEN"}), 400

    # Try cache first
    cached = get_cached_eval(fen)
    if cached:
        return jsonify({
            'evaluation': cached['eval_cp'],
            'evaluation_normalized': cached['eval_norm'],
            'top_moves': cached['top_moves'],
            'move_analysis': cached['move_analysis'],
            'cached': True
        })

    # Compute via shared instance
    result = get_or_compute_eval(board)

    return jsonify({
        'evaluation': result['eval_cp'],
        'evaluation_normalized': result['eval_norm'],
        'top_moves': result['top_moves'],
        'move_analysis': result['move_analysis'],
        'cached': False
    })

@training_bp.route('/api/train/games')
def train_games():
    return jsonify({"games": recent_games[:20]})

@training_bp.route('/api/train/push', methods=['POST'])
def train_push():
    t = get_trainer()
    success = t.push_checkpoint()
    return jsonify({
        "status": "pushed" if success else "failed",
    })

@training_bp.route('/api/train/model')
def download_model():
    model_path = os.path.join(LOCAL_MODEL_DIR, 'checkpoint.pt')
    if not os.path.exists(model_path):
        return jsonify({"error": "No model available"}), 404
    from flask import send_file
    return send_file(model_path, mimetype='application/octet-stream')

@training_bp.route('/api/train/reset', methods=['POST'])
def train_reset():
    global trainer, recent_games, current_game_moves
    import glob
    for f in glob.glob(os.path.join(LOCAL_MODEL_DIR, '*.pt')):
        os.remove(f)
    trainer = None
    recent_games = []
    current_game_moves = []
    # Clear eval cache on reset
    with eval_cache_lock:
        eval_cache.clear()
    return jsonify({"status": "reset"})

def init_and_start(trainer_ref=None):
    """Initialize the trainer and start side games.

    Must be called explicitly (not from module-level code).
    """
    global trainer
    if trainer_ref is not None:
        trainer = trainer_ref
    _eager_init()
    start_side_games(trainer)
