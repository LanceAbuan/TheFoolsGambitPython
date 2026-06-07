"""Flask API for training management, live SSE streaming, and Stockfish integration."""
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

import torch
torch.backends.cudnn.benchmark = False

from .trainer import Trainer, LOCAL_MODEL_DIR
from .tensorize import board_to_tensor
from .stockfish_engine import StockfishPlayer
from .critic_game import CriticGame

training_bp = Blueprint('training', __name__)

trainer = None
training_thread = None
_lock = threading.Lock()
recent_games = []
sse_event_queue = queue.Queue(maxsize=200)
# ---- GAME STATE ----
# Game 0 = main game. Games 1-2 = side games (self-play only, reduced MCTS).
current_game_moves = []  # legacy: still used for main game status endpoint
current_game_status = "idle"

NUM_GAMES = 3  # 1 main + 2 side games
# Per-game state
game_moves = [[] for _ in range(NUM_GAMES)]
game_fens = ['' for _ in range(NUM_GAMES)]
game_statuses = ['idle' for _ in range(NUM_GAMES)]
game_locks = [threading.Lock() for _ in range(NUM_GAMES)]

# Lock-free snapshots for HTTP status endpoint (written by side games after releasing locks)
side_game_snapshots = [{'status': 'idle', 'moves': []} for _ in range(NUM_GAMES)]

# Per-game SF instances for side games (main game still uses shared _stockfish_instance)
_side_game_sfs = [None] * NUM_GAMES  # index 0 unused; 1-2 for side games

# ---- SINGLE SHARED STOCKFISH (main game) ----
_stockfish_instance = None
_stockfish_lock = threading.Lock()

# ---- EVAL CACHE ----
# LRU cache: FEN -> {eval_cp, eval_norm, top_moves, move_analysis, timestamp}
eval_cache = OrderedDict()
eval_cache_lock = threading.Lock()
EVAL_CACHE_MAX = 2000
EVAL_CACHE_TTL = 600  # seconds

# ---- PERSISTENT STORAGE ----
CACHE_FILE = "eval_cache.json"

def load_persistent_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f'[CACHE] Error loading cache: {e}', flush=True)
    return {}

def save_persistent_cache(cache):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        print(f'[CACHE] Error saving cache: {e}', flush=True)

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
        print(f'[EVAL] Stockfish eval error: {e}', flush=True)
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
    """Cache an evaluation result. LRU eviction when full."""
    eval_data['timestamp'] = time.time()
    with eval_cache_lock:
        if fen in eval_cache:
            eval_cache.move_to_end(fen)
        else:
            if len(eval_cache) >= EVAL_CACHE_MAX:
                eval_cache.popitem(last=False)
        eval_cache[fen] = eval_data
    
    # Update persistent cache
    persistent_cache[fen] = eval_data
    save_persistent_cache(persistent_cache)

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
    """Get the single shared Stockfish instance (lazy init).

    If the shared engine has crashed, it is automatically restarted.
    """
    global _stockfish_instance
    with _stockfish_lock:
        if _stockfish_instance is None:
            try:
                _stockfish_instance = StockfishPlayer(depth=10, threads=2, hash_mb=256)
                print('[SF] Shared Stockfish instance created', flush=True)
            except Exception as e:
                print(f'[SF] Failed to create Stockfish: {e}', flush=True)
                return None
        # Health-check: if the engine died, restart it in-place
        if not _stockfish_instance._is_alive():
            try:
                _stockfish_instance._restart()
                print('[SF] Shared Stockfish restarted (was dead)', flush=True)
            except Exception as e:
                print(f'[SF] Failed to restart Stockfish: {e}', flush=True)
                return None
        return _stockfish_instance


# ---- EAGER INIT: trainer + side games live on boot ----
def _eager_init():
    """Initialize the trainer and side game Stockfish instances immediately on server boot."""
    global trainer
    try:
        # Main game Stockfish
        sf = StockfishPlayer(depth=10, threads=2, hash_mb=256)
        _stockfish_instance = sf
        print('[SF] Main Stockfish instance created', flush=True)
        
        trainer = Trainer(stockfish=sf)
        print('[BOOT] Trainer initialized eagerly', flush=True)
        
        # Side game Stockfish instances
        for i in range(1, NUM_GAMES):
            _side_game_sfs[i] = StockfishPlayer(depth=10, threads=2, hash_mb=256)
            print(f'[SF] Side game {i} Stockfish instance created', flush=True)
            
    except Exception as e:
        print(f'[BOOT] Eager init failed: {e}', flush=True)
        import traceback
        traceback.print_exc()


_eager_init()

# Defer side games until after start_side_games is defined
# (Called at bottom of file once all functions are in scope)

# Placeholder — will be set at module bottom after start_side_games is defined
_side_games_started = False

def _start_side_games_on_boot():
    """Called at module bottom once start_side_games is defined."""
    global _side_games_started
    if _side_games_started:
        return
    if trainer is not None:
        try:
            start_side_games(trainer)
            print('[BOOT] Side games started on boot', flush=True)
            _side_games_started = True
        except Exception as e:
            print(f'[BOOT] Side game start failed: {e}', flush=True)

def send_sse(data, event=None):
    # Auto-use data["type"] as event name if not explicitly provided
    if event is None and "type" in data:
        event = data["type"]
    # Push to queue; generator will yield it
    try:
        sse_event_queue.put_nowait({
            "event": event,
            "data": json.dumps(data)
        })
    except queue.Full:
        pass  # Drop if queue full

import queue
mcts_progress_queue = queue.Queue(maxsize=100)

def send_mcts_progress(move_num, sim_count, total_sims, top_moves):
    """Send MCTS search progress to dashboard."""
    try:
        mcts_progress_queue.put_nowait({
            'type': 'mcts_progress',
            'move': move_num,
            'sims': sim_count,
            'total': total_sims,
            'top_moves': top_moves[:5],
            'timestamp': time.time()
        })
    except queue.Full:
        pass

def update_game_state(moves, game_id=0):
    """Update move state for a specific game slot. game_id=0 is main game."""
    global current_game_moves
    with game_locks[game_id]:
        game_moves[game_id] = list(moves)
        # Keep main game state in sync for legacy endpoints
        if game_id == 0:
            current_game_moves = list(moves)
        # Track current FEN
        board = chess.Board()
        for san in moves:
            try:
                board.push_san(san)
            except:
                break
        game_fens[game_id] = board.fen()
    # Write lock-free snapshot for HTTP status endpoint (outside lock)
    if game_id > 0:
        side_game_snapshots[game_id] = {'status': 'playing', 'moves': list(moves)}

def stream_game_progress(game_id=0, timestamp=None):
    """Stream game moves + current position eval via SSE for a specific game."""
    global current_game_moves, current_game_status
    
    with game_locks[game_id]:
        move_sans = list(game_moves[game_id])
        status = game_statuses[game_id]
    
    board = chess.Board()
    for san in move_sans:
        try:
            board.push_san(san)
        except:
            break
    
    fen = board.fen()
    eval_data = get_cached_eval(fen) or compute_eval_with_stockfish(board)
    
    ts_str = f" [{timestamp}]" if timestamp else ""
    
    if game_id == 0:
        send_sse({
            'type': 'game_progress',
            'game_id': game_id,
            'moves': move_sans,
            'status': status,
            'fen': fen,
            'eval': eval_data.get('eval_cp', 0),
            'eval_norm': eval_data.get('eval_norm', 0.0),
            'is_check': board.is_check(),
            'is_checkmate': board.is_checkmate(),
            'is_stalemate': board.is_stalemate(),
            'timestamp': timestamp
        })
    else:
        # Side games: lighter stream (no eval)
        send_sse({
            'type': 'game_progress',
            'game_id': game_id,
            'moves': move_sans,
            'status': status,
            'fen': fen,
            'is_check': board.is_check(),
            'is_checkmate': board.is_checkmate(),
            'is_stalemate': board.is_stalemate(),
            'timestamp': timestamp
        })


def stream_game_progress_main():
    """Backward-compatible stream_game_progress for main game (game_id=0)."""
    global current_game_moves, current_game_status
    
    with _lock:
        move_sans = list(current_game_moves)
        status = current_game_status
    
    board = chess.Board()
    for san in move_sans:
        try:
            board.push_san(san)
        except Exception:
            break
    
    eval_data = get_or_compute_eval(board)
    
    move_qualities = []
    b = chess.Board()
    
    current_pre_eval = get_or_compute_eval(b)
    
    for idx, san in enumerate(move_sans):
        try:
            move = b.parse_san(san)
            b.push(move)
            current_post_eval = get_or_compute_eval(b)
            
            best_eval_before = current_pre_eval['top_moves'][0]['evaluation'] if current_pre_eval['top_moves'] else 0
            diff = abs(best_eval_before - current_post_eval['eval_cp'])
            quality = _classify_move_quality(diff)
            
            move_qualities.append({
                'index': idx,
                'san': san,
                'quality': quality,
                'eval_before': current_pre_eval['eval_cp'],
                'eval_after': current_post_eval['eval_cp'],
                'diff': diff,
            })
            current_pre_eval = current_post_eval
        except Exception:
            break
    
    final_eval = current_post_eval if 'current_post_eval' in locals() else eval_data
    
    send_sse({
        'type': 'game_progress',
        'game_id': 0,
        'moves': move_sans,
        'fen': board.fen(),
        'status': status,
        'eval': {
            'cp': final_eval['eval_cp'],
            'norm': final_eval['eval_norm'],
            'top_moves': final_eval['top_moves'],
            'move_analysis': final_eval['move_analysis'],
        },
        'move_qualities': move_qualities,
        'is_check': board.is_check(),
        'is_checkmate': board.is_checkmate(),
        'is_stalemate': board.is_stalemate(),
        'timestamp': time.time()
    })


def stream_status_update():
    t = get_trainer()
    with _lock:
        status = t.get_status()
        recent_games_snapshot = list(recent_games[:5])
        current_game_moves_snapshot = list(current_game_moves)
        current_game_status_snapshot = current_game_status

    # Use lock-free snapshots for side games (no deadlock)
    side_statuses = {}
    for gid in range(1, NUM_GAMES):
        snap = side_game_snapshots[gid]
        side_statuses[str(gid)] = {
            'status': snap['status'],
            'moves': list(snap['moves'])
        }
    
    # Use current_game_status as the authoritative status source
    status['status'] = current_game_status_snapshot

    status['recent_games'] = recent_games_snapshot
    status['current_game'] = {
        'moves': current_game_moves_snapshot,
        'status': current_game_status_snapshot
    }
    status['side_games'] = side_statuses
    
    board = chess.Board()
    for san in current_game_moves_snapshot:
        try:
            board.push_san(san)
        except Exception:
            break
    eval_data = get_cached_eval(board.fen()) or compute_eval_with_stockfish(board)
    status['current_eval'] = {
        'cp': eval_data['eval_cp'],
        'norm': eval_data['eval_norm'],
        'top_moves': eval_data['top_moves'],
        'move_analysis': eval_data['move_analysis'],
    }
    send_sse({
        'type': 'status_update',
        'data': status
    })

def mcts_select_move(board):
    """Run MCTS search and return a UCI move string."""
    import numpy as np
    from .tensorize import move_to_idx
    t = get_trainer()
    visit_counts = t.selfplay.mcts.search(board)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    probs = visit_counts[[move_to_idx(m) for m in legal_moves]]
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(len(legal_moves)) / len(legal_moves)
    move_idx = np.random.choice(len(legal_moves), p=probs)
    return legal_moves[move_idx].uci()

@training_bp.route("/api/train/stream")
def sse_stream():
    """SSE endpoint for real-time training updates."""
    def generate():
        import time as _time
        while True:
            try:
                item = sse_event_queue.get(timeout=0.5)
                if item.get("event"):
                    yield f"event: {item['event']}\ndata: {item['data']}\n\n"
                else:
                    yield f"data: {item['data']}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Connection"] = "close"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@training_bp.route('/api/train/status')
def train_status():
    t = get_trainer()
    with _lock:
        recent_games_snapshot = list(recent_games[:5])
        current_game_moves_snapshot = list(current_game_moves)
        current_game_status_snapshot = current_game_status
        
    status = t.get_status()
    status['status'] = current_game_status_snapshot
    status['recent_games'] = recent_games_snapshot
    status['current_game'] = {
        'moves': current_game_moves_snapshot,
        'status': current_game_status_snapshot
    }
    
    board = chess.Board()
    for san in current_game_moves_snapshot:
        try:
            board.push_san(san)
        except Exception:
            break
    eval_data = get_cached_eval(board.fen()) or compute_eval_with_stockfish(board)
    status['current_eval'] = {
        'cp': eval_data['eval_cp'],
        'norm': eval_data['eval_norm'],
        'top_moves': eval_data['top_moves'],
        'move_analysis': eval_data['move_analysis'],
    }
    
    try:
        status['gpu_available'] = torch.cuda.is_available()
        if status['gpu_available']:
            status['gpu_name'] = torch.cuda.get_device_name(0)
            status['gpu_memory'] = f"{torch.cuda.memory_allocated(0) / 1024**2:.0f}MB"
    except Exception:
        status['gpu_available'] = False
    
    status['save_interval'] = 'every 2 games'
    status['hf_upload_interval'] = 'every 50 training steps'
    
    # Use lock-free snapshots for side games (no deadlock)
    side_statuses = {}
    for gid in range(1, NUM_GAMES):
        snap = side_game_snapshots[gid]
        side_statuses[str(gid)] = {
            'status': snap['status'],
            'moves': list(snap['moves'])
        }
    status['side_games'] = side_statuses
    
    return jsonify(status)

@training_bp.route('/api/train/start', methods=['POST'])
def train_start():
    global training_thread, current_game_moves, current_game_status
    
    with _lock:
        if training_thread and training_thread.is_alive():
            return jsonify({"error": "Training already running"}), 409
    
    data = request.get_json(silent=True) or {}
    games_per_cycle = data.get('games_per_cycle', 10)
    steps_per_cycle = data.get('steps_per_cycle', 100)
    mcts_sims = data.get('mcts_simulations', 800)
    use_stockfish = data.get('use_stockfish', True)
    
    t = get_trainer()
    t.selfplay.num_mcts_simulations = mcts_sims
    t.running = True
    
    with _lock:
        current_game_status = "starting"

    training_thread = threading.Thread(target=run_training, args=(t, games_per_cycle, steps_per_cycle, use_stockfish), daemon=True)
    training_thread.start()

    return jsonify({"status": "started", "trainer": t.get_status()})

def run_training(t, games_per_cycle, steps_per_cycle, use_stockfish):
    try:
        print(f'[TRAIN] Starting training thread: games={games_per_cycle}, steps={steps_per_cycle}, mcts={t.selfplay.num_mcts_simulations}', flush=True)
        # Start 2 side game workers
        start_side_games(t)
        while t.running:
            for i in range(games_per_cycle):
                if not t.running:
                    break
                
                print(f'[TRAIN] --- Starting game {i+1}/{games_per_cycle} ---', flush=True)
                with _lock:
                    current_game_moves = []
                    current_game_status = "self-play"
                with game_locks[0]:
                    game_moves[0] = []
                    game_statuses[0] = "self-play"
                
                send_sse({'type': 'game_start', 'mode': 'self-play'})
                
                if use_stockfish:
                    sf = get_stockfish()
                    if sf:
                        with _lock:
                            current_game_status = "supervised"
                        print(f'[TRAIN] Entering Critic mode for game {i+1}', flush=True)
                        send_sse({'type': 'game_start', 'mode': 'critic'})
                        sg = CriticGame(t.model, sf, temperature=0.15)
                        game_data = sg.play(on_move=lambda moves: (
                            update_game_state(moves),
                            stream_game_progress()
                        ))
                        t.buffer.add(game_data.get('examples', []))
                        t.games_played += 1
                        t.last_game_pgn = game_data.get('pgn', '')
                        t.last_game_result = game_data.get('result', '*')
                        t.last_game_moves = game_data.get('moves', [])
                    else:
                        print(f'[TRAIN] Warning: Stockfish unavailable, falling back to self-play', flush=True)
                        game_data = t.play_game(on_move=lambda moves: (
                            update_game_state(moves),
                            stream_game_progress()
                        ))
                else:
                    print(f'[TRAIN] Entering self-play mode (no critic) for game {i+1}', flush=True)
                    game_data = t.play_game(on_move=lambda moves: (
                        update_game_state(moves),
                        stream_game_progress()
                    ))
                
                print(f'[TRAIN] Game {i+1} finished. Moves: {len(game_data.get("moves", []))}', flush=True)
                
                with _lock:
                    current_game_moves = game_data.get('moves', [])
                
                if game_data.get('pgn'):
                    with _lock:
                        recent_games.append({
                            'game_num': t.games_played,
                            'pgn': game_data.get('pgn', ''),
                            'result': game_data.get('result', '*'),
                            'mode': game_data.get('mode', 'self-play'),
                            'timestamp': time.time()
                        })
                    stream_game_progress_main()
                    stream_status_update()
                
                if use_stockfish and (i + 1) % 2 == 0:
                    sf = get_stockfish()
                    if sf:
                        print(f'[TRAIN] Entering Stockfish mode for game {i+1}', flush=True)
                        with _lock:
                            current_game_status = "stockfish"
                        send_sse({'type': 'game_start', 'mode': 'stockfish'})
                        sf_game = sf.play_game(
                            opponent_move_fn=mcts_select_move
                        )
                        with _lock:
                            current_game_moves = sf_game.get('moves', [])
                        if sf_game.get('pgn'):
                            with _lock:
                                recent_games.append({
                                    'game_num': len(recent_games) + 1,
                                    'pgn': sf_game.get('pgn', ''),
                                    'result': sf_game.get('result', '*'),
                                    'mode': 'stockfish',
                                    'timestamp': time.time()
                                })
                            stream_game_progress_main()
                            stream_status_update()
                
                if (i + 1) % 2 == 0:
                    with _lock:
                        current_game_status = "training"
                    steps_between = max(10, steps_per_cycle // max(games_per_cycle // 2, 1))
                    print(f'[TRAIN] Running {steps_between} training steps...', flush=True)
                    for _ in range(steps_between):
                        if not t.running:
                            break
                        result = t.train_step()
                        if result:
                            send_sse({'type': 'train_step', 'data': result})
                            stream_status_update()
                    t.save_checkpoint()
                    with _lock:
                        current_game_status = "self-play"
                
            for _ in range(steps_per_cycle):
                if not t.running:
                    break
                with _lock:
                    current_game_status = "training"
                print(f'[TRAIN] Running final training steps ({_+1}/{steps_per_cycle})...', flush=True)
                result = t.train_step()
                if result:
                    send_sse({'type': 'train_step', 'data': result})
            
            if not t.running:
                break
            
            with _lock:
                current_game_status = "training"
            result = t.train_step()
            if result:
                send_sse({'type': 'train_step', 'data': result})
        
        t.save_checkpoint()
        with _lock:
            current_game_status = "checkpoint"
        print('[TRAIN] Checkpoint saved.', flush=True)
        time.sleep(0.1)
    except Exception as e:
        print(f'[TRAIN] ERROR: {e}', flush=True)
        import traceback
        traceback.print_exc()
        with _lock:
            current_game_status = "error"
    finally:
        stop_side_games()
    
# ---- SIDE GAME WORKERS ----
# Each side game runs in its own thread with its own SF instance.
_side_game_threads = [None] * NUM_GAMES
_side_game_running = [False] * NUM_GAMES

def run_side_game(game_id, trainer):
    """Run a self-play game in a side slot, streaming via SSE with game_id."""
    from .selfplay import SelfPlayGame
    
    # Side games use reduced MCTS (75 sims vs 500 main) to preserve main game speed
    side_mcts = max(75, trainer.selfplay.num_mcts_simulations // 8)
    try:
        sp = SelfPlayGame(trainer.model, num_mcts_simulations=side_mcts, stockfish=_side_game_sfs[game_id])
        print(f'[SIDE-GAME {game_id}] Initialized', flush=True)
    except Exception as e:
        print(f'[SIDE-GAME {game_id}] Initialization failed: {e}', flush=True)
        return
    
    try:
        with game_locks[game_id]:
            game_statuses[game_id] = 'self-play'
            game_moves[game_id] = []
        
        send_sse({'type': 'game_start', 'game_id': game_id, 'mode': 'self-play'})
        
        game_data = sp.play(on_move=lambda moves: (
            update_game_state(moves, game_id),
            stream_game_progress(game_id, time.time())
        ))
        print(f'[SIDE-GAME {game_id}] Completed game with {len(game_data.get("moves", []))} moves', flush=True)
        
        # Feed into shared replay buffer
        if game_data.get('examples'):
            trainer.buffer.add(game_data['examples'])
            trainer.games_played += 1
        
        with game_locks[game_id]:
            game_statuses[game_id] = 'idle'
        
        send_sse({'type': 'game_end', 'game_id': game_id, 'result': game_data.get('result', '*')})
    except Exception as e:
        print(f'[SIDE-GAME {game_id}] Runtime error: {e}', flush=True)
        import traceback
        traceback.print_exc()
    finally:
        with game_locks[game_id]:
            game_statuses[game_id] = 'idle'

def side_game_loop(game_id, trainer):
    """Loop: play side games continuously until stopped."""
    while _side_game_running[game_id] and trainer.running:
        run_side_game(game_id, trainer)
        time.sleep(0.5)

def start_side_games(trainer):
    """Start 2 side game workers (indices 1-2)."""
    for gid in range(1, NUM_GAMES):
        if _side_game_threads[gid] and _side_game_threads[gid].is_alive():
            continue
        _side_game_running[gid] = True
        t = threading.Thread(target=side_game_loop, args=(gid, trainer), daemon=True)
        _side_game_threads[gid] = t
        t.start()
        print(f'[SIDE-GAME] Started side game {gid}', flush=True)

def stop_side_games():
    """Stop all side game workers."""
    for gid in range(1, NUM_GAMES):
        _side_game_running[gid] = False
        if _side_game_threads[gid]:
            _side_game_threads[gid].join(timeout=5)
            _side_game_threads[gid] = None



@training_bp.route('/api/train/stop', methods=['POST'])
def train_stop():
    t = get_trainer()
    t.running = False
    t.status = "stopped"
    
    with _lock:
        current_game_status = "stopped"
    
    t.save_checkpoint()
    t.push_checkpoint()
    send_sse({'type': 'training_stopped'})
    return jsonify({"status": "stopped", "trainer": t.get_status()})

@training_bp.route('/api/train/play', methods=['POST'])
def train_play():
    global current_game_moves, current_game_status
    t = get_trainer()
    
    with _lock:
        current_game_moves = []
        current_game_status = "playing"
    
    send_sse({'type': 'game_start', 'mode': 'self-play'})
    
    game_data = t.play_game(on_move=lambda moves: (
        update_game_state(moves),
        stream_game_progress()
    ))
    
    with _lock:
        current_game_moves = game_data.get('moves', [])
        if game_data.get('pgn'):
            recent_games.append({
                'game_num': game_data.get('games_played', t.games_played),
                'pgn': game_data.get('pgn', ''),
                'result': game_data.get('result', '*'),
                'mode': 'self-play',
                'timestamp': time.time()
            })
    
    stream_game_progress_main()
    return jsonify(game_data)

@training_bp.route('/api/train/play-supervised', methods=['POST'])
def train_play_supervised():
    global current_game_moves, current_game_status
    t = get_trainer()
    sf = get_stockfish()
    if not sf:
        return jsonify({"error": "Stockfish not available"}), 500
    
    with _lock:
        current_game_moves = []
        current_game_status = "supervised"
    
    send_sse({'type': 'game_start', 'mode': 'critic'})
    
    sg = CriticGame(t.model, sf, temperature=0.15)
    game_data = sg.play(on_move=lambda moves: (
        update_game_state(moves),
        stream_game_progress()
    ))
    
    with _lock:
        current_game_moves = game_data.get('moves', [])
        t.buffer.add(game_data.get('examples', []))
        t.games_played += 1
        t.last_game_pgn = game_data.get('pgn', '')
        t.last_game_result = game_data.get('result', '*')
        t.last_game_moves = game_data.get('moves', [])
        recent_games.append({
            'game_num': t.games_played,
            'pgn': game_data.get('pgn', ''),
            'result': game_data.get('result', '*'),
            'mode': 'critic',
            'timestamp': time.time()
        })
    
    stream_game_progress_main()

    return jsonify({
        'moves': game_data.get('moves', []),
        'pgn': game_data.get('pgn', ''),
        'result': game_data.get('result', '*'),
        'examples': len(game_data.get('examples', [])),
        'buffer_size': len(t.buffer),
        'games_played': t.games_played,
    })

@training_bp.route('/api/train/play-stockfish', methods=['POST'])
def train_play_stockfish():
    global current_game_moves, current_game_status
    sf = get_stockfish()
    if not sf:
        return jsonify({"error": "Stockfish not available"}), 500
    
    with _lock:
        current_game_moves = []
        current_game_status = "stockfish"
    
    send_sse({'type': 'game_start', 'mode': 'stockfish'})
    
    t = get_trainer()
    sf_game = sf.play_game(
        opponent_move_fn=mcts_select_move
    )
    
    with _lock:
        current_game_moves = sf_game.get('moves', [])
        if sf_game.get('pgn'):
            recent_games.append({
                'game_num': len(recent_games) + 1,
                'pgn': sf_game.get('pgn', ''),
                'result': sf_game.get('result', '*'),
                'mode': 'stockfish',
                'timestamp': time.time()
            })
    
    stream_game_progress_main()
    return jsonify(sf_game)

@training_bp.route('/api/train/step', methods=['POST'])
def train_step():
    t = get_trainer()
    result = t.train_step()
    if result is None:
        return jsonify({"error": "Not enough data in buffer"}), 400
    send_sse({'type': 'train_step', 'data': result})
    return jsonify(result)

@training_bp.route('/api/train/evaluate', methods=['POST'])
def train_evaluate():
    """Evaluate a FEN — served from cache or shared Stockfish (NO temp instance)."""
    data = request.get_json(silent=True) or {}
    fen = data.get('fen', chess.STARTING_FEN)
    
    board = chess.Board(fen)
    if not board.is_valid():
        return jsonify({"error": "Invalid FEN"}), 400
    
    # Try cache first
    cached = get_cached_eval(fen)
    if cached:
        return jsonify({
            "stockfish": {"centipawns": cached['eval_cp']},
            "top_moves": cached['top_moves'],
            "nn_value": cached['eval_norm'],
            "nn_evaluation": f"{'White' if cached['eval_norm'] > 0 else 'Black'} +{abs(cached['eval_norm']):.2f}" if abs(cached['eval_norm']) > 0.1 else "Equal",
            "cached": True
        })
    
    # Compute via shared instance
    result = get_or_compute_eval(board)
    
    return jsonify({
        "stockfish": {"centipawns": result['eval_cp']},
        "top_moves": result['top_moves'],
        "nn_value": result['eval_norm'],
        "nn_evaluation": f"{'White' if result['eval_norm'] > 0 else 'Black'} +{abs(result['eval_norm']):.2f}" if abs(result['eval_norm']) > 0.1 else "Equal",
        "cached": False
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

# Defer side game start until after all functions are defined
_start_side_games_on_boot()
