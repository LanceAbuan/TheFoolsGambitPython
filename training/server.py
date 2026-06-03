"""Flask API for training management, live SSE streaming, and Stockfish integration."""
import os
import sys
import json
import time
import threading
import chess
from flask import Blueprint, jsonify, request, Response
from datetime import datetime
from huggingface_hub import HfApi, login

from .trainer import Trainer, LOCAL_MODEL_DIR
from .tensorize import board_to_tensor
from .stockfish_engine import StockfishPlayer

training_bp = Blueprint('training', __name__)

trainer = None
training_thread = None
_lock = threading.Lock()
recent_games = []
sse_clients = set()
current_game_moves = []
current_game_status = "idle"


def get_trainer():
    global trainer
    if trainer is None:
        trainer = Trainer()
    return trainer


def get_stockfish():
    try:
        return StockfishPlayer()
    except Exception:
        return None


def send_sse(data, event=None):
    global sse_clients
    if event:
        msg = f"event: {event}\n"
        msg += f"data: {json.dumps(data)}\n\n"
    else:
        msg = f"data: {json.dumps(data)}\n\n"
    to_remove = set()
    for client in sse_clients:
        try:
            client.send(msg)
        except Exception:
            to_remove.add(client)
    for client in to_remove:
        sse_clients.discard(client)


# Thread-safe queue for MCTS progress events
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
        pass  # Drop if dashboard is slow


# Thread-safe queue for MCTS progress events
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
        pass  # Drop if dashboard is slow


def stream_game_progress():
    global current_game_moves, current_game_status
    send_sse({
        'type': 'game_progress',
        'moves': current_game_moves[-50:],
        'status': current_game_status,
        'timestamp': time.time()
    })


def stream_status_update():
    t = get_trainer()
    status = t.get_status()
    status['recent_games'] = recent_games[:5]
    send_sse({
        'type': 'status_update',
        'data': status
    })


@training_bp.route('/api/train/stream')
def sse_stream():
    """SSE endpoint for real-time training updates."""
    client_socket = request.environ.get('werkzeug.socket')
    sse_clients.add(client_socket)
    def generate():
        import time as _time
        try:
            while True:
                yield ''
                _time.sleep(0.5)
        except GeneratorExit:
            pass
        finally:
            sse_clients.discard(client_socket)
    resp = Response(generate(), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    resp.headers['Connection'] = 'keep-alive'
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    return resp


@training_bp.route('/api/train/status')
def train_status():
    t = get_trainer()
    status = t.get_status()
    status['recent_games'] = recent_games[:5]
    status['current_game'] = {
        'moves': current_game_moves[-20:],
        'status': current_game_status
    }
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
    use_stockfish = data.get('use_stockfish', False)

    t = get_trainer()
    t.selfplay.num_mcts_simulations = mcts_sims
    t.running = True
    current_game_status = "starting"

    def run_training():
        try:
            print(f'[TRAIN] Starting training thread: games={games_per_cycle}, steps={steps_per_cycle}, mcts={t.selfplay.num_mcts_simulations}', flush=True)
            while t.running:
                for i in range(games_per_cycle):
                    if not t.running:
                        break
                    global current_game_moves
                    current_game_moves = []
                    current_game_status = "self-play"
                    print(f'[TRAIN] Starting game {i+1}/{games_per_cycle}', flush=True)
                    send_sse({'type': 'game_start', 'mode': 'self-play'})

                    game_data = t.play_game(on_move=lambda moves: (
                        setattr(sys.modules[__name__], 'current_game_moves', list(moves)),
                        stream_game_progress()
                    ))
                    print(f'[TRAIN] Game {i+1} done, moves: {len(game_data.get("moves", []))}', flush=True)
                current_game_moves = game_data.get('moves', [])

                if game_data.get('pgn'):
                    recent_games.append({
                        'game_num': game_data.get('games_played', t.games_played),
                        'pgn': game_data.get('pgn', ''),
                        'result': game_data.get('result', '*'),
                        'mode': 'self-play',
                        'timestamp': time.time()
                    })
                    stream_game_progress()

                # Optional: Stockfish game
                if use_stockfish and (i + 1) % 2 == 0:
                    sf = get_stockfish()
                    if sf:
                        current_game_status = "stockfish"
                        send_sse({'type': 'game_start', 'mode': 'stockfish'})
                        sf_game = sf.play_game(
                            opponent_move_fn=lambda b: t.selfplay.mcts.search(b)
                        )
                        current_game_moves = sf_game.get('moves', [])
                        if sf_game.get('pgn'):
                            recent_games.append({
                                'game_num': len(recent_games) + 1,
                                'pgn': sf_game.get('pgn', ''),
                                'result': sf_game.get('result', '*'),
                                'mode': 'stockfish',
                                'timestamp': time.time()
                            })
                            stream_game_progress()

            # Training steps
            for _ in range(steps_per_cycle):
                if not t.running:
                    break
                current_game_status = "training"
                result = t.train_step()
                if result:
                    send_sse({'type': 'train_step', 'data': result})

            t.save_checkpoint()
            current_game_status = "checkpoint"
            time.sleep(0.1)
        except Exception as e:
            print(f'[TRAIN] ERROR: {e}', flush=True)
            import traceback
            traceback.print_exc()
            current_game_status = "error"

    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()

    return jsonify({"status": "started", "trainer": t.get_status()})


@training_bp.route('/api/train/stop', methods=['POST'])
def train_stop():
    global current_game_status
    t = get_trainer()
    t.running = False
    t.status = "stopped"
    current_game_status = "stopped"
    t.save_checkpoint()
    t.push_checkpoint()
    send_sse({'type': 'training_stopped'})
    return jsonify({"status": "stopped", "trainer": t.get_status()})


@training_bp.route('/api/train/play', methods=['POST'])
def train_play():
    global current_game_moves, current_game_status
    t = get_trainer()
    current_game_moves = []
    current_game_status = "playing"
    send_sse({'type': 'game_start', 'mode': 'self-play'})

    game_data = t.play_game(on_move=lambda moves: (
        setattr(sys.modules[__name__], 'current_game_moves', list(moves)),
        stream_game_progress()
    ))
    current_game_moves = game_data.get('moves', [])

    if game_data.get('pgn'):
        recent_games.append({
            'game_num': game_data.get('games_played', t.games_played),
            'pgn': game_data.get('pgn', ''),
            'result': game_data.get('result', '*'),
            'mode': 'self-play',
            'timestamp': time.time()
        })
        stream_game_progress()

    return jsonify(game_data)


@training_bp.route('/api/train/play-stockfish', methods=['POST'])
def train_play_stockfish():
    global current_game_moves, current_game_status
    sf = get_stockfish()
    if not sf:
        return jsonify({"error": "Stockfish not available"}), 500

    current_game_moves = []
    current_game_status = "stockfish"
    send_sse({'type': 'game_start', 'mode': 'stockfish'})

    t = get_trainer()
    sf_game = sf.play_game(
        opponent_move_fn=lambda b: t.selfplay.mcts.search(b)
    )
    current_game_moves = sf_game.get('moves', [])

    recent_games.append({
        'game_num': len(recent_games) + 1,
        'pgn': sf_game.get('pgn', ''),
        'result': sf_game.get('result', '*'),
        'mode': 'stockfish',
        'timestamp': time.time()
    })
    stream_game_progress()

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
    data = request.get_json(silent=True) or {}
    fen = data.get('fen', chess.START_FEN)

    t = get_trainer()
    board = chess.Board(fen)

    if not board.is_valid():
        return jsonify({"error": "Invalid FEN"}), 400

    # Get NN evaluation
    board_tensor = board_to_tensor(board)
    legal_mask = None
    legal_moves = list(board.legal_moves)
    if legal_moves:
        import torch
        legal_mask = torch.zeros(4096)
        for m in legal_moves:
            legal_mask[m.from_square * 64 + m.to_square] = 1.0

    policy_probs, value = t.model.evaluate(board_tensor, legal_mask)
    policy_probs = policy_probs.detach().numpy()

    top_moves = []
    sorted_indices = policy_probs.argsort()[::-1][:10]
    for idx in sorted_indices:
        from_sq = idx // 64
        to_sq = idx % 64
        move = chess.Move(from_sq, to_sq)
        top_moves.append({
            'move': board.san(move),
            'probability': float(policy_probs[idx]),
            'uci': move.uci()
        })

    eval_str = "Equal"
    if value > 0.3:
        eval_str = f"White +{value:.2f}"
    elif value < -0.3:
        eval_str = f"Black +{abs(value):.2f}"

    # Get Stockfish evaluation if available
    sf_eval = None
    sf = get_stockfish()
    if sf:
        try:
            cp = sf.get_evaluation(board)
            sf_eval = {'centipawns': cp}
        except Exception:
            pass

    return jsonify({
        "nn_value": float(value),
        "nn_evaluation": eval_str,
        "nn_top_moves": top_moves,
        "stockfish": sf_eval,
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
    return jsonify({"status": "reset"})
