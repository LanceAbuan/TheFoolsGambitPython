"""Flask API for training management and live status.

Endpoints:
- GET  /api/train/status     - Training status, metrics, progress
- POST /api/train/start      - Start training loop (self-play + training)
- POST /api/train/stop       - Stop training
- POST /api/train/play       - Play a single self-play game
- POST /api/train/step       - Run a single training step
- POST /api/train/evaluate   - Evaluate a position with the trained model
- GET  /api/train/model      - Download trained model weights

The training server runs alongside the game API. It can be started independently
for training runs, or integrated with the main app.

Usage:
    python training/server.py
    # or as part of the main app
    from training.server import training_bp
    app.register_blueprint(training_bp)
"""
import os
import threading
import json
import time
import chess
from flask import Blueprint, jsonify, request

from .trainer import Trainer, MODEL_DIR
from .tensorize import board_to_tensor

training_bp = Blueprint('training', __name__)

trainer = None
training_thread = None
_lock = threading.Lock()


def get_trainer():
    """Get or create the singleton trainer instance."""
    global trainer
    if trainer is None:
        trainer = Trainer()
    return trainer


@training_bp.route('/api/train/status')
def train_status():
    """Get current training status."""
    t = get_trainer()
    return jsonify(t.get_status())


@training_bp.route('/api/train/start', methods=['POST'])
def train_start():
    """Start the training loop.
    
    Request body (optional):
    {
        "games_per_cycle": 10,
        "steps_per_cycle": 100,
        "mcts_simulations": 800
    }
    """
    global training_thread
    
    with _lock:
        if training_thread and training_thread.is_alive():
            return jsonify({"error": "Training already running"}), 409
    
    data = request.get_json(silent=True) or {}
    games_per_cycle = data.get('games_per_cycle', 10)
    steps_per_cycle = data.get('steps_per_cycle', 100)
    mcts_sims = data.get('mcts_simulations', 800)
    
    t = get_trainer()
    t.selfplay.num_mcts_simulations = mcts_sims
    t.running = True
    
    def run_training():
        while t.running:
            for _ in range(games_per_cycle):
                if not t.running:
                    break
                t.play_game()
            
            for _ in range(steps_per_cycle):
                if not t.running:
                    break
                t.train_step()
            
            t.save_checkpoint()
            time.sleep(0.1)
    
    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
    
    return jsonify({"status": "started", "trainer": t.get_status()})


@training_bp.route('/api/train/stop', methods=['POST'])
def train_stop():
    """Stop the training loop."""
    t = get_trainer()
    t.running = False
    t.status = "stopped"
    t.save_checkpoint()
    
    return jsonify({"status": "stopped", "trainer": t.get_status()})


@training_bp.route('/api/train/play', methods=['POST'])
def train_play():
    """Play a single self-play game and collect training data."""
    t = get_trainer()
    result = t.play_game()
    return jsonify(result)


@training_bp.route('/api/train/step', methods=['POST'])
def train_step():
    """Run a single training step."""
    t = get_trainer()
    result = t.train_step()
    if result is None:
        return jsonify({"error": "Not enough data in buffer"}), 400
    return jsonify(result)


@training_bp.route('/api/train/evaluate', methods=['POST'])
def train_evaluate():
    """Evaluate a board position with the trained model.
    
    Request body:
    {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    }
    
    Returns:
    {
        "value": 0.0,           # evaluation in [-1, 1]
        "policy": {...},        # top moves with probabilities
        "evaluation": "Equal"   # human-readable eval
    }
    """
    data = request.get_json(silent=True) or {}
    fen = data.get('fen', chess.START_FEN)
    
    t = get_trainer()
    board = chess.Board(fen)
    
    if not board.is_valid():
        return jsonify({"error": "Invalid FEN"}), 400
    
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
    
    return jsonify({
        "value": float(value),
        "evaluation": eval_str,
        "top_moves": top_moves,
    })


@training_bp.route('/api/train/model')
def download_model():
    """Download the trained model weights."""
    model_path = os.path.join(MODEL_DIR, 'checkpoint.pt')
    if not os.path.exists(model_path):
        return jsonify({"error": "No model available"}), 404
    
    from flask import send_file
    return send_file(model_path, mimetype='application/octet-stream')


@training_bp.route('/api/train/reset', methods=['POST'])
def train_reset():
    """Reset training: clear buffer, reinitialize model."""
    global trainer
    
    import glob
    for f in glob.glob(os.path.join(MODEL_DIR, '*.pt')):
        os.remove(f)
    
    trainer = None
    return jsonify({"status": "reset"})
