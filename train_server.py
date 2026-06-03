"""Standalone training server.

Run this to start the training API on port 5001:
    python train_server.py

Endpoints:
    GET  /api/train/status     - Training status and metrics
    POST /api/train/start      - Start training loop
    POST /api/train/stop       - Stop training
    POST /api/train/play       - Play single self-play game
    POST /api/train/step       - Run single training step
    POST /api/train/evaluate   - Evaluate a position
    POST /api/train/reset      - Reset training state
"""
import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, jsonify, after_this_request
from training.server import training_bp

app = Flask(__name__)
app.register_blueprint(training_bp)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'https://gambit.lanceabuan.tech'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


def auto_start_training():
    """Auto-start the training loop when server boots."""
    import time as _time
    import requests as _req
    _time.sleep(5)
    try:
        _req.post('http://localhost:5001/api/train/start', json={
            'games_per_cycle': 10,
            'steps_per_cycle': 50,
            'mcts_simulations': 100,
            'use_stockfish': False
        }, timeout=5)
        print('[AUTO] Training started automatically.')
    except Exception as e:
        print(f'[AUTO] Failed to start training: {e}')


if __name__ == '__main__':
    t = threading.Thread(target=auto_start_training, daemon=True)
    t.start()
    app.run(debug=False, host='0.0.0.0', port=5001)
