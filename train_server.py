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

from flask import Flask, jsonify, after_this_request, request
from training.server import training_bp

app = Flask(__name__)
app.register_blueprint(training_bp)

ALLOWED_ORIGINS = {
    'https://gambit.lanceabuan.tech',
    'http://localhost:5001',
    'http://127.0.0.1:5001',
}

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin', '')
    if origin in ALLOWED_ORIGINS or not origin:
        response.headers['Access-Control-Allow-Origin'] = origin if origin else '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, X-Requested-With'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Vary'] = 'Origin'
    return response

@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        resp = app.make_default_options_response()
        return resp


def auto_start_training():
    """Auto-start the training loop when server boots."""
    import time as _time
    import requests as _req
    _time.sleep(5)
    try:
        _req.post('http://localhost:5001/api/train/start', json={
            'games_per_cycle': 20,
            'steps_per_cycle': 50,
            'mcts_simulations': 100,
            'use_stockfish': True
        }, timeout=5)
        print('[AUTO] Training started automatically.')
    except Exception as e:
        print(f'[AUTO] Failed to start training: {e}')


if __name__ == '__main__':
    t = threading.Thread(target=auto_start_training, daemon=True)
    t.start()
    app.run(debug=False, host='0.0.0.0', port=5001)
