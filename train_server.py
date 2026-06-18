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
import logging
logging.basicConfig(level=logging.INFO)
import sys
import os
import threading

from flask import Flask, jsonify, after_this_request, request, send_from_directory
from training.server import training_bp

app = Flask(__name__, static_folder=None)
app.register_blueprint(training_bp)

REACT_DIST = os.path.join(os.path.dirname(__file__), "client", "dist")

@app.route("/")
def index():
    """Serve the React app."""
    return send_from_directory(REACT_DIST, "index.html")

@app.route("/assets/<path:path>")
def serve_react_assets(path):
    """Vite puts hashed assets under /assets/."""
    return send_from_directory(os.path.join(REACT_DIST, "assets"), path)

@app.route("/static/<path:path>")
def serve_static(path):
    """Serve static files (favicon, chessboard.css) from React build's public/ copy."""
    return send_from_directory(os.path.join(REACT_DIST, "static"), path)

@app.route("/<path:path>")
def static_files(path):
    """Catch-all: try React build first, then project root."""
    import os as _os
    react_path = _os.path.join(REACT_DIST, path)
    if _os.path.exists(react_path):
        return send_from_directory(REACT_DIST, path)
    return send_from_directory(_os.path.dirname(__file__), path)

ALLOWED_ORIGINS = {
    'https://gambit.lanceabuan.tech',
    'https://api.lanceabuan.tech',
    'http://localhost:5001',
    'http://127.0.0.1:5001',
    'http://localhost:5173',   # Vite dev server
}

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin', '')
    # Allow known origins, any *.lanceabuan.tech, any localhost, or non-browser requests
    if not origin:
        response.headers['Access-Control-Allow-Origin'] = '*'
    elif origin in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
    elif '.lanceabuan.tech' in origin:
        response.headers['Access-Control-Allow-Origin'] = origin
    elif origin.startswith('http://localhost'):
        response.headers['Access-Control-Allow-Origin'] = origin
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
            'mcts_simulations': 600,
            'use_stockfish': True
        }, timeout=5)
        print('[AUTO] Training started automatically.')
    except Exception as e:
        print(f'[AUTO] Failed to start training: {e}')


if __name__ == '__main__':
    # Initialize trainer and start side-game processes (must be in __main__
    # so spawned child processes don't re-execute this)
    from training.server import init_and_start
    init_and_start()

    t = threading.Thread(target=auto_start_training, daemon=True)
    t.start()
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
