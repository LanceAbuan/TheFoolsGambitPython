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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask
from training.server import training_bp

app = Flask(__name__)
app.register_blueprint(training_bp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
