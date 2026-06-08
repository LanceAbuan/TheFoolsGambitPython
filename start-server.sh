#!/usr/bin/env bash
# Start The Fool's Gambit chess training server in background
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Kill any existing instance
pkill -f 'python3 app.py' 2>/dev/null || true
sleep 1

cd "$SCRIPT_DIR/TheFoolsGambitPython/backend"

export HF_TOKEN="${HF_TOKEN:-}"
export HF_REPO=LanceAbuan/chess-alpha-zero
export MODEL_DIR=/home/lance/.chess-models
export STOCKFISH_PATH=/home/lance/.local/bin/stockfish

nohup python3 app.py >> "$SCRIPT_DIR/server.log" 2>&1 &
echo "Server started (PID: $!)"
