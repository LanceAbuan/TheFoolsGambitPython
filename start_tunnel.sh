#!/bin/bash
# Start training server + Cloudflare Tunnel
# Set your HF_TOKEN in .env file before running

cd /home/lance/TheFoolsGambitPython

# Load .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=== Chess Training Server ==="
echo "HF Repo: ${HF_REPO:-not set}"
echo "HF Token: ${HF_TOKEN:+set}"
echo "Model Dir: ${MODEL_DIR:-/tmp/chess-models}"

# Start Cloudflare Tunnel in background
echo "Starting Cloudflare Tunnel..."
/home/lance/.local/bin/cloudflared tunnel --url http://localhost:5001 &
TUNNEL_PID=$!
echo "Tunnel PID: $TUNNEL_PID"

# Cleanup on exit
cleanup() {
    echo "Stopping tunnel (PID: $TUNNEL_PID)..."
    kill $TUNNEL_PID 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

sleep 3

echo "Starting training server..."
python3 train_server.py
