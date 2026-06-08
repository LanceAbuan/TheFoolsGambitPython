#!/usr/bin/env bash
# Start The Fool's Gambit training server
# Usage: bash start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/TheFoolsGambitPython/backend"

# Load env vars
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi

exec python3 app.py
