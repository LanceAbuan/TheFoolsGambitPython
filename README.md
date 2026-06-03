# The Fool's Gambit

> A custom chess AI platform — train your own model with self-play + Stockfish, deploy it online, and watch it learn in real time.

[![CI](https://github.com/LanceAbuan/TheFoolsGambitPython/actions/workflows/ci.yml/badge.svg)](https://github.com/LanceAbuan/TheFoolsGambitPython/actions/workflows/ci.yml)
[![Deployed on Vercel](https://img.shields.io/badge/Deployed-Vercel-black?style=flat&logo=vercel)](https://gambit.lanceabuan.tech)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Live Demo:** [https://gambit.lanceabuan.tech](https://gambit.lanceabuan.tech)

---

## What This Is

The Fool's Gambit combines **three systems** into one repository:

| Component | Where It Runs | Purpose |
|-----------|---------------|---------|
| **Web App** | Vercel (free) | Play against the AI, watch live training |
| **Training Server** | Local GPU machine | Run self-play games, train neural network |
| **Model Registry** | Hugging Face Hub | Persist trained checkpoints across sessions |

The AI improves by playing chess games against itself (MCTS + neural net) and against Stockfish (engine teacher). Games feed a replay buffer. The neural network trains on that buffer. Checkpoints push to Hugging Face automatically.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                │
│  index.html  │  chessboard.js  │  chess.js  │  jQuery          │
│  Play board + Live training board (SSE) + Training status panel │
└───────────┬───────────────────────────────┬─────────────────────┘
            │ POST /api/*                   │ GET /api/train/*
            │ (game logic + AI moves)       │ (training controls)
            ▼                               ▼
┌──────────────────────┐      ┌──────────────────────────────────┐
│   API (Vercel)       │      │  TRAINING SERVER (Local GPU)     │
│  api/index.js        │      │  training/server.py (Flask)      │
│                      │      │                                  │
│  • New game          │      │  • /api/train/status             │
│  • Make move         │      │  • /api/train/start              │
│  • AI move (minimax) │      │  • /api/train/stop               │
│  • Undo              │      │  • /api/train/play               │
│  • Proxy /api/train* │      │  • /api/train/play-stockfish     │
│    → Cloudflare Tunnel│      │  • /api/train/step               │
│                      │      │  • /api/train/stream (SSE)       │
└──────────┬───────────┘      └──────────┬───────────────────────┘
           │                             │
           │ imports                     │ uses
           ▼                             ▼
┌──────────────────────┐      ┌──────────────────────────────────┐
│  Game Logic          │      │  TRAINING PIPELINE               │
│  (chess.js in Node)  │      │                                  │
└──────────────────────┘      │  trainer.py  → self-play loop    │
                              │  selfplay.py → MCTS engine       │
                              │  model.py    → neural net        │
                              │  tensorize.py → board → tensor   │
                              │  stockfish_engine.py → SF teacher│
                              │                                  │
                              │  Hugging Face Hub                │
                              │  (auto-push every 50 steps)      │
                              └──────────────────────────────────┘
```

### Key Design Decisions

- **Stateless frontend** — sends FEN with every request, no server sessions
- **Vercel handles game API** — serverless, scales to zero, free tier
- **Local GPU trains** — RTX 3090 runs self-play + backprop
- **Cloudflare Tunnel** — exposes local training server to Vercel without port forwarding
- **HF Hub persistence** — checkpoints survive local machine restarts
- **SSE streaming** — live training board updates without WebSockets

---

## Quick Start (3 Minutes)

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Git**
- **GPU (recommended)** — any NVIDIA GPU with CUDA support; RTX 3090 tested

### 1. Clone and Install

```bash
git clone https://github.com/LanceAbuan/TheFoolsGambitPython.git
cd TheFoolsGambitPython
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
pip install stockfish  # Stockfish Python wrapper
```

### 3. Get Stockfish Engine

```bash
# Option A: Download pre-built binary
wget https://stockfishchess.org/files/stockfish_16_linux_x64.bz2
bzip2 -d stockfish_16_linux_x64.bz2
chmod +x stockfish_16_linux_x64
mv stockfish_16_linux_x64 /home/$USER/.local/bin/stockfish

# Option B: Install from package manager
sudo apt install stockfish
```

### 4. Configure Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
HF_TOKEN=hf_your_token_here     # Hugging Face write token
HF_REPO=your_username/model     # HF repo name
MODEL_DIR=~/.chess-models       # Local checkpoint directory
TRAINING_URL=http://localhost:8000  # Local training server URL
STOCKFISH_PATH=~/.local/bin/stockfish  # Path to Stockfish binary
```

#### Getting a Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **Write** scope
3. Optionally create a repo at https://huggingface.co/new for your model

### 5. Start the Training Server

```bash
source .env
python -m training.server
```

The training server runs on `http://localhost:8000`.

### 6. Start the Frontend (Local Dev)

```bash
npx vercel dev
```

Or serve the static files directly:

```bash
npx serve -p 3000
```

Open **http://localhost:3000** in your browser.

---

## Full Setup Guide

### Deploying the Web App on Vercel

**Automatic (via GitHub):** Pushing to `main` triggers a Vercel deployment automatically. Ensure your Vercel project is linked to this GitHub repository.

**Manual:**
```bash
# Install Vercel CLI
npm i -g vercel

# Login and deploy
vercel login
vercel deploy --prod
```

Set these environment variables in your Vercel project settings:
- `TRAINING_URL` — the public URL of your training server (see Cloudflare Tunnel below)

### Deploying the Training Server on Railway

**Automatic:** Link your Railway project to this GitHub repository. Railway will deploy on push to `main` using the `railway.json` config.

**Manual:**
```bash
railway login
railway up
```

Set these environment variables in your Railway project settings:
- `HF_TOKEN` — your Hugging Face write token
- `HF_REPO` — your HF model repo name
- `MODEL_DIR` — local checkpoint directory (default: `~/.chess-models`)
- `STOCKFISH_PATH` — path to Stockfish binary

### Setting Up Cloudflare Tunnel (Local Development Only)

> **Note:** If you deploy the training server on Railway, you don't need a tunnel. Use the Railway URL directly as `TRAINING_URL`.

This exposes your local training server so Vercel can reach it:

```bash
# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared

# Start tunnel (uses .env vars)
./start_tunnel.sh
```

The tunnel prints a URL like `https://random-name.trycloudflare.com`. Set this as `TRAINING_URL` in your Vercel project.

### Starting Training

Via the web UI: click "Start Training" in the training panel.

Via API:
```bash
curl -X POST http://localhost:8000/api/train/start \
  -H "Content-Type: application/json" \
  -d '{"games_per_cycle": 10, "steps_per_cycle": 100, "mcts_simulations": 800, "use_stockfish": true}'
```

Via curl to check status:
```bash
curl http://localhost:8000/api/train/status
```

---

## Project Structure

```
TheFoolsGambitPython/
├── api/
│   └── index.js          # Node.js serverless handler (Vercel)
├── training/
│   ├── server.py         # Flask API for training endpoints + SSE
│   ├── trainer.py        # Training loop, HF push/download, replay buffer
│   ├── selfplay.py       # MCTS engine, self-play game generation
│   ├── model.py          # AlphaZero-style residual neural network
│   ├── tensorize.py      # Board → tensor conversion
│   └── stockfish_engine.py # Stockfish wrapper for evaluation + play
├── static/
│   ├── chessboard.js     # Board rendering library
│   ├── chessboard.css    # Board styles
│   ├── chess.js          # Chess logic library (client-side)
│   ├── jquery.min.js     # jQuery
│   └── img/chesspieces/  # Wikipedia piece images
├── index.html            # Frontend (game board + training panel)
├── start_tunnel.sh       # Launch training server + Cloudflare Tunnel
├── vercel.json           # Vercel routing config
├── requirements.txt      # Python dependencies
├── .env.example          # Template for environment variables
└── .github/
    └── workflows/ci.yml  # CI pipeline (tests + lint)
```

---

## API Reference

### Game Endpoints (Vercel)

#### `POST /api/new-game`
Start a new game.
```json
// Request
{ "aiDepth": 3 }
// Response
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "legal": ["a2a3", "a2a4", "b1a3", ...],
  "turn": "white",
  "pgn": "",
  "status": "active",
  "result": null,
  "in_check": false
}
```

#### `POST /api/make-move`
Apply a player move.
```json
// Request
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "uci": "e2e4",
  "legal": ["e2e3", "e2e4", ...]
}
```

#### `POST /api/ai-move`
Get AI's best move.
```json
// Request
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
  "legal": ["e7e5", "e7e6", ...],
  "aiDepth": 3
}
```

#### `POST /api/undo`
Undo the last move.

### Training Endpoints (Local Server)

#### `GET /api/train/status`
Get training status, metrics, and recent games.

#### `POST /api/train/start`
Start the training loop.
```json
{
  "games_per_cycle": 10,
  "steps_per_cycle": 100,
  "mcts_simulations": 800,
  "use_stockfish": true
}
```

#### `POST /api/train/stop`
Stop training and push latest checkpoint to HF.

#### `POST /api/train/play`
Play a single self-play game.

#### `POST /api/train/play-stockfish`
Play a Stockfish vs neural network game.

#### `GET /api/train/stream`
Server-Sent Events endpoint for real-time training updates.
Events: `game_start`, `game_progress`, `train_step`, `training_stopped`

#### `POST /api/train/step`
Run a single training step.

#### `POST /api/train/evaluate`
Get neural network + Stockfish evaluation for a position.

#### `GET /api/train/games`
List recent training games.

#### `POST /api/train/push`
Manually push checkpoint to Hugging Face.

#### `POST /api/train/reset`
Clear all checkpoints and replay buffer.

---

## Training Pipeline Details

### How It Works

1. **Self-Play**: The neural network plays against itself using MCTS (Monte Carlo Tree Search). Each game produces board states, policy targets, and value targets.
2. **Stockfish Games**: Stockfish plays against the neural network, providing strong expert moves.
3. **Replay Buffer**: All game data is stored in a buffer.
4. **Training Step**: Mini-batches are sampled from the buffer. The network learns to predict both the best move (policy) and the game outcome (value).
5. **Checkpoint**: After each cycle, the model saves locally and pushes to Hugging Face.

### Neural Network Architecture

AlphaZero-style ResNet with:
- 12 input channels (piece types × color)
- 4 residual blocks with 64 filters
- Shared trunk for policy + value heads
- ~2M parameters

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_mcts_simulations` | 800 | MCTS simulations per move |
| `c_puct` | 1.0 | Exploration constant |
| `batch_size` | 64 | Training batch size |
| `lr` | 0.001 | Learning rate |
| `buffer_size` | 50,000 | Max replay buffer size |
| `push_interval` | 50 | Steps between HF pushes |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | Hugging Face write token |
| `HF_REPO` | Yes | — | HF repo (e.g., `user/model`) |
| `MODEL_DIR` | No | `~/.chess-models` | Local checkpoint directory |
| `TRAINING_URL` | No | `http://localhost:8000` | Training server URL |
| `STOCKFISH_PATH` | No | `~/.local/bin/stockfish` | Stockfish binary path |

---

## Running Without GPU

The training server works without GPU but will be slow:
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python -m training.server
```

The web app and game API work perfectly without a GPU — they don't do any training.

---

## Running Without Hugging Face

Training works without HF — checkpoints stay local only:
```bash
# Don't set HF_TOKEN or HF_REPO
python -m training.server
```

Auto-push is skipped. Use `/api/train/push` manually if you later add credentials.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Stockfish not found | Set `STOCKFISH_PATH` or install: `sudo apt install stockfish` |
| CUDA out of memory | Reduce `batch_size` in `trainer.py` |
| SSE not working | Ensure `TRAINING_URL` is correct and tunnel is running |
| HF push fails | Check token has Write scope at huggingface.co/settings/tokens |
| Vercel can't reach training | Verify Cloudflare Tunnel is running; check `TRAINING_URL` |
| Frontend shows "Training unavailable" | Training server not running or `TRAINING_URL` misconfigured |
| Vercel not deploying on merge | Check Vercel dashboard → Project Settings → Git → Deployment Protection; ensure "main" branch is set for production |
| Railway deployment fails | Check Railway logs; verify `railway.json` startCommand works; ensure Python deps are in `requirements.txt` |

---

## License

MIT — see [LICENSE](LICENSE) for details.
