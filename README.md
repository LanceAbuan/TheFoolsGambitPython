# Fool's Gambit ♟️🧠

> **An AlphaZero-inspired chess AI that learns entirely through self-play, with a live-training dashboard you can watch in real time.**

[![Python](https://img.shields.io/badge/Python-3.10+-3fb950?logo=python)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-58a6ff?logo=react)](https://react.dev)
[![Mantine](https://img.shields.io/badge/Mantine-7-339af0?logo=mantine)](https://mantine.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-8b949e)](#license)

---

## 🔥 What Is This?

Fool's Gambit is a **chess neural network** that learns by playing against itself — no Grandmaster games, no human data, no supervised learning. It discovers chess entirely from scratch using **Monte Carlo Tree Search** guided by a neural network, the same approach behind DeepMind's AlphaZero.

The twist: **you can watch it train live**. The dashboard streams moves, evaluations, training metrics, and even 9 parallel side games in real time via Server-Sent Events.

**Live at:** [gambit.lanceabuan.tech](https://gambit.lanceabuan.tech)

---

## 🏗️ Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                   Flask Backend (Python)                     │
│  ┌──────────┐  ┌───────────┐  ┌────────┐  ┌────────────┐  │
│  │ Self-Play │  │   MCTS    │  │ Critic │  │  Trainer   │  │
│  │  Engine   │←─│  Search   │←─│  Game   │  │ (PyTorch)  │  │
│  └────┬─────┘  └─────┬─────┘  └────┬───┘  └──────┬─────┘  │
│       │              │              │              │        │
│       └──────────────┴──────────────┴──────────────┘        │
│                          │  BatchEvaluator                   │
│                          ▼  (GPU batching)                   │
│  ┌──────────────────────────────────────────────────┐       │
│  │            ChessNet (NN) — PyTorch                │       │
│  │  2 Residual Blocks → Policy Head + Value Head     │       │
│  └──────────────────────────────────────────────────┘       │
│       │                                                     │
│       ▼ shared StockfishPlayer (depth 10)                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Stockfish — leaf eval blending + ELO calibration │       │
│  └──────────────────────────────────────────────────┘       │
│                                                             │
│  SSE Stream ──────▶ ┌───────────────────────────────────┐  │
│                     │   React Frontend (Vercel)          │  │
│                     │  ┌──────────────────────────────┐  │  │
│                     │  │  Live Board + Eval Bar        │  │  │
│                     │  │  Move List + Analysis Table   │  │  │
│                     │  │  Metrics Dashboard            │  │  │
│                     │  │  9 Side Game Mini-Boards      │  │  │
│                     │  │  Event Log + Recent Games     │  │  │
│                     │  └──────────────────────────────┘  │  │
│                     └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Machine Learning Pipeline

### Neural Network — `ChessNet`

A lightweight **AlphaZero-style** architecture:

| Component | Details |
|-----------|---------|
| **Input** | 16-channel board tensor (8×8×16) — piece positions, castling rights, en passant, side to move |
| **Residual Tower** | 2 blocks × 32 filters × 3×3 convolutions + batch norm + ReLU |
| **Policy Head** | 1×1 conv (4 filters) → FC → 4096 logits (one per from→to square pair) |
| **Value Head** | 1×1 conv (32 filters) → FC(32) → tanh → scalar in [-1, +1] |
| **Parameters** | ~200K (deliberately small — fast enough for real-time training) |

### Self-Play Loop

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Current │────▶│   MCTS   │────▶│  Policy  │────▶│  Play    │
│ Position│     │  Search  │     │ + Value  │     │  Move    │
└─────────┘     └──────────┘     └──────────┘     └────┬─────┘
       ▲                                                │
       └────────────────────────────────────────────────┘
```

1. **MCTS Search** — 500 simulations per move (8 parallel threads)
   - **Selection:** PUCT algorithm — `score = Q(s,a) + cpuct * P(s,a) * sqrt(N) / (1 + n)`
   - **Expansion:** NN generates policy priors for new nodes
   - **Evaluation:** Blend `0.6 × Stockfish(depth=10) + 0.4 × NN value` with Gaussian noise (σ=0.1)
   - **Backpropagation:** Average value back up the tree

2. **Training Example** — Each position records:
   - `board_tensor`: 16×8×8 encoded position
   - `policy`: MCTS visit count distribution (the "improved" policy target)
   - `value`: Game outcome (+1 win, 0 draw, -1 loss)
   - `chosen_eval`: Stockfish evaluation of the chosen move

3. **Training Step** — Sample batch from replay buffer → minimize:
   - `Loss = PolicyCrossEntropy + ValueMSE + L2_regularization`

### Critic Mode (Alternative Training)

In critic mode, the NN picks its own moves while Stockfish evaluates **every legal move** and builds a weighted policy target. This lets the network develop its own style while being guided toward good positions:
- Stockfish evaluates all legal moves via `evaluate_legal_moves_batch`
- Target policy = softmax over shifted evaluations
- NN + small critic bias → move selection

### Training Loop

```
Every cycle:
  1. Play N self-play games (or critic games) — collect positions
  2. Add positions to replay buffer (max 100K)
  3. Sample mini-batches → train NN
  4. Every 50 steps: play 10 games vs Stockfish → estimate ELO
  5. Every 50 steps: push checkpoint to Hugging Face Hub
```

### ELO Calibration

The network plays **10 games against Stockfish** (depth 10) every 50 training steps:
- **Base ELO:** 200
- **Each win:** +400/10 = +40 ELO
- **Estimate:** `ELO = 200 + win_rate × 400 - 200`

---

## 🎯 Move Quality Classification

When Stockfish analyzes a position, it ranks every legal move by centipawn difference from the best move:

| Label | Threshold | Color |
|-------|-----------|-------|
| **Best** | ≤ 5 cp diff | `#3fb950` |
| **Good** | ≤ 15 cp dif | `#58a6ff` |
| **OK** | ≤ 50 cp dif | `#8b949e` |
| **Bad** | ≤ 200 cp dif | `#f0883e` |
| **Blunder** | > 200 cp dif | `#f85149` |

---

## ⚙️ Configuration

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| MCTS Simulations (main) | 500 |
| MCTS Simulations (side) | 350 |
| MCTS Threads | 8 |
| PUCT Constant | 1.0 |
| Dirichlet Noise (ε/α) | 0.25 / 0.03 |
| Stockfish Blend | 0.6 SF / 0.4 NN |
| Stockfish Depth | 10 |
| Batch Size | 64 |
| Learning Rate | 1×10⁻³ |
| Replay Buffer | 100K positions |
| Policy Loss Weight | 1.0 |
| Value Loss Weight | 1.0 |
| L2 Regularization | 1×10⁻⁴ |
| Resign Threshold | -0.8 NN value |
| ELO Calibration Interval | 50 steps |
| Hugging Face Push Interval | 50 steps |

---

## 🖥️ Frontend (React + Mantine)

A dark-themed real-time dashboard:

| Component | Description |
|-----------|-------------|
| **LiveBoard** | Main chess board (react-chessboard) |
| **EvalBar** | White/black evaluation bar — flips with board |
| **PlayerInfoBar** | Player cards with turn indicator |
| **BoardNav** | Move navigation (first/prev/play/next/last) |
| **SideBoard (×9)** | Mini boards for parallel side games |
| **MoveList** | Move history with click-to-navigate |
| **AnalysisTable** | Stockfish analysis of top moves |
| **MetricsGrid** | Training metrics (step, games, loss, ELO) |
| **SSEEventLog** | Real-time event stream |
| **RecentGames** | Finished games list |

Data flows from backend → browser via **Server-Sent Events** (SSE). The frontend updates in real time with no polling.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ with CUDA-capable GPU (optional but recommended)
- Stockfish chess engine (`stockfish` in PATH or set `STOCKFISH_PATH`)
- Node.js 18+

### Backend Setup

```bash
# Clone and install
git clone https://github.com/LanceAbuan/TheFoolsGambitPython
cd TheFoolsGambitPython
pip install -r requirements.txt

# Set environment
export STOCKFISH_PATH=/path/to/stockfish   # optional
export HF_TOKEN=hf_...                      # optional — for model persistence

# Run training server
python training/server.py
```

### Frontend Setup

```bash
cd client
npm install

# Development
npm run dev

# Production build
npm run build
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STOCKFISH_PATH` | auto-detect | Path to Stockfish binary |
| `HF_TOKEN` | — | Hugging Face token for model persistence |
| `HF_REPO` | `LanceAbuan/chess-alpha-zero` | HF repo for checkpoints |
| `MODEL_DIR` | `/tmp/chess-models` | Local model storage |
| `CUDA_VISIBLE_DEVICES` | all | GPU selection |

---

## 📚 Deep Dives

- **MCTS Implementation** → `training/selfplay.py`
- **Critic-Guided Training** → `training/critic_game.py`
- **Stockfish Integration** → `training/stockfish_engine.py`
- **Neural Network Architecture** → `training/model.py`
- **Training Loop & Checkpointing** → `training/trainer.py`
- **Flask API & SSE Stream** → `training/server.py`
- **Board Tensor Encoding** → `training/tensorize.py`

---

## 📄 License

MIT
