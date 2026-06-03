# The Fool's Gambit

> A custom chess AI platform — train your own model, let people play against it, and watch it learn in real time.

[![CI](https://github.com/LanceAbuan/TheFoolsGambitPython/actions/workflows/ci.yml/badge.svg)](https://github.com/LanceAbuan/TheFoolsGambitPython/actions/workflows/ci.yml)
[![Deployed on Vercel](https://img.shields.io/badge/Deployed-Vercel-black?style=flat&logo=vercel)](https://gambit.lanceabuan.tech)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## High-Level Goal

This repository has two core objectives:

1. **Host a chess game and train a custom chess model** — The model trains by playing matches against strong engines like Stockfish, using Stockfish's evaluations as a reward signal to guide learning.
2. **Host a website where people can:**
   - **Play against the model** in real time through a web interface
   - **Watch the model train** by observing live matches as it plays against Stockfish

The vision is a single repo containing both the training pipeline and the web app, with everything deployable and playable.

---

## Links

| Resource | URL |
|----------|-----|
| **Live Site** | [https://gambit.lanceabuan.tech](https://gambit.lanceabuan.tech) |
| **GitHub Repo** | [LanceAbuan/TheFoolsGambitPython](https://github.com/LanceAbuan/TheFoolsGambitPython) |
| **Vercel Dashboard** | [vercel.com › thefoolsgambitpython](https://vercel.com/lanceabuans-projects/thefoolsgambitpython) |
| **CI/CD Pipeline** | [GitHub Actions](https://github.com/LanceAbuan/TheFoolsGambitPython/actions) |

---

## Project Overview

The Fool's Gambit is a web-based chess application where players face off against an AI opponent. The frontend renders an interactive board using Chessground, while the backend handles game logic, move validation, and AI decision-making.

Currently the AI uses a **minimax algorithm with alpha-beta pruning** (depth 3), **piece-square tables (PST)** for positional evaluation, and a **mobility bonus** for piece activity. The long-term goal is to replace this with a custom-trained neural network model.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend                          │
│  index.html  │  Chessground  │  jQuery  │  CSS      │
│  (vanilla JS game state management, board rendering) │
└──────────────────────┬──────────────────────────────┘
                       │  POST /api/*  (FEN + move data)
                       ▼
┌─────────────────────────────────────────────────────┐
│                   API Layer                          │
│  api/index.py  (Python serverless on Vercel)        │
│                                                     │
│  Routes:                                            │
│    POST /api/new-game    → start fresh game         │
│    POST /api/make-move   → apply player move        │
│    POST /api/ai-move     → generate AI move         │
│    POST /api/undo        → undo last move           │
└──────────────────────┬──────────────────────────────┘
                       │  imports
                       ▼
┌─────────────────────────────────────────────────────┐
│                 Game Engine                           │
│  game.py  (stateless: FEN + UCI → new state)        │
│  ai.py    (minimax, alpha-beta, PST evaluation)     │
└─────────────────────────────────────────────────────┘
```

### Stateless Design

The frontend maintains all game state (FEN, legal moves, move history). Every API call is independent — no server-side sessions. This enables horizontal scaling and zero cold-start penalty on state restoration.

| Field | Description |
|-------|-------------|
| `fen` | FEN string of the current position |
| `legal` | List of legal moves in UCI format |
| `turn` | `"white"` or `"black"` |
| `pgn` | SAN notation of the last move played |
| `status` | `"active"`, `"checkmate"`, `"stalemate"`, `"draw"` |
| `result` | Human-readable game result (null if active) |
| `in_check` | Whether the side to move is in check |

### Data Flow

1. Player clicks on the board → frontend sends `POST /api/make-move` with UCI move + current FEN + legal moves
2. API validates and applies the move → returns new game state
3. Frontend updates board and triggers `POST /api/ai-move` with FEN + legal moves
4. API runs minimax search → returns new state with AI's move applied
5. Frontend renders AI's move and repeats

---

## Tech Stack

### Frontend
- **HTML/CSS/JS** — vanilla, no framework
- **[Chessground](https://github.com/lichess-org/chessground)** — interactive board widget (from Lichess)
- **jQuery** — AJAX calls and DOM manipulation

### Backend
- **[python-chess](https://github.com/niklasf/python-chess)** — game logic, move validation, FEN/PGN

### AI Engine
- **Minimax with alpha-beta pruning** — depth 3, capture-first move ordering
- **Piece-square tables (PST)** — positional evaluation per piece type
- **Mobility bonus** — rewards positions with more legal moves

### Deployment
- **[Vercel](https://vercel.com)** — Python serverless function for `/api/*`, automatic deploy on push to `main`
- **GitHub Actions** — CI pipeline runs lint + smoke tests on every PR and push

---

## Setup & Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+ (for Vercel CLI)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Running Locally with Flask

```bash
cd TheFoolsGambitPython/backend
pip install flask gunicorn
python -m flask --app app run
```

The app will be available at `http://localhost:5000`.

### Running with Vercel CLI

```bash
npx vercel dev
```

This starts a local preview at `http://localhost:3000` with the full app (frontend + API).

### Running the Python Engine Standalone

```python
import sys
sys.path.insert(0, "TheFoolsGambitPython/backend")
from game import new_game, make_move, ai_move

# Start a new game
state = new_game()
print(state["fen"])

# Play a move
state = make_move(state["fen"], "e2e4")
print(state["legal"])

# Get an AI move
state = ai_move(state["fen"], ai_depth=3)
print(state["pgn"])
```

---

## API Endpoints

All routes are handled by `api/index.py`. The frontend sends FEN + move data with every request.

### `POST /api/new-game`

**Request:** `{ "aiDepth": 3 }` (optional)
**Response:**
```json
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

### `POST /api/make-move`

**Request:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "uci": "e2e4",
  "legal": ["e2e3", "e2e4", ...]
}
```

### `POST /api/ai-move`

**Request:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
  "legal": ["e7e5", "e7e6", ...],
  "aiDepth": 3
}
```

### `POST /api/undo`

**Request:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
}
```

### Response State Object

| Field | Type | Description |
|-------|------|-------------|
| `fen` | string | FEN string of the current position |
| `legal` | string[] | Legal moves in UCI format |
| `turn` | string | `"white"` or `"black"` |
| `pgn` | string | SAN notation of the last move |
| `status` | string | `"active"`, `"checkmate"`, `"stalemate"`, `"draw"` |
| `result` | string \| null | Human-readable game result |
| `in_check` | boolean | Whether the side to move is in check |

---

## Deployment

### Automatic Deploy (Push to `main`)

Every push to `main` triggers a Vercel production deploy. The `vercel.json` rewrites route all `/api/*` requests to `api/index.py`.

```json
{
  "version": 2,
  "github": { "enabled": true },
  "rewrites": [
    { "source": "/api/(.*)", "destination": "/api/index" }
  ]
}
```

### Manual Deploy

```bash
# Preview
vercel deploy

# Production
vercel deploy --prod
```

### CI/CD Pipeline

GitHub Actions runs on every PR and push to `main`:
1. **Lint** — flake8 on all Python files
2. **Smoke tests** — validate game engine (new game, moves, AI, illegal moves)
3. **Flask tests** — verify API endpoints respond correctly

PRs cannot merge if the CI pipeline fails. The `main` branch is protected — no direct pushes allowed.

---

## Future Roadmap

- [x] **Convert API to Python** — replace `api/index.js` with Python serverless (#2, #9)
- [ ] **Custom model training pipeline** — train a neural network by playing against Stockfish (#3)
- [ ] **Live training matches** — let visitors watch the model play against Stockfish in real time (#4)
- [ ] **Model serving** — replace minimax AI with the trained model for online play (#5)
- [ ] **Player matchmaking** — support human vs human games (#6)
- [ ] **Game replay** — full PGN export and move-by-move replay (#7)