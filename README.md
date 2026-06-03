# The Fool's Gambit

> A custom chess AI platform — train your own model, let people play against it, and watch it learn in real time.

## High-Level Goal

This repository has two core objectives:

1. **Host a chess game and train a custom chess model** — The model trains by playing matches against strong engines like Stockfish, using Stockfish's evaluations as a reward signal to guide learning.
2. **Host a website where people can:**
   - **Play against the model** in real time through a web interface
   - **Watch the model train** by observing live matches as it plays against Stockfish

The vision is a single repo containing both the training pipeline and the web app, with everything deployable and playable.

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
                       │  POST /api/*
                       ▼
┌─────────────────────────────────────────────────────┐
│                   API Layer                          │
│  api/index.js  (Node.js Edge runtime on Vercel)     │
│                                                     │
│  Routes:                                            │
│    POST /api/new-game    → start fresh game         │
│    POST /api/make-move   → apply player move        │
│    POST /api/ai-move     → generate AI move         │
└──────────────────────┬──────────────────────────────┘
                       │  (logic currently duplicated in JS)
                       ▼
┌─────────────────────────────────────────────────────┐
│                 Game Engine                           │
│  engine.py  (stateless: FEN + UCI → new state)      │
│  ai.py      (minimax, alpha-beta, PST evaluation)   │
│                                                     │
│  Python files represent the intended backend —       │
│  currently the logic is ported to JS in api/index.js │
│  for Vercel Edge runtime compatibility.              │
└─────────────────────────────────────────────────────┘
```

### Component Breakdown

| Layer | Files | Description |
|-------|-------|-------------|
| **Frontend** | `index.html`, `static/` | Single-page app with Chessground board, move history, timers, promotion dialog, game-over overlay |
| **API** | `api/index.js` | Vercel Edge function handling all game requests; currently contains the full chess logic in JS |
| **Game Engine** | `engine.py` | Stateless engine: `new_game()`, `play_move(fen, uci)`, `ai_move(fen)` — returns state dict with FEN, legal moves, turn, status |
| **AI** | `ai.py` | `AIMoveGenerator` class with configurable depth; minimax + alpha-beta pruning, PST-based evaluation, capture-first move ordering |
| **Config** | `vercel.json`, `package.json`, `requirements.txt` | Deployment config, Node dependencies (`chess.js`), Python dependencies (`chess`) |

### Data Flow

1. Player clicks on the board → frontend sends `POST /api/make-move` with UCI move + current FEN
2. API validates and applies the move → returns new game state (FEN, legal moves, turn, status)
3. Frontend updates board and triggers `POST /api/ai-move`
4. API runs minimax search → returns new state with AI's move applied
5. Frontend renders AI's move and repeats

---

## Tech Stack

### Frontend
- **HTML/CSS/JS** — vanilla, no framework
- **[Chessground](https://github.com/lichess-org/chessground)** — interactive board widget (from Lichess)
- **jQuery** — AJAX calls and DOM manipulation

### Backend
- **[chess.js](https://github.com/jhlywa/chess.js)** — game logic, move validation, FEN/PGN (Node.js, used in API)
- **[python-chess](https://github.com/niklasf/python-chess)** — same logic in Python (target backend)

### AI Engine
- **Minimax with alpha-beta pruning** — depth 3, capture-first move ordering
- **Piece-square tables (PST)** — positional evaluation per piece type
- **Mobility bonus** — rewards positions with more legal moves

### Deployment
- **[Vercel](https://vercel.com)** — Edge runtime, serverless function for `/api/*`, GitHub integration for deploy on push

---

## Setup & Local Development

### Prerequisites
- Node.js 18+
- Python 3.10+
- [Vercel CLI](https://vercel.com/docs/cli) (optional, for local preview)

### Install Dependencies

```bash
# Node dependencies (chess.js for the API layer)
npm install

# Python dependencies (for engine.py and ai.py)
pip install -r requirements.txt
```

### Running Locally

The API runs as a Vercel Edge function. To preview locally:

```bash
npx vercel dev
```

This starts a local server at `http://localhost:3000` with the full app (frontend + API).

### Running the Python Engine Standalone

The Python modules can be used independently:

```python
from engine import new_game, play_move, ai_move

# Start a new game
state = new_game()
print(state["fen"])

# Play a move
state = play_move(state["fen"], "e2e4")
print(state["legal"])

# Get an AI move
state = ai_move(state["fen"], depth=3)
print(state["pgn"])
```

---

## API Endpoints

All routes are handled by a single Vercel Edge function at `api/index.js`. The frontend routes requests by passing `__path` in the request body.

### `POST /api/new-game`

Starts a fresh game.

**Request:** `{}`
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

Applies a player move to the current position.

**Request:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "uci": "e2e4",
  "legal": ["e2e3", "e2e4", ...]
}
```
**Response:** Same state object as above, with updated FEN and `pgn` set to the SAN of the played move.

### `POST /api/ai-move`

Generates an AI move at the given position.

**Request:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
  "depth": 3
}
```
**Response:** Same state object with the AI's move applied.

### Response State Object

| Field | Type | Description |
|-------|------|-------------|
| `fen` | string | FEN string of the current position |
| `legal` | string[] | List of legal moves in UCI format |
| `turn` | string | `"white"` or `"black"` |
| `pgn` | string | SAN notation of the last move played |
| `status` | string | `"active"`, `"checkmate"`, `"stalemate"`, `"draw"` |
| `result` | string \| null | Human-readable game result (null if active) |
| `in_check` | boolean | Whether the side to move is in check |

---

## Deployment

The project is configured for Vercel with GitHub integration:

```json
{
  "version": 2,
  "github": { "enabled": true },
  "rewrites": [
    { "source": "/api/:path*", "destination": "/api/index" }
  ]
}
```

All `/api/*` requests are routed to `api/index.js` (Edge runtime). Push to `main` to trigger automatic deployment.

---

## Future Roadmap

- [ ] **Convert API to Python** — replace `api/index.js` with a Python serverless function that wires `engine.py` and `ai.py` (see [#2](https://github.com/LanceAbuan/TheFoolsGambitPython/issues/2))
- [ ] **Custom model training pipeline** — train a neural network by playing against Stockfish, using Stockfish evaluations as reward signal
- [ ] **Live training matches** — let visitors watch the model play against Stockfish in real time
- [ ] **Model serving** — replace minimax AI with the trained model for online play
- [ ] **Player matchmaking** — support human vs human games
- [ ] **Game replay** — full PGN export and move-by-move replay
