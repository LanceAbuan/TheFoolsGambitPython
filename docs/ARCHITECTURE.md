# Architecture Deep Dive

> *Everything you want to know about how Fool's Gambit works, from the neural network to the dashboard.*

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Neural Network Design](#2-neural-network-design)
3. [Board Encoding](#3-board-encoding)
4. [Monte Carlo Tree Search](#4-monte-carlo-tree-search)
5. [Stockfish Integration](#5-stockfish-integration)
6. [Training Pipeline](#6-training-pipeline)
7. [Critic-Guided Training](#7-critic-guided-training)
8. [Scoring & Evaluation](#8-scoring--evaluation)
9. [Side Games & Parallelism](#9-side-games--parallelism)
10. [SSE Event Stream](#10-sse-event-stream)
11. [Frontend Architecture](#11-frontend-architecture)
12. [ELO Estimation](#12-elo-estimation)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Flask Process (Python)                        │
│                                                                      │
│  ┌──────────────┐     ┌─────────────────┐     ┌─────────────────┐  │
│  │ Main Training │     │ Side Game Worker │     │ Side Game Worker │  │
│  │ Thread        │     │ 1 (thread)       │...  │ 9 (thread)       │  │
│  │ (trainer.py)  │     │ (server.py:311)  │     │ (server.py:311)  │  │
│  └──────┬───────┘     └────────┬────────┘     └────────┬────────┘  │
│         │                      │                       │            │
│         └──────────────────────┴───────────────────────┘            │
│                                │                                    │
│                        ┌───────▼────────┐                           │
│                        │  BatchEvaluator │ ← shared across all      │
│                        │ (selfplay.py:36)│   games for GPU batching │
│                        └───────┬────────┘                           │
│                                │                                    │
│                        ┌───────▼────────┐                           │
│                        │    ChessNet     │ ← single model instance  │
│                        │  (model.py:44)  │                           │
│                        └────────────────┘                           │
│                                                                      │
│  ┌────────────────┐    ┌─────────────────┐                          │
│  │ StockfishPlayer│    │ SSE Event Queue  │ → HTTP SSE clients      │
│  │ (shared, depth │    │ (server.py:32)   │                          │
│  │  10, thread-   │    └─────────────────┘                          │
│  │  safe wrapper) │                                                 │
│  └────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### The Two Training Modes

The system operates in one of two mutually exclusive modes:

| Mode | Description | When |
|------|-------------|------|
| **Self-Play** | NN + MCTS plays both sides. Training data = MCTS visit counts as policy targets. | Default for pure RL |
| **Critic** | NN picks moves, Stockfish evaluates all legal moves to build policy targets. | User-specified via API |

The mode is set by the `play_mode` variable in `server.py` (line 170), toggled via `POST /api/train/mode`.

---

## 2. Neural Network Design

### ChessNet Architecture (`training/model.py`)

```
Input: 8×8×16 board tensor
│
├─ Conv2D(16 → 32, 3×3) + BatchNorm + ReLU
│
├─ ResidualBlock × 2
│   ├─ Conv2D(32 → 32, 3×3) + BatchNorm + ReLU
│   ├─ Conv2D(32 → 32, 3×3) + BatchNorm
│   └─ + skip connection + ReLU
│
├─ Policy Head                          ├─ Value Head
│  ├─ Conv2D(32 → 4, 1×1) + BN + ReLU  │  ├─ Conv2D(32 → 32, 1×1) + BN + ReLU
│  ├─ Flatten (4×64 = 256)              │  ├─ Flatten (32×64 = 2048)
│  └─ FC(256 → 4096)                    │  ├─ FC(2048 → 32) + ReLU
│                                       │  └─ FC(32 → 1) + tanh
└───────────────────────────────────────┴────────────────────────────
```

**Why so small?** — 2 residual blocks × 32 filters (~200K parameters) is deliberately tiny compared to AlphaZero's 20 blocks × 256 filters. This is intentional:
- **Real-time training**: The model can run hundreds of forward passes per second on a single GPU
- **Interactive demo**: Users can see the network improve in minutes, not days
- **Side games**: 9 parallel games share the same model via `BatchEvaluator`

### Forward Pass

```python
# model.py:72-88
def forward(self, x):
    x = self.input_bn(self.input_conv(x))
    x = self.residual_tower(x)        # 2 residual blocks
    
    policy = self.policy_fc(self.policy_bn(self.policy_conv(x)))
    value = torch.tanh(self.value_fc2(self.value_fc1(self.value_bn(self.value_conv(x)))))
    
    return policy, value  # (batch, 4096), (batch, 1)
```

The policy output is **raw logits** over 4096 possible moves (64 from-squares × 64 to-squares). Illegal moves are masked before softmax.

The value output is **tanh-activated** to [-1, +1]:
- `+1.0` = position is winning for the side to move
- `-1.0` = position is losing for the side to move
- `0.0` = equal

---

## 3. Board Encoding

The board is encoded as a **16-channel 8×8 tensor** (`training/tensorize.py`):

| Channels | Content |
|----------|---------|
| 0 | White pawns |
| 1 | White knights |
| 2 | White bishops |
| 3 | White rooks |
| 4 | White queens |
| 5 | White king |
| 6 | Black pawns |
| 7 | Black knights |
| 8 | Black bishops |
| 9 | Black rooks |
| 10 | Black queens |
| 11 | Black king |
| 12 | En passant target square (all zeros if none) |
| 13 | Castling rights — white kingside |
| 14 | Castling rights — white queenside |
| 15 | Side to move (+1 for white, 0 for black, broadcast) |

This encoding lets the network learn piece relationships, king safety, and tactical patterns directly from the spatial structure — no feature engineering required.

---

## 4. Monte Carlo Tree Search

### Core Algorithm (`training/selfplay.py:150`)

MCTS balances **exploration vs exploitation** to find strong moves without evaluating every branch:

```
function MCTS_SEARCH(board, num_simulations):
    root = BUILD_ROOT(board)
    for _ in range(num_simulations):
        node = root
        // 1. SELECT — traverse to unexplored leaf
        while node is expanded:
            node = argmax(Q + cpuct * P * sqrt(N) / (1 + n))
            board.push(node.move)
        
        // 2. EXPAND — create children for legal moves
        if not terminal(node):
            EXPAND(node, board)
        
        // 3. EVALUATE — NN forward pass (+ optional Stockfish blend)
        value = EVALUATE(board)
        
        // 4. BACKPROPAGATE — update stats up the tree
        for each node in path:
            node.n += 1
            node.Q = (node.Q * (n-1) + value) / n
            value = -value
    
    return root.visit_counts  // improved policy
```

### Key Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `cpuct` | 1.0 | Exploration constant — higher = more exploration |
| `noise_epsilon` | 0.25 | Dirichlet noise mix — prevents policy overconfidence |
| `noise_alpha` | 0.03 | Dirichlet concentration — lower = sparser noise |
| `max_depth` | 15 | Max plies per simulation |
| `NUM_MCTS_THREADS` | 8 | Parallel simulations per search |

### Parallel MCTS

Simulations are split across 8 threads using `ThreadPoolExecutor`:

```python
# selfplay.py:171-185
sims_per_thread = max(1, num_simulations // NUM_MCTS_THREADS)

def _worker(sims):
    for _ in range(sims):
        self._simulate(root, board.copy(), max_depth=15)

with ThreadPoolExecutor(max_workers=NUM_MCTS_THREADS) as executor:
    futures = [executor.submit(_worker, n) for n in ...]
    for f in futures:
        f.result()
```

Each thread operates on a **copy of the board** and all tree mutations are protected by `self._tree_lock`. The NN evaluation (the slow part) runs lock-free.

### PUCT Selection

```
score = Q(s,a) + cpuct * P(s,a) * sqrt(N_parent) / (1 + n_child)
```

Where:
- `Q(s,a)` = average game outcome from taking action `a` in state `s` (exploitation)
- `P(s,a)` = neural network's prior probability of action `a` (knowledge)
- `N_parent` = visit count of the parent node
- `n_child` = visit count of this child node
- `cpuct` = exploration constant

The formula ensures that promising moves (high Q or high P) are explored first, while under-visited moves (low n) get a bonus.

---

## 5. Stockfish Integration

### Architecture (`training/stockfish_engine.py`)

Stockfish runs as a **single shared subprocess** with thread-safe access:

```python
class StockfishPlayer:
    def __init__(self):
        self._lock = threading.RLock()
        self._engine = SF(path=STOCKFISH_PATH, depth=10, ...)
```

All public methods acquire `_lock` before touching the engine. This prevents race conditions when the training thread, MCTS, and HTTP request handlers all share one Stockfish instance.

**Crash Recovery:** If Stockfish dies (OOM, segfault), the wrapper auto-restarts the subprocess and retries once before returning a safe fallback (centipawn 0).

### Stockfish-Guided Rollouts

In MCTS, leaf nodes are evaluated by blending Stockfish with the NN:

```python
# selfplay.py:~350
sf_value = stockfish.get_evaluation(board) / 2000.0  # normalize to [-1, 1]
sf_value += np.random.normal(0, SF_EVAL_NOISE_SIGMA)  # add Gaussian noise
blended = SF_LEAF_BLEND * sf_value + (1 - SF_LEAF_BLEND) * nn_value
```

- **Blend ratio:** 0.6 Stockfish / 0.4 NN
- **Noise sigma:** 0.1 (prevents memorization of Stockfish evaluations)
- **Stockfish depth:** 10 (fast enough for real-time, strong enough to guide)

The NN still generates policy priors — Stockfish only improves the **value signal** at leaf nodes. This means the network develops its own strategic understanding while learning accurate position evaluation from Stockfish.

### Critic Mode

In critic mode, Stockfish evaluates **every legal move** before the NN chooses:

```python
# critic_game.py:50-68
eval_map = stockfish.evaluate_legal_moves_batch(board)
# Build weighted policy target from evaluations
eval_shifted = eval_values - min(eval_values) + 1e-10
weights = eval_shifted / eval_shifted.sum()
target_policy[move_to_idx(move)] = weights[idx]
# NN picks move with small critic bias
logits += critic_bias  # bias = cp * 0.001
```

The `evaluate_legal_moves_batch()` method sends all legal moves to Stockfish in one API call (using Stockfish's `get_top_moves` with `MultiPV=N`), avoiding N sequential queries.

---

## 6. Training Pipeline

### TrainingBuffer (`trainer.py:95`)

A thread-safe circular buffer (deque) storing up to 100K training examples:

```python
class TrainingBuffer:
    def add(self, examples):
        self.buffer.extend(examples)  # thread-safe, maxlen caps it
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return stack board_tensors, policies, values
```

### Training Step (`trainer.py:246`)

```python
def train_step(self):
    batch = self.buffer.sample(BATCH_SIZE)  # 64
    pred_policy, pred_value = self.model(tensors)
    
    p_loss = cross_entropy(pred_policy, policies)
    v_loss = mse_loss(pred_value, values)
    l2 = sum(p.pow(2).sum() for p in self.model.parameters()) * L2_REG
    loss = POLICY_WEIGHT * p_loss + VALUE_WEIGHT * v_loss + l2
    
    loss.backward()
    optimizer.step()
```

| Loss Component | Weight | Purpose |
|---------------|--------|---------|
| Policy Cross-Entropy | 1.0 | Learn which moves are good (from MCTS visit counts) |
| Value MSE | 1.0 | Learn position evaluation (from game outcomes) |
| L2 Regularization | 1×10⁻⁴ | Prevent overfitting |

### Training Loop (`server.py`)

```
Every cycle:
  1. Play N self-play games (collect positions + outcomes)
  2. Add examples to TrainingBuffer
  3. For each training step:
     a. Sample batch from buffer
     b. Forward pass → loss
     c. Backward pass → gradient descent
  4. Every 50 steps: calibrate ELO vs Stockfish
  5. Every 50 steps: push checkpoint to Hugging Face Hub
  6. Every 100 steps: save local checkpoint
```

### Hugging Face Persistence

Checkpoints are auto-pushed to Hugging Face Hub every 50 training steps:

```python
push_to_hf(checkpoint_path, metadata={
    'step': step,
    'games_played': games_played,
    'loss': loss,
    'last_game_pgn': pgn,
    'pushed_at': datetime.utcnow().isoformat(),
})
```

This means **the model persists even if the training server crashes**. On restart, it downloads the latest checkpoint from HF.

---

## 7. Critic-Guided Training

Critic mode (`training/critic_game.py`) takes a different approach:

**Instead of MCTS, the NN picks moves directly** with a small bias from Stockfish's evaluations. Stockfish then evaluates the resulting position to create the training target.

### How It Works

```python
# For each position:
# 1. Stockfish evaluates every legal move
eval_values = [stockfish.evaluate(move) for move in legal_moves]

# 2. Build target policy from evaluations (softmax over shifted values)
weights = softmax(eval_values - min(eval_values))

# 3. NN picks a move (not the best one — exploration!)
logits = nn(board)
critic_bias[move] = cp * 0.001
probs = softmax(logits + critic_bias)
chosen = sample(probs)

# 4. Record training example
example = {
    'board_tensor': board_tensor,
    'policy': target_policy,  # Stockfish's opinion = teacher
    'value': current_eval / 2000.0,  # position evaluation
}
```

**This is similar to behavioral cloning** — the NN learns to imitate Stockfish's move preferences. But the exploration noise (temperature = 0.3) and the NN's own preferences (it samples, it doesn't always pick the best) let it discover novel strategies.

---

## 8. Scoring & Evaluation

### Centipawns → Display

All internal evaluations are in **centipawns** (1 pawn = 100 cp). The frontend divides by 100 for display:

```
Stockfish returns: +68 cp       →  Display: +0.68
Stockfish returns: -724 cp      →  Display: -7.24
Stockfish returns: 0 cp         →  Display: +0.00
```

### Move Quality Classification

When a position is analyzed, Stockfish returns the top 5 moves. Each is classified by centipawn difference from the best:

```python
def _classify_move_quality(diff_cp):
    if diff_cp <= 5:     return 'best'      # Essentially equal
    if diff_cp <= 15:    return 'good'       # Slight inaccuracy
    if diff_cp <= 50:    return 'ok'         # Minor mistake
    if diff_cp <= 200:   return 'bad'        # Clear mistake
    return 'blunder'                          # Game-losing move
```

5 cp = 0.05 pawns. A "blunder" threshold of 200 cp (2 pawns) means the move gives away at least a minor piece worth of advantage.

### Resign Logic

The NN value head outputs a position evaluation in [-1, +1]. If it drops below `RESIGN_THRESHOLD = -0.8` for consecutive moves, the engine resigns:

```python
# selfplay.py (resign check during play)
if nn_value < RESIGN_THRESHOLD:
    # Check again on next move to avoid premature resignation
    consecutive_bad += 1
    if consecutive_bad >= 3:
        resign()
```

---

## 9. Side Games & Parallelism

### Architecture

In addition to the main training game, the system runs **9 side games** in parallel threads. Each is an independent self-play game:

```python
# server.py:481-495
num_sims = 350  # per side game

for gid in range(1, NUM_GAMES):  # games 2-10
    t = threading.Thread(
        target=_side_game_worker,
        args=(gid, model, num_sims, shared_evaluator, event_queue, buffer)
    )
    t.start()
```

| Aspect | Main Game | Side Games |
|--------|-----------|------------|
| **MCTS Sims** | 500 | 350 |
| **Model** | Shared ChessNet | Shared ChessNet (via BatchEvaluator) |
| **GPU Batching** | Direct + BatchEvaluator | BatchEvaluator only |
| **Training Data** | Added to buffer | Added to buffer |
| **Display** | Full board + controls | Mini board in grid |
| **Stockfish** | Shared instance | One per game (simpler) |

### Why?

Side games serve three purposes:
1. **Fill the training buffer faster** — more games = more diverse positions
2. **Explore more** — lower sims = lower-quality play = more variety
3. **Visual feedback** — the 3×3 grid of mini-boards makes the training feel alive

### Shared GPU Batching

All games share a single `BatchEvaluator` (`selfplay.py:36`). It collects individual NN requests into batches and runs a single GPU forward pass:

```
Time ──────────────────────────────────────────►
Game 1: [request]..........................[result]
Game 2: ...[request]......................[result]  
Game 3: ......[request]...................[result]
         │                         ▲
         ▼ Batch (size=3)          │
    [Forward Pass]──────────────────┘
```

- **Batch size:** Up to 512
- **Max wait time:** 10ms (collects requests arriving within a 10ms window)
- **OOM handling:** If GPU OOMs, batch is split in half recursively

---

## 10. SSE Event Stream

### Architecture (`server.py:35-100`)

Server-Sent Events power the real-time dashboard. A `StreamManager` manages the broadcast pipeline:

```
Game Thread    →  SSE Event Queue  →  StreamManager  →  HTTP SSE Clients
  (push events)      (thread-safe      (fan-out to        (browser)
                      Queue max 5K)     client queues)
```

### Event Types

| Event | Trigger | Payload |
|-------|---------|---------|
| `move` | Move played in any game | `{fen, move, game_id, moves_uci, status}` |
| `side_move` | Move in side game | `{game_id, fen, move_count}` |
| `analysis` | New Stockfish eval | `{eval_cp, eval_norm, move_analysis[], top_moves[]}` |
| `status` | Training status change | `{step, games_played, loss, buffer_size, ...}` |
| `game_start` | New game begins | `{game_type, game_id}` |
| `finished` | Game ends | `{pgn, result, game_id}` |

### Client Connection

Browsers connect to `/api/train/stream` and receive a continuous stream of JSON events:

```javascript
const evtSource = new EventSource('/api/train/stream');
evtSource.addEventListener('move', (e) => updateBoard(JSON.parse(e.data)));
evtSource.addEventListener('analysis', (e) => updateAnalysis(JSON.parse(e.data)));
```

Keepalive comments are sent every 15 seconds to prevent proxy timeouts.

---

## 11. Frontend Architecture

### Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Framework | React 18 (Vite) | Component rendering |
| UI Library | Mantine 7 | Pre-built components (Paper, Badge, Table, etc.) |
| Icons | Tabler Icons | Icon set for sidebar sections |
| Chess Board | react-chessboard | Board rendering + FEN display |
| State | React Context | Global game state (GameContext) |
| Real-time | EventSource (SSE) | Live data stream |
| Hosting | Vercel | Static frontend deployment |

### Component Tree

```
<App>
  ├─ <AppShell>
  │   ├─ <TopBar>                  — Title, SSE status, flip/fullscreen
  │   └─ <Main>
  │       ├─ Main Column
  │       │   ├─ <PlayerInfoBar>   — Top player info + turn indicator
  │       │   ├─ <EvalBar>         — White/black evaluation bar
  │       │   ├─ <LiveBoard>       — Main chess board
  │       │   ├─ <PlayerInfoBar>   — Bottom player info
  │       │   ├─ <BoardNav>        — Move navigation controls
  │       │   └─ <SideBoard> ×9    — Side game mini-boards (3×3 grid)
  │       └─ Sidebar (grid layout)
  │           ├─ <MoveList>        — Move history
  │           ├─ <SSEEventLog>     — Real-time event log
  │           ├─ <AnalysisTable>   — Stockfish analysis
  │           ├─ <MetricsGrid>     — Training metrics
  │           └─ <RecentGames>     — Finished games list
  └─ <FullscreenOverlay>           — Fullscreen board modal
```

### State Management

State flows through a single `GameContext`:

```typescript
interface GameState {
  allMoves: string[];
  fenCache: string[];
  currentViewIndex: number;
  boardOrientation: 'white' | 'black';
  analysis: object | null;
  trainingStatus: TrainingStatus | null;
  sseEvents: SSEEvent[];
  sseStatus: string;
  whatHappening: string;
  sideFens: Record<number, string>;
  sideMoveCounts: Record<number, number>;
  isFullscreen: boolean;
}
```

Updates come from:
1. **SSE hooks** (`useSSE`, `useStatus`, `useAnalysis`) — receive real-time data
2. **User interactions** — board navigation, flip, fullscreen

### The SectionCard Pattern

All sidebar panels use a shared `SectionCard` component for consistent styling:

```tsx
<SectionCard icon={<IconX size={16} />} title="Panel Title" rightSection={<Badge/>}>
  {content}
</SectionCard>
```

This ensures every panel has the same `Paper` background (`#161B22`), border (`#30363D`), padding (`md`), and header styling.

---

## 12. ELO Estimation

ELO is estimated by playing **10 games against Stockfish** (depth 10) every 50 training steps:

```python
# trainer.py:378-396
base_elo = 200

# Map win_rate [0,1] → [-200, +400] ELO offset
# win_rate = 0.0 → ELO = 200 (no offset)
# win_rate = 0.5 → ELO = 200 + 200 = 400
# win_rate = 1.0 → ELO = 200 + 400 = 600
elo_offset = int(win_rate * 400 - 200)
return max(base_elo, base_elo + elo_offset)
```

| Win Rate vs SF | Estimated ELO |
|----------------|---------------|
| 0% (always loses) | 200 |
| 25% | 300 |
| 50% | 400 |
| 75% | 500 |
| 100% (always wins) | 600 |

The base ELO of 200 represents the starting point — a randomly initialized network. As training progresses, the network climbs the scale until it plateaus against Stockfish.

### Stockfish Calibration Games

```
Setup:
- White = NN (raw policy, no MCTS — fastest play)
- Black = Stockfish (depth 10)
- Max 200 moves per game
- Score: 1.0 for NN win, 0.5 for draw, 0.0 for loss

Result tracking: last 50 calibration games stored
→ ELO = average over the most recent 10 games
```

---

## Performance Optimizations

### GPU Batch Evaluator

The `BatchEvaluator` is the key to high throughput:

- **Micro-batching:** Collects requests for up to 10ms before sending to GPU
- **OOM recovery:** Splits batch in half on CUDA OOM
- **Zero-copy:** Board tensors are pre-allocated and reused
- **Thread-safe:** Uses `threading.Event` for lock-free result delivery

### Incremental Position Cache (Stockfish)

Stockfish's internal transposition table is preserved between consecutive positions:

```python
# stockfish_engine.py:40-41
self._cached_fen = starting_fen
self._cached_board = chess.Board()
```

When the next position is a legal next move from the cached position, Stockfish uses `set_position` (which keeps the TT warm) instead of reconstructing from scratch.

### Eval Cache (Server-side)

Position evaluations are cached in an LRU cache with TTL:

```python
eval_cache = OrderedDict()    # FEN → {eval_cp, eval_norm, top_moves, ...}
EVAL_CACHE_MAX = 2000         # Max entries
EVAL_CACHE_TTL = 600          # 10 minute TTL
```

The cache is also persisted to disk (`eval_cache.json`) and loaded on server restart.

---

## Model Checkpointing

```
Checkpoint strategy:
├── Every 100 steps → Local disk (/tmp/chess-models/checkpoint.pt)
├── Every 50 steps  → Hugging Face Hub (LanceAbuan/chess-alpha-zero)
├── On restart      → Download latest from HF (if token set)
└── Metadata        → Includes step, games, loss, PGN of last game
```

This means **you can stop the server at any time and resume where you left off** — the model state is always backed up.

---

## Glossary

| Term | Definition |
|------|------------|
| **Centipawn** | 1/100 of a pawn. Standard chess evaluation unit. |
| **CPuct** | Exploration constant in PUCT selection formula. |
| **Critic Mode** | Training mode where Stockfish critiques NN move choices. |
| **ELO** | Rating system. Higher = stronger play. |
| **FEN** | Forsyth–Edwards Notation. Compact chess position encoding. |
| **MCTS** | Monte Carlo Tree Search. Search algorithm that balances exploration vs exploitation. |
| **MultiPV** | Stockfish mode that returns evaluations for multiple moves. |
| **NN** | Neural Network. |
| **Policy** | Probability distribution over legal moves. What to play. |
| **PUCT** | Polynomial Upper Confidence Trees. Selection formula for MCTS. |
| **Residual Block** | Neural network building block with skip connections. |
| **Self-Play** | Training by playing against yourself. |
| **SSE** | Server-Sent Events. HTTP-based real-time streaming protocol. |
| **Value** | Position evaluation scalar [-1, +1]. Who's winning. |
