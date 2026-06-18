import { Title, Text, Container, Divider, Box, Stack, Table, Code } from '@mantine/core';

const mono = { whiteSpace: 'pre' as const, fontFamily: '"JetBrains Mono", "Fira Code", monospace', fontSize: '13px', lineHeight: '1.5' };
const codeBox = { ...mono, padding: '1rem', background: '#161B22', border: '1px solid #30363D', borderRadius: '6px', overflowX: 'auto' as const };
const inlineCode = { background: '#161B22', border: '1px solid #30363D', borderRadius: '3px', padding: '1px 5px', fontFamily: '"JetBrains Mono", monospace', fontSize: '13px' };
const sideNote = { borderLeft: '2px solid #30363D', paddingLeft: '1rem', marginTop: '0.5rem' };
const tableHeader = { background: '#161B22', color: '#8B949E', fontSize: '12px', textTransform: 'uppercase' as const, letterSpacing: '0.5px' as const };

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Box>
      <Title order={2} mt="lg" mb="md" style={{ color: '#58A6FF' }}>{title}</Title>
      {children}
    </Box>
  );
}

function SubSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Box mt="md">
      <Title order={3} mb="sm" style={{ color: '#C9D1D9' }}>{title}</Title>
      {children}
    </Box>
  );
}

function Pre({ children }: { children: string }) {
  return <Text style={codeBox}>{children}</Text>;
}

function Key({ children }: { children: string }) {
  return <Code style={inlineCode}>{children}</Code>;
}

function Note({ children }: { children: React.ReactNode }) {
  return <Box mt="sm" mb="sm" style={sideNote}>{children}</Box>;
}

function StyledTable({ headers, rows }: { headers: string[]; rows: (string | React.ReactNode)[][] }) {
  return (
    <Box mt="sm" mb="sm" style={{ overflowX: 'auto' }}>
      <Table style={{ fontSize: '14px', borderCollapse: 'collapse' }}>
        <Table.Thead>
          <Table.Tr style={tableHeader}>
            {headers.map((h, i) => <Table.Th key={i} style={{ padding: '8px 12px', borderBottom: '1px solid #30363D' }}>{h}</Table.Th>)}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {rows.map((row, ri) => (
            <Table.Tr key={ri} style={{ borderBottom: '1px solid #21262D' }}>
              {row.map((cell, ci) => <Table.Td key={ci} style={{ padding: '8px 12px' }}>{cell}</Table.Td>)}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Box>
  );
}

export default function Readme() {
  return (
    <Container size={800} py="xl">
      <Title order={1} mb="md" style={{ color: '#F0F6FC' }}>Architecture Deep Dive</Title>
      <Text c="dimmed" style={{ fontStyle: 'italic' }}>
        Everything you want to know about how Fool's Gambit works, from the neural network to the dashboard.
      </Text>

      <Divider my="xl" />

      <Stack gap="xl">

        {/* ─── 1. System Architecture ─── */}
        <Section title="1. System Architecture">
          <Text mb="md">The entire system runs inside a single Flask process. Training, side games, Stockfish evaluation, and SSE streaming all share the same address space:</Text>
          <Pre>{`┌─────────────────────────────────────────────────────────────────────┐
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
└─────────────────────────────────────────────────────────────────────┘`}</Pre>

          <SubSection title="The Two Training Modes">
            <Text>The system operates in one of two mutually exclusive modes:</Text>
            <StyledTable
              headers={['Mode', 'Description', 'When']}
              rows={[
                ['Self-Play', 'NN + MCTS plays both sides. Training data = MCTS visit counts as policy targets.', 'Default for pure RL'],
                ['Critic', 'NN picks moves, Stockfish evaluates all legal moves to build policy targets.', 'User-specified via API'],
              ]}
            />
            <Text mt="sm">The mode is toggled at runtime. Critic mode is generally stronger early on because Stockfish provides a better training signal than the random initial network.</Text>
          </SubSection>
        </Section>

        {/* ─── 2. Neural Network Design ─── */}
        <Section title="2. Neural Network Design">
          <Text mb="md">The <Key>ChessNet</Key> architecture is a tiny but efficient ResNet with two heads — one for policy (which move to play) and one for value (who's winning).</Text>
          <Pre>{`Input: 8×8×16 board tensor
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
└───────────────────────────────────────┴────────────────────────────`}</Pre>

          <SubSection title="Why So Small?">
            <Text>~200K parameters vs AlphaZero's 20 blocks × 256 filters (~20M params). This is intentional:</Text>
            <ul style={{ color: '#C9D1D9', lineHeight: '1.8' }}>
              <li><b>Real-time training:</b> The model runs hundreds of forward passes per second on a single GPU</li>
              <li><b>Interactive demo:</b> Users see the network improve in minutes, not days</li>
              <li><b>Side games:</b> 9 parallel games share the same model via <Key>BatchEvaluator</Key></li>
              <li><b>Memory:</b> The entire model + optimizer state fits in ~10MB of GPU memory</li>
            </ul>
          </SubSection>

          <SubSection title="Forward Pass">
            <Pre>{`def forward(self, x):
    x = self.input_bn(self.input_conv(x))
    x = self.residual_tower(x)        # 2 residual blocks

    policy = self.policy_fc(self.policy_bn(self.policy_conv(x)))
    value = torch.tanh(self.value_fc2(self.value_fc1(self.value_bn(self.value_conv(x)))))

    return policy, value  # (batch, 4096), (batch, 1)`}</Pre>
            <Text mt="sm">
              The <b>policy output</b> is raw logits over 4096 possible moves (64 from-squares × 64 to-squares).
              Illegal moves are masked before softmax. The <b>value output</b> is tanh-activated to [-1, +1]:
            </Text>
            <ul style={{ color: '#C9D1D9', lineHeight: '1.8' }}>
              <li><Key>+1.0</Key> = position is winning for the side to move</li>
              <li><Key>-1.0</Key> = position is losing for the side to move</li>
              <li><Key>0.0</Key> = equal</li>
            </ul>
          </SubSection>
        </Section>

        {/* ─── 3. Board Encoding ─── */}
        <Section title="3. Board Encoding">
          <Text mb="md">The board is encoded as a 16-channel 8×8 tensor (<Key>training/tensorize.py</Key>). This is the input representation the neural network sees:</Text>
          <StyledTable
            headers={['Channels', 'Content']}
            rows={[
              ['0', 'White pawns'],
              ['1', 'White knights'],
              ['2', 'White bishops'],
              ['3', 'White rooks'],
              ['4', 'White queens'],
              ['5', 'White king'],
              ['6', 'Black pawns'],
              ['7', 'Black knights'],
              ['8', 'Black bishops'],
              ['9', 'Black rooks'],
              ['10', 'Black queens'],
              ['11', 'Black king'],
              ['12', 'En passant target square (all zeros if none)'],
              ['13', 'Castling rights — white kingside'],
              ['14', 'Castling rights — white queenside'],
              ['15', 'Side to move (+1 for white, 0 for black, broadcast)'],
            ]}
          />
          <Text mt="sm">
            Each piece channel is a binary mask (1 where the piece exists, 0 elsewhere). The spatial structure lets the network learn piece relationships, king safety, and tactical patterns directly — no hand-crafted features like "bishop pair" or "pawn structure" are needed.
          </Text>
          <Note>
            <Text size="sm" c="#8B949E">This encoding is inspired by the AlphaZero paper but simplified: we use 16 channels instead of 119 (no repetitions, no 8-step history, no color planes). This keeps the network footprint small enough for real-time training.</Text>
          </Note>
        </Section>

        {/* ─── 4. Monte Carlo Tree Search ─── */}
        <Section title="4. Monte Carlo Tree Search">
          <Text mb="md">MCTS is the search algorithm that selects moves by balancing <b>exploration</b> (trying new moves) vs <b>exploitation</b> (playing known-good moves). Every time the NN makes a move, it runs hundreds of MCTS simulations first.</Text>

          <SubSection title="The Four Phases">
            <Pre>{`function MCTS_SEARCH(board, num_simulations):
    root = BUILD_ROOT(board)
    for _ in range(num_simulations):
        node = root
        // 1. SELECT — traverse to unexplored leaf
        while node.is_expanded:
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

    return root.visit_counts  // improved policy`}</Pre>
          </SubSection>

          <SubSection title="PUCT Selection Formula">
            <Text mb="sm">The selection formula that balances exploration vs exploitation:</Text>
            <Text style={{ fontStyle: 'italic', color: '#58A6FF', fontSize: '16px', textAlign: 'center', padding: '1rem' }}>
              score = Q(s,a) + cpuct × P(s,a) × √N_parent / (1 + n_child)
            </Text>
            <StyledTable
              headers={['Symbol', 'Meaning', 'Effect']}
              rows={[
                ['Q(s,a)', 'Average game outcome from taking move a in position s', 'Exploitation — prefer moves that have worked before'],
                ['P(s,a)', 'NN prior probability of move a', 'Knowledge — prefer moves the NN thinks are good'],
                ['N_parent', 'Visit count of the parent node', 'Decays bonus as parent is explored more'],
                ['n_child', 'Visit count of this child', 'Under-visited moves get a bonus'],
                ['cpuct', 'Exploration constant (1.0)', 'Higher = more exploration'],
              ]}
            />
          </SubSection>

          <SubSection title="Parallel MCTS">
            <Text mb="sm">Simulations are split across 8 threads for maximum CPU utilization:</Text>
            <Pre>{`# selfplay.py:171
NUM_MCTS_THREADS = 8
sims_per_thread = max(1, num_simulations // NUM_MCTS_THREADS)

def _worker(sims):
    for _ in range(sims):
        self._simulate(root, board.copy(), max_depth=15)

with ThreadPoolExecutor(max_workers=NUM_MCTS_THREADS) as executor:
    futures = [executor.submit(_worker, n) for n in ...]
    for f in futures:
        f.result()  # wait for all threads`}</Pre>
            <Text mt="sm">Each thread operates on a <b>copy of the board</b> to avoid race conditions. All tree mutations (visit counts, Q-values) are protected by a single <Key>_tree_lock</Key>. The NN evaluation — the bottleneck — runs lock-free since it&apos;s read-only on the model weights.</Text>
          </SubSection>

          <SubSection title="Key Parameters">
            <StyledTable
              headers={['Parameter', 'Value', 'Effect']}
              rows={[
                ['cpuct', '1.0', 'Exploration constant — higher = more exploration'],
                ['noise_epsilon', '0.25', 'Dirichlet noise mix — prevents policy overconfidence'],
                ['noise_alpha', '0.03', 'Dirichlet concentration — lower = sparser noise'],
                ['max_depth', '15', 'Max plies per simulation (prevents infinite loops)'],
                ['NUM_MCTS_THREADS', '8', 'Parallel threads per MCTS search'],
              ]}
            />
            <Text mt="sm">
              <b>Dirichlet noise</b> is added to the root node's prior probabilities at the start of each search. This ensures the network explores moves it wouldn't otherwise consider, preventing policy collapse where the NN only plays its top move.
            </Text>
          </SubSection>
        </Section>

        {/* ─── 5. Stockfish Integration ─── */}
        <Section title="5. Stockfish Integration">
          <Text mb="md">Stockfish runs as a single shared subprocess with thread-safe access. The wrapper (<Key>stockfish_engine.py</Key>) handles locking, crash recovery, and position caching.</Text>

          <SubSection title="Thread-Safe Wrapper">
            <Pre>{`class StockfishPlayer:
    def __init__(self):
        self._lock = threading.RLock()
        self._engine = SF(path=STOCKFISH_PATH, depth=10, ...)`}</Pre>
            <Text mt="sm">All public methods acquire <Key>_lock</Key> before touching the engine. This prevents race conditions when the training thread, MCTS, and HTTP request handlers all share one Stockfish instance.</Text>
          </SubSection>

          <SubSection title="Crash Recovery">
            <Text>If Stockfish dies (OOM, segfault), the wrapper auto-restarts the subprocess and retries once. If it fails again, it returns a safe fallback value (<Key>centipawn 0</Key>) so training never crashes from a Stockfish failure.</Text>
          </SubSection>

          <SubSection title="Stockfish-Guided Rollouts">
            <Text mb="sm">In MCTS, leaf nodes are evaluated by blending Stockfish with the NN value. This gives the network a stronger value signal than pure NN self-play:</Text>
            <Pre>{`# selfplay.py:~340
sf_value = stockfish.get_evaluation(board) / 2000.0  # normalize to [-1, 1]
noise = np.random.normal(0, SF_EVAL_NOISE_SIGMA)       # add Gaussian noise
sf_noisy = clamp(sf_value + noise, -1.0, 1.0)
blended = SF_LEAF_BLEND * sf_noisy + (1 - SF_LEAF_BLEND) * nn_value`}</Pre>
            <StyledTable
              headers={['Parameter', 'Value', 'Purpose']}
              rows={[
                ['SF_LEAF_BLEND', '0.6', 'Blend ratio — 60% Stockfish, 40% NN'],
                ['SF_EVAL_NOISE_SIGMA', '0.1', 'Gaussian noise prevents memorization of Stockfish evaluations'],
                ['Stockfish depth', '10', 'Fast enough for real-time, strong enough to guide'],
              ]}
            />
            <Note>
              <Text size="sm" c="#8B949E">The NN still generates policy priors — Stockfish only improves the <b>value signal</b> at leaf nodes. This means the network develops its own strategic understanding while learning accurate position evaluation from Stockfish.</Text>
            </Note>
          </SubSection>

          <SubSection title="Position Cache">
            <Text>Stockfish&apos;s internal transposition table is preserved between consecutive positions. When the next position is a legal next move from the cached position, Stockfish uses <Key>set_position</Key> (which keeps the TT warm) instead of reconstructing from scratch. This speeds up sequential evaluations by ~10×.</Text>
          </SubSection>
        </Section>

        {/* ─── 6. Training Pipeline ─── */}
        <Section title="6. Training Pipeline">
          <Text mb="md">Training follows the RL loop: play games → collect experience → learn from it → repeat.</Text>

          <SubSection title="TrainingBuffer">
            <Text>A thread-safe circular buffer (deque) storing up to 100K training examples:</Text>
            <Pre>{`class TrainingBuffer:
    def add(self, examples):
        self.buffer.extend(examples)  # thread-safe, maxlen caps it

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return stack board_tensors, policies, values`}</Pre>
            <Text mt="sm">Side games and the main game both call <Key>buffer.add()</Key>. The buffer drains randomly during training — every example has an equal chance of being learned from, regardless of which game produced it.</Text>
          </SubSection>

          <SubSection title="Training Step">
            <Pre>{`def train_step(self):
    batch = self.buffer.sample(BATCH_SIZE)   # 64 examples
    if actual_batch_size < MIN_BATCH_SIZE:   # need at least 16
        return None

    pred_policy, pred_value = self.model(tensors)
    p_loss = cross_entropy(pred_policy, policies)
    v_loss = mse_loss(pred_value, values)
    l2 = sum(p.pow(2).sum() for p in self.model.parameters()) * L2_REG
    loss = POLICY_WEIGHT * p_loss + VALUE_WEIGHT * v_loss + l2

    loss.backward()
    optimizer.step()`}</Pre>

            <StyledTable
              headers={['Loss Component', 'Weight', 'Purpose']}
              rows={[
                ['Policy Cross-Entropy', '1.0', 'Learn which moves are good (from MCTS visit counts)'],
                ['Value MSE', '1.0', 'Learn position evaluation (from game outcomes)'],
                ['L2 Regularization', '1×10⁻⁴', 'Prevent overfitting'],
              ]}
            />
          </SubSection>

          <SubSection title="Training Loop">
            <Text mb="sm">The training loop is a continuous cycle that plays 1 main game, waits for 9 side games, then runs a few gradient steps:</Text>
            <Pre>{`Every batch:
  1. Play 1 main game (NN + MCTS, 500 sims)
  2. Wait for 9 side games to complete (+350 sims each)
     → All 10 games contribute examples to TrainingBuffer
  3. Run 2 training steps:
     a. Sample batch of 64 from buffer
     b. Forward pass → compute loss
     c. Backward pass → update weights
  4. Every 20 batches:
     a. Save checkpoint to disk
     b. Push checkpoint to Hugging Face Hub`}</Pre>
            <Text mt="sm">
              This means the user sees the step counter increment after every 10-game batch — roughly every few minutes — instead of waiting hours for a batch of 20 main games to finish before any training happens.
            </Text>
          </SubSection>

          <SubSection title="Optimizer &amp; Schedule">
            <Text>The optimizer is <Key>Adam</Key> with a fixed learning rate of <Key>0.001</Key>. No learning rate schedule is used — the small network converges well with constant LR. Gradient norms are clipped at 1.0 to prevent instability.</Text>
          </SubSection>

          <SubSection title="Model Persistence">
            <Text>Checkpoints follow a three-tier strategy:</Text>
            <Pre>{`Checkpoint strategy:
├── Every 20 batches → Local disk (/tmp/chess-models/checkpoint.pt)
├── Every 20 batches → Hugging Face Hub (LanceAbuan/chess-alpha-zero)
├── On restart      → Download latest from HF (if token set)
└── Metadata        → Includes step, games, loss, PGN of last game`}</Pre>
            <Text mt="sm">This means <b>you can stop the server at any time and resume where you left off</b>. The model state is always backed up.</Text>
          </SubSection>
        </Section>

        {/* ─── 7. Critic-Guided Training ─── */}
        <Section title="7. Critic-Guided Training">
          <Text mb="md">Critic mode (<Key>training/critic_game.py</Key>) takes a fundamentally different approach: instead of MCTS, the NN picks moves directly, and Stockfish provides the training signal.</Text>

          <SubSection title="How It Works">
            <Pre>{`# For each position:
# 1. Stockfish evaluates every legal move
eval_values = [stockfish.evaluate(move) for move in legal_moves]

# 2. Build target policy from evaluations (softmax over shifted values)
eval_shifted = eval_values - min(eval_values) + 1e-10
weights = eval_shifted / eval_shifted.sum()
target_policy[move_to_idx(move)] = weights[idx]

# 3. NN picks a move (with exploration)
logits = nn(board)
logits += critic_bias  # bias = cp * 0.001 — slight Stockfish preference
probs = softmax(logits + critic_bias)
chosen = sample(probs)

# 4. Record training example
example = {
    'board_tensor': board_tensor,
    'policy': target_policy,    # Stockfish's opinion = teacher
    'value': current_eval / 2000.0,
}`}</Pre>
          </SubSection>

          <SubSection title="Self-Play vs Critic">
            <StyledTable
              headers={['Aspect', 'Self-Play', 'Critic']}
              rows={[
                ['Move Selection', 'MCTS (500 sims) refines NN policy', 'NN samples from policy + small Stockfish bias'],
                ['Policy Target', 'MCTS visit counts', 'Stockfish evaluation distribution'],
                ['Value Target', 'Actual game outcome', 'Stockfish position evaluation'],
                ['Speed', '~30-60s per game', '~10-20s per game'],
                ['When to Use', 'After the NN has basic chess knowledge', 'Early training / bootstrapping'],
              ]}
            />
            <Note>
              <Text size="sm" c="#8B949E">Critic mode is similar to <b>behavioral cloning</b> — the NN learns to imitate Stockfish's move preferences. But the exploration noise (temperature = 0.3) and the NN's own sampling let it discover novel strategies that Stockfish wouldn't find.</Text>
            </Note>
          </SubSection>
        </Section>

        {/* ─── 8. Scoring & Evaluation ─── */}
        <Section title="8. Scoring &amp; Evaluation">
          <SubSection title="Centipawns → Display">
            <Text>All internal evaluations are in centipawns (1 pawn = 100 cp). The frontend divides by 100 for display:</Text>
            <Pre>{`Stockfish returns: +68 cp       →  Display: +0.68
Stockfish returns: -724 cp      →  Display: -7.24
Stockfish returns: mate in 5    →  Display: #+5 (shown as +10.0)`}</Pre>
          </SubSection>

          <SubSection title="Move Quality Classification">
            <Text mb="sm">When a position is analyzed, Stockfish returns the top moves. Each is classified by centipawn difference from the best move:</Text>
            <Pre>{`def classify_move_quality(diff_cp):
    if diff_cp <= 5:     return 'best'     # Essentially equal
    if diff_cp <= 15:    return 'good'      # Slight inaccuracy
    if diff_cp <= 50:    return 'ok'        # Minor mistake
    if diff_cp <= 200:   return 'bad'       # Clear mistake
    return 'blunder'                         # Game-losing move (≥2 pawns)`}</Pre>
          </SubSection>

          <SubSection title="Resign Logic">
            <Text mb="sm">The NN value head outputs a position evaluation in [-1, +1]. If it drops below threshold for consecutive moves, the engine resigns:</Text>
            <Pre>{`# selfplay.py
RESIGN_THRESHOLD = -0.8

if nn_value < RESIGN_THRESHOLD:
    consecutive_bad += 1
    if consecutive_bad >= 3:
        resign()  # position is hopeless`}</Pre>
            <Text mt="sm">The 3-move grace period prevents premature resignation from a single blunder. A value of -0.8 corresponds to roughly -800 centipawns — a losing position equivalent to being down a rook.</Text>
          </SubSection>
        </Section>

        {/* ─── 9. Side Games & Parallelism ─── */}
        <Section title="9. Side Games &amp; Parallelism">
          <Text mb="md">In addition to the main training game, 9 side games run in parallel threads. Each is an independent self-play game using the same model.</Text>

          <SubSection title="Comparison">
            <StyledTable
              headers={['Aspect', 'Main Game', 'Side Games']}
              rows={[
                ['MCTS Sims', '500', '350'],
                ['Model', 'Shared ChessNet', 'Shared ChessNet (via BatchEvaluator)'],
                ['GPU Batching', 'Direct + BatchEvaluator', 'BatchEvaluator only'],
                ['Training Data', 'Added to buffer', 'Added to buffer'],
                ['Display', 'Full board + controls', 'Mini board in 3×3 grid'],
                ['Stockfish', 'Shared instance', 'One per game (simpler)'],
              ]}
            />
          </SubSection>

          <SubSection title="Why Side Games?">
            <Text>Side games serve three purposes:</Text>
            <ol style={{ color: '#C9D1D9', lineHeight: '1.8' }}>
              <li><b>Fill the training buffer faster</b> — 9× more games means 9× more diverse positions per minute</li>
              <li><b>Explore more</b> — lower sims = lower-quality play = more variety in the buffer</li>
              <li><b>Visual feedback</b> — the 3×3 grid of mini-boards makes training feel alive</li>
            </ol>
          </SubSection>

          <SubSection title="Shared GPU Batching (BatchEvaluator)">
            <Text mb="sm">All games share a single <Key>BatchEvaluator</Key> (<Key>selfplay.py:36</Key>). It collects individual NN requests into batches and runs a single GPU forward pass:</Text>
            <Pre>{`Time ──────────────────────────────────────────►
Game 1: [request]..........................[result]
Game 2: ...[request]......................[result]
Game 3: ......[request]...................[result]
         │                         ▲
         ▼ Batch (size=3)          │
    [Forward Pass]──────────────────┘`}</Pre>
            <StyledTable
              headers={['Parameter', 'Value', 'Effect']}
              rows={[
                ['Batch size', 'Up to 512', 'Collects pending requests into one GPU call'],
                ['Max wait time', '10ms', 'How long to wait for more requests before firing the batch'],
                ['OOM handling', 'Split in half', 'If GPU OOMs, batch is split recursively until it fits'],
              ]}
            />
            <Text mt="sm">
              Without batching, 10 concurrent games would require 10 sequential GPU forward passes, creating a bottleneck. With batching, they all run in roughly the time of one pass — a 10× throughput improvement.
            </Text>
          </SubSection>
        </Section>

        {/* ─── 10. SSE Event Stream ─── */}
        <Section title="10. SSE Event Stream">
          <Text mb="md">Server-Sent Events (SSE) power the real-time dashboard. A <Key>StreamManager</Key> manages the broadcast pipeline from game threads to browser.</Text>

          <SubSection title="Pipeline">
            <Pre>{`Game Thread    →  SSE Event Queue  →  StreamManager  →  HTTP SSE Clients
  (push events)      (thread-safe      (fan-out to        (browser)
                      Queue max 5K)     client queues)`}</Pre>
          </SubSection>

          <SubSection title="Event Types">
            <StyledTable
              headers={['Event', 'Trigger', 'Payload']}
              rows={[
                ['game_start', 'New main game begins', '{game_id, timestamp}'],
                ['game_progress', 'Move played in any game', '{game_id, move, status, timestamp}'],
                ['status_update', 'Training status change', '{step, games_played, loss, buffer_size, cycle, ...}'],
                ['mcts_progress', 'MCTS search update', '{simulations, best_move, ...}'],
              ]}
            />
          </SubSection>

          <SubSection title="Client Connection">
            <Text mb="sm">Browsers connect to <Key>/api/train/stream</Key> and receive a continuous stream of JSON events:</Text>
            <Pre>{`const evtSource = new EventSource('/api/train/stream');
evtSource.addEventListener('game_progress', (e) => {
  const data = JSON.parse(e.data);
  if (data.game_id > 0) {
    updateSideGame(data.game_id, data.move);
  } else {
    updateMainBoard(data.move);
  }
});`}</Pre>
            <Text mt="sm">Keepalive comments (lines starting with <Key>:</Key>) are sent every 15 seconds to prevent proxy timeouts. If the connection drops, the browser auto-reconnects within 3 seconds.</Text>
          </SubSection>
        </Section>

        {/* ─── 11. Frontend Architecture ─── */}
        <Section title="11. Frontend Architecture">
          <SubSection title="Technology Stack">
            <StyledTable
              headers={['Layer', 'Technology', 'Purpose']}
              rows={[
                ['Framework', 'React 18 (Vite)', 'Component rendering + build tooling'],
                ['UI Library', 'Mantine 7', 'Pre-built components (Paper, Badge, Table, etc.)'],
                ['Icons', 'Tabler Icons', 'Consistent iconography across sidebar panels'],
                ['Chess Board', 'react-chessboard', 'Board rendering + FEN display + move animations'],
                ['State', 'React Context', 'Global game state (GameContext) + useReducer'],
                ['Real-time', 'EventSource (SSE)', 'Live data stream from Flask backend'],
                ['Routing', 'react-router-dom', 'Dashboard vs Docs page routing'],
                ['Hosting', 'Vercel (static)', 'Frontend deployment + API proxy'],
              ]}
            />
          </SubSection>

          <SubSection title="Component Tree">
            <Pre>{`<App>
  ├─ <Routes>
  │   ├─ "/" → <Dashboard>
  │   │   ├─ <AppShell>
  │   │   │   ├─ <TopBar>                — Title, SSE badge, Docs/Flip/FS buttons
  │   │   │   └─ Main + Sidebar
  │   │   │       ├─ Main Column
  │   │   │       │   ├─ <PlayerInfoBar> — Top player (Self-Play NN)
  │   │   │       │   ├─ <EvalBar>       — White/black evaluation bar
  │   │   │       │   ├─ <LiveBoard>     — Main chess board (672px max)
  │   │   │       │   ├─ <PlayerInfoBar> — Bottom player (AI)
  │   │   │       │   ├─ <BoardNav>      — Move navigation (< > ↕)
  │   │   │       │   └─ <SideBoard> ×9  — Side game mini-boards (3×3 grid)
  │   │   │       └─ Sidebar (stacked)
  │   │   │           ├─ <MoveList>      — Move history with eval
  │   │   │           ├─ <SSEEventLog>   — Real-time event feed
  │   │   │           ├─ <AnalysisTable> — Stockfish analysis per move
  │   │   │           ├─ <MetricsGrid>   — Step/Loss/ELO/Buffer/Batch
  │   │   │           └─ <RecentGames>   — Finished games table
  │   │   └─ <FullscreenOverlay>         — Fullscreen board modal
  │   └─ "/docs" → <Docs>
  │       └─ <Readme>                    — This page
  └─ (GameContext provider wraps everything)`}</Pre>
          </SubSection>

          <SubSection title="State Management">
            <Text mb="sm">State flows through a single <Key>GameContext</Key> using <Key>useReducer</Key>:</Text>
            <Pre>{`interface GameState {
  allMoves: string[];           // Full SAN move list (main game)
  fenCache: string[];           // FEN after each move
  currentViewIndex: number;     // Which move we're viewing
  boardOrientation: 'white' | 'black';
  analysis: object | null;      // Stockfish analysis result
  trainingStatus: TrainingStatus | null;
  sseEvents: SSEEvent[];        // Event log
  sseStatus: string;            // 'Connected' | 'Connecting...' | 'Error'
  whatHappening: string;        // Status text shown in header
  sideFens: Record<number, string>;      // Per-game FENs for side boards
  sideMoveCounts: Record<number, number>; // Per-game move counts
  isFullscreen: boolean;
  autoFollow: boolean;          // Auto-scroll to latest move
}`}</Pre>
            <Text mt="sm">Updates come from two sources:</Text>
            <ol style={{ color: '#C9D1D9', lineHeight: '1.8' }}>
              <li><b>SSE hooks</b> (<Key>useSSE</Key>, <Key>useStatus</Key>, <Key>useAnalysis</Key>) — receive real-time data from the backend</li>
              <li><b>User interactions</b> — board navigation buttons, flip board, fullscreen toggle</li>
            </ol>
          </SubSection>

          <SubSection title="The SectionCard Pattern">
            <Text mb="sm">All sidebar panels use a shared <Key>SectionCard</Key> wrapper for consistent styling:</Text>
            <Pre>{`// SectionCard.tsx — reusable sidebar panel wrapper
<Paper p="md" radius="md"
  style={{ background: '#161B22', border: '1px solid #30363D' }}>
  <Group gap="xs" mb="xs">
    {icon}
    <Text size="xs" fw={700} c="#8B949E" tt="uppercase">
      {title}
    </Text>
    {rightSection && <div style={{ marginLeft: 'auto' }}>{rightSection}</div>}
  </Group>
  {children}
</Paper>`}</Pre>
            <Text mt="sm">This ensures every panel has identical background, border, padding, and header styling — the visual consistency you see across the entire sidebar.</Text>
          </SubSection>
        </Section>

        {/* ─── 12. ELO Estimation ─── */}
        <Section title="12. ELO Estimation">
          <Text mb="md">ELO is estimated by playing games against Stockfish (depth 10) at regular intervals during training.</Text>

          <SubSection title="The Formula">
            <Pre>{`# trainer.py:378-396
CALIBRATION_INTERVAL = 50  # steps between calibrations
base_elo = 200

# win_rate is the fraction of games won against Stockfish
# Range: [0.0, 1.0]
elo_offset = int(win_rate * 400 - 200)
return max(base_elo, base_elo + elo_offset)`}</Pre>
            <Text mt="sm">The formula maps win rate to ELO with a base of 200:</Text>
            <StyledTable
              headers={['Win Rate vs SF', 'ELO Offset', 'Estimated ELO']}
              rows={[
                ['0% (always loses)', '-200', '200'],
                ['25%', '-100', '300'],
                ['50%', '0', '400'],
                ['75%', '+100', '500'],
                ['100% (always wins)', '+200', '600'],
              ]}
            />
          </SubSection>

          <SubSection title="Calibration Games Setup">
            <Pre>{`Calibration match:
  - White = NN (raw policy, no MCTS — fastest play)
  - Black = Stockfish (depth 10)
  - Max 200 moves per game
  - Score: 1.0 for NN win, 0.5 for draw, 0.0 for loss
  - ELO = average over the most recent 10 calibration games`}</Pre>
            <Text mt="sm">
              The NN plays without MCTS during calibration to get pure network strength (MCTS would mask the network's weaknesses by searching deeper). A base ELO of 200 represents a randomly initialized network — as training progresses, the network climbs the scale.
            </Text>
            <Note>
              <Text size="sm" c="#8B949E"><b>Important:</b> Stockfish depth 10 is deliberately weaker than full-strength Stockfish. This ensures the NN can actually win some games, giving a meaningful ELO curve. Full-strength Stockfish (~3400 ELO) would crush the NN at every stage and always return 0% win rate.</Text>
            </Note>
          </SubSection>
        </Section>

        {/* ─── 13. Eval Cache ─── */}
        <Section title="13. Position Evaluation Cache">
          <Text mb="md">Position evaluations are cached server-side to avoid redundant Stockfish queries. The cache uses an LRU eviction policy with a TTL:</Text>
          <Pre>{`eval_cache = OrderedDict()     # FEN → {eval_cp, eval_norm, top_moves, move_analysis}
EVAL_CACHE_MAX = 2000           # Max entries — oldest evicted when full
EVAL_CACHE_TTL = 600            # 10 minute TTL — refreshed on re-request

# Persisted to disk on shutdown, loaded on restart
try:
    with open('eval_cache.json', 'r') as f:
        eval_cache.update(json.load(f))
except FileNotFoundError:
    pass`}</Pre>
          <Text mt="sm">
            The cache is also persisted to disk and loaded on server restart. This means frequently analyzed positions (like the starting position) are instant on subsequent requests, even after a server restart.
          </Text>
        </Section>

        {/* ─── 14. Glossary ─── */}
        <Section title="Glossary">
          <Text mb="md">Key terms used throughout the system:</Text>
          <StyledTable
            headers={['Term', 'Definition']}
            rows={[
              ['Centipawn', '1/100 of a pawn. Standard chess evaluation unit.'],
              ['CPuct', 'Exploration constant in PUCT selection formula.'],
              ['Critic Mode', 'Training mode where Stockfish critiques NN move choices.'],
              ['ELO', 'Rating system. Higher = stronger play. Base is 200.'],
              ['FEN', 'Forsyth–Edwards Notation. Compact chess position encoding.'],
              ['MCTS', 'Monte Carlo Tree Search. Search algorithm balancing exploration vs exploitation.'],
              ['MultiPV', 'Stockfish mode that returns evaluations for multiple moves simultaneously.'],
              ['NN', 'Neural Network — the ChessNet model.'],
              ['Policy', 'Probability distribution over legal moves. "What to play."'],
              ['PUCT', 'Polynomial Upper Confidence Trees. The selection formula for MCTS.'],
              ['Residual Block', 'NN building block with skip connections (prevents vanishing gradients).'],
              ['Self-Play', 'Training by playing against yourself — the classic RL approach.'],
              ['SSE', 'Server-Sent Events. HTTP-based real-time streaming protocol.'],
              ['Value', 'Position evaluation scalar [-1, +1]. "Who\'s winning."'],
              ['TrainingBuffer', 'Thread-safe circular deque storing up to 100K examples.'],
              ['BatchEvaluator', 'Micro-batching wrapper that coalesces GPU requests across threads.'],
              ['PGN', 'Portable Game Notation. Standard text format for recording chess games.'],
            ]}
          />
        </Section>

      </Stack>
    </Container>
  );
}
