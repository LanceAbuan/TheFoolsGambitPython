import { Title, Text, Container, Divider, Box, Stack } from '@mantine/core';

export default function Readme() {
  return (
    <Container size={800} py="xl">
      <Title order={1} mb="md">Architecture Deep Dive</Title>
      <Text c="dimmed" style={{ fontStyle: 'italic' }}>
        Everything you want to know about how Fool's Gambit works, from the neural network to the dashboard.
      </Text>
      
      <Divider my="xl" />
      
      <Stack gap="xl">
        <Box>
          <Title order={2}>System Architecture</Title>
          <Text style={{ whiteSpace: 'pre', fontFamily: 'monospace', padding: '1rem', background: '#161B22', border: '1px solid #30363D' }}>
{`┌─────────────────────────────────────────────────────────────────────┐
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
└─────────────────────────────────────────────────────────────────────┘`}
          </Text>
          <Text mt="md">The system operates in one of two mutually exclusive modes:</Text>
          <Box mt="sm" style={{ borderLeft: '2px solid #30363D', paddingLeft: '1rem' }}>
            <Text><b>Self-Play:</b> NN + MCTS plays both sides. Training data = MCTS visit counts as policy targets.</Text>
            <Text mt="xs"><b>Critic:</b> NN picks moves, Stockfish evaluates all legal moves to build policy targets.</Text>
          </Box>
        </Box>

        <Box>
          <Title order={2}>Neural Network Design</Title>
          <Text>
            The <b>ChessNet</b> architecture is a tiny but efficient ResNet.
          </Text>
          <Text mt="sm" style={{ whiteSpace: 'pre', fontFamily: 'monospace', padding: '1rem', background: '#161B22', border: '1px solid #30363D' }}>
{`Input: 8×8×16 board tensor
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
└───────────────────────────────────────┴────────────────────────────`}
          </Text>
          <Text mt="md">
            Why so small? Because it allows for <b>real-time training</b>. We can run hundreds of forward passes per second on a single GPU, letting users see the network improve in minutes.
          </Text>
        </Box>

        <Box>
          <Title order={2}>Monte Carlo Tree Search</Title>
          <Text>
            MCTS balances exploration vs exploitation using the <b>PUCT</b> formula:
          </Text>
          <Text mt="sm" style={{ fontStyle: 'italic', color: '#8B949E' }}>
            score = Q(s,a) + cpuct * P(s,a) * sqrt(N_parent) / (1 + n_child)
          </Text>
          <Text mt="md">
            The system uses <b>Parallel MCTS</b>, distributing simulations across 8 threads. The NN provides the prior probabilities (P), and Stockfish provides a blended evaluation for leaf nodes.
          </Text>
        </Box>
      </Stack>
    </Container>
  );
}
