import { Group, Text, Paper, Badge } from '@mantine/core';
import { useGame } from '../GameContext';
import LiveBoard from '../components/Board/LiveBoard';
import EvalBar from '../components/Board/EvalBar';
import PlayerInfoBar from '../components/Board/PlayerInfoBar';
import BoardNav from '../components/Board/BoardNav';
import MoveList from '../components/Analysis/MoveList';
import type { AnalysisResult } from '../types';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, ReferenceLine } from 'recharts';

function EngineEvaluation() {
  const { state } = useGame();
  const analysis = state.analysis as AnalysisResult | null;
  const evalScore = analysis ? (analysis.evaluation / 100) : 0;
  const normalizedEval = analysis?.evaluation_normalized ?? 0;

  // Generate dummy eval history for chart (in real app, this would come from state)
  const evalHistory = state.allMoves.map((_: string, i: number) => ({
    move: i + 1,
    eval: (Math.sin(i * 0.3) * 0.5 + normalizedEval * 0.5).toFixed(2),
  }));

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 8 }}>
        Engine Evaluation
      </Text>
      <Group justify="space-between" align="flex-end">
        <Text size="3xl" fw={700} c={evalScore > 0 ? '#3fb950' : evalScore < 0 ? '#f85149' : '#C9D1D9'}>
          {evalScore > 0 ? '+' : ''}{evalScore.toFixed(2)}
        </Text>
        <div style={{ textAlign: 'right' }}>
          <Text size="xs" c="#6E7681">Depth 24</Text>
          <Text size="xs" c="#6E7681">Stockfish 16</Text>
        </div>
      </Group>
      <Text size="sm" c="#8B949E" mt={4}>
        {evalScore > 0.1 ? 'White is better' : evalScore < -0.1 ? 'Black is better' : 'Equal position'}
      </Text>

      {/* Mini eval chart */}
      <div style={{ height: 60, marginTop: 12 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={evalHistory}>
            <XAxis dataKey="move" hide />
            <YAxis hide domain={[-2, 2]} />
            <ReferenceLine y={0} stroke="#30363D" />
            <Line
              type="monotone"
              dataKey="eval"
              stroke="#58a6ff"
              strokeWidth={1.5}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Paper>
  );
}

function TopMoves() {
  const { state } = useGame();
  const analysis = state.analysis as AnalysisResult | null;
  const moves = analysis?.move_analysis?.slice(0, 4) || [];

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 8 }}>
        Top Moves
      </Text>
      {moves.length === 0 ? (
        <Text size="sm" c="#6E7681">No analysis available</Text>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {moves.map((move: { san: string; eval: string }, i: number) => {
            const evalVal = parseFloat(move.eval) || 0;
            const winPct = Math.max(0, Math.min(100, 50 + evalVal * 10));
            return (
              <Group key={i} justify="space-between" gap="xs">
                <Group gap="xs">
                  <Text size="sm" c="#6E7681" style={{ width: 16 }}>{i + 1}.</Text>
                  <Text size="sm" fw={600} c="#C9D1D9" style={{ fontFamily: 'ui-monospace, monospace' }}>
                    {move.san}
                  </Text>
                </Group>
                <Group gap="xs">
                  <Text size="xs" c={evalVal > 0 ? '#3fb950' : evalVal < 0 ? '#f85149' : '#8B949E'}>
                    {evalVal > 0 ? '+' : ''}{evalVal.toFixed(2)}
                  </Text>
                  <Text size="xs" c="#6E7681">{winPct.toFixed(0)}%</Text>
                </Group>
              </Group>
            );
          })}
        </div>
      )}
    </Paper>
  );
}

function AnalysisMetrics() {
  const { state } = useGame();
  const analysis = state.analysis as AnalysisResult | null;
  const rows = analysis?.move_analysis || [];

  const bestMove = rows.find((r: { quality: string }) => r.quality === 'best');
  const blunder = rows.find((r: { quality: string }) => r.quality === 'blunder');
  const mistake = rows.find((r: { quality: string }) => r.quality === 'bad');

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 8 }}>
        Analysis
      </Text>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <div>
          <Text size="xs" c="#6E7681">Accuracy</Text>
          <Group gap="xs">
            <Badge size="sm" color="blue" variant="filled">NN: 88.7</Badge>
            <Badge size="sm" color="gray" variant="filled">SF: 92.1</Badge>
          </Group>
        </div>
        <div>
          <Text size="xs" c="#6E7681">Best Move</Text>
          <Text size="sm" fw={600} c="#3fb950" style={{ fontFamily: 'ui-monospace, monospace' }}>
            {bestMove?.san || '—'}
          </Text>
        </div>
        <div>
          <Text size="xs" c="#6E7681">Mistake</Text>
          <Text size="sm" fw={600} c="#d6b81e" style={{ fontFamily: 'ui-monospace, monospace' }}>
            {mistake?.san || '—'}
          </Text>
        </div>
        <div>
          <Text size="xs" c="#6E7681">Blunder</Text>
          <Text size="sm" fw={600} c="#f85149" style={{ fontFamily: 'ui-monospace, monospace' }}>
            {blunder?.san || '—'}
          </Text>
        </div>
      </div>
    </Paper>
  );
}

function EvalChart() {
  const { state } = useGame();

  // Generate dummy eval history for chart
  const evalHistory = state.allMoves.map((_: string, i: number) => ({
    move: i + 1,
    eval: (Math.sin(i * 0.3) * 0.8 + Math.random() * 0.2).toFixed(2),
  }));

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 8 }}>
        Evaluation Over Time
      </Text>
      <div style={{ height: 120 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={evalHistory}>
            <XAxis dataKey="move" stroke="#6E7681" tick={{ fontSize: 10 }} />
            <YAxis stroke="#6E7681" tick={{ fontSize: 10 }} domain={[-2, 2]} />
            <ReferenceLine y={0} stroke="#30363D" />
            <Line
              type="monotone"
              dataKey="eval"
              stroke="#58a6ff"
              strokeWidth={1.5}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Paper>
  );
}

function OpeningInfo() {
  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 8 }}>
        Opening Info
      </Text>
      <Text size="sm" fw={600} c="#C9D1D9">Sicilian Defense</Text>
      <Text size="xs" c="#6E7681">Najdorf Variation</Text>
      <Group justify="space-between" mt={8}>
        <Text size="xs" c="#6E7681">Move 12</Text>
        <Text size="xs" c="#6E7681">Popularity: 45%</Text>
      </Group>
    </Paper>
  );
}

export default function AnalysisView() {
  const { state } = useGame();
  const whiteToMove = state.allMoves.length % 2 === 0;

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* Left: Board */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px 0', overflow: 'auto' }}>
        <Group gap="xs" mb={8}>
          <Text fw={700} size="lg" c="#C9D1D9">
            Analysis
          </Text>
        </Group>

        <PlayerInfoBar
          name="Self-Play (NN)"
          detail="vs Self"
          isTurn={!whiteToMove}
          color="top"
        />

        <div style={{ display: 'flex', alignItems: 'stretch', gap: 8, width: '100%', maxWidth: 672 }}>
          <EvalBar />
          <LiveBoard />
        </div>

        <PlayerInfoBar
          name="Fool's Gambit AI"
          detail="Training"
          isTurn={whiteToMove}
          color="bottom"
        />

        <BoardNav />
      </div>

      {/* Right: Analysis panels */}
      <div
        style={{
          width: 320,
          flexShrink: 0,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          padding: '16px',
          overflow: 'auto',
          borderLeft: '1px solid #30363D',
        }}
      >
        <EngineEvaluation />
        <TopMoves />

        <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
          <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 8 }}>
            Move List
          </Text>
          <MoveList />
        </Paper>

        <AnalysisMetrics />
        <EvalChart />
        <OpeningInfo />
      </div>
    </div>
  );
}
