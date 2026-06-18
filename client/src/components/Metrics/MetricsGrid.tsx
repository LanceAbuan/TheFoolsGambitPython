import { SimpleGrid, Text } from '@mantine/core';
import { useGame } from '../../GameContext';
import type { CycleInfo } from '../../types';

interface MetricCardProps {
  label: string;
  value: string;
}

function MetricCard({ label, value }: MetricCardProps) {
  return (
    <div className="metric-card">
      <Text className="metric-label">{label}</Text>
      <Text className="metric-value">{value}</Text>
    </div>
  );
}

function cycleLabel(c: CycleInfo, status: string): { line1: string; line2?: string } {
  if (status === 'playing' || status === 'critic' || status === 'self-play') {
    return { line1: `Game ${Math.min(c.games_this_cycle + 1, c.games_per_cycle)} / ${c.games_per_cycle}` };
  }
  if (status === 'training') {
    return { line1: `Step ${c.steps_this_cycle} / ${c.steps_per_cycle}` };
  }
  if (status === 'stockfish') {
    return { line1: 'Calibrating ELO', line2: 'vs Stockfish' };
  }
  // idle
  return { line1: `${c.games_per_cycle} games + ${c.steps_per_cycle} steps`, line2: 'Waiting for start...' };
}

export default function MetricsGrid() {
  const { state } = useGame();
  const s = state.trainingStatus;

  if (!s) {
    return (
      <div>
        <div className="section-header">Metrics</div>
        <Text size="sm" c="dimmed">No data</Text>
      </div>
    );
  }

  const c = s.cycle;
  const bufferLabel = c
    ? `${(s.buffer_size ?? 0).toLocaleString()} / ${c.min_buffer_for_train} min`
    : (s.buffer_size ?? 0).toLocaleString();

  const metrics = [
    { label: 'Status', value: s.status || '-' },
    { label: 'Step', value: (s.step ?? 0).toLocaleString() },
    { label: 'Games', value: (s.games_played ?? 0).toLocaleString() },
    { label: 'Loss', value: s.loss != null ? s.loss.toFixed(4) : '—' },
    { label: 'ELO', value: s.estimated_elo != null ? `~${s.estimated_elo}` : '—' },
    { label: 'Buffer', value: bufferLabel },
    { label: 'Side', value: (s.side_games_completed ?? 0).toLocaleString() },
  ];

  const cycleData = c ? cycleLabel(c, s.status) : null;

  return (
    <div>
      <div className="section-header">Metrics</div>
      <SimpleGrid cols={2} spacing="sm">
        {metrics.map((m) => (
          <MetricCard key={m.label} label={m.label} value={m.value} />
        ))}
      </SimpleGrid>
      {c && cycleData && (
        <div
          className="metric-card"
          style={{
            marginTop: 8,
            border: '1px solid #4a4540',
            background: '#2f2b27',
          }}
        >
          <Text className="metric-label">Cycle</Text>
          <Text className="metric-value" style={{ fontSize: 16 }}>{cycleData.line1}</Text>
          {cycleData.line2 && (
            <Text style={{ fontSize: 11, color: '#6b6560', marginTop: 2 }}>{cycleData.line2}</Text>
          )}
        </div>
      )}
    </div>
  );
}
