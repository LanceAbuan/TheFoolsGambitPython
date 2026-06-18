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

function cycleLabel(c: CycleInfo, status: string): string {
  const isPlaying = status === 'playing' || status === 'critic' || status === 'self-play';
  const isTraining = status === 'training';
  if (c.games_per_cycle > 0 && isPlaying) {
    return `Game ${Math.min(c.games_this_cycle + 1, c.games_per_cycle)}/${c.games_per_cycle}`;
  }
  if (c.steps_per_cycle > 0 && isTraining) {
    return `Train ${c.steps_this_cycle}/${c.steps_per_cycle}`;
  }
  return '—';
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

  const cycleText = c ? cycleLabel(c, s.status) : null;

  return (
    <div>
      <div className="section-header">Metrics</div>
      <SimpleGrid cols={2} spacing="sm">
        {metrics.map((m) => (
          <MetricCard key={m.label} label={m.label} value={m.value} />
        ))}
      </SimpleGrid>
      {cycleText && cycleText !== '—' && (
        <div className="metric-card" style={{ marginTop: 8 }}>
          <Text className="metric-label">Cycle</Text>
          <Text className="metric-value" style={{ fontSize: 14 }}>{cycleText}</Text>
        </div>
      )}
    </div>
  );
}
