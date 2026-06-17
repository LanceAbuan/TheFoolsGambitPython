import { SimpleGrid, Text } from '@mantine/core';
import { useGame } from '../../GameContext';

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

  const metrics = [
    { label: 'Status', value: s.status || '-' },
    { label: 'Step', value: (s.step ?? 0).toLocaleString() },
    { label: 'Games', value: (s.games_played ?? 0).toLocaleString() },
    { label: 'Loss', value: s.loss != null ? s.loss.toFixed(4) : '—' },
    { label: 'ELO', value: s.estimated_elo != null ? `~${s.estimated_elo}` : '—' },
    { label: 'Buffer', value: (s.buffer_size ?? 0).toLocaleString() },
    { label: 'Side', value: (s.side_games_completed ?? 0).toLocaleString() },
  ];

  return (
    <div>
      <div className="section-header">Metrics</div>
      <SimpleGrid cols={2} spacing="sm">
        {metrics.map((m) => (
          <MetricCard key={m.label} label={m.label} value={m.value} />
        ))}
      </SimpleGrid>
    </div>
  );
}
