import { SimpleGrid, Text, Paper, Badge } from '@mantine/core';
import { IconActivity } from '@tabler/icons-react';
import { useGame } from '../../GameContext';
import type { CycleInfo } from '../../types';
import SectionCard from '../Layout/SectionCard';

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <Paper p="sm" radius="sm" style={{ background: '#0D1117', border: '1px solid #30363D' }}>
      <Text size="xs" c="#6E7681" tt="uppercase" style={{ letterSpacing: '0.5px' }}>
        {label}
      </Text>
      <Text size="xl" fw={700} c="#C9D1D9" mt={2}>
        {value}
      </Text>
    </Paper>
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
  return { line1: `${c.games_per_cycle} games + ${c.steps_per_cycle} steps`, line2: 'Waiting for start...' };
}

export default function MetricsGrid() {
  const { state } = useGame();
  const s = state.trainingStatus;

  if (!s) {
    return (
      <SectionCard icon={<IconActivity size={16} color="#8B949E" />} title="Metrics">
        <Text size="sm" c="dimmed">No data</Text>
      </SectionCard>
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

  const statusColor = s.status === 'running' || s.status === 'playing' ? 'green' : 'gray';

  return (
    <SectionCard
      icon={<IconActivity size={16} color="#8B949E" />}
      title="Metrics"
      rightSection={
        <Badge size="sm" color={statusColor} variant="filled">
          {s.status}
        </Badge>
      }
    >
      <SimpleGrid cols={2} spacing="xs">
        {metrics.map((m) => (
          <MetricCard key={m.label} label={m.label} value={m.value} />
        ))}
      </SimpleGrid>

      {c && cycleData && (
        <Paper p="sm" radius="sm" mt="sm" style={{ background: '#0D1117', border: '1px solid #30363D' }}>
          <Text size="xs" c="#6E7681" tt="uppercase" style={{ letterSpacing: '0.5px' }}>
            Cycle
          </Text>
          <Text size="md" fw={700} c="#C9D1D9" mt={2}>
            {cycleData.line1}
          </Text>
          {cycleData.line2 && (
            <Text size="xs" c="#6E7681" mt={2}>{cycleData.line2}</Text>
          )}
        </Paper>
      )}
    </SectionCard>
  );
}
