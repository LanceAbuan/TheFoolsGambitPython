import { Group, Text, Paper, Badge, Progress, SimpleGrid } from '@mantine/core';
import { useGame } from '../GameContext';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import type { TrainingStatus, SystemResources } from '../types';

function MetricCard({ label, value, data }: { label: string; value: string; data?: { t: number; v: number }[] }) {
  const chartData = data?.map((d, i) => ({ x: i, y: d.v })) || [];

  return (
    <Paper p="sm" radius="sm" style={{ background: '#0D1117', border: '1px solid #21262D' }}>
      <Text size="xs" c="#6E7681" tt="uppercase" style={{ letterSpacing: '0.5px' }}>
        {label}
      </Text>
      <Text size="xl" fw={700} c="#C9D1D9" mt={2}>
        {value}
      </Text>
      {chartData.length > 1 && (
        <div style={{ height: 30, marginTop: 4 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <Line
                type="monotone"
                dataKey="y"
                stroke="#58a6ff"
                strokeWidth={1}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Paper>
  );
}

function TrainingStatusCard({ s }: { s: TrainingStatus | null }) {
  const statusColor = s?.status === 'running' || s?.status === 'playing' || s?.status === 'critic' || s?.status === 'self-play' ? 'green'
    : s?.status === 'training' ? 'blue'
    : s?.status === 'stockfish' ? 'yellow'
    : 'gray';

  const gamesPlayed = s?.games_played ?? 0;
  const gamesPerCycle = s?.cycle?.games_per_cycle ?? 10;
  const stepsThisCycle = s?.cycle?.steps_this_cycle ?? 0;
  const stepsPerCycle = s?.cycle?.steps_per_cycle ?? 20;

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Group justify="space-between" mb="sm">
        <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px' }}>
          Training Status
        </Text>
        <Badge size="sm" color={statusColor} variant="filled">
          {s?.status || 'idle'}
        </Badge>
      </Group>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <Group justify="space-between">
          <Text size="sm" c="#8B949E">Self-Play Games</Text>
          <Text size="sm" fw={600} c="#C9D1D9">
            {gamesPlayed} / {gamesPerCycle}
          </Text>
        </Group>
        <Progress
          value={(gamesPlayed / gamesPerCycle) * 100}
          color="green"
          size="sm"
        />

        <Group justify="space-between" mt={4}>
          <Text size="sm" c="#8B949E">Cycle Progress</Text>
          <Text size="sm" fw={600} c="#C9D1D9">
            Step {stepsThisCycle} / {stepsPerCycle}
          </Text>
        </Group>

        <Group justify="space-between">
          <Text size="sm" c="#8B949E">Training Iteration</Text>
          <Text size="sm" fw={600} c="#C9D1D9">
            {(s?.step ?? 0).toLocaleString()}
          </Text>
        </Group>

        <Group justify="space-between">
          <Text size="sm" c="#8B949E">Buffer Size</Text>
          <Text size="sm" fw={600} c="#C9D1D9">
            {(s?.buffer_size ?? 0).toLocaleString()}
          </Text>
        </Group>
      </div>
    </Paper>
  );
}

function PerformanceCard({ s }: { s: TrainingStatus | null }) {
  const elo = s?.estimated_elo;
  const sfResults = (s as any)?.sf_calibration_results || [];
  const recentWins = sfResults.filter((r: number) => r === 1.0).length;
  const recentGames = sfResults.length;
  const winRate = recentGames > 0 ? ((recentWins / recentGames) * 100).toFixed(1) : '—';

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 12 }}>
        Performance
      </Text>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div>
          <Text size="xs" c="#6E7681">ELO (estimated)</Text>
          <Group gap="xs" align="baseline">
            <Text size="2xl" fw={700} c="#C9D1D9">
              {elo ?? '—'}
            </Text>
          </Group>
        </div>

        <div>
          <Text size="xs" c="#6E7681">Win Rate (vs Stockfish, last {recentGames})</Text>
          <Group gap="xs" align="baseline">
            <Text size="xl" fw={700} c="#C9D1D9">{winRate}%</Text>
          </Group>
        </div>

        <div>
          <Text size="xs" c="#6E7681">Total Games Played</Text>
          <Group gap="xs" align="baseline">
            <Text size="xl" fw={700} c="#C9D1D9">{s?.games_played ?? 0}</Text>
          </Group>
        </div>
      </div>
    </Paper>
  );
}

function TopOpeningsCard() {
  // Placeholder — would need backend tracking of opening names
  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 12 }}>
        Top Openings
      </Text>
      <Text size="sm" c="#6E7681">No opening data yet</Text>
    </Paper>
  );
}

function ResourcesCard({ resources }: { resources: SystemResources | null | undefined }) {
  if (!resources) {
    return (
      <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
        <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 12 }}>
          Resources
        </Text>
        <Text size="sm" c="#6E7681">Loading...</Text>
      </Paper>
    );
  }

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 12 }}>
        Resources
      </Text>

      <SimpleGrid cols={2} spacing="md">
        <div>
          <Group justify="space-between" mb={4}>
            <Text size="xs" c="#8B949E">CPU Usage</Text>
            <Text size="xs" fw={600} c="#C9D1D9">{resources.cpu_percent}%</Text>
          </Group>
          <Progress value={resources.cpu_percent} color={resources.cpu_percent > 90 ? 'red' : resources.cpu_percent > 70 ? 'yellow' : 'green'} size="sm" />
          <Text size="xs" c="#6E7681" mt={2}>{resources.cpu_count} cores</Text>
        </div>

        <div>
          <Group justify="space-between" mb={4}>
            <Text size="xs" c="#8B949E">RAM</Text>
            <Text size="xs" fw={600} c="#C9D1D9">{resources.ram.used_gb} / {resources.ram.total_gb} GB</Text>
          </Group>
          <Progress value={resources.ram.percent} color={resources.ram.percent > 90 ? 'red' : resources.ram.percent > 70 ? 'yellow' : 'blue'} size="sm" />
        </div>

        {resources.gpu && (
          <>
            <div>
              <Group justify="space-between" mb={4}>
                <Text size="xs" c="#8B949E">GPU</Text>
                <Text size="xs" fw={600} c="#C9D1D9">{resources.gpu.utilization_percent ?? '—'}%</Text>
              </Group>
              {(resources.gpu.utilization_percent ?? 0) > 0 && (
                <Progress value={resources.gpu.utilization_percent ?? 0} color="green" size="sm" />
              )}
              <Text size="xs" c="#6E7681" mt={2}>{resources.gpu.name}</Text>
            </div>

            <div>
              <Group justify="space-between" mb={4}>
                <Text size="xs" c="#8B949E">VRAM</Text>
                <Text size="xs" fw={600} c="#C9D1D9">{resources.gpu.memory_used_gb} / {resources.gpu.memory_total_gb} GB</Text>
              </Group>
              <Progress value={resources.gpu.memory_percent} color="blue" size="sm" />
            </div>
          </>
        )}
      </SimpleGrid>
    </Paper>
  );
}

function ThroughputCard({ s }: { s: TrainingStatus | null }) {
  const t = s?.throughput;

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 12 }}>
        Throughput
      </Text>

      <SimpleGrid cols={2} spacing="md">
        <div>
          <Text size="xs" c="#6E7681">Games / Hour</Text>
          <Text size="lg" fw={700} c="#C9D1D9">{t?.games_per_hour ?? '—'}</Text>
        </div>
        <div>
          <Text size="xs" c="#6E7681">Positions / Sec</Text>
          <Text size="lg" fw={700} c="#C9D1D9">{t?.positions_per_sec?.toLocaleString() ?? '—'}</Text>
        </div>
        <div>
          <Text size="xs" c="#6E7681">Train Steps / Sec</Text>
          <Text size="lg" fw={700} c="#C9D1D9">{t?.train_steps_per_sec ?? '—'}</Text>
        </div>
        <div>
          <Text size="xs" c="#6E7681">Side Games</Text>
          <Text size="lg" fw={700} c="#C9D1D9">{s?.side_games_completed ?? 0}</Text>
        </div>
      </SimpleGrid>
    </Paper>
  );
}

function RecentImprovementsCard() {
  // Would need backend historical tracking of ELO/win rate changes
  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 12 }}>
        Recent Improvements
      </Text>
      <Text size="sm" c="#6E7681">No improvement data yet</Text>
    </Paper>
  );
}

export default function TrainingView() {
  const { state } = useGame();
  const s = state.trainingStatus;
  const metricHistory = state.metricHistory;

  return (
    <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
      <Text size="xl" fw={700} c="#C9D1D9" mb={4}>Training View</Text>
      <Text size="sm" c="#6E7681" mb={24}>Training metrics and performance</Text>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr 320px', gap: 16 }}>
        {/* Left column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <TrainingStatusCard s={s} />
          <PerformanceCard s={s} />
        </div>

        {/* Center column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px' }}>
            Metrics
          </Text>
          <SimpleGrid cols={3} spacing="sm">
            <MetricCard
              label="Policy Loss"
              value={s?.policy_loss != null ? s.policy_loss.toFixed(4) : '—'}
              data={metricHistory.policy_loss}
            />
            <MetricCard
              label="Value Loss"
              value={s?.value_loss != null ? s.value_loss.toFixed(4) : '—'}
              data={metricHistory.value_loss}
            />
            <MetricCard
              label="Total Loss"
              value={s?.loss != null ? s.loss.toFixed(4) : '—'}
              data={metricHistory.loss}
            />
            <MetricCard
              label="ELO"
              value={s?.estimated_elo != null ? `~${s.estimated_elo}` : '—'}
              data={metricHistory.elo}
            />
            <MetricCard
              label="Learning Rate"
              value={s?.learning_rate != null ? s.learning_rate.toExponential(1) : '—'}
            />
            <MetricCard
              label="Buffer Size"
              value={(s?.buffer_size ?? 0).toLocaleString()}
              data={metricHistory.buffer_size}
            />
          </SimpleGrid>

          <ResourcesCard resources={s?.resources} />
          <ThroughputCard s={s} />
        </div>

        {/* Right column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <TopOpeningsCard />
          <RecentImprovementsCard />
        </div>
      </div>
    </div>
  );
}
