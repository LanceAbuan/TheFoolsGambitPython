import { Group, Text, Paper, Badge, Progress, SimpleGrid } from '@mantine/core';
import { useGame } from '../GameContext';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

function MetricCard({ label, value, data }: { label: string; value: string; data?: number[] }) {
  const chartData = data?.map((v: number, i: number) => ({ x: i, y: v })) || [];

  return (
    <Paper p="sm" radius="sm" style={{ background: '#0D1117', border: '1px solid #21262D' }}>
      <Text size="xs" c="#6E7681" tt="uppercase" style={{ letterSpacing: '0.5px' }}>
        {label}
      </Text>
      <Text size="xl" fw={700} c="#C9D1D9" mt={2}>
        {value}
      </Text>
      {chartData.length > 0 && (
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

function TrainingStatusCard() {
  const { state } = useGame();
  const s = state.trainingStatus;
  const statusColor = s?.status === 'running' || s?.status === 'playing' ? 'green' : 'gray';

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
            {s?.games_played ?? 0} / {s?.cycle?.games_per_cycle ?? 10}
          </Text>
        </Group>
        <Progress
          value={((s?.games_played ?? 0) / (s?.cycle?.games_per_cycle ?? 10)) * 100}
          color="green"
          size="sm"
        />

        <Group justify="space-between" mt={4}>
          <Text size="sm" c="#8B949E">Estimated Time Remaining</Text>
          <Text size="sm" fw={600} c="#C9D1D9">2h 15m</Text>
        </Group>

        <Group justify="space-between">
          <Text size="sm" c="#8B949E">Training Iteration</Text>
          <Text size="sm" fw={600} c="#C9D1D9">
            {(s?.step ?? 0).toLocaleString()}
          </Text>
        </Group>

        <Group justify="space-between">
          <Text size="sm" c="#8B949E">Current Checkpoint</Text>
          <Text size="sm" fw={600} c="#C9D1D9">checkpoint_1245.pt</Text>
        </Group>
      </div>
    </Paper>
  );
}

function PerformanceCard() {
  const { state } = useGame();
  const s = state.trainingStatus;

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 12 }}>
        Performance
      </Text>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div>
          <Text size="xs" c="#6E7681">ELO</Text>
          <Group gap="xs" align="baseline">
            <Text size="2xl" fw={700} c="#C9D1D9">
              {s?.estimated_elo ?? '—'}
            </Text>
            <Text size="xs" c="#3fb950">+18</Text>
          </Group>
        </div>

        <div>
          <Text size="xs" c="#6E7681">Win Rate (vs Stockfish)</Text>
          <Group gap="xs" align="baseline">
            <Text size="xl" fw={700} c="#C9D1D9">56.3%</Text>
            <Text size="xs" c="#3fb950">+2.1%</Text>
          </Group>
        </div>

        <div>
          <Text size="xs" c="#6E7681">Average Game Length</Text>
          <Group gap="xs" align="baseline">
            <Text size="xl" fw={700} c="#C9D1D9">78.4</Text>
            <Text size="xs" c="#3fb950">+5.2</Text>
          </Group>
        </div>
      </div>
    </Paper>
  );
}

function TopOpeningsCard() {
  const openings = [
    { name: 'Sicilian Defense', nn: 32, sf: 58 },
    { name: 'French Defense', nn: 21, sf: 55 },
    { name: 'Caro-Kann', nn: 12, sf: 53 },
    { name: 'English Opening', nn: 8, sf: 57 },
    { name: "Queen's Gambit", nn: 7, sf: 54 },
  ];

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 12 }}>
        Top Openings
      </Text>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <Group gap="xs" c="#6E7681" mb={4}>
          <Text size="xs" style={{ flex: 1 }}></Text>
          <Text size="xs" style={{ width: 40, textAlign: 'right' }}>NN</Text>
          <Text size="xs" style={{ width: 40, textAlign: 'right' }}>Win Rate</Text>
        </Group>
        {openings.map((o, i) => (
          <Group key={i} gap="xs">
            <Text size="sm" c="#6E7681" style={{ width: 16 }}>{i + 1}.</Text>
            <Text size="sm" c="#C9D1D9" style={{ flex: 1 }}>{o.name}</Text>
            <Text size="sm" c="#8B949E" style={{ width: 40, textAlign: 'right' }}>{o.nn}%</Text>
            <Badge size="sm" color={o.sf >= 55 ? 'green' : 'gray'} variant="filled" style={{ width: 40, justifyContent: 'center' }}>
              {o.sf}%
            </Badge>
          </Group>
        ))}
      </div>
    </Paper>
  );
}

function ResourcesCard() {
  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 12 }}>
        Resources
      </Text>

      <SimpleGrid cols={2} spacing="md">
        <div>
          <Group justify="space-between" mb={4}>
            <Text size="xs" c="#8B949E">GPU Usage</Text>
            <Text size="xs" fw={600} c="#C9D1D9">97%</Text>
          </Group>
          <Progress value={97} color="green" size="sm" />
          <Text size="xs" c="#6E7681" mt={2}>CUDA</Text>
        </div>

        <div>
          <Group justify="space-between" mb={4}>
            <Text size="xs" c="#8B949E">VRAM</Text>
            <Text size="xs" fw={600} c="#C9D1D9">14.2 / 16 GB</Text>
          </Group>
          <Progress value={88.75} color="blue" size="sm" />
        </div>

        <div>
          <Group justify="space-between" mb={4}>
            <Text size="xs" c="#8B949E">CPU Usage</Text>
            <Text size="xs" fw={600} c="#C9D1D9">82%</Text>
          </Group>
          <Progress value={82} color="yellow" size="sm" />
          <Text size="xs" c="#6E7681" mt={2}>PyTorch</Text>
        </div>

        <div>
          <Group justify="space-between" mb={4}>
            <Text size="xs" c="#8B949E">RAM</Text>
            <Text size="xs" fw={600} c="#C9D1D9">22.1 / 32 GB</Text>
          </Group>
          <Progress value={69} color="blue" size="sm" />
        </div>
      </SimpleGrid>
    </Paper>
  );
}

function ThroughputCard() {
  const metrics = [
    { label: 'Games / Hour', value: '423' },
    { label: 'Positions / Sec', value: '18,432' },
    { label: 'MCTS / Sec', value: '11,302' },
    { label: 'Training Steps / Sec', value: '128' },
  ];

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 12 }}>
        Throughput
      </Text>

      <SimpleGrid cols={2} spacing="md">
        {metrics.map((m) => (
          <div key={m.label}>
            <Text size="xs" c="#6E7681">{m.label}</Text>
            <Text size="lg" fw={700} c="#C9D1D9">{m.value}</Text>
          </div>
        ))}
      </SimpleGrid>
    </Paper>
  );
}

function RecentImprovementsCard() {
  const improvements = [
    { text: '+18 ELO vs Stockfish', time: '2h ago', color: '#3fb950' },
    { text: '+2.1% Win Rate vs Last Checkpoint', time: '3h ago', color: '#3fb950' },
    { text: '-0.03 Policy Loss', time: '4h ago', color: '#3fb950' },
  ];

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 12 }}>
        Recent Improvements
      </Text>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {improvements.map((imp, i) => (
          <Group key={i} justify="space-between">
            <Text size="sm" c={imp.color}>{imp.text}</Text>
            <Text size="xs" c="#6E7681">{imp.time}</Text>
          </Group>
        ))}
      </div>
    </Paper>
  );
}

export default function TrainingView() {
  // Generate dummy metric data for sparklines
  const policyLossData = Array.from({ length: 20 }, (_, i) => 0.192 + Math.sin(i * 0.5) * 0.02);
  const valueLossData = Array.from({ length: 20 }, (_, i) => 0.281 + Math.cos(i * 0.4) * 0.03);
  const entropyData = Array.from({ length: 20 }, (_, i) => 1.45 + Math.sin(i * 0.6) * 0.1);

  return (
    <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
      <Text size="xl" fw={700} c="#C9D1D9" mb={4}>Training View</Text>
      <Text size="sm" c="#6E7681" mb={24}>Training metrics and performance</Text>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr 320px', gap: 16 }}>
        {/* Left column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <TrainingStatusCard />
          <PerformanceCard />
        </div>

        {/* Center column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px' }}>
            Metrics
          </Text>
          <SimpleGrid cols={3} spacing="sm">
            <MetricCard label="Policy Loss" value="0.192" data={policyLossData} />
            <MetricCard label="Value Loss" value="0.281" data={valueLossData} />
            <MetricCard label="Entropy" value="1.45" data={entropyData} />
            <MetricCard label="Learning Rate" value="0.00025" />
            <MetricCard label="Grad Norm" value="2.34" />
            <MetricCard label="Explained Var" value="0.67" />
          </SimpleGrid>

          <ResourcesCard />
          <ThroughputCard />
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
