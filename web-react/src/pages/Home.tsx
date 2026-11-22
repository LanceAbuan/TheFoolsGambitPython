import {
  Badge,
  Button,
  Card,
  Container,
  Group,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from '@mantine/core';
import {
  IconActivityHeartbeat,
  IconArrowRight,
  IconChess,
  IconInfoCircle,
} from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import { useMetrics } from '../lib/hooks/useMetrics';

function Stat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <Card>
      <Text fz="sm" c="dimmed">{label}</Text>
      <Text fz={28} fw={600} mt={4} className="tabular" lh={1.15}>
        {value}
      </Text>
      <Text fz="xs" c="dimmed" mt={4}>
        {hint}
      </Text>
    </Card>
  );
}

const DESTINATIONS = [
  {
    to: '/play',
    icon: IconChess,
    title: 'Play the bot',
    body: 'Sit across from the current champion. Legal moves, promotion, undo, resign.',
  },
  {
    to: '/watch',
    icon: IconActivityHeartbeat,
    title: 'Watch it train',
    body: 'Live self-play boards, the training journal, the Elo curve, arena gates.',
  },
  {
    to: '/about',
    icon: IconInfoCircle,
    title: 'How it works',
    body: 'An AlphaZero-shaped loop on one GPU: encoder, ResNet, search, gated ladder.',
  },
];

export function Home() {
  const { metrics } = useMetrics();
  const live = metrics?.trainer_alive === true;
  const elo = typeof metrics?.elo === 'number' ? Math.round(metrics.elo) : '—';
  const games = typeof metrics?.games_played === 'number' ? metrics.games_played : '—';
  const champion = metrics?.champion ?? '—';

  return (
    <Container size={1180} pt={{ base: 32, sm: 48 }} pb={64}>
      <Stack gap="lg">
        <Stack gap="md" maw={720}>
          <Group gap={8}>
            <Badge
              variant="light"
              color={live ? 'emerald' : 'gray'}
              size="sm"
              leftSection={<span className={live ? 'live-dot' : 'live-dot offline'} aria-hidden />}
            >
              {live ? 'Training live' : 'Training offline'}
            </Badge>
          </Group>

          <Title order={1}>A chess bot that learns in public.</Title>

          <Text size="lg" c="dimmed" maw={620}>
            The Fool plays itself on a local RTX 3080 and climbs a gated checkpoint ladder.
            You play whatever version currently holds the crown.
          </Text>

          <Group mt={4}>
            <Button component={Link} to="/play" size="md" rightSection={<IconArrowRight size={16} />}>
              Play the bot
            </Button>
            <Button component={Link} to="/watch" size="md" variant="default">
              Watch it train
            </Button>
          </Group>
        </Stack>

        <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="md">
          <Stat
            label="Estimated Elo"
            value={String(elo)}
            hint="Live rating from the arena ledger"
          />
          <Stat
            label="Games played"
            value={String(games)}
            hint="Self-play plus gated matches"
          />
          <Stat label="Champion" value={champion} hint="Current promoted checkpoint" />
        </SimpleGrid>

        <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="md">
          {DESTINATIONS.map(({ to, icon: Icon, title, body }) => (
            <Card key={to} component={Link} to={to} style={{ textDecoration: 'none' }}>
              <Group gap={10} mb={8}>
                <Icon size={18} stroke={1.6} />
                <Text fw={600}>{title}</Text>
              </Group>
              <Text fz="sm" c="dimmed" lh={1.55}>
                {body}
              </Text>
            </Card>
          ))}
        </SimpleGrid>
      </Stack>
    </Container>
  );
}
