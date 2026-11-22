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
    <Card padding="lg">
      <Text fz="xs" c="dimmed" tt="uppercase" lts="0.14em">
        {label}
      </Text>
      <Text fz={30} fw={650} mt={6} className="tabular" lh={1.1}>
        {value}
      </Text>
      <Text fz="xs" c="dimmed" mt={6}>
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
    body: 'Take the board against the current champion. Legal moves, promotion, undo, resign — all server-side.',
  },
  {
    to: '/watch',
    icon: IconActivityHeartbeat,
    title: 'Watch it train',
    body: 'Live self-play boards, the training journal, the Elo curve and every arena gate result.',
  },
  {
    to: '/about',
    icon: IconInfoCircle,
    title: 'How it works',
    body: 'An AlphaZero-shaped loop: 29-plane encoder, ResNet, PUCT search, self-play and a gated ladder.',
  },
];

export function Home() {
  const { metrics } = useMetrics();
  const live = metrics?.trainer_alive === true;
  const elo = typeof metrics?.elo === 'number' ? Math.round(metrics.elo) : '—';
  const games = typeof metrics?.games_played === 'number' ? metrics.games_played : '—';
  const champion = metrics?.champion ?? '—';

  return (
    <div className="app-glow">
      <Container size={1180} pt={{ base: 48, sm: 80 }} pb={80}>
        <Stack gap="xl">
          <Stack gap="md" maw={760}>
            <Badge
              variant="light"
              color={live ? 'emerald' : 'gray'}
              size="lg"
              radius="sm"
              leftSection={
                <span
                  aria-hidden
                  style={{
                    width: 7,
                    height: 7,
                    borderRadius: 99,
                    background: live
                      ? 'var(--mantine-color-emerald-6)'
                      : 'var(--mantine-color-gray-5)',
                    display: 'inline-block',
                  }}
                />
              }
            >
              {live ? 'TRAINING LIVE' : 'TRAINING OFFLINE'}
            </Badge>

            <Title order={1}>A chess bot that learns in public.</Title>

            <Text size="lg" c="dimmed" maw={620}>
              The Fool plays itself on a local RTX 3080, climbs a gated checkpoint ladder,
              and lets you sit across the board from whatever version currently holds the
              crown.
            </Text>

            <Group mt={6}>
              <Button
                component={Link}
                to="/play"
                size="md"
                rightSection={<IconArrowRight size={17} />}
              >
                Play the bot
              </Button>
              <Button component={Link} to="/watch" size="md" variant="default">
                Watch it train
              </Button>
            </Group>
          </Stack>

          <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="md" mt="md">
            <Stat
              label="Estimated Elo"
              value={String(elo)}
              hint="Live rating series from the arena ledger"
            />
            <Stat
              label="Games played"
              value={String(games)}
              hint="Self-play plus gated evaluation matches"
            />
            <Stat label="Champion" value={champion} hint="Current promoted checkpoint" />
          </SimpleGrid>

          <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="md">
            {DESTINATIONS.map(({ to, icon: Icon, title, body }) => (
              <Card key={to} component={Link} to={to} style={{ textDecoration: 'none' }}>
                <Group gap={10} mb={10}>
                  <Icon size={20} stroke={1.7} />
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
    </div>
  );
}
