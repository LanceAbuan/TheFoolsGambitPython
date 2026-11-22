/**
 * About: what The Fool is, how it trains, where it stands right now.
 * Deliberately plain — the only live element is the status strip fed by /api/metrics.
 */
import { Link } from 'react-router-dom';
import { Anchor, Card, Container, List, SimpleGrid, Stack, Text, Title } from '@mantine/core';
import { useMetrics } from '../lib/hooks/useMetrics';

export function About() {
  const { metrics } = useMetrics();
  const elo = typeof metrics?.elo === 'number' ? metrics.elo : null;
  const alive = metrics?.trainer_alive === true;

  return (
    <Container size={840} py={{ base: 24, sm: 40 }}>
      <Stack gap="lg">
        <Stack gap={4}>
          <Title order={2}>About</Title>
          <Text c="dimmed" fz="sm">
            A chess bot trained from zero chess knowledge on one GPU, in the open.
          </Text>
        </Stack>

        <Card>
          <Title order={3} mb="sm">What it is</Title>
          <Text fz="sm" c="dimmed" lh={1.6}>
            The Fool is a chess engine that learns by playing itself — the AlphaZero loop,
            scaled down to a single RTX 3080. It started knowing only the rules of chess.
            Every point of strength it has was earned by beating an earlier version of itself
            in a rated match; nothing about its play was hand-set.
          </Text>
        </Card>

        <Card>
          <Title order={3} mb="sm">How training works</Title>
          <List spacing="xs" size="sm" lh={1.55}>
            <List.Item>
              <Text span fw={600} inherit>
                Self-play
              </Text>{' '}
              — the current network plays thousands of quick games against itself to generate
              data.
            </List.Item>
            <List.Item>
              <Text span fw={600} inherit>
                Learn
              </Text>{' '}
              — the network is nudged toward the moves that led to wins.
            </List.Item>
            <List.Item>
              <Text span fw={600} inherit>
                Gate
              </Text>{' '}
              — each newly trained checkpoint must beat the reigning champion in a match to
              take the crown.
            </List.Item>
            <List.Item>
              <Text span fw={600} inherit>
                Repeat
              </Text>{' '}
              — every promotion is recorded in the arena ledger with an Elo rating, the same
              scale chess clubs use.
            </List.Item>
          </List>
        </Card>

        <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="md">
          <Card>
            <Text fz="sm" c="dimmed">Status</Text>
            <Text fw={600} mt={4}>
              {alive ? 'Training loop active' : 'Between training runs'}
            </Text>
          </Card>
          <Card>
            <Text fz="sm" c="dimmed">Strength</Text>
            <Text fw={600} mt={4} className="tabular">
              {elo !== null ? `Elo ${Math.round(elo)}` : 'Not yet rated'}
            </Text>
          </Card>
          <Card>
            <Text fz="sm" c="dimmed">Next step</Text>
            <Anchor component={Link} to="/play" fw={600} mt={4} display="inline-block">
              Play the champion
            </Anchor>
          </Card>
        </SimpleGrid>

        <Text fz="sm" c="dimmed">
          Follow the loop as it happens on the{' '}
          <Anchor component={Link} to="/watch">Watch page</Anchor> — live boards, the training
          journal, and every gate result.
        </Text>
      </Stack>
    </Container>
  );
}
