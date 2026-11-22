/**
 * About: the card without a number. Ported from the Svelte AboutView with the
 * same intent — the tarot medallion, the three story cards, the plain-English
 * training story (FS-8b: zero chess knowledge assumed) and an honest status
 * strip fed by live metrics. Colors are theme tokens via light-dark(), so the
 * medallion renders in both color schemes with no theme-specific overrides.
 */
import { Link } from 'react-router-dom';
import { Card, Container, Group, SimpleGrid, Stack, Text, Title, UnstyledButton } from '@mantine/core';
import { bestKnownElo, type MetricsSnapshot } from '../lib/api/http';
import { useMetrics } from '../lib/hooks/useMetrics';
import { useEffect, useState } from 'react';

function JourneyLink({ children }: { children: string }) {
  return (
    <UnstyledButton component={Link} to="/watch" className="text-link">
      {children} <span aria-hidden>→</span>
    </UnstyledButton>
  );
}

export function About() {
  const { metrics } = useMetrics();
  const [elo, setElo] = useState<number | null>(null);

  useEffect(() => {
    setElo(bestKnownElo(metrics as MetricsSnapshot | null));
  }, [metrics]);

  const alive = metrics?.trainer_alive === true;

  return (
    <Container size={980} py={{ base: 24, sm: 40 }}>
      <Stack gap="lg">
        <Group justify="space-between" align="flex-start" gap="md" wrap="nowrap">
          <Stack gap={2}>
            <Text className="eyebrow">The card without a number</Text>
            <Title order={2}>About The Fool</Title>
            <Text c="dimmed" fz="sm">
              A perpetual learner, forever on the journey.
            </Text>
          </Stack>
          <Text className="mono chapter-mark" fz={11} c="dimmed" visibleFrom="sm" mt={6}>
            0 <span aria-hidden>·</span> THE BEGINNING
          </Text>
        </Group>

        {/* ---- hero: the medallion + the study in curiosity ---- */}
        <Card padding={0} className="about-hero">
          <div className="tarot-card" aria-hidden>
            <div className="tarot-inner">
              <div className="tarot-number">0</div>
              <div className="tarot-sun">✹</div>
              <div className="tarot-moon">☾</div>
              <div className="tarot-mountains" />
              <div className="tarot-traveler">♟</div>
              <div className="tarot-stars">✦&ensp;✧&ensp;✦</div>
              <div className="tarot-title">THE FOOL</div>
            </div>
          </div>
          <div className="about-intro">
            <Text className="eyebrow">A study in curiosity</Text>
            <Title order={3} className="about-title" mt={14} mb={16}>
              Every master
              <br />
              was once a beginner.
            </Title>
            <Text className="about-copy">
              The Fool is a chess-playing machine learning experiment inspired by the tarot&apos;s most
              curious card. It begins without certainty, explores without fear, and improves through
              every game it plays.
            </Text>
            <Text className="about-copy" mt={12}>
              Like the traveler stepping off the edge of the familiar, The Fool approaches chess as an
              endless landscape of possibilities.
            </Text>
            <JourneyLink>Follow its journey</JourneyLink>
          </div>
        </Card>

        {/* ---- three story cards ---- */}
        <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="md">
          <Card padding="lg" className="story-card">
            <span className="story-icon" aria-hidden>
              ✧
            </span>
            <Text className="eyebrow" mt={14}>
              I · The journey
            </Text>
            <Title order={4} mt={8} mb={8} fw={500}>
              Learning through play
            </Title>
            <Text className="about-copy" fz="md">
              Self-play gives The Fool a place to experiment. Each position is a question; each move, a
              hypothesis. Over time, patterns emerge from the games it has played.
            </Text>
          </Card>
          <Card padding="lg" className="story-card">
            <span className="story-icon" aria-hidden>
              ☼
            </span>
            <Text className="eyebrow" mt={14}>
              II · The mind
            </Text>
            <Title order={4} mt={8} mb={8} fw={500}>
              A model in progress
            </Title>
            <Text className="about-copy" fz="md">
              The long-term goal is to train a chess model, measure its progress, and make that process
              visible. The interface is designed to reveal the thinking and learning behind the moves.
            </Text>
          </Card>
          <Card padding="lg" className="story-card">
            <span className="story-icon" aria-hidden>
              ☾
            </span>
            <Text className="eyebrow" mt={14}>
              III · The unknown
            </Text>
            <Title order={4} mt={8} mb={8} fw={500}>
              There is always more
            </Title>
            <Text className="about-copy" fz="md">
              Chess has more possible games than can be explored in a lifetime. The Fool&apos;s journey
              has no final destination—only new positions, new lessons, and another move.
            </Text>
          </Card>
        </SimpleGrid>

        {/* ---- the training loop, in plain English (FS-8b) ---- */}
        <Card padding="lg" className="learn-panel">
          <Text className="eyebrow">The training loop · in plain English</Text>
          <Title order={4} mt={8} mb={12} fw={500}>
            How The Fool learns
          </Title>
          <Text className="about-copy">
            The Fool learns chess the way the AlphaZero project taught a machine the game: it starts
            knowing only the rules, then plays thousands of quick games against itself. After each round
            it studies what happened — which guesses were wrong, which moves led to wins — and nudges its
            internal web of numbers (a neural network) a little closer to good chess. Every so often, a
            freshly trained version has to earn its title: it plays a short match against the reigning
            champion, and only takes over if it wins. Each promotion is recorded in the arena ledger you
            can see on the Watch page, along with the version&apos;s Elo rating — the same rating scale
            chess clubs use, where higher means stronger. Nothing about its strength is hand-set: every
            point was earned by beating an earlier version of itself. Training runs on this machine&apos;s
            own GPU, and the Watch page follows the journal as it happens.
          </Text>
          <Text className="about-copy learn-close" mt={12}>
            No chess knowledge needed to follow along — you can play the current champion at any time.
          </Text>
          <JourneyLink>Watch the training live</JourneyLink>
        </Card>

        {/* ---- honest status strip (live metrics, never a static claim) ---- */}
        <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="md">
          <Card padding="md" className="status-cell">
            <Text component="small" className="eyebrow">
              Project status
            </Text>
            <Text fw={500} fz={17} mt={4}>
              {alive ? 'Training loop active' : 'Between training runs'}
            </Text>
          </Card>
          <Card padding="md" className="status-cell">
            <Text component="small" className="eyebrow">
              Model strength
            </Text>
            <Text fw={500} fz={17} mt={4}>
              {elo !== null ? `Elo ${Math.round(elo)} · self-trained` : 'Not yet rated'}
            </Text>
          </Card>
          <Card padding="md" className="status-cell">
            <Text component="small" className="eyebrow">
              The next step
            </Text>
            <Text fw={500} fz={17} mt={4}>
              Play the champion yourself
            </Text>
          </Card>
        </SimpleGrid>
      </Stack>
    </Container>
  );
}
