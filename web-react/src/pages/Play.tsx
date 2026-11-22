import { useEffect, useState } from 'react';
import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Card,
  Container,
  Group,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Text,
  Title,
  Tooltip,
} from '@mantine/core';
import {
  IconArrowBackUp,
  IconArrowsUpDown,
  IconFlag,
  IconInfoCircle,
  IconKeyboard,
  IconPlayerPlay,
  IconRefresh,
} from '@tabler/icons-react';
import { Board } from '../components/Board';
import { MoveList } from '../components/MoveList';
import { useGame } from '../lib/hooks/useGame';
import { useMetrics } from '../lib/hooks/useMetrics';
import { bestKnownElo, type Color, type Opponent } from '../lib/api/http';
import { game } from '../lib/state/game';

const OPPONENT_META: Record<Opponent, { name: string; blurb: string }> = {
  fool: { name: 'The Fool', blurb: 'Built-in practice bot. Runs offline in your browser.' },
  stockfish: { name: 'Stockfish', blurb: 'Traditional chess engine. The strongest opponent.' },
  nn: { name: 'The Fool (trained)', blurb: 'The neural network trained on this site.' },
};

const DIFFICULTY = [
  { value: 'wandering', label: 'Wandering · Beginner' },
  { value: 'curious', label: 'Curious · Intermediate' },
  { value: 'enlightened', label: 'Enlightened · Advanced' },
];

const TIP_OPPONENT =
  'Three ways to play. The Fool — a practice bot that runs entirely in your browser, no connection needed. Stockfish — a traditional chess engine, the strongest opponent here. The Fool (trained) — the neural network this site is teaching to play; its rating comes from real arena games.';

const TIP_DIFFICULTY =
  'How hard the opponent tries, in effort terms: how many candidate moves it inspects per turn — deeper thinking, slower moves. Wandering is a gentle start; Enlightened digs the deepest.';

const TIP_ELO_CARD =
  'A chess rating — higher means a stronger player. A casual club player is around 1200; world champions are near 2800. The trained bot earns its number by beating earlier versions of itself in arena matches. A dash (—) means this opponent has no rating yet.';

const GLYPH_KEY = 'fsg:unicodePieces';

function readGlyphs(): boolean {
  try {
    return localStorage.getItem(GLYPH_KEY) === '1';
  } catch {
    return false;
  }
}

function Hint({ text }: { text: string }) {
  return (
    <Tooltip label={text} multiline w={300} position="top" withArrow>
      <IconInfoCircle size={14} className="hint-icon" aria-label="More information" />
    </Tooltip>
  );
}

export function Play() {
  const state = useGame();
  const { metrics } = useMetrics();

  const [opponent, setOpponent] = useState<Opponent>('fool');
  const [difficulty, setDifficulty] = useState('curious');
  const [side, setSide] = useState('White');
  const [starting, setStarting] = useState(false);
  const [glyphs, setGlyphs] = useState(readGlyphs);

  const nnElo = bestKnownElo(metrics);
  const nnReady = nnElo !== null;
  const card = OPPONENT_META[opponent];

  const setGlyphsPersist = (value: boolean) => {
    setGlyphs(value);
    try {
      localStorage.setItem(GLYPH_KEY, value ? '1' : '0');
    } catch {
      /* storage unavailable — keep the session-local value */
    }
  };

  useEffect(() => {
    void game.probeBackend();
  }, []);

  const modeLabel =
    state.mode === 'live' && state.gameId
      ? `LIVE · ${state.gameId.toUpperCase()}`
      : state.backendUp === false
        ? 'OFFLINE · DEMO'
        : 'DEMO · PRACTICE';

  const humanColor = state.playColor === 'white' ? 'w' : 'b';

  const statusText = (() => {
    if (state.notice) return state.notice;
    if (state.endReason === 'abandoned') return 'Auto-resigned after 30 idle minutes.';
    if (state.resigned) return 'You resigned.';
    if (state.status === 'checkmate') {
      return game.winner === humanColor ? 'Checkmate — you win.' : 'Checkmate — The Fool wins.';
    }
    if (state.status === 'stalemate') return 'Stalemate — draw.';
    if (state.status === 'draw') return 'Draw.';
    if (state.status === 'finished') {
      if (game.winner === null) return 'Draw.';
      return game.winner === humanColor ? 'Game over — you win.' : 'Game over — The Fool wins.';
    }
    if (state.thinking) return opponent === 'nn' ? 'The Fool (trained) is thinking…' : 'The Fool is thinking…';
    if (state.turn !== humanColor) return 'The opponent is on the move…';
    if (state.inCheck) return 'You are in check.';
    return 'Awaiting your move';
  })();

  const opponentHelp = (() => {
    if (opponent === 'stockfish')
      return 'A traditional chess engine. It calculates the strongest move it can find; it does not learn.';
    if (opponent === 'nn') {
      return nnReady
        ? 'The neural network this site has been training. Its rating is earned in arena matches against earlier versions of itself.'
        : 'The trained bot. There is no champion checkpoint yet, so it stays locked — training runs live on the Watch page, and this option unlocks when one is crowned.';
    }
    return 'A built-in practice bot that runs in your browser — no connection needed. A relaxed first game.';
  })();

  async function beginJourney() {
    if (starting) return;
    setStarting(true);
    const color: Color =
      side === 'Random' ? (Math.random() < 0.5 ? 'white' : 'black') : (side.toLowerCase() as Color);
    try {
      await game.startGame(color, opponent, difficulty);
    } finally {
      setStarting(false);
    }
  }

  return (
    <Container size={1180} py={{ base: 24, sm: 32 }}>
      <Stack gap="lg">
        <Title order={2}>Play</Title>

        <div className="board-layout">
          <div>
            <Card padding="md">
              <Group justify="space-between" gap="xs" wrap="nowrap">
                <Badge variant="light" color={state.mode === 'live' ? 'emerald' : 'gray'} radius="sm" size="sm">
                  {modeLabel}
                </Badge>
                <Tooltip
                  label="Keyboard: arrows move the cursor, Enter selects or moves, Esc cancels."
                  multiline
                  w={260}
                  withArrow
                  position="bottom"
                >
                  <ActionIcon variant="subtle" color="gray" size="sm" aria-label="Keyboard shortcuts">
                    <IconKeyboard size={15} />
                  </ActionIcon>
                </Tooltip>
              </Group>

              <Box mt="md">
                <Board
                  fen={state.fen}
                  lastMove={state.lastMove}
                  checkSquare={state.checkSquare}
                  orientation={state.orientation}
                  interactive={!game.over}
                  legalTargets={(sq) => game.legalTargets(sq)}
                  onMove={(from, to, promotion) => game.playerMove(from, to, promotion)}
                  unicode={glyphs}
                />
              </Box>

              <SimpleGrid cols={{ base: 2, xs: 4 }} spacing="xs" mt="md">
                <Button
                  variant="default"
                  size="sm"
                  leftSection={<IconArrowBackUp size={15} />}
                  disabled={!game.canUndo}
                  onClick={() => game.undo()}
                >
                  Undo
                </Button>
                <Button
                  variant="default"
                  size="sm"
                  leftSection={<IconFlag size={15} />}
                  disabled={game.over}
                  onClick={() => game.resign()}
                >
                  Resign
                </Button>
                <Button
                  variant="default"
                  size="sm"
                  leftSection={<IconArrowsUpDown size={15} />}
                  onClick={() => game.flip()}
                >
                  Flip
                </Button>
                <Button
                  variant="default"
                  size="sm"
                  leftSection={<IconRefresh size={15} />}
                  disabled={starting}
                  onClick={() => game.newGame()}
                >
                  New game
                </Button>
              </SimpleGrid>

              <Text
                fz="sm"
                ta="center"
                mt="sm"
                c={state.notice ? 'orange' : 'dimmed'}
                className={state.thinking ? 'status-think' : undefined}
                aria-live="polite"
              >
                {statusText}
              </Text>
            </Card>
          </div>

          <div>
            <Stack gap="md">
              <Card padding="lg">
                <Group justify="space-between" align="flex-start" wrap="nowrap" gap="sm">
                  <Stack gap={4} style={{ minWidth: 0 }}>
                    <Title order={3}>{card.name}</Title>
                    <Text fz="xs" c="dimmed" lineClamp={2}>
                      {card.blurb}
                    </Text>
                  </Stack>
                  <Stack gap={0} align="flex-end" style={{ flexShrink: 0 }}>
                    {opponent === 'nn' && nnElo !== null ? (
                      <>
                        <Text fz={20} fw={600} lh={1.1} className="tabular">
                          {Math.round(nnElo)}
                        </Text>
                        <Group gap={4} wrap="nowrap">
                          <Text fz="xs" c="dimmed">
                            Elo
                          </Text>
                          <Hint text={TIP_ELO_CARD} />
                        </Group>
                      </>
                    ) : (
                      <Text fz="xs" c="dimmed">
                        No rating yet
                      </Text>
                    )}
                  </Stack>
                </Group>
              </Card>

              <Card padding="lg">
                <Text fw={600} mb="sm">
                  Game settings
                </Text>
                <Stack gap="sm">
                  <Group gap={6} wrap="nowrap">
                    <Text component="label" htmlFor="opponent" fz="sm">
                      Opponent
                    </Text>
                    <Hint text={TIP_OPPONENT} />
                  </Group>
                  <Select
                    id="opponent"
                    value={opponent}
                    onChange={(value) => value && setOpponent(value as Opponent)}
                    allowDeselect={false}
                    data={[
                      { value: 'fool', label: 'The Fool — practice bot (offline)' },
                      { value: 'stockfish', label: 'Stockfish — classic engine' },
                      {
                        value: 'nn',
                        label: nnReady
                          ? `The Fool (trained) · Elo ${Math.round(nnElo!)}`
                          : 'The Fool (trained) — unlocks when a champion exists',
                        disabled: !nnReady,
                      },
                    ]}
                  />
                  <Text fz="xs" c={opponent === 'nn' && nnReady ? 'emerald' : 'dimmed'} lh={1.6}>
                    {opponentHelp}
                  </Text>

                  <Group gap={6} wrap="nowrap">
                    <Text component="label" htmlFor="difficulty" fz="sm">
                      Difficulty
                    </Text>
                    <Hint text={TIP_DIFFICULTY} />
                  </Group>
                  <Select
                    id="difficulty"
                    value={difficulty}
                    onChange={(value) => value && setDifficulty(value)}
                    allowDeselect={false}
                    data={DIFFICULTY}
                  />

                  <Text component="label" htmlFor="side" fz="sm">
                    Play as
                  </Text>
                  <Select
                    id="side"
                    value={side}
                    onChange={(value) => value && setSide(value)}
                    allowDeselect={false}
                    data={['White', 'Black', 'Random']}
                  />

                  <Switch
                    label="Unicode glyph pieces"
                    size="sm"
                    checked={glyphs}
                    onChange={(e) => setGlyphsPersist(e.currentTarget.checked)}
                  />

                  <Button
                    fullWidth
                    leftSection={<IconPlayerPlay size={16} />}
                    disabled={starting}
                    onClick={() => void beginJourney()}
                  >
                    {starting ? 'Starting…' : 'Start game'}
                  </Button>
                </Stack>
              </Card>

              <Card padding="lg">
                <Text fw={600} mb="xs">
                  Move history
                </Text>
                <MoveList sans={state.sans} />
              </Card>
            </Stack>
          </div>
        </div>
      </Stack>
    </Container>
  );
}
