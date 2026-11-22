import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Badge,
  Button,
  Card,
  Container,
  Divider,
  Group,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from '@mantine/core';
import {
  IconActivityHeartbeat,
  IconChevronLeft,
  IconChevronRight,
  IconPlayerPause,
  IconPlayerPlay,
  IconTrophy,
} from '@tabler/icons-react';
import { Chess } from 'chess.js';
import { Board } from '../components/Board';
import { EloChart } from '../components/EloChart';
import { ExpectedBar, evalScoreText, useFoolEval } from '../components/ExpectedBar';
import { useWatchFeed, type JournalEntry, type LiveSlot } from '../lib/hooks/useWatchFeed';
import { games as fetchGames, type ArenaResult } from '../lib/api/http';
import {
  chartPoints,
  clock,
  fmtWhen,
  pct,
  scoreText,
  shortCkpt,
  tsMs,
} from '../lib/watch/format';

/* ---- sample replay (honest: labelled illustrative, never engine output) ---- */
const SAMPLE_LINE = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'Nf6', 'O-O', 'Be7', 'd3', 'd6', 'c3', 'a6'];
interface Frame {
  fen: string;
  label: string;
  lm: { from: string; to: string } | null;
}
const SAMPLE_FRAMES: Frame[] = (() => {
  const c = new Chess();
  const out: Frame[] = [{ fen: c.fen(), label: 'Start', lm: null }];
  SAMPLE_LINE.forEach((san, i) => {
    const mv = c.move(san);
    const no = Math.floor(i / 2) + 1;
    out.push({ fen: c.fen(), label: i % 2 === 0 ? `${no}. ${san}` : `${no}… ${san}`, lm: { from: mv.from, to: mv.to } });
  });
  return out;
})();

/** Score with losses derived: arena rows may omit `losses` (games − wins − draws). */
function gateScoreOf(row: ArenaResult): { w: number; d: number; l: number } {
  const w = Number(row.wins ?? 0);
  const d = Number(row.draws ?? 0);
  const l =
    row.losses !== undefined && row.losses !== null
      ? Number(row.losses)
      : Math.max(0, Number(row.games ?? 0) - w - d);
  return { w, d, l };
}

function SampleReplay({ compact = false }: { compact?: boolean }) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => setFrame((f) => (f >= SAMPLE_FRAMES.length - 1 ? 0 : f + 1)), 1100);
    return () => clearInterval(timer);
  }, [playing]);

  const current = SAMPLE_FRAMES[frame];

  return (
    <div className={compact ? 'sample compact' : 'sample'}>
      <Group justify="space-between" gap="xs">
        <Group gap={8}>
          <Text fz="sm" fw={600}>
            Sample game
          </Text>
          <Badge variant="light" color="gray" radius="sm" size="xs">
            Replay · illustrative
          </Badge>
        </Group>
        <Text fz="xs" c="dimmed" className="mono">
          {current.label} · {frame + 1}/{SAMPLE_FRAMES.length}
        </Text>
      </Group>

      <div className="sample-board">
        <Board fen={current.fen} lastMove={current.lm} orientation="white" interactive={false} />
      </div>

      <Group gap="xs" mt="sm" justify="center">
        <Button
          variant="default"
          size="compact-sm"
          aria-label="Previous position"
          onClick={() => {
            setPlaying(false);
            setFrame((f) => Math.max(0, f - 1));
          }}
        >
          <IconChevronLeft size={16} />
        </Button>
        <Button
          variant={playing ? 'light' : 'default'}
          size="compact-sm"
          leftSection={playing ? <IconPlayerPause size={15} /> : <IconPlayerPlay size={15} />}
          onClick={() => setPlaying((p) => !p)}
        >
          {playing ? 'Pause' : 'Replay'}
        </Button>
        <Button
          variant="default"
          size="compact-sm"
          aria-label="Next position"
          onClick={() => {
            setPlaying(false);
            setFrame((f) => Math.min(SAMPLE_FRAMES.length - 1, f + 1));
          }}
        >
          <IconChevronRight size={16} />
        </Button>
      </Group>
      <Text fz={11} c="dimmed" ta="center" mt={6}>
        An opening line for atmosphere — not the bot&apos;s evaluation.
      </Text>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Stack gap={2}>
      <Text fz="sm" c="dimmed">
        {label}
      </Text>
      <Text fz={20} fw={600} lh={1.15} className="tabular">
        {value}
      </Text>
      {hint ? (
        <Text fz="xs" c="dimmed" lineClamp={1}>
          {hint}
        </Text>
      ) : null}
    </Stack>
  );
}

function ArenaRow({ row }: { row: ArenaResult }) {
  const ckpt = typeof row.ckpt === 'string' ? row.ckpt : typeof row.checkpoint === 'string' ? row.checkpoint : '';
  const opponent = typeof row.opponent === 'string' ? row.opponent : '';
  const score = gateScoreOf(row);
  const promoted = row.promoted === true;
  return (
    <div className="arena-row">
      <Group justify="space-between" gap="xs" wrap="nowrap" align="flex-start">
        <div className="arena-tags">
          <Text fz="sm" fw={600}>
            {shortCkpt(ckpt)}
            <Text span fz="xs" fw={400} c="dimmed">
              {' '}
              vs {shortCkpt(opponent)}
            </Text>
          </Text>
          <Text fz="xs" c="dimmed" className="mono">
            {fmtWhen(row.ts)} · {score.w + score.d + score.l} games
          </Text>
        </div>
        <Badge variant={promoted ? 'filled' : 'light'} color={promoted ? 'emerald' : 'gray'} radius="sm" size="xs">
          {promoted ? 'Promoted' : 'Held'}
        </Badge>
      </Group>
      {/* The mono score text carries exact W/D/L — a bar adds nothing on
          degenerate matches (e.g. all draws) and was removed. */}
      <Group gap={10} mt={7} wrap="nowrap">
        <Text fz="xs" className="mono" c="dimmed" style={{ whiteSpace: 'nowrap' }}>
          {score.w}W {score.d}D {score.l}L
        </Text>
        <Text fz="xs" fw={600} className="mono">
          {scoreText(score.w, score.d, score.l)}
        </Text>
      </Group>
    </div>
  );
}

function JournalRow({ entry }: { entry: JournalEntry }) {
  return (
    <div className="journal-row">
      <Text fz={11} className="mono journal-time">
        {clock(entry.at)}
      </Text>
      <span className={`journal-level ${entry.level.toLowerCase()}`}>{entry.level}</span>
      <Text fz="xs" className="journal-text">
        {entry.text}
      </Text>
    </div>
  );
}

function FeaturedGame({ slot, movesPerSec }: { slot: LiveSlot; movesPerSec: number }) {
  const recent = slot.moves.slice(-10);
  const evalState = useFoolEval(slot.ended ? null : slot.fen);
  const evalLabel =
    evalState.state === 'thinking'
      ? 'Fool is scoring…'
      : evalState.state === 'ready'
        ? `Fool eval · white ${evalScoreText(evalState.cp, evalState.mate)}`
        : evalState.state === 'unavailable'
          ? 'Fool eval unavailable'
          : 'Fool eval';
  return (
    <div className="live-feature">
      <div className="live-board">
        <Board
          fen={slot.fen}
          lastMove={slot.lastUci ? { from: slot.lastUci.slice(0, 2), to: slot.lastUci.slice(2, 4) } : null}
          orientation="white"
          interactive={false}
        />
        <div className="featured-eval" aria-live="off">
          <ExpectedBar value={evalState.expectedWhite} />
          <Text fz="xs" c="dimmed" className="mono" mt={4}>
            {evalLabel}
          </Text>
        </div>
      </div>
      <Stack gap="sm" className="live-context">
        <Stack gap={2}>
          <Title order={3}>Game {slot.game}</Title>
          <Text fz="xs" c="dimmed">
            {slot.runId ? `run ${slot.runId}` : 'self-play'} · ply {slot.ply}
            {slot.ended ? ` · ended ${slot.ended}` : ''}
          </Text>
        </Stack>

        <Group gap={6}>
          <Badge variant="light" color="emerald" radius="sm" size="sm" leftSection={<IconActivityHeartbeat size={12} />}>
            {movesPerSec > 0 ? `${movesPerSec.toFixed(1)} moves/s` : 'streaming'}
          </Badge>
          {slot.lastSan ? (
            <Badge variant="default" radius="sm" size="sm" className="mono">
              {slot.lastSan}
            </Badge>
          ) : null}
        </Group>

        <div>
          <Text fz="sm" fw={600} mb={6}>
            Recent moves
          </Text>
          <div className="live-movelist">
            {recent.length === 0 ? (
              <Text fz="xs" c="dimmed">
                Waiting for the first move…
              </Text>
            ) : (
              recent.map((m) => (
                <span key={`${m.ply}-${m.uci}`} className="move-chip">
                  {m.san ?? m.uci}
                </span>
              ))
            )}
          </div>
        </div>

        <Text fz="xs" c="dimmed" mt="auto">
          Boards update live; a jump can mean several plies arrived at once.
        </Text>
      </Stack>
    </div>
  );
}

export function Watch() {
  const { status, metrics, metricsError, journal, slots, movesPerSec } = useWatchFeed();
  const [arena, setArena] = useState<ArenaResult[] | null>(null);
  const [arenaError, setArenaError] = useState(false);
  const [showAllArena, setShowAllArena] = useState(false);
  const feedRef = useRef<HTMLDivElement | null>(null);
  const [follow, setFollow] = useState(true);

  const arenaGames = metrics?.arena_games;

  useEffect(() => {
    let alive = true;
    fetchGames('arena', 50)
      .then((r) => {
        if (alive) {
          setArena(r.games);
          setArenaError(false);
        }
      })
      .catch(() => {
        if (alive) setArenaError(true);
      });
    return () => {
      alive = false;
    };
  }, [arenaGames]);

  const arenaSorted = useMemo(() => {
    const rows = arena ?? [];
    return [...rows].sort((a, b) => {
      const ta = tsMs(a.ts);
      const tb = tsMs(b.ts);
      if (Number.isNaN(ta) && Number.isNaN(tb)) return 0;
      if (Number.isNaN(ta)) return 1;
      if (Number.isNaN(tb)) return -1;
      return tb - ta;
    });
  }, [arena]);

  const latestGate = arenaSorted[0] ?? null;
  const champion = metrics?.champion ?? null;
  const trainerAlive = metrics?.trainer_alive === true;
  const gate = metrics?.gate_in_progress === true;
  const elo = typeof metrics?.elo === 'number' ? Math.round(metrics.elo) : null;
  const chart = useMemo(() => chartPoints(metrics?.checkpoints, champion), [metrics?.checkpoints, champion]);

  // Stable featured game: the slot list is sorted by last update, so taking
  // slots[0] directly makes the big board jump to whichever parallel game moved
  // last. Pin a game until it actually leaves the feed (ended / TTL), then
  // re-pin to the newest one.
  const [pinnedGame, setPinnedGame] = useState<number | null>(null);
  const featured = useMemo(() => {
    if (pinnedGame !== null) {
      const pinned = slots.find((s) => s.game === pinnedGame);
      if (pinned) return pinned;
    }
    return slots[0] ?? null;
  }, [pinnedGame, slots]);
  useEffect(() => {
    if (featured && featured.game !== pinnedGame) setPinnedGame(featured.game);
  }, [featured, pinnedGame]);
  const thumbs = slots.filter((s) => s.game !== featured?.game).slice(0, 4);

  useEffect(() => {
    if (follow && feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [journal.length, follow]);

  const liveBadge = status === 'open' && trainerAlive;
  const latestScore = latestGate ? gateScoreOf(latestGate) : null;
  const gateScore = latestScore ? scoreText(latestScore.w, latestScore.d, latestScore.l) : null;

  return (
    <Container size={1180} py={{ base: 24, sm: 32 }}>
      <Stack gap="lg">
        <Group justify="space-between" align="flex-end" gap="md" wrap="nowrap">
          <Title order={2}>Watch</Title>
          <Badge
            variant="light"
            color={liveBadge ? 'emerald' : 'gray'}
            radius="sm"
            size="md"
            leftSection={<span className={liveBadge ? 'live-dot pulse' : 'live-dot offline'} aria-hidden />}
          >
            {liveBadge ? 'Live' : status === 'open' ? 'Idle' : 'Connecting'}
          </Badge>
        </Group>

        {/* ---- live games ---- */}
        <Card padding="md">
          <Group justify="space-between" gap="xs" mb="sm">
            <Group gap={8}>
              <Text fw={600}>Live games</Text>
              <Text fz="xs" c="dimmed">
                {featured
                  ? 'self-play, streaming move by move'
                  : gate
                    ? 'paused during the title match'
                    : trainerAlive
                      ? 'waiting for the next game'
                      : 'trainer offline'}
              </Text>
            </Group>
            {gate ? (
              <Badge variant="light" color="emerald" radius="sm" size="sm" leftSection={<IconTrophy size={12} />}>
                Title match
              </Badge>
            ) : featured ? (
              <Text fz="xs" c="dimmed" className="mono">
                {slots.length} board{slots.length === 1 ? '' : 's'} streaming
              </Text>
            ) : null}
          </Group>

          {featured ? (
            <>
              <FeaturedGame slot={featured} movesPerSec={movesPerSec} />
              {thumbs.length > 0 ? (
                <div className="thumb-grid">
                  {thumbs.map((slot) => (
                    <div key={slot.game} className="thumb">
                      <Board
                        mini
                        fen={slot.fen}
                        lastMove={
                          slot.lastUci ? { from: slot.lastUci.slice(0, 2), to: slot.lastUci.slice(2, 4) } : null
                        }
                        orientation="white"
                        interactive={false}
                      />
                      <Group justify="space-between" gap={4} mt={5} wrap="nowrap">
                        <Text fz={11} className="mono" c="dimmed">
                          G{slot.game} · {slot.ply}
                        </Text>
                        <Text fz={11} className="mono" c="dimmed">
                          {slot.lastSan ?? slot.lastUci ?? ''}
                        </Text>
                      </Group>
                    </div>
                  ))}
                </div>
              ) : null}
            </>
          ) : (
            <div className="gate-empty">
              {gate ? (
                <Stack gap="xs" maw={460}>
                  <Group gap={8}>
                    <span className="live-dot pulse" aria-hidden />
                    <Text fw={600}>A title match is running</Text>
                  </Group>
                  <Text fz="sm" c="dimmed">
                    The champion is defending the crown against the newest checkpoint. Gate games are
                    played in the arena, not streamed move by move — live boards resume when the match ends.
                  </Text>
                  {champion ? (
                    <Text fz="sm">
                      Champion <b className="mono">{champion}</b>
                      {champion ? ` · Elo ${Math.round(metrics?.elo ?? 0)}` : ''}
                    </Text>
                  ) : null}
                  {latestGate && latestScore ? (
                    <Text fz="xs" c="dimmed">
                      Last result: {shortCkpt(latestGate.ckpt)} vs {shortCkpt(latestGate.opponent)} ·{' '}
                      {scoreText(latestScore.w, latestScore.d, latestScore.l)} ·{' '}
                      {latestGate.promoted ? 'promoted' : 'champion held'}
                    </Text>
                  ) : null}
                </Stack>
              ) : (
                <Stack gap="xs" maw={520}>
                  <Text fw={600}>{trainerAlive ? 'No game is streaming right now' : 'Training is offline'}</Text>
                  <Text fz="sm" c="dimmed">
                    {trainerAlive
                      ? 'The socket is connected, but no self-play frame has arrived yet. The board below is a sample opening line.'
                      : 'The trainer process is not running, so there are no live positions. Start the run to see boards here.'}
                  </Text>
                </Stack>
              )}
              <div className="gate-sample">
                <SampleReplay compact />
              </div>
            </div>
          )}
        </Card>

        {/* ---- progress + numbers ---- */}
        <div className="watch-row">
          <Card padding="md">
            <Group justify="space-between" gap="xs" mb={6}>
              <Text fw={600}>Training progress</Text>
              <Text fz="xs" c="dimmed" className="mono">
                {metrics?.updated_at ? `updated ${clock(metrics.updated_at)}` : 'awaiting telemetry'}
              </Text>
            </Group>
            <EloChart points={chart} />
            <Text fz="xs" c="dimmed" mt={4}>
              Elo after each checkpoint&apos;s gauntlet. The ring marks the current champion
              {champion ? ` (${shortCkpt(champion)})` : ''}. Source: arena ladder.
            </Text>
          </Card>

          <Card padding="md">
            <Text fw={600} mb="md">
              By the numbers
            </Text>
            <SimpleGrid cols={2} spacing="lg" verticalSpacing="md">
              <Stat
                label="Run"
                value={metrics?.regime ?? '—'}
                hint={trainerAlive ? 'trainer process alive' : 'trainer not detected'}
              />
              <Stat
                label="Champion"
                value={champion ? shortCkpt(champion) : '—'}
                hint="current title holder"
              />
              <Stat label="Estimated Elo" value={elo !== null ? String(elo) : '—'} hint="arena ladder" />
              <Stat
                label="Gate"
                value={gate ? 'Title match' : 'Idle'}
                hint={gateScore ? `last result ${gateScore}` : 'no gates yet'}
              />
              <Stat
                label="Self-play games"
                value={typeof metrics?.games_played === 'number' ? String(metrics.games_played) : '—'}
                hint="against itself"
              />
              <Stat
                label="Arena games"
                value={typeof metrics?.arena_games === 'number' ? String(metrics.arena_games) : '—'}
                hint="gated matches"
              />
              <Stat label="Win rate" value={pct(metrics?.win_rate)} hint="latest match" />
              <Stat
                label="Training loss"
                value={typeof metrics?.loss === 'number' ? metrics.loss.toFixed(2) : '—'}
                hint="lower is better"
              />
            </SimpleGrid>
            {metricsError ? (
              <Text fz="xs" c="orange" mt="md">
                Metrics REST fallback failed — showing the last socket snapshot.
              </Text>
            ) : null}
          </Card>
        </div>

        {/* ---- journal + arena ---- */}
        <div className="watch-row">
          <Card padding="md" className="journal-card">
            <Group justify="space-between" gap="xs" mb={8}>
              <Text fw={600}>Training journal</Text>
              <Text fz="xs" c="dimmed" className="mono">
                {movesPerSec > 0 ? `${movesPerSec.toFixed(1)} moves/s` : 'quiet'}
              </Text>
            </Group>
            <div
              className="journal-feed"
              ref={feedRef}
              onScroll={() => {
                const el = feedRef.current;
                if (el) setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 32);
              }}
              role="log"
              aria-live="polite"
            >
              {journal.length === 0 ? (
                <Text fz="xs" c="dimmed" p="sm">
                  No journal lines yet. Iterations are logged roughly once an hour; live moves stream above.
                </Text>
              ) : (
                journal.map((entry) => <JournalRow key={entry.id} entry={entry} />)
              )}
            </div>
            <Text fz="xs" c="dimmed" mt={6}>
              Structured training events, translated to plain language. Newest at the bottom.
            </Text>
          </Card>

          <Card padding="md" className="arena-card">
            <Group justify="space-between" gap="xs" mb={8}>
              <Text fw={600}>Arena ledger</Text>
              <Text fz="xs" c="dimmed" className="mono">
                {arena ? `${arena.length} matches` : arenaError ? 'unavailable' : 'loading'}
              </Text>
            </Group>

            {arenaError ? (
              <Text fz="xs" c="dimmed">
                The arena ledger is unreachable right now. Gate history will appear here when the API is back.
              </Text>
            ) : arenaSorted.length === 0 ? (
              <Text fz="xs" c="dimmed">
                No gate matches recorded yet.
              </Text>
            ) : (
              <Stack gap={0}>
                {(showAllArena ? arenaSorted : arenaSorted.slice(0, 6)).map((row, i) => (
                  <div key={`${row.ckpt ?? row.checkpoint ?? i}-${row.ts ?? i}`}>
                    {i > 0 ? <Divider my={8} color="var(--mantine-color-default-border)" /> : null}
                    <ArenaRow row={row} />
                  </div>
                ))}
              </Stack>
            )}

            {arenaSorted.length > 6 ? (
              <Button
                variant="subtle"
                size="compact-sm"
                mt="sm"
                onClick={() => setShowAllArena((v) => !v)}
                rightSection={showAllArena ? <IconChevronLeft size={14} style={{ transform: 'rotate(-90deg)' }} /> : <IconChevronRight size={14} style={{ transform: 'rotate(90deg)' }} />}
              >
                {showAllArena ? 'Show fewer' : `Show all ${arenaSorted.length} matches`}
              </Button>
            ) : null}
          </Card>
        </div>
      </Stack>
    </Container>
  );
}
