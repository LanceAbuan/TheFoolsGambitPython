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
  IconClock,
  IconPlayerPause,
  IconPlayerPlay,
  IconTrophy,
} from '@tabler/icons-react';
import { Chess } from 'chess.js';
import { Board } from '../components/Board';
import { EloChart } from '../components/EloChart';
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
          <Text fz="xs" fw={700} lts="0.12em" tt="uppercase">
            Sample game
          </Text>
          <Badge variant="light" color="gray" radius="sm" size="xs">
            REPLAY · ILLUSTRATIVE
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
      <Text fz={10} fw={650} lts="0.14em" tt="uppercase" c="dimmed">
        {label}
      </Text>
      <Text fz={26} fw={650} lh={1.05} className="tabular">
        {value}
      </Text>
      {hint ? (
        <Text fz={11} c="dimmed" lineClamp={1}>
          {hint}
        </Text>
      ) : null}
    </Stack>
  );
}

function WdlBar({ wins, draws, losses }: { wins: number; draws: number; losses: number }) {
  const total = Math.max(wins + draws + losses, 1);
  return (
    <div className="wdl" aria-hidden>
      {wins > 0 ? <span className="wdl-w" style={{ flex: wins / total }} /> : null}
      {draws > 0 ? <span className="wdl-d" style={{ flex: draws / total }} /> : null}
      {losses > 0 ? <span className="wdl-l" style={{ flex: losses / total }} /> : null}
    </div>
  );
}

function ArenaRow({ row }: { row: ArenaResult }) {
  const ckpt = typeof row.ckpt === 'string' ? row.ckpt : typeof row.checkpoint === 'string' ? row.checkpoint : '';
  const opponent = typeof row.opponent === 'string' ? row.opponent : '';
  const wins = row.wins ?? 0;
  const draws = row.draws ?? 0;
  const losses = row.losses ?? 0;
  const promoted = row.promoted === true;
  return (
    <div className="arena-row">
      <Group justify="space-between" gap="xs" wrap="nowrap" align="flex-start">
        <div className="arena-tags">
          <Text fz={12.5} fw={600}>
            {shortCkpt(ckpt)}
            <Text span fz={11.5} fw={400} c="dimmed">
              {' '}
              vs {shortCkpt(opponent)}
            </Text>
          </Text>
          <Text fz={10.5} c="dimmed" className="mono">
            {fmtWhen(row.ts)} · {wins + draws + losses} games
          </Text>
        </div>
        <Badge variant={promoted ? 'filled' : 'light'} color={promoted ? 'emerald' : 'gray'} radius="sm" size="xs">
          {promoted ? 'PROMOTED' : 'HELD'}
        </Badge>
      </Group>
      <Group gap={8} mt={7} wrap="nowrap">
        <WdlBar wins={wins} draws={draws} losses={losses} />
        <Text fz={11} className="mono" c="dimmed" style={{ whiteSpace: 'nowrap' }}>
          {wins}W {draws}D {losses}L
        </Text>
        <Text fz={12} fw={650} className="mono">
          {scoreText(wins, draws, losses)}
        </Text>
      </Group>
    </div>
  );
}

function JournalRow({ entry }: { entry: JournalEntry }) {
  return (
    <div className="journal-row">
      <Text fz={10.5} className="mono journal-time">
        {clock(entry.at)}
      </Text>
      <span className={`journal-level ${entry.level.toLowerCase()}`}>{entry.level}</span>
      <Text fz={12.5} className="journal-text">
        {entry.text}
      </Text>
    </div>
  );
}

function FeaturedGame({ slot, movesPerSec }: { slot: LiveSlot; movesPerSec: number }) {
  const recent = slot.moves.slice(-10);
  return (
    <div className="live-feature">
      <div className="live-board">
        <Board
          fen={slot.fen}
          lastMove={slot.lastUci ? { from: slot.lastUci.slice(0, 2), to: slot.lastUci.slice(2, 4) } : null}
          orientation="white"
          interactive={false}
        />
      </div>
      <Stack gap="sm" className="live-context">
        <Stack gap={2}>
          <Text fz={10} fw={650} lts="0.14em" tt="uppercase" c="dimmed">
            Featured game
          </Text>
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
          <Text fz={10} fw={650} lts="0.14em" tt="uppercase" c="dimmed" mb={6}>
            Recent moves
          </Text>
          <div className="live-movelist">
            {recent.length === 0 ? (
              <Text fz={12} c="dimmed">
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

        <Text fz={11} c="dimmed" mt="auto">
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
  const featured = slots[0] ?? null;
  const thumbs = slots.slice(1, 5);

  useEffect(() => {
    if (follow && feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [journal.length, follow]);

  const liveBadge = status === 'open' && trainerAlive;
  const gateScore =
    latestGate && typeof latestGate.wins === 'number' && typeof latestGate.draws === 'number' && typeof latestGate.losses === 'number'
      ? scoreText(latestGate.wins, latestGate.draws, latestGate.losses)
      : null;

  return (
    <Container size={1180} py={{ base: 24, sm: 32 }}>
      <Stack gap="lg">
        <Group justify="space-between" align="flex-end" gap="md" wrap="nowrap">
          <Stack gap={2}>
            <Text className="eyebrow">An observation of growth</Text>
            <Title order={2}>Watch</Title>
          </Stack>
          <Group gap={8} wrap="nowrap">
            <Badge
              variant="light"
              color={liveBadge ? 'emerald' : 'gray'}
              radius="sm"
              size="md"
              leftSection={<span className={liveBadge ? 'live-dot pulse' : 'live-dot offline'} aria-hidden />}
            >
              {liveBadge ? 'TRAINING LIVE' : status === 'open' ? 'TRAINING IDLE' : 'CONNECTING'}
            </Badge>
          </Group>
        </Group>

        {/* ---- status strip ---- */}
        <Card padding="md">
          <SimpleGrid cols={{ base: 2, md: 4 }} spacing="lg" verticalSpacing="md">
            <Stat
              label="Run"
              value={metrics?.regime ?? '—'}
              hint={trainerAlive ? 'trainer process alive' : 'trainer not detected'}
            />
            <Stat
              label="Estimated Elo"
              value={elo !== null ? String(elo) : '—'}
              hint={champion ? `champion ${shortCkpt(champion)}` : 'no champion yet'}
            />
            <Stat
              label="Gate"
              value={gate ? 'Title match' : 'Idle'}
              hint={gateScore ? `last result ${gateScore}` : 'no gate recorded'}
            />
            <Stat
              label="Arena games"
              value={typeof metrics?.arena_games === 'number' ? String(metrics.arena_games) : '—'}
              hint={typeof metrics?.games_played === 'number' ? `${metrics.games_played} self-play games` : undefined}
            />
          </SimpleGrid>
        </Card>

        {/* ---- live games ---- */}
        <Card padding="md">
          <Group justify="space-between" gap="xs" mb="sm">
            <Group gap={8}>
              <span className={featured ? 'live-dot pulse' : 'live-dot offline'} aria-hidden />
              <Text fz="xs" fw={700} lts="0.14em" tt="uppercase">
                Live games
              </Text>
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
                TITLE MATCH
              </Badge>
            ) : featured ? (
              <Text fz={11} c="dimmed" className="mono">
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
                        <Text fz={10.5} className="mono" c="dimmed">
                          G{slot.game} · {slot.ply}
                        </Text>
                        <Text fz={10.5} className="mono" c="dimmed">
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
                  {latestGate ? (
                    <Text fz={12} c="dimmed">
                      Last result: {shortCkpt(latestGate.ckpt)} vs {shortCkpt(latestGate.opponent)} ·{' '}
                      {scoreText(latestGate.wins ?? 0, latestGate.draws ?? 0, latestGate.losses ?? 0)} ·{' '}
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
        <div className="watch-row chart-row">
          <Card padding="md">
            <Group justify="space-between" gap="xs" mb={6}>
              <Text fw={600}>Training progress</Text>
              <Text fz={10.5} c="dimmed" className="mono">
                {metrics?.updated_at ? `UPDATED ${clock(metrics.updated_at)}` : 'AWAITING TELEMETRY'}
              </Text>
            </Group>
            <EloChart points={chart} />
            <Text fz={11} c="dimmed" mt={4}>
              Elo after each checkpoint&apos;s gauntlet. The ring marks the current champion
              {champion ? ` (${shortCkpt(champion)})` : ''}. Source: arena ladder.
            </Text>
          </Card>

          <Card padding="md">
            <Text fw={600} mb="md">
              By the numbers
            </Text>
            <SimpleGrid cols={2} spacing="lg" verticalSpacing="lg">
              <Stat
                label="Self-play games"
                value={typeof metrics?.games_played === 'number' ? String(metrics.games_played) : '—'}
                hint="against itself"
              />
              <Stat label="Win rate" value={pct(metrics?.win_rate)} hint="latest match" />
              <Stat
                label="Training loss"
                value={typeof metrics?.loss === 'number' ? metrics.loss.toFixed(2) : '—'}
                hint="lower is better"
              />
              <Stat
                label="Gate record"
                value={latestGate ? (latestGate.promoted ? 'Promoted' : 'Held') : '—'}
                hint={latestGate ? fmtWhen(latestGate.ts) : 'no gates yet'}
              />
            </SimpleGrid>
            {metricsError ? (
              <Text fz={11} c="orange" mt="md">
                Metrics REST fallback failed — showing the last socket snapshot.
              </Text>
            ) : null}
          </Card>
        </div>

        {/* ---- journal + arena ---- */}
        <div className="watch-row feed-row">
          <Card padding="md" className="journal-card">
            <Group justify="space-between" gap="xs" mb={8}>
              <Group gap={8}>
                <Text fw={600}>Training journal</Text>
                <Badge variant="light" color="gray" radius="sm" size="xs">
                  LIVE FEED
                </Badge>
              </Group>
              <Group gap={6} wrap="nowrap">
                <IconActivityHeartbeat size={13} className="hint-icon" aria-hidden />
                <Text fz={10.5} c="dimmed" className="mono">
                  {movesPerSec > 0 ? `${movesPerSec.toFixed(1)} moves/s` : 'quiet'}
                </Text>
              </Group>
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
                <Text fz={12.5} c="dimmed" p="sm">
                  No journal lines yet. Iterations are logged roughly once an hour; live moves stream above.
                </Text>
              ) : (
                journal.map((entry) => <JournalRow key={entry.id} entry={entry} />)
              )}
            </div>
            <Text fz={10.5} c="dimmed" mt={6}>
              Structured training events, translated to plain language. Newest at the bottom.
            </Text>
          </Card>

          <Card padding="md" className="arena-card">
            <Group justify="space-between" gap="xs" mb={8}>
              <Text fw={600}>Arena ledger</Text>
              <Text fz={10.5} c="dimmed" className="mono">
                {arena ? `${arena.length} MATCHES` : arenaError ? 'UNAVAILABLE' : 'LOADING'}
              </Text>
            </Group>

            {arenaError ? (
              <Text fz={12.5} c="dimmed">
                The arena ledger is unreachable right now. Gate history will appear here when the API is back.
              </Text>
            ) : arenaSorted.length === 0 ? (
              <Text fz={12.5} c="dimmed">
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

        <Group gap={6} justify="center" c="dimmed">
          <IconClock size={12} />
          <Text fz={10.5} className="mono">
            {lastUpdated(metrics?.updated_at)}
          </Text>
        </Group>
      </Stack>
    </Container>
  );
}

function lastUpdated(ts: number | undefined): string {
  if (typeof ts !== 'number' || !Number.isFinite(ts)) return 'AWAITING TELEMETRY';
  const age = Math.max(0, Math.round(Date.now() / 1000 - ts));
  if (age < 90) return `UPDATED ${age}s AGO`;
  return `UPDATED ${Math.round(age / 60)} MIN AGO`;
}
