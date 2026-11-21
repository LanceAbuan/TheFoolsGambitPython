<script lang="ts">
  import { Chess } from 'chess.js';
  import Board from '../lib/components/Board.svelte';
  import Tip from '../lib/components/Tip.svelte';
  import {
    metrics as fetchMetrics,
    games as fetchGames,
    type ArenaResult,
    type MetricsSnapshot,
  } from '../lib/api/http';
  import { socket, type ServerMessage, type SocketStatus } from '../lib/api/ws';

  interface Props {
    /** Wired by App.svelte so the primer can send newcomers to Play/About. */
    onnav?: (tab: 'play' | 'watch' | 'about') => void;
  }
  let { onnav }: Props = $props();

  // FS-8b: the "What am I looking at?" primer — collapsed by default, one click
  // to expand. No persistence: every visit starts collapsed so the page stays calm.
  let primerOpen = $state(false);

  // A real miniature game, replayed frame by frame through the playback controls.
  const line = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'Nf6', 'O-O', 'Be7', 'd3', 'd6', 'c3', 'a6'];
  const frames: { fen: string; label: string }[] = (() => {
    const c = new Chess();
    const out = [{ fen: c.fen(), label: 'START' }];
    line.forEach((san, i) => {
      c.move(san);
      out.push({ fen: c.fen(), label: `${Math.floor(i / 2) + 1}. ${i % 2 === 0 ? san : out.at(-1)!.label.split(' ').slice(1).join(' ') + ' ' + san}` });
    });
    return out;
  })();
  // Cleaner labels: build "1. e4", "1… e5", "2. Nf3" style.
  for (let i = 1; i < frames.length; i++) {
    const moveNo = Math.floor((i - 1) / 2) + 1;
    frames[i].label = i % 2 === 1 ? `${moveNo}. ${line[i - 1]}` : `${moveNo}… ${line[i - 1]}`;
  }

  let frame = $state(0);
  let playing = $state(false);
  let timer: ReturnType<typeof setInterval> | null = null;

  const current = $derived(frames[frame]);
  const score = $derived(['+0.18', '+0.18', '+0.21', '+0.09', '+0.31', '+0.12', '+0.34', '+0.22', '+0.28', '+0.19', '+0.24', '+0.15', '+0.21'][frame] ?? '+0.18');
  const depth = $derived(String(6 + (frame % 5)).padStart(2, '0') + ' PLY');

  function toggle(): void {
    if (playing) {
      stop();
    } else {
      playing = true;
      timer = setInterval(() => {
        if (frame >= frames.length - 1) {
          frame = 0;
        } else {
          frame += 1;
        }
      }, 1100);
    }
  }
  function stop(): void {
    playing = false;
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }
  function step(delta: number): void {
    stop();
    frame = Math.min(frames.length - 1, Math.max(0, frame + delta));
  }

  // ---- live telemetry: WS `train` topic (§14.0) with REST one-shot fallback ----
  let conn = $state<SocketStatus>('connecting');
  let liveMetrics = $state<MetricsSnapshot | null>(null);

  function clockTime(ts: string): string {
    const d = new Date(ts);
    return Number.isNaN(d.getTime())
      ? new Date().toLocaleTimeString('en-GB', { hour12: false })
      : d.toLocaleTimeString('en-GB', { hour12: false });
  }

  // ---- training journal (FS-6/FS-8): a scrolling mono feed of `log` frames.
  // The frame shape is deliberately parsed defensively: the contract pins
  // {level, message}, FS-6 ships {line}, and a bare string is accepted too.
  interface JournalLine {
    id: string;
    time: string;
    level: string;
    text: string;
  }
  const JOURNAL_CAP = 200;
  let journal = $state<JournalLine[]>([]);
  let journalSeq = 0;
  let feedEl: HTMLDivElement | undefined = $state();
  let follow = $state(true);

  function logLineFrom(msg: ServerMessage): { text: string; level: string } | null {
    const p: unknown = msg.payload;
    if (typeof p === 'string' && p.length > 0) return { text: p, level: 'log' };
    if (p && typeof p === 'object') {
      const o = p as Record<string, unknown>;
      const text = [o.line, o.message, o.msg, o.text].find(
        (v): v is string => typeof v === 'string' && v.length > 0,
      );
      if (text) return { text, level: typeof o.level === 'string' ? o.level.toUpperCase() : 'LOG' };
    }
    return null;
  }

  function pushJournal(msg: ServerMessage): void {
    const parsed = logLineFrom(msg);
    if (!parsed) return;
    journalSeq += 1;
    const entry: JournalLine = {
      id: `${typeof msg.seq === 'number' ? msg.seq : 'x'}-${journalSeq}`,
      time: clockTime(msg.ts),
      level: parsed.level,
      text: parsed.text,
    };
    // Oldest → newest reading order; hard cap so a chatty run cannot grow the DOM forever.
    journal = [...journal, entry].slice(-JOURNAL_CAP);
  }

  function onFeedScroll(): void {
    if (!feedEl) return;
    follow = feedEl.scrollHeight - feedEl.scrollTop - feedEl.clientHeight < 40;
  }
  // Auto-follow: pin to the newest line unless the reader scrolled up to browse.
  $effect(() => {
    const count = journal.length;
    if (follow && feedEl && count > 0) feedEl.scrollTop = feedEl.scrollHeight;
  });

  // ---- metrics parsing: base fields are pinned, FS-1/FS-2 additions are optional ----
  function mapOfNumbers(v: unknown): Record<string, number> | undefined {
    if (!v || typeof v !== 'object' || Array.isArray(v)) return undefined;
    const out: Record<string, number> = {};
    for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
      if (typeof val === 'number' && Number.isFinite(val)) out[k] = val;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  }

  /** regime → { ckpt-tag → elo } (FS-2 nested map); non-numeric leaves dropped. */
  function mapOfRegimeElo(v: unknown): Record<string, Record<string, number>> | undefined {
    if (!v || typeof v !== 'object' || Array.isArray(v)) return undefined;
    const out: Record<string, Record<string, number>> = {};
    for (const [regime, ckpts] of Object.entries(v as Record<string, unknown>)) {
      if (!ckpts || typeof ckpts !== 'object' || Array.isArray(ckpts)) continue;
      const inner: Record<string, number> = {};
      for (const [tag, val] of Object.entries(ckpts as Record<string, unknown>)) {
        if (typeof val === 'number' && Number.isFinite(val)) inner[tag] = val;
      }
      if (Object.keys(inner).length > 0) out[regime] = inner;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  }

  function normalizeMetrics(m: unknown): MetricsSnapshot | null {
    if (!m || typeof m !== 'object') return null;
    const o = m as Record<string, unknown>;
    const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
    return {
      elo: num(o.elo),
      loss: num(o.loss),
      win_rate: num(o.win_rate),
      games_played: typeof o.games_played === 'number' ? o.games_played : 0,
      regime: typeof o.regime === 'string' ? o.regime : null,
      champion: typeof o.champion === 'string' ? o.champion : null,
      arena_games: num(o.arena_games) ?? undefined,
      best_streak: num(o.best_streak) ?? undefined,
      elo_by_regime: mapOfRegimeElo(o.elo_by_regime),
      regime_elo: mapOfNumbers(o.regime_elo),
      checkpoints: mapOfNumbers(o.checkpoints),
    };
  }

  // Per-regime Elo leaderboard (FS-2): the map is regime → { ckpt-tag → elo };
  // each regime is represented by its best checkpoint. Fall back to the flat
  // `regime_elo` map, then to a single row for the current regime; else nothing.
  const regimeElo = $derived.by(() => {
    const m = liveMetrics;
    if (!m) return [];
    const map = m.elo_by_regime;
    if (map && typeof map === 'object' && !Array.isArray(map)) {
      const rows = Object.entries(map)
        .map(([regime, ckpts]) => {
          let best = -Infinity;
          if (ckpts && typeof ckpts === 'object') {
            for (const v of Object.values(ckpts)) {
              if (typeof v === 'number' && Number.isFinite(v) && v > best) best = v;
            }
          }
          return { regime, elo: best === -Infinity ? null : best };
        })
        .filter((r): r is { regime: string; elo: number } => r.elo !== null);
      if (rows.length > 0) return rows.sort((a, b) => b.elo - a.elo);
    }
    const flat = m.regime_elo;
    if (flat && Object.keys(flat).length > 0) {
      return Object.entries(flat).map(([regime, elo]) => ({ regime, elo }));
    }
    if (m.regime && typeof m.elo === 'number') return [{ regime: m.regime, elo: m.elo }];
    return [];
  });

  /** Ledger `ts` (epoch seconds or ISO) → short local stamp; '' when unusable. */
  function fmtWhen(ts: number | string | undefined): string {
    if (ts === undefined) return '';
    const d = typeof ts === 'number' ? new Date(ts < 1e12 ? ts * 1000 : ts) : new Date(ts);
    if (Number.isNaN(d.getTime())) return '';
    return d.toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false });
  }

  // ---- arena ledger (FS-2 adds /api/games?kind=arena; absent → graceful note) ----
  let arena = $state<ArenaResult[] | null>(null);
  let arenaCountSeen = $state<number | null>(null);

  async function refreshArena(): Promise<void> {
    try {
      const r = await fetchGames('arena');
      arena = r.games;
    } catch {
      arena = null; // endpoint absent/unreachable — the fallback copy stays honest
    }
  }

  // ---- FS-8b: honest rating chart -------------------------------------------
  // The points are REAL when the arena ledger has rows: each approved checkpoint
  // contributes the Elo recorded right after its gauntlet. Secondary real source
  // is the metrics checkpoint ladder. With neither, the decorative curve stays
  // but is explicitly labelled SAMPLE — it never masquerades as data.
  type ChartPoint = { label: string; elo: number };

  const chartData = $derived.by(() => {
    const rows = (arena ?? []).filter((r) => typeof r.elo === 'number');
    if (rows.length > 0) {
      const byTs = [...rows].sort(
        (a, b) => (typeof a.ts === 'number' ? a.ts : 0) - (typeof b.ts === 'number' ? b.ts : 0),
      );
      const pts: ChartPoint[] = [];
      for (const r of byTs) {
        const label = typeof r.ckpt === 'string' && r.ckpt.length > 0 ? r.ckpt : `#${pts.length + 1}`;
        // Same checkpoint re-gauntleted later → keep the latest rating.
        const existing = pts.findIndex((p) => p.label === label);
        if (existing >= 0) pts[existing] = { label, elo: r.elo as number };
        else pts.push({ label, elo: r.elo as number });
      }
      return { real: true as const, pts };
    }
    const ladder = liveMetrics?.checkpoints;
    if (ladder && Object.keys(ladder).length > 0) {
      return { real: true as const, pts: Object.entries(ladder).map(([label, elo]) => ({ label, elo })) };
    }
    return { real: false as const, pts: [] as ChartPoint[] };
  });

  /** Real data → scaled geometry; no data → null (sample SVG renders instead). */
  const chartView = $derived.by(() => {
    if (!chartData.real || chartData.pts.length === 0) return null;
    const elos = chartData.pts.map((p) => p.elo);
    const lo = Math.min(...elos);
    const hi = Math.max(...elos);
    const span = Math.max(hi - lo, 40);
    const min = lo - span * 0.15;
    const max = hi + span * 0.15;
    const W = 520;
    const H = 190;
    const xy = chartData.pts.map((p, i) => ({
      ...p,
      x: chartData.pts.length === 1 ? W / 2 : (i / (chartData.pts.length - 1)) * W,
      y: H - ((p.elo - min) / (max - min)) * (H - 12) - 6,
    }));
    const d = xy.length > 1 ? `M${xy.map((p) => `${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' L')}` : null;
    const ticks = [max, (max + min) / 2, min].map((v) => Math.round(v).toLocaleString('en-US'));
    return { xy, d, ticks };
  });

  const yTicks = $derived(chartView ? chartView.ticks : ['1800', '1600', '1400', '1200', '1000']);

  // ---- FS-8b: plain-English tooltip copy (bodies are pure clarity) ----------
  const TIP_ELO =
    'A chess rating — higher means a stronger player. A casual club player is around 1200 and world champions are near 2800. The Fool starts at 800 (a beginner) and climbs by beating versions of itself.';
  const TIP_GAMES =
    'How many games The Fool has played against itself. Like a musician practicing — every game is a repetition, and the bot studies each one afterwards.';
  const TIP_WIN_RATE =
    "How often the newest version beats the previous version in a head-to-head match. A version only becomes 'champion' — and takes over — if it wins.";
  const TIP_LOSS =
    "A score for how wrong The Fool's guesses were during its latest study session. High = still confused; falling over time = it's learning. Some noise is normal.";
  const TIP_STREAK = 'Its longest run of consecutive wins against earlier versions of itself.';
  const TIP_BOARD =
    "This board replays a sample game so you can see what The Fool's practice looks like: it plays both sides, one move at a time. It is an illustration — the real training games run on the machine doing the training. The score and depth figures below come from the sample, too.";
  const TIP_JOURNAL =
    "The bot's training diary, newest at the bottom. Each entry is one round of practice: how many games it played, how wrong its guesses were, and its current rating. Some lines show raw JSON — that is the diary in its rawest form; you never need to read it.";
  const TIP_ARENA =
    'Every time The Fool learns something new, the new version must beat the old one in a short match before it takes over. This is the record of those matches — one row per challenge.';
  const TIP_LEADERBOARD =
    "The Fool can train in different 'schools' — different practice methods with their own schedules. This compares how each school's champion is rated.";
  const chipTip = $derived(
    conn === 'open'
      ? "You're watching the training in real time — this page refreshes itself every couple of seconds while the bot learns."
      : 'The training feed is not connected right now, so this page shows sample numbers. When a run is live, this chip lights up and the page updates itself every couple of seconds.',
  );
  const telemetryTip = $derived(
    liveMetrics !== null
      ? 'These numbers come straight from the live training run — they update on their own.'
      : 'The bot is not connected right now, so these are sample numbers showing what the panel looks like. Real figures appear as soon as training telemetry arrives.',
  );
  const chartTip = $derived(
    chartData.real
      ? "When The Fool finishes training a new version, that version must beat the old one in a short match (a 'gauntlet') before it takes over. Each point is the official rating recorded right after one of those matches — newest on the right. Up means the challenger won the title. These points are real: they come from the arena ledger below."
      : "When The Fool finishes training a new version, that version must beat the old one in a short match (a 'gauntlet') before it takes over. This panel will plot its rating after each of those matches. No matches are recorded yet, so the curve shown is a labelled sample — not real data.",
  );

  function onMessage(msg: ServerMessage): void {
    if (msg.topic !== 'train') return;
    if (msg.type === 'metrics') {
      const m = normalizeMetrics(msg.payload);
      if (m) {
        liveMetrics = m;
        // Refetch the ledger only when the arena counter actually moves.
        if (typeof m.arena_games === 'number' && m.arena_games !== arenaCountSeen) {
          arenaCountSeen = m.arena_games;
          void refreshArena();
        }
      }
    } else if (msg.type === 'log') {
      pushJournal(msg);
    }
  }

  $effect(() => {
    const offMessage = socket.onMessage(onMessage);
    const offStatus = socket.onStatus((s) => (conn = s));
    socket.start();
    socket.subscribe('train');
    void fetchMetrics()
      .then((m) => {
        if (liveMetrics === null) liveMetrics = normalizeMetrics(m);
      })
      .catch(() => {});
    void refreshArena();
    return () => {
      offMessage();
      offStatus();
      socket.stop();
    };
  });

  // Live values when the backend reports them; illustrative numbers otherwise.
  const eloText = $derived(liveMetrics?.elo != null ? Math.round(liveMetrics.elo).toLocaleString('en-US') : '1,625');
  const lossText = $derived(liveMetrics?.loss != null ? liveMetrics.loss.toFixed(3) : '0.384');
  const winRateText = $derived(
    liveMetrics?.win_rate != null
      ? `${((liveMetrics.win_rate <= 1 ? liveMetrics.win_rate * 100 : liveMetrics.win_rate)).toFixed(1)}%`
      : '54.7%',
  );
  const gamesText = $derived(
    liveMetrics?.games_played != null ? liveMetrics.games_played.toLocaleString('en-US') : '12,482',
  );
  // Honesty: the streak only displays when the run reports it; the illustrative
  // sample keeps its number under the ILLUSTRATIVE DATA chip.
  const streakText = $derived.by(() => {
    if (liveMetrics === null) return '08 W';
    const s = liveMetrics.best_streak;
    return typeof s === 'number' ? `${Math.round(s)} W` : '—';
  });
  const telemetryLabel = $derived(liveMetrics !== null ? 'LIVE TELEMETRY' : 'ILLUSTRATIVE DATA');
  const regimeText = $derived(liveMetrics?.regime ?? null);
  const journalLabel = $derived(journal.length > 0 ? (conn === 'open' ? 'LIVE FEED' : 'BUFFERED') : 'IDLE');
  const chip = $derived(
    conn === 'open' ? 'LIVE · CONNECTED' : conn === 'connecting' ? 'LIVE · CONNECTING' : 'DEMO · OFFLINE',
  );
</script>

<main class="page-wrap">
  <div class="page-heading">
    <div>
      <div class="eyebrow">AN OBSERVATION OF GROWTH</div>
      <h1>Watch</h1>
      <p>Observe The Fool as it learns the game.</p>
    </div>
    <div class="training-state" class:off={conn !== 'open'}>
      <i class="live-dot"></i>
      <span>{chip}</span>
      <Tip text={chipTip} label="What does this connection chip mean?" />
    </div>
  </div>

  <!-- FS-8b: the newcomer primer — the whole training loop in story form. -->
  <section class="panel primer">
    <button
      class="primer-toggle"
      aria-expanded={primerOpen}
      aria-controls="primer-body"
      onclick={() => (primerOpen = !primerOpen)}
    >
      <span class="primer-mark">✦</span>
      <span class="primer-title">New here? <b>What am I looking at?</b></span>
      <span class="primer-hint">{primerOpen ? 'HIDE −' : 'SHOW +'}</span>
    </button>
    {#if primerOpen}
      <div class="primer-body" id="primer-body">
        <p>
          The Fool is a chess bot that is teaching itself, and this page is its training diary. The whole loop, in
          order:
        </p>
        <ol>
          <li><b>It plays itself.</b> Thousands of fast games, both sides, no teacher — just the rules and curiosity.</li>
          <li><b>It studies.</b> After each round it reviews every game and adjusts its guesses about good moves.</li>
          <li><b>It must defend the title.</b> A newly trained version only becomes the champion if it beats the previous champion in a short match.</li>
          <li><b>Its rating climbs.</b> Every approved champion adds a point on the Elo scale — higher means stronger — recorded in the arena ledger.</li>
          <li><b>You can play the champion.</b> The current version is always ready for a game on the Play page.</li>
        </ol>
        <p class="primer-close">
          No chess knowledge needed to watch — and you can play it yourself on the
          <button class="text-link primer-link" onclick={() => onnav?.('play')}>Play page</button>.
          The full story lives on the
          <button class="text-link primer-link" onclick={() => onnav?.('about')}>About page</button>.
        </p>
      </div>
    {/if}
  </section>

  <div class="watch-grid">
    <section class="panel training-board">
      <div class="panel-top">
        <span><i class="live-dot"></i> SELF-PLAY TABLE <Tip text={TIP_BOARD} label="About the self-play board" /></span>
        <span class="tiny-label">SAMPLE GAME · REPLAY</span>
      </div>
      <div class="board-shell small-board">
        <div class="coordinates top-coords">a　 b　 c　 d　 e　 f　 g　 h</div>
        <Board mini fen={current.fen} />
        <div class="coordinates bottom-coords">a　 b　 c　 d　 e　 f　 g　 h</div>
      </div>
      <div class="playback">
        <button class="round-button" onclick={() => step(-1)} aria-label="previous position">←</button>
        <button class="round-button play-toggle" onclick={toggle} aria-label={playing ? 'pause' : 'play'}>
          {playing ? '❙❙' : '▶'}
        </button>
        <button class="round-button" onclick={() => step(1)} aria-label="next position">→</button>
        <span>POSITION {String(frame + 1).padStart(2, '0')} / {String(frames.length).padStart(2, '0')}</span>
        <input type="range" min="0" max={frames.length - 1} bind:value={frame} oninput={stop} aria-label="timeline" />
      </div>
      <div class="analysis-row">
        <div><small>CURRENT MOVE</small><strong>{current.label}</strong></div>
        <div><small>POSITION SCORE</small><strong>{score}</strong></div>
        <div><small>SEARCH DEPTH</small><strong>{depth}</strong></div>
      </div>
      {#if regimeText}
        <p class="regime-line">REGIME <b>{regimeText}</b>{#if liveMetrics?.champion} · CHAMPION <b>{liveMetrics.champion}</b>{/if}</p>
      {/if}
    </section>

    <section class="panel progress-panel">
      <div class="section-title">
        <span>Training progress <Tip text={chartTip} label="What does this chart show?" /></span>
        <span class="tiny-label">{telemetryLabel} <Tip text={telemetryTip} label="Is this data real?" /></span>
      </div>
      <div class="metric-main">
        <strong>{eloText}</strong>
        <span>ESTIMATED ELO <Tip text={TIP_ELO} label="What is Estimated Elo?" /></span>
        <em>{liveMetrics !== null ? 'LIVE · WS FEED' : '↗ +42 THIS WEEK'}</em>
      </div>
      <div class="chart-wrap">
        <div class="chart-y">{#each yTicks as t (t)}<span>{t}</span>{/each}</div>
        {#if chartView}
          <svg
            class="line-chart"
            viewBox="0 0 520 190"
            preserveAspectRatio="none"
            role="img"
            aria-label="Real Elo rating after each arena-approved checkpoint"
          >
            <path class="chart-grid" d="M0 20H520 M0 60H520 M0 100H520 M0 140H520 M0 180H520" />
            {#if chartView.d}
              <path class="chart-line" d={chartView.d} />
            {/if}
            {#each chartView.xy as p, i (`${i}-${p.label}`)}
              <rect class="chart-pt" x={p.x - 3.5} y={p.y - 3.5} width="7" height="7" />
            {/each}
          </svg>
        {:else}
          <svg class="line-chart" viewBox="0 0 520 190" preserveAspectRatio="none" role="img" aria-label="Sample rating curve, rising trend — an illustration until real arena results arrive">
            <path class="chart-grid" d="M0 20H520 M0 60H520 M0 100H520 M0 140H520 M0 180H520" />
            <path
              class="chart-area"
              d="M0 164 L35 151 L70 157 L105 130 L140 137 L175 112 L210 118 L245 92 L280 103 L315 76 L350 83 L385 57 L420 62 L455 35 L490 42 L520 18 L520 180 L0 180Z"
            />
            <path
              class="chart-line chart-line-sample"
              d="M0 164 L35 151 L70 157 L105 130 L140 137 L175 112 L210 118 L245 92 L280 103 L315 76 L350 83 L385 57 L420 62 L455 35 L490 42 L520 18"
            />
          </svg>
        {/if}
      </div>
      <div class="chart-legend">
        {#if chartView}
          <span><i></i> RATING AFTER EACH GAUNTLET · REAL</span>
          <span>SOURCE · ARENA LEDGER</span>
        {:else}
          <span><i></i> SAMPLE CURVE</span>
          <span>REAL PLOTS APPEAR AFTER THE FIRST ARENA MATCH</span>
        {/if}
      </div>
      <div class="stat-grid">
        <div>
          <small>GAMES PLAYED <Tip text={TIP_GAMES} label="What counts as games played?" /></small>
          <strong>{gamesText}</strong>
        </div>
        <div>
          <small>WIN RATE <Tip text={TIP_WIN_RATE} label="What is win rate here?" /></small>
          <strong>{winRateText}</strong>
        </div>
        <div>
          <small>TRAINING LOSS <Tip text={TIP_LOSS} label="What is training loss?" /></small>
          <strong>{lossText}</strong>
        </div>
        <div>
          <small>BEST STREAK <Tip text={TIP_STREAK} label="What is the best streak?" /></small>
          <strong>{streakText}</strong>
        </div>
      </div>
    </section>

    <section class="panel log-panel">
      <div class="section-title">
        <span>Training journal <Tip text={TIP_JOURNAL} label="What is the training journal?" /></span>
        <span class="tiny-label">{journalLabel}</span>
      </div>
      <div class="journal" bind:this={feedEl} onscroll={onFeedScroll}>
        {#if journal.length === 0}
          <p class="journal-empty">Awaiting training telemetry — the journal fills as the pipeline runs.</p>
        {:else}
          {#each journal as j (j.id)}
            <p class="journal-line">
              <time>{j.time}</time>
              <b data-level={j.level}>{j.level}</b>
              <span>{j.text}</span>
            </p>
          {/each}
        {/if}
      </div>
      <p class="journal-foot">{journal.length} LINE{journal.length === 1 ? '' : 'S'} · FOLLOW {follow ? 'ON' : 'PAUSED'}</p>
    </section>

    <section class="panel lb-panel">
      <div class="section-title">
        <span>Regime leaderboard <Tip text={TIP_LEADERBOARD} label="What is the regime leaderboard?" /></span>
        <span class="tiny-label">PER-REGIME ELO</span>
      </div>
      {#if regimeElo.length === 0}
        <p class="panel-empty">No per-regime Elo reported yet — the ledger appears once arenas are scored.</p>
      {:else}
        <div class="lb-rows">
          {#each regimeElo as row, i (`${row.regime}-${i}`)}
            <div class="lb-row">
              <span class="lb-rank">{String(i + 1).padStart(2, '0')}</span>
              <span class="lb-name">{row.regime}</span>
              <span class="lb-elo">{Math.round(row.elo).toLocaleString('en-US')}</span>
            </div>
          {/each}
        </div>
      {/if}
    </section>

    <section class="panel arena-panel">
      <div class="section-title">
        <span>Arena results <Tip text={TIP_ARENA} label="What is the arena ledger?" /></span>
        <span class="tiny-label">LEDGER</span>
      </div>
      {#if arena === null}
        <p class="panel-empty">Arena ledger not available yet.</p>
      {:else if arena.length === 0}
        <p class="panel-empty">No arena games recorded yet — checkpoints are gauntleted as training runs.</p>
      {:else}
        <div class="arena-rows">
          {#each arena as a, i (`${a.run_id ?? 'run'}-${a.ckpt ?? i}`)}
            <div class="arena-row">
              <span class="arena-title">
                {[a.regime, a.ckpt].filter((v) => typeof v === 'string' && v.length > 0).join(' / ')
                  || a.run_id
                  || 'Arena run'}
                <em>vs {a.opponent ?? 'sparring partner'}</em>
              </span>
              <span class="arena-sub">
                {[
                  typeof a.games === 'number' ? `${a.games} games` : null,
                  typeof a.wins === 'number' && typeof a.draws === 'number'
                    ? `${a.wins}W ${a.draws}D${typeof a.losses === 'number' ? ` ${a.losses}L` : ''}`
                    : (a.result ?? null),
                  typeof a.elo === 'number' ? `elo ${Math.round(a.elo)}` : null,
                  fmtWhen(a.ts),
                ]
                  .filter((v) => typeof v === 'string' && v.length > 0)
                  .join(' · ') || '—'}
              </span>
            </div>
          {/each}
        </div>
      {/if}
    </section>

    <section class="panel quote-panel">
      <span class="quote-star">✧</span>
      <p>“It is not the destination,<br />but the journey that teaches.”</p>
      <small>THE FOOL'S JOURNAL · ENTRY 001</small>
    </section>
  </div>
</main>

<style>
  /* ---- FS-8b: newcomer primer (spans the full grid width) ---- */
  .primer {
    grid-column: 1 / -1;
    padding: 14px 20px;
  }
  .primer-toggle {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    background: none;
    border: 0;
    padding: 0;
    cursor: pointer;
    color: var(--paper);
    text-align: left;
    font-family: inherit;
  }
  .primer-toggle:focus-visible {
    outline: 1px solid var(--gold-bright);
    outline-offset: 3px;
  }
  .primer-mark {
    color: var(--gold);
    font-size: 13px;
  }
  .primer-title {
    font: 15px var(--serif);
  }
  .primer-title b {
    color: var(--gold-bright);
    font-weight: 600;
  }
  .primer-hint {
    margin-left: auto;
    font: 9px var(--mono);
    letter-spacing: 0.14em;
    color: #927e63;
    white-space: nowrap;
  }
  .primer-body {
    margin-top: 12px;
    border-top: 1px solid #594634;
    padding-top: 12px;
  }
  .primer-body > p {
    margin: 0 0 10px;
    font: 14px/1.6 var(--serif);
    color: var(--paper-dim);
  }
  .primer-body ol {
    margin: 0 0 10px;
    padding-left: 20px;
    display: grid;
    gap: 7px;
  }
  .primer-body li {
    font: 13.5px/1.6 var(--serif);
    color: var(--paper-dim);
  }
  .primer-body li b {
    color: var(--paper);
    font-weight: 600;
  }
  .primer-close {
    margin: 0;
  }
  .primer-close .primer-link {
    margin: 0 2px;
    padding-bottom: 1px;
    font-size: 13px;
  }
  /* sample curve reads as an illustration: dashed stroke, no marker squares */
  .chart-line-sample {
    stroke-dasharray: 6 5;
    opacity: 0.85;
  }
  .chart-pt {
    fill: var(--gold-bright);
    stroke: #2a2118;
    stroke-width: 1;
  }

  .training-state.off .live-dot {
    background: var(--red);
    box-shadow: 0 0 9px var(--red);
  }
  .regime-line {
    font: 9px var(--mono);
    letter-spacing: 0.12em;
    color: #8f7b61;
    border-top: 1px solid #594634;
    padding-top: 10px;
    margin: 12px 0 0;
  }
  .regime-line b {
    color: var(--gold-bright);
    font-weight: 500;
  }
  /* journal: a terminal-style mono feed, auto-following, capped at 200 lines */
  .journal {
    max-height: 240px;
    overflow-y: auto;
    background: #17131b;
    border: 1px solid #44382f;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .journal-line {
    display: flex;
    gap: 9px;
    align-items: baseline;
    font: 11px var(--mono);
    line-height: 1.5;
    margin: 0;
    color: #cbb694;
  }
  .journal-line time {
    color: #7d6b53;
    flex: none;
  }
  .journal-line b {
    font-weight: 400;
    flex: none;
    min-width: 44px;
    color: #a9b88b;
  }
  .journal-line b[data-level='ERROR'],
  .journal-line b[data-level='WARN'] {
    color: #c98a80;
  }
  .journal-line span {
    word-break: break-word;
    overflow-wrap: anywhere;
  }
  .journal-empty {
    font: 11px var(--mono);
    color: #7d6b53;
    margin: 0;
  }
  .journal-foot {
    font: 8px var(--mono);
    letter-spacing: 0.12em;
    color: #6d5d48;
    margin: 8px 0 0;
  }
  .panel-empty {
    font: 11px var(--mono);
    color: #7d6b53;
    line-height: 1.6;
    margin: 6px 0;
  }
  .lb-rows {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .lb-row {
    display: grid;
    grid-template-columns: 30px 1fr auto;
    align-items: baseline;
    gap: 10px;
    padding: 9px 10px;
    background: #241e27;
    border: 1px solid #44382f;
    font: 12px var(--mono);
  }
  .lb-rank {
    color: #7d6b53;
  }
  .lb-name {
    color: var(--paper-dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .lb-elo {
    color: var(--gold-bright);
  }
  .arena-rows {
    display: flex;
    flex-direction: column;
  }
  .arena-row {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 8px 0;
    border-bottom: 1px solid #44382f;
  }
  .arena-title {
    font: 12px var(--mono);
    color: var(--paper-dim);
  }
  .arena-title em {
    font-style: normal;
    color: #9c896d;
  }
  .arena-sub {
    font: 10px var(--mono);
    color: #9c896d;
  }
</style>
