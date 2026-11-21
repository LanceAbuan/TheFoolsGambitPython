<script lang="ts">
  import { Chess } from 'chess.js';
  import Board from '../lib/components/Board.svelte';
  import {
    metrics as fetchMetrics,
    games as fetchGames,
    type ArenaResult,
    type MetricsSnapshot,
  } from '../lib/api/http';
  import { socket, type ServerMessage, type SocketStatus } from '../lib/api/ws';

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
      {chip}
    </div>
  </div>

  <div class="watch-grid">
    <section class="panel training-board">
      <div class="panel-top">
        <span><i class="live-dot"></i> SELF-PLAY TABLE</span>
        <span class="tiny-label">GAME #12,483</span>
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
      <div class="section-title"><span>Training progress</span><span class="tiny-label">{telemetryLabel}</span></div>
      <div class="metric-main">
        <strong>{eloText}</strong>
        <span>ESTIMATED ELO</span>
        <em>{liveMetrics !== null ? 'LIVE · WS FEED' : '↗ +42 THIS WEEK'}</em>
      </div>
      <div class="chart-wrap">
        <div class="chart-y"><span>1800</span><span>1600</span><span>1400</span><span>1200</span><span>1000</span></div>
        <svg class="line-chart" viewBox="0 0 520 190" preserveAspectRatio="none" role="img" aria-label="Rating over training, rising trend">
          <path class="chart-grid" d="M0 20H520 M0 60H520 M0 100H520 M0 140H520 M0 180H520" />
          <path
            class="chart-area"
            d="M0 164 L35 151 L70 157 L105 130 L140 137 L175 112 L210 118 L245 92 L280 103 L315 76 L350 83 L385 57 L420 62 L455 35 L490 42 L520 18 L520 180 L0 180Z"
          />
          <path
            class="chart-line"
            d="M0 164 L35 151 L70 157 L105 130 L140 137 L175 112 L210 118 L245 92 L280 103 L315 76 L350 83 L385 57 L420 62 L455 35 L490 42 L520 18"
          />
        </svg>
      </div>
      <div class="chart-legend"><span><i></i> RATING OVER TRAINING</span><span>LAST 30 SESSIONS</span></div>
      <div class="stat-grid">
        <div><small>GAMES PLAYED</small><strong>{gamesText}</strong></div>
        <div><small>WIN RATE</small><strong>{winRateText}</strong></div>
        <div><small>TRAINING LOSS</small><strong>{lossText}</strong></div>
        <div><small>BEST STREAK</small><strong>08 W</strong></div>
      </div>
    </section>

    <section class="panel log-panel">
      <div class="section-title"><span>Training journal</span><span class="tiny-label">{journalLabel}</span></div>
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
      <div class="section-title"><span>Regime leaderboard</span><span class="tiny-label">PER-REGIME ELO</span></div>
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
      <div class="section-title"><span>Arena results</span><span class="tiny-label">LEDGER</span></div>
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
