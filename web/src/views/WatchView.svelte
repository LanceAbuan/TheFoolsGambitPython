<script lang="ts">
  import { Chess } from 'chess.js';
  import Board from '../lib/components/Board.svelte';
  import { metrics as fetchMetrics, type MetricsSnapshot } from '../lib/api/http';
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
  let journal = $state<{ time: string; title: string; detail: string }[]>([
    { time: '23:48:12', title: 'Game completed', detail: 'The Fool won by checkmate.' },
    { time: '23:47:58', title: 'Position evaluated', detail: 'Advantage: +0.18 pawns.' },
    { time: '23:47:43', title: 'New variation explored', detail: 'Candidate move: Nf6.' },
    { time: '23:47:21', title: 'Self-play started', detail: 'Journey #12,483 began.' },
  ]);

  function clockTime(ts: string): string {
    const d = new Date(ts);
    return Number.isNaN(d.getTime())
      ? new Date().toLocaleTimeString('en-GB', { hour12: false })
      : d.toLocaleTimeString('en-GB', { hour12: false });
  }

  function onMessage(msg: ServerMessage): void {
    if (msg.topic !== 'train') return;
    if (msg.type === 'metrics') {
      const m = msg.payload as Partial<MetricsSnapshot>;
      liveMetrics = {
        elo: typeof m.elo === 'number' ? m.elo : null,
        loss: typeof m.loss === 'number' ? m.loss : null,
        win_rate: typeof m.win_rate === 'number' ? m.win_rate : null,
        games_played: typeof m.games_played === 'number' ? m.games_played : 0,
      };
    } else if (msg.type === 'log') {
      const p = msg.payload as { level?: string; message?: string };
      if (p.message) {
        journal = [
          { time: clockTime(msg.ts), title: (p.level ?? 'log').toUpperCase(), detail: p.message },
          ...journal,
        ].slice(0, 8);
      }
    }
  }

  $effect(() => {
    const offMessage = socket.onMessage(onMessage);
    const offStatus = socket.onStatus((s) => (conn = s));
    socket.start();
    socket.subscribe('train');
    void fetchMetrics()
      .then((m) => {
        if (liveMetrics === null) liveMetrics = m;
      })
      .catch(() => {});
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
      <div class="section-title"><span>Training journal</span><span class="tiny-label">LATEST EVENTS</span></div>
      <div class="log-list">
        {#each journal as entry, i (`${i}-${entry.time}-${entry.title}`)}
          <div><time>{entry.time}</time><i class="log-dot"></i><span><b>{entry.title}</b><small>{entry.detail}</small></span></div>
        {/each}
      </div>
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
</style>
