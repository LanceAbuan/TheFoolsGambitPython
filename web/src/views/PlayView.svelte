<script lang="ts">
  import Board from '../lib/components/Board.svelte';
  import MoveScroll from '../lib/components/MoveScroll.svelte';
  import { game } from '../lib/game.svelte';
  import { metrics as fetchMetrics, bestKnownElo, type MetricsSnapshot, type Opponent } from '../lib/api/http';
  import { settings, setSetting } from '../lib/settings.svelte';

  let difficulty = $state('Curious · Intermediate');
  let side = $state('White');
  let opponent = $state<Opponent>('fool');
  let starting = $state(false);

  // FS-1 contract: `POST /play/new {"opponent":"nn"}` answers 503 "no trained
  // model yet…" until a checkpoint exists. We gate the option on the live
  // metrics Elo instead of guessing, and re-poll so the option lights up the
  // moment the backend lands a trained model — no redeploy needed.
  let nnMetrics = $state<MetricsSnapshot | null>(null);
  const nnReady = $derived(bestKnownElo(nnMetrics) !== null);
  const nnEloText = $derived(bestKnownElo(nnMetrics) !== null ? String(Math.round(bestKnownElo(nnMetrics)!)) : null);

  const OPPONENT_META: Record<Opponent, { name: string; blurb: string }> = {
    fool: { name: 'The Fool', blurb: 'Practice spirit of the house.' },
    stockfish: { name: 'Stockfish', blurb: 'The shadow that calculates.' },
    nn: { name: 'The Fool (NN)', blurb: 'A mind trained on the board.' },
  };
  const cardName = $derived(OPPONENT_META[opponent].name);
  const cardBlurb = $derived(OPPONENT_META[opponent].blurb);
  /** Live Elo belongs to the trained NN; the other foes have no ledger number yet. */
  const ratingText = $derived(opponent === 'nn' && nnEloText !== null ? nnEloText : '—');

  /** Panel-top mode chip: honest about where the moves are coming from. */
  const modeLabel = $derived.by(() => {
    if (game.mode === 'live' && game.gameId) return `LIVE · ${game.gameId.toUpperCase()}`;
    if (game.backendUp === false) return 'OFFLINE · DEMO';
    return 'DEMO · PRACTICE';
  });

  const statusText = $derived.by(() => {
    if (game.notice) return game.notice;
    if (game.endReason === 'abandoned') return 'Claimed by time — auto-resigned after 30 idle minutes.';
    if (game.resigned) return 'You resign — The Fool prevails.';
    if (game.status === 'checkmate') {
      const iWon = game.winner === (game.playColor === 'white' ? 'w' : 'b');
      return iWon ? 'Checkmate — you prevail. ✦' : 'Checkmate — The Fool prevails.';
    }
    if (game.status === 'stalemate') return 'Stalemate — the fates are balanced.';
    if (game.status === 'draw') return 'A draw — the fates are balanced.';
    if (game.thinking) return opponent === 'nn' ? 'The Fool (NN) is thinking…' : 'The Fool is thinking…';
    if (game.turn !== (game.playColor === 'white' ? 'w' : 'b')) return 'The opponent is on the move…';
    if (game.inCheck) return 'You are in check.';
    return 'Awaiting your move';
  });

  function strength(): string {
    if (difficulty.startsWith('Wandering')) return 'wandering';
    if (difficulty.startsWith('Enlightened')) return 'enlightened';
    return 'curious';
  }

  async function beginJourney(): Promise<void> {
    if (starting) return;
    starting = true;
    const color =
      side === 'Random' ? (Math.random() < 0.5 ? 'white' : 'black') : (side.toLowerCase() as 'white' | 'black');
    try {
      await game.startGame(color, opponent, strength());
    } finally {
      starting = false;
    }
  }

  async function pollMetrics(): Promise<void> {
    try {
      nnMetrics = await fetchMetrics();
    } catch {
      /* backend down — the option keeps its "not trained yet" guard */
    }
  }

  $effect(() => {
    void game.probeBackend();
    void pollMetrics();
    const t = setInterval(() => void pollMetrics(), 30_000);
    return () => clearInterval(t);
  });
</script>

<main class="page-wrap">
  <div class="page-heading">
    <div>
      <div class="eyebrow">THE GAME OF CHANCE</div>
      <h1>Play</h1>
      <p>Take a seat across from The Fool.</p>
    </div>
    <div class="chapter-mark">I <span>·</span> THE FIRST STEP</div>
  </div>

  <div class="play-grid">
    <section class="panel board-panel">
      <div class="panel-top">
        <span><i class="live-dot"></i> YOUR JOURNEY</span>
        <span class="tiny-label">{modeLabel}</span>
      </div>
      <div class="board-shell">
        <div class="coordinates top-coords">a　 b　 c　 d　 e　 f　 g　 h</div>
        <Board
          fen={game.fen}
          lastMove={game.lastMove}
          checkSquare={game.checkSquare}
          orientation={game.orientation}
          interactive={!game.over}
          legalTargets={(sq) => game.legalTargets(sq)}
          onmove={(f, t, p) => game.playerMove(f, t, p)}
        />
        <div class="coordinates bottom-coords">a　 b　 c　 d　 e　 f　 g　 h</div>
      </div>
      <div class="board-legend">
        <span>♙ YOU <i class="you-dot"></i></span>
        <span>VS</span>
        <span><i class="fool-dot"></i> {cardName.toUpperCase()} ♟</span>
      </div>
      <div class="board-controls">
        <button class="button" disabled={game.over || !game.canUndo} onclick={() => game.undo()}>↶ Undo</button>
        <button class="button" disabled={game.over} onclick={() => game.resign()}>⚑ Resign</button>
        <button class="button" onclick={() => game.flip()}>⇅ Flip board</button>
        <button class="button" disabled={starting} onclick={() => game.newGame()}>↻ New game</button>
      </div>
      <p class="kb-hint">KEYBOARD · ARROWS MOVE CURSOR · ENTER SELECTS/MOVES · ESC CANCELS</p>
      <p class="notice" class:think={game.thinking} class:alert={game.notice !== null}>{statusText}</p>
    </section>

    <aside class="side-stack">
      <section class="panel opponent-card">
        <div class="mini-fool">{opponent === 'nn' ? '✦' : '0'}</div>
        <div>
          <div class="eyebrow">YOUR OPPONENT</div>
          <h2>{cardName}</h2>
          <p><i class="live-dot"></i> {statusText}</p>
        </div>
        <div class="rating">{ratingText}<small>EST. ELO</small></div>
      </section>

      <section class="panel settings">
        <div class="section-title"><span>Game settings</span><span>✧</span></div>
        <label for="opponent">Opponent</label>
        <select id="opponent" bind:value={opponent}>
          <option value="fool">The Fool · Practice spirit</option>
          <option value="stockfish">Stockfish · The shadow</option>
          <option value="nn" disabled={!nnReady}>
            {nnReady ? `The Fool (NN) · Elo ${nnEloText}` : 'The Fool (NN) — not trained yet'}
          </option>
        </select>
        {#if opponent === 'nn' && !nnReady}
          <p class="nn-hint">No trained model yet — available once training produces a champion.</p>
        {:else if opponent === 'nn'}
          <p class="nn-hint ok">Trained mind ready · Elo {nnEloText}</p>
        {/if}
        <label for="difficulty">Difficulty</label>
        <select id="difficulty" bind:value={difficulty}>
          <option>Wandering · Beginner</option>
          <option>Curious · Intermediate</option>
          <option>Enlightened · Advanced</option>
        </select>
        <label for="side">Play as</label>
        <select id="side" bind:value={side}>
          <option>White</option>
          <option>Black</option>
          <option>Random</option>
        </select>
        <label class="check-row" for="unicode-pieces">
          <input
            id="unicode-pieces"
            type="checkbox"
            checked={settings.unicodePieces}
            onchange={(e) => setSetting('unicodePieces', (e.currentTarget as HTMLInputElement).checked)}
          />
          <span>Unicode glyph pieces</span>
        </label>
        <button class="button gold full" disabled={starting} onclick={beginJourney}>
          {starting ? 'Summoning The Fool…' : 'Begin a new journey'} <span>→</span>
        </button>
      </section>

      <section class="panel moves">
        <div class="section-title"><span>Move history</span><span class="tiny-label">PAST STEPS</span></div>
        <MoveScroll sans={game.sans} />
      </section>

      <div class="quote">“The journey of a thousand miles<br />begins with a single step.”</div>
    </aside>
  </div>
</main>

<style>
  .notice.think {
    animation: pulse 1.8s infinite;
  }
  .notice.alert {
    color: var(--gold-bright);
  }
  .button:disabled {
    opacity: 0.45;
    cursor: default;
  }
  .button:disabled:hover {
    background: transparent;
  }
  .kb-hint {
    text-align: center;
    font: 8px var(--mono);
    letter-spacing: 0.14em;
    color: #7d6b53;
    margin: 8px 0 0;
  }
  .nn-hint {
    font: 10px var(--mono);
    line-height: 1.5;
    color: #8e7c65;
    margin: 7px 0 0;
  }
  .nn-hint.ok {
    color: #a9b88b;
  }
  .check-row {
    display: flex !important;
    align-items: center;
    gap: 9px;
    cursor: pointer;
    user-select: none;
  }
  .check-row input {
    width: 14px;
    height: 14px;
    accent-color: #b99558;
    cursor: pointer;
  }
</style>
