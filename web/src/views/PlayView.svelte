<script lang="ts">
  import Board from '../lib/components/Board.svelte';
  import MoveScroll from '../lib/components/MoveScroll.svelte';
  import Tip from '../lib/components/Tip.svelte';
  import { game } from '../lib/game.svelte';
  import { metrics as fetchMetrics, bestKnownElo, type MetricsSnapshot, type Opponent } from '../lib/api/http';
  import { settings, setSetting } from '../lib/settings.svelte';

  interface Props {
    /** Wired by App.svelte: the "not trained yet" hint sends newcomers to Watch. */
    onnav?: (tab: 'play' | 'watch' | 'about') => void;
  }
  let { onnav }: Props = $props();

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

  // FS-8b: plain-English helper line + tooltip copy. Blurbs name the option in
  // the same breath they describe it, so no prior knowledge is assumed.
  const TIP_OPPONENT =
    'Three ways to play. The Fool — a friendly offline practice bot that lives in your browser, no connection needed. Stockfish — a world-class traditional chess engine, the strongest opponent here. The Fool (NN) — the bot that trained itself by playing thousands of games; its rating is real.';
  const TIP_DIFFICULTY =
    'How hard the opponent tries, in effort terms: how many candidate moves it inspects per turn — deeper thinking, slower moves. Wandering is a gentle start; Enlightened digs the deepest.';
  const TIP_ELO_CARD =
    "A chess rating — higher means a stronger player. A casual club player is around 1200; world champions are near 2800. The Fool's number is earned by beating earlier versions of itself in arena matches. A dash (—) means this opponent has no rating yet.";
  const opponentHelp = $derived.by(() => {
    if (opponent === 'stockfish')
      return 'A world-class traditional chess engine — the strongest opponent here. It calculates moves; it does not learn.';
    if (opponent === 'nn') {
      return nnReady
        ? 'The bot that trained itself by playing thousands of games against itself. Its rating is real.'
        : 'The self-trained bot. No trained model yet — training happens live on the Watch page, and the option unlocks the moment a champion is produced.';
    }
    return 'An offline practice bot that lives in your browser — no connection needed. A relaxed first game.';
  });

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
    // F-11: server-decided endings the board cannot express (reaper, server
    // draws). The reaper's own copy is matched earlier via endReason.
    if (game.status === 'finished') {
      if (game.winner === null) return 'A draw — the fates are balanced.';
      const iWon = game.winner === (game.playColor === 'white' ? 'w' : 'b');
      return iWon ? 'The game is decided — you prevail. ✦' : 'The game is decided — The Fool prevails.';
    }
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
        <!-- Undo after game over is LIVE (2026-11-21): the backend 200s the
             take-back on mate/stalemate/resigned games and re-opens the board
             from the response snapshot; abandoned (reaped) games still 409,
             and canUndo hides the button for exactly those. The store still
             coordinates by contract: 200 = re-render from the response, 409 =
             keep the rendered state. -->
        <button class="button" disabled={!game.canUndo} onclick={() => game.undo()}>↶ Undo</button>
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
        <div class="rating">
          {ratingText}
          <small>EST. ELO <Tip text={TIP_ELO_CARD} label="What is EST. ELO?" /></small>
        </div>
      </section>

      <section class="panel settings">
        <div class="section-title"><span>Game settings</span><span>✧</span></div>
        <div class="label-row">
          <label for="opponent">Opponent</label>
          <Tip text={TIP_OPPONENT} label="About the opponent options" />
        </div>
        <select id="opponent" bind:value={opponent}>
          <option value="fool">The Fool · Practice spirit</option>
          <option value="stockfish">Stockfish · The shadow</option>
          <option value="nn" disabled={!nnReady}>
            {nnReady ? `The Fool (NN) · Elo ${nnEloText}` : 'The Fool (NN) — not trained yet'}
          </option>
        </select>
        <p class="select-help" class:ok={opponent === 'nn' && nnReady}>
          {opponentHelp}
          {#if opponent === 'nn' && !nnReady}
            {' '}
            <button class="text-link help-link" onclick={() => onnav?.('watch')}>Watch the training live</button>
          {/if}
        </p>
        <div class="label-row">
          <label for="difficulty">Difficulty</label>
          <Tip text={TIP_DIFFICULTY} label="What does difficulty change?" />
        </div>
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
  .kb-hint {
    text-align: center;
    font: 8px var(--mono);
    letter-spacing: 0.14em;
    color: var(--faint);
    margin: 8px 0 0;
  }
  /* FS-8b: label + tooltip row, and the plain-English helper under the select */
  .label-row {
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .label-row label {
    margin-bottom: 0;
  }
  .select-help {
    font: 10px var(--mono);
    line-height: 1.6;
    color: var(--muted);
    margin: 7px 0 0;
  }
  .select-help.ok {
    color: var(--green);
  }
  .select-help .help-link {
    margin: 0;
    padding-bottom: 1px;
    font-size: 10px;
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
    accent-color: var(--accent);
    cursor: pointer;
  }
</style>
