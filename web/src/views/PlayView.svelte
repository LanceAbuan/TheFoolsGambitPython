<script lang="ts">
  import Board from '../lib/components/Board.svelte';
  import MoveScroll from '../lib/components/MoveScroll.svelte';
  import { game } from '../lib/game.svelte';

  let difficulty = $state('Curious · Intermediate');
  let side = $state('White');
  let starting = $state(false);

  /** Panel-top mode chip: honest about where the moves are coming from. */
  const modeLabel = $derived.by(() => {
    if (game.mode === 'live' && game.gameId) return `LIVE · ${game.gameId.toUpperCase()}`;
    if (game.backendUp === false) return 'OFFLINE · DEMO';
    return 'DEMO · PRACTICE';
  });

  const statusText = $derived.by(() => {
    if (game.notice) return game.notice;
    if (game.resigned) return 'You resign — The Fool prevails.';
    if (game.status === 'checkmate') {
      const iWon = game.winner === (game.playColor === 'white' ? 'w' : 'b');
      return iWon ? 'Checkmate — you prevail. ✦' : 'Checkmate — The Fool prevails.';
    }
    if (game.status === 'stalemate') return 'Stalemate — the fates are balanced.';
    if (game.status === 'draw') return 'A draw — the fates are balanced.';
    if (game.thinking) return 'The Fool is thinking…';
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
      await game.startGame(color, 'fool', strength());
    } finally {
      starting = false;
    }
  }

  $effect(() => {
    void game.probeBackend();
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
        <span><i class="fool-dot"></i> THE FOOL ♟</span>
      </div>
      <div class="board-controls">
        <button class="button" disabled={game.over || !game.canUndo} onclick={() => game.undo()}>↶ Undo</button>
        <button class="button" disabled={game.over} onclick={() => game.resign()}>⚑ Resign</button>
        <button class="button" onclick={() => game.flip()}>⇅ Flip board</button>
        <button class="button" disabled={starting} onclick={() => game.newGame()}>↻ New game</button>
      </div>
      <p class="notice" class:think={game.thinking} class:alert={game.notice !== null}>{statusText}</p>
    </section>

    <aside class="side-stack">
      <section class="panel opponent-card">
        <div class="mini-fool">0</div>
        <div>
          <div class="eyebrow">YOUR OPPONENT</div>
          <h2>The Fool</h2>
          <p><i class="live-dot"></i> {statusText}</p>
        </div>
        <div class="rating">—<small>EST. ELO</small></div>
      </section>

      <section class="panel settings">
        <div class="section-title"><span>Game settings</span><span>✧</span></div>
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
</style>
