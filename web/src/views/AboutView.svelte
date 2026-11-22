<script lang="ts">
  import { metrics as fetchMetrics, bestKnownElo } from '../lib/api/http';

  interface Props {
    onnav?: (tab: 'play' | 'watch' | 'about') => void;
  }
  let { onnav }: Props = $props();

  // FS-8b: one-shot metrics fetch so the status strip tells the truth about the
  // model instead of repeating the static mock's "Not yet connected".
  let elo = $state<number | null>(null);
  $effect(() => {
    fetchMetrics()
      .then((m) => (elo = bestKnownElo(m)))
      .catch(() => {});
  });
</script>

<main class="page-wrap about-wrap">
  <div class="page-heading">
    <div>
      <div class="eyebrow">THE CARD WITHOUT A NUMBER</div>
      <h1>About The Fool</h1>
      <p>A perpetual learner, forever on the journey.</p>
    </div>
    <div class="chapter-mark">0 <span>·</span> THE BEGINNING</div>
  </div>

  <section class="about-hero panel">
    <div class="tarot-card">
      <div class="tarot-inner">
        <div class="tarot-number">0</div>
        <div class="tarot-sun">✹</div>
        <div class="tarot-moon">☾</div>
        <div class="tarot-mountains"></div>
        <div class="tarot-traveler">♟</div>
        <div class="tarot-stars">✦　✧　✦</div>
        <!-- Classical themes drop the tarot scene; this engraved piece
             medallion keeps the card intentional instead of empty. -->
        <div class="tarot-minimal" aria-hidden="true"><span>♞</span></div>
        <div class="tarot-title">THE FOOL</div>
      </div>
    </div>
    <div class="about-intro">
      <div class="eyebrow">A STUDY IN CURIOSITY</div>
      <h2>Every master<br />was once a beginner.</h2>
      <p>
        The Fool is a chess-playing machine learning experiment inspired by the tarot's most curious card. It begins
        without certainty, explores without fear, and improves through every game it plays.
      </p>
      <p>
        Like the traveler stepping off the edge of the familiar, The Fool approaches chess as an endless landscape of
        possibilities.
      </p>
      <button class="text-link" onclick={() => onnav?.('watch')}>Follow its journey <span>→</span></button>
    </div>
  </section>

  <section class="about-columns">
    <article class="panel story-card">
      <span class="story-icon">✧</span>
      <div class="eyebrow">I · THE JOURNEY</div>
      <h3>Learning through play</h3>
      <p>
        Self-play gives The Fool a place to experiment. Each position is a question; each move, a hypothesis. Over
        time, patterns emerge from the games it has played.
      </p>
    </article>
    <article class="panel story-card">
      <span class="story-icon">☼</span>
      <div class="eyebrow">II · THE MIND</div>
      <h3>A model in progress</h3>
      <p>
        The long-term goal is to train a chess model, measure its progress, and make that process visible. The
        interface is designed to reveal the thinking and learning behind the moves.
      </p>
    </article>
    <article class="panel story-card">
      <span class="story-icon">☾</span>
      <div class="eyebrow">III · THE UNKNOWN</div>
      <h3>There is always more</h3>
      <p>
        Chess has more possible games than can be explored in a lifetime. The Fool's journey has no final
        destination—only new positions, new lessons, and another move.
      </p>
    </article>
  </section>

  <!-- FS-8b: the whole training story in plain English, zero knowledge assumed. -->
  <section class="panel learn-panel">
    <div class="eyebrow">THE TRAINING LOOP · IN PLAIN ENGLISH</div>
    <h3>How The Fool learns</h3>
    <p>
      The Fool learns chess the way the AlphaZero project taught a machine the game: it starts knowing only the
      rules, then plays thousands of quick games against itself. After each round it studies what happened — which
      guesses were wrong, which moves led to wins — and nudges its internal web of numbers (a neural network) a
      little closer to good chess. Every so often, a freshly trained version has to earn its title: it plays a short
      match against the reigning champion, and only takes over if it wins. Each promotion is recorded in the arena
      ledger you can see on the Watch page, along with the version's Elo rating — the same rating scale chess clubs
      use, where higher means stronger. Nothing about its strength is hand-set: every point was earned by beating an
      earlier version of itself. Training runs on this machine's own GPU, and the Watch page follows the journal as
      it happens.
    </p>
    <p class="learn-close">
      No chess knowledge needed to follow along — you can play the current champion at any time.
    </p>
    <button class="text-link" onclick={() => onnav?.('watch')}>Watch the training live <span>→</span></button>
  </section>

  <section class="panel status-strip">
    <div>
      <small>PROJECT STATUS</small>
      <strong>{elo !== null ? 'Training loop active' : 'Awaiting the first training run'}</strong>
    </div>
    <div>
      <small>MODEL STRENGTH</small>
      <strong>{elo !== null ? `Elo ${Math.round(elo)} · self-trained` : 'Not yet rated'}</strong>
    </div>
    <div>
      <small>THE NEXT STEP</small>
      <strong>Play the champion yourself</strong>
    </div>
  </section>
</main>
