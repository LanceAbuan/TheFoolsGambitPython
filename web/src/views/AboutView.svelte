<script lang="ts">
  interface Props {
    onnav?: (tab: 'play' | 'watch' | 'about') => void;
  }
  let { onnav }: Props = $props();
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

  <section class="panel status-strip">
    <div><small>PROJECT STATUS</small><strong>Client live · backend next</strong></div>
    <div><small>MODEL STRENGTH</small><strong>Not yet connected</strong></div>
    <div><small>THE NEXT STEP</small><strong>Begin training</strong></div>
  </section>
</main>
